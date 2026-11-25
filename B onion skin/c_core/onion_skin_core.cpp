#include "onion_skin_core.h"
#include "thread_pool.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <string>
#include <memory>
#include <mutex>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <future>
#include <atomic>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

// 64-bit FNV-1a Hash
uint64_t fnv1a_64(const char* str) {
    if (!str) return 0;
    uint64_t hash = 14695981039346656037ULL;
    while (*str) {
        hash ^= (uint8_t)(*str++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline uint64_t make_cache_key(uint64_t obj_id, int32_t frame) {
    // Murmur/SplitMix64 style combination
    uint64_t k = obj_id ^ (static_cast<uint64_t>(static_cast<uint32_t>(frame)) * 0x9e3779b97f4a7c15ULL);
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return k;
}

// Data storage for a cached or merged mesh
struct MeshData {
    uint64_t object_id = 0;
    std::vector<float> vertices;     // 3 * vertex_count
    std::vector<float> normals;      // 3 * vertex_count
    std::vector<int32_t> indices;    // index_count
    int32_t vertex_count = 0;
    int32_t index_count = 0;
    int32_t prim_type = ONION_PRIM_TRIS;
    float aabb_min[3] = { 1e30f,  1e30f,  1e30f};
    float aabb_max[3] = {-1e30f, -1e30f, -1e30f};
    float centroid[3] = {0.0f, 0.0f, 0.0f};

    void to_result(OnionMergedMeshResult* res) const {
        if (!res) return;
        res->vertices = vertices.empty() ? nullptr : vertices.data();
        res->normals = normals.empty() ? nullptr : normals.data();
        res->indices = indices.empty() ? nullptr : indices.data();
        res->vertex_count = vertex_count;
        res->index_count = index_count;
        res->prim_type = prim_type;
        std::memcpy(res->aabb_min, aabb_min, 3 * sizeof(float));
        std::memcpy(res->aabb_max, aabb_max, 3 * sizeof(float));
        std::memcpy(res->centroid, centroid, 3 * sizeof(float));
    }
};

struct GhostMarker {
    int32_t frame = 0;
    std::string tag;
    MeshData mesh;
};

// Global singleton state
std::mutex g_state_mutex;
MeshData g_last_merged_result; // Reusable buffer for immediate merge results

// Frame Cache
std::unordered_map<uint64_t, std::shared_ptr<MeshData>> g_frame_cache;
std::list<uint64_t> g_cache_lru;
std::unordered_map<uint64_t, std::list<uint64_t>::iterator> g_cache_lru_map;
size_t g_max_cache_entries = 500;

// Ghost Markers
std::vector<GhostMarker> g_ghost_markers;

// Thread pool support
std::atomic<bool> g_initialized{false};
unsigned int g_worker_threads = 4;

} // namespace

extern "C" {

uint64_t onion_hash_name(const char* name) {
    return fnv1a_64(name);
}

int onion_core_init(int thread_count) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (thread_count <= 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        g_worker_threads = hw > 0 ? std::min(hw, 32u) : 4;
    } else {
        g_worker_threads = static_cast<unsigned int>(thread_count);
    }
    g_initialized = true;
    return 1;
}

void onion_core_shutdown(void) {
    onion_fast_pool_shutdown();
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_frame_cache.clear();
    g_cache_lru.clear();
    g_cache_lru_map.clear();
    g_ghost_markers.clear();
    g_last_merged_result = MeshData();
    g_initialized = false;
}

// ---------------------------------------------------------------------------
// SIMD AVX2 / FMA Vector and Normal Transformation
// ---------------------------------------------------------------------------

static void transform_vertices_chunk(
    const float* in_v,
    float* out_v,
    int32_t count,
    const float* m,
    float* local_min,
    float* local_max,
    double* local_sum
) {
    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

    int32_t i = 0;

#if defined(__AVX2__) && defined(__FMA__)
    __m256 m00 = _mm256_set1_ps(m[0]);
    __m256 m01 = _mm256_set1_ps(m[1]);
    __m256 m02 = _mm256_set1_ps(m[2]);
    __m256 m03 = _mm256_set1_ps(m[3]);

    __m256 m10 = _mm256_set1_ps(m[4]);
    __m256 m11 = _mm256_set1_ps(m[5]);
    __m256 m12 = _mm256_set1_ps(m[6]);
    __m256 m13 = _mm256_set1_ps(m[7]);

    __m256 m20 = _mm256_set1_ps(m[8]);
    __m256 m21 = _mm256_set1_ps(m[9]);
    __m256 m22 = _mm256_set1_ps(m[10]);
    __m256 m23 = _mm256_set1_ps(m[11]);

    __m256 vmin_x = _mm256_set1_ps(1e30f);
    __m256 vmin_y = _mm256_set1_ps(1e30f);
    __m256 vmin_z = _mm256_set1_ps(1e30f);

    __m256 vmax_x = _mm256_set1_ps(-1e30f);
    __m256 vmax_y = _mm256_set1_ps(-1e30f);
    __m256 vmax_z = _mm256_set1_ps(-1e30f);

    __m256 vsum_x = _mm256_setzero_ps();
    __m256 vsum_y = _mm256_setzero_ps();
    __m256 vsum_z = _mm256_setzero_ps();

    // Process 8 vertices per iteration
    for (; i <= count - 8; i += 8) {
        // Load 8 vertices (24 floats) into 3 registers: vx, vy, vz
        // v0: [x0,y0,z0, x1,y1,z1, x2,y2]
        // v1: [z2, x3,y3,z3, x4,y4,z4, x5]
        // v2: [y5,z5, x6,y6,z6, x7,y7,z7]
        alignas(32) float vx_arr[8];
        alignas(32) float vy_arr[8];
        alignas(32) float vz_arr[8];

        for (int k = 0; k < 8; ++k) {
            vx_arr[k] = in_v[(i + k) * 3 + 0];
            vy_arr[k] = in_v[(i + k) * 3 + 1];
            vz_arr[k] = in_v[(i + k) * 3 + 2];
        }

        __m256 vx = _mm256_load_ps(vx_arr);
        __m256 vy = _mm256_load_ps(vy_arr);
        __m256 vz = _mm256_load_ps(vz_arr);

        // x' = m00*x + m01*y + m02*z + m03
        __m256 rx = _mm256_fmadd_ps(m00, vx, _mm256_fmadd_ps(m01, vy, _mm256_fmadd_ps(m02, vz, m03)));
        // y' = m10*x + m11*y + m12*z + m13
        __m256 ry = _mm256_fmadd_ps(m10, vx, _mm256_fmadd_ps(m11, vy, _mm256_fmadd_ps(m12, vz, m13)));
        // z' = m20*x + m21*y + m22*z + m23
        __m256 rz = _mm256_fmadd_ps(m20, vx, _mm256_fmadd_ps(m21, vy, _mm256_fmadd_ps(m22, vz, m23)));

        vmin_x = _mm256_min_ps(vmin_x, rx);
        vmin_y = _mm256_min_ps(vmin_y, ry);
        vmin_z = _mm256_min_ps(vmin_z, rz);

        vmax_x = _mm256_max_ps(vmax_x, rx);
        vmax_y = _mm256_max_ps(vmax_y, ry);
        vmax_z = _mm256_max_ps(vmax_z, rz);

        vsum_x = _mm256_add_ps(vsum_x, rx);
        vsum_y = _mm256_add_ps(vsum_y, ry);
        vsum_z = _mm256_add_ps(vsum_z, rz);

        _mm256_store_ps(vx_arr, rx);
        _mm256_store_ps(vy_arr, ry);
        _mm256_store_ps(vz_arr, rz);

        for (int k = 0; k < 8; ++k) {
            out_v[(i + k) * 3 + 0] = vx_arr[k];
            out_v[(i + k) * 3 + 1] = vy_arr[k];
            out_v[(i + k) * 3 + 2] = vz_arr[k];
        }
    }

    // Horizontal reduction for AVX results
    alignas(32) float red_min[8], red_max[8], red_sum[8];

    _mm256_store_ps(red_min, vmin_x);
    for (int k = 0; k < 8; ++k) min_x = std::min(min_x, red_min[k]);
    _mm256_store_ps(red_min, vmin_y);
    for (int k = 0; k < 8; ++k) min_y = std::min(min_y, red_min[k]);
    _mm256_store_ps(red_min, vmin_z);
    for (int k = 0; k < 8; ++k) min_z = std::min(min_z, red_min[k]);

    _mm256_store_ps(red_max, vmax_x);
    for (int k = 0; k < 8; ++k) max_x = std::max(max_x, red_max[k]);
    _mm256_store_ps(red_max, vmax_y);
    for (int k = 0; k < 8; ++k) max_y = std::max(max_y, red_max[k]);
    _mm256_store_ps(red_max, vmax_z);
    for (int k = 0; k < 8; ++k) max_z = std::max(max_z, red_max[k]);

    _mm256_store_ps(red_sum, vsum_x);
    for (int k = 0; k < 8; ++k) sum_x += red_sum[k];
    _mm256_store_ps(red_sum, vsum_y);
    for (int k = 0; k < 8; ++k) sum_y += red_sum[k];
    _mm256_store_ps(red_sum, vsum_z);
    for (int k = 0; k < 8; ++k) sum_z += red_sum[k];
#endif

    // Remaining tail elements
    for (; i < count; ++i) {
        float x = in_v[i * 3 + 0];
        float y = in_v[i * 3 + 1];
        float z = in_v[i * 3 + 2];

        float rx = m[0]*x + m[1]*y + m[2]*z + m[3];
        float ry = m[4]*x + m[5]*y + m[6]*z + m[7];
        float rz = m[8]*x + m[9]*y + m[10]*z + m[11];

        out_v[i * 3 + 0] = rx;
        out_v[i * 3 + 1] = ry;
        out_v[i * 3 + 2] = rz;

        min_x = std::min(min_x, rx);
        min_y = std::min(min_y, ry);
        min_z = std::min(min_z, rz);

        max_x = std::max(max_x, rx);
        max_y = std::max(max_y, ry);
        max_z = std::max(max_z, rz);

        sum_x += rx;
        sum_y += ry;
        sum_z += rz;
    }

    local_min[0] = min_x; local_min[1] = min_y; local_min[2] = min_z;
    local_max[0] = max_x; local_max[1] = max_y; local_max[2] = max_z;
    local_sum[0] = sum_x; local_sum[1] = sum_y; local_sum[2] = sum_z;
}

static void transform_normals_chunk(
    const float* in_n,
    float* out_n,
    int32_t count,
    const float* m
) {
    if (!in_n || !out_n || count <= 0) return;

    int32_t i = 0;

#if defined(__AVX2__) && defined(__FMA__)
    __m256 r00 = _mm256_set1_ps(m[0]);
    __m256 r01 = _mm256_set1_ps(m[1]);
    __m256 r02 = _mm256_set1_ps(m[2]);

    __m256 r10 = _mm256_set1_ps(m[4]);
    __m256 r11 = _mm256_set1_ps(m[5]);
    __m256 r12 = _mm256_set1_ps(m[6]);

    __m256 r20 = _mm256_set1_ps(m[8]);
    __m256 r21 = _mm256_set1_ps(m[9]);
    __m256 r22 = _mm256_set1_ps(m[10]);

    __m256 eps = _mm256_set1_ps(1e-12f);
    __m256 three = _mm256_set1_ps(3.0f);
    __m256 half = _mm256_set1_ps(0.5f);

    for (; i <= count - 8; i += 8) {
        alignas(32) float nx_arr[8], ny_arr[8], nz_arr[8];
        for (int k = 0; k < 8; ++k) {
            nx_arr[k] = in_n[(i + k) * 3 + 0];
            ny_arr[k] = in_n[(i + k) * 3 + 1];
            nz_arr[k] = in_n[(i + k) * 3 + 2];
        }

        __m256 nx = _mm256_load_ps(nx_arr);
        __m256 ny = _mm256_load_ps(ny_arr);
        __m256 nz = _mm256_load_ps(nz_arr);

        __m256 tx = _mm256_fmadd_ps(r00, nx, _mm256_fmadd_ps(r01, ny, _mm256_mul_ps(r02, nz)));
        __m256 ty = _mm256_fmadd_ps(r10, nx, _mm256_fmadd_ps(r11, ny, _mm256_mul_ps(r12, nz)));
        __m256 tz = _mm256_fmadd_ps(r20, nx, _mm256_fmadd_ps(r21, ny, _mm256_mul_ps(r22, nz)));

        // Fast normalization: rsqrt with Newton-Raphson refinement
        __m256 len2 = _mm256_add_ps(_mm256_fmadd_ps(tx, tx, _mm256_mul_ps(ty, ty)), _mm256_mul_ps(tz, tz));
        len2 = _mm256_max_ps(len2, eps);

        __m256 nr0 = _mm256_rsqrt_ps(len2);
        // nr1 = 0.5 * nr0 * (3.0 - len2 * nr0 * nr0)
        __m256 nr1 = _mm256_mul_ps(_mm256_mul_ps(half, nr0), _mm256_fnmadd_ps(_mm256_mul_ps(len2, nr0), nr0, three));

        tx = _mm256_mul_ps(tx, nr1);
        ty = _mm256_mul_ps(ty, nr1);
        tz = _mm256_mul_ps(tz, nr1);

        _mm256_store_ps(nx_arr, tx);
        _mm256_store_ps(ny_arr, ty);
        _mm256_store_ps(nz_arr, tz);

        for (int k = 0; k < 8; ++k) {
            out_n[(i + k) * 3 + 0] = nx_arr[k];
            out_n[(i + k) * 3 + 1] = ny_arr[k];
            out_n[(i + k) * 3 + 2] = nz_arr[k];
        }
    }
#endif

    for (; i < count; ++i) {
        float nx = in_n[i * 3 + 0];
        float ny = in_n[i * 3 + 1];
        float nz = in_n[i * 3 + 2];

        float tx = m[0]*nx + m[1]*ny + m[2]*nz;
        float ty = m[4]*nx + m[5]*ny + m[6]*nz;
        float tz = m[8]*nx + m[9]*ny + m[10]*nz;

        float len = std::sqrt(tx*tx + ty*ty + tz*tz);
        if (len > 1e-8f) {
            float inv = 1.0f / len;
            tx *= inv; ty *= inv; tz *= inv;
        }

        out_n[i * 3 + 0] = tx;
        out_n[i * 3 + 1] = ty;
        out_n[i * 3 + 2] = tz;
    }
}

static void offset_indices_chunk(
    const int32_t* in_idx,
    int32_t* out_idx,
    int32_t count,
    int32_t offset
) {
    if (!in_idx || !out_idx || count <= 0) return;
    int32_t i = 0;

#if defined(__AVX2__)
    __m256i v_offset = _mm256_set1_epi32(offset);
    for (; i <= count - 8; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_idx + i));
        v = _mm256_add_epi32(v, v_offset);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_idx + i), v);
    }
#endif

    for (; i < count; ++i) {
        out_idx[i] = in_idx[i] + offset;
    }
}

// ---------------------------------------------------------------------------
// Main Mesh Merger
// ---------------------------------------------------------------------------

int onion_core_merge_meshes(
    const OnionMeshChunkInput* chunks,
    int chunk_count,
    int wireframe,
    OnionMergedMeshResult* out_result
) {
    if (!chunks || chunk_count <= 0 || !out_result) return 0;

    std::lock_guard<std::mutex> lock(g_state_mutex);

    // 1. Calculate totals
    int32_t total_verts = 0;
    int32_t total_indices = 0;
    for (int i = 0; i < chunk_count; ++i) {
        if (chunks[i].vertex_count > 0 && chunks[i].index_count > 0 && chunks[i].vertices && chunks[i].indices) {
            total_verts += chunks[i].vertex_count;
            total_indices += chunks[i].index_count;
        }
    }

    if (total_verts <= 0 || total_indices <= 0) {
        return 0;
    }

    // 2. Pre-allocate flat memory in reusable buffer
    g_last_merged_result.vertices.resize(total_verts * 3);
    g_last_merged_result.indices.resize(total_indices);
    g_last_merged_result.vertex_count = total_verts;
    g_last_merged_result.index_count = total_indices;
    g_last_merged_result.prim_type = wireframe ? ONION_PRIM_LINES : ONION_PRIM_TRIS;

    if (!wireframe) {
        g_last_merged_result.normals.resize(total_verts * 3);
    } else {
        g_last_merged_result.normals.clear();
    }

    float* out_v = g_last_merged_result.vertices.data();
    float* out_n = wireframe ? nullptr : g_last_merged_result.normals.data();
    int32_t* out_idx = g_last_merged_result.indices.data();

    float global_min[3] = { 1e30f,  1e30f,  1e30f};
    float global_max[3] = {-1e30f, -1e30f, -1e30f};
    double global_sum[3] = {0.0, 0.0, 0.0};

    // 3. Transform and merge each chunk (multi-threaded via persistent FastThreadPool)
    struct ValidChunkInfo {
        int chunk_idx;
        int32_t v_off;
        int32_t idx_off;
        float c_min[3];
        float c_max[3];
        double c_sum[3];
    };
    std::vector<ValidChunkInfo> valid_infos;
    valid_infos.reserve(static_cast<size_t>(chunk_count));
    int32_t v_offset = 0;
    int32_t idx_offset = 0;
    for (int c = 0; c < chunk_count; ++c) {
        const auto& chunk = chunks[c];
        if (chunk.vertex_count <= 0 || chunk.index_count <= 0 || !chunk.vertices || !chunk.indices) {
            continue;
        }
        ValidChunkInfo info{};
        info.chunk_idx = c;
        info.v_off = v_offset;
        info.idx_off = idx_offset;
        valid_infos.push_back(info);
        v_offset += chunk.vertex_count;
        idx_offset += chunk.index_count;
    }

    auto process_valid_chunk = [&](size_t vi) {
        auto& info = valid_infos[vi];
        const auto& chunk = chunks[info.chunk_idx];
        transform_vertices_chunk(
            chunk.vertices,
            out_v + (static_cast<size_t>(info.v_off) * 3),
            chunk.vertex_count,
            chunk.matrix,
            info.c_min, info.c_max, info.c_sum
        );

        if (!wireframe && chunk.normals) {
            float norm_mat[16];
            std::memcpy(norm_mat, chunk.matrix, 16 * sizeof(float));
            const float m0 = chunk.matrix[0], m1 = chunk.matrix[1], m2 = chunk.matrix[2];
            const float m4 = chunk.matrix[4], m5 = chunk.matrix[5], m6 = chunk.matrix[6];
            const float m8 = chunk.matrix[8], m9 = chunk.matrix[9], m10 = chunk.matrix[10];
            const float c00 = m5 * m10 - m6 * m9;
            const float c01 = -(m4 * m10 - m6 * m8);
            const float c02 = m4 * m9 - m5 * m8;
            const float c10 = -(m1 * m10 - m2 * m9);
            const float c11 = m0 * m10 - m2 * m8;
            const float c12 = -(m0 * m9 - m1 * m8);
            const float c20 = m1 * m6 - m2 * m5;
            const float c21 = -(m0 * m6 - m2 * m4);
            const float c22 = m0 * m5 - m1 * m4;
            const float det = m0 * c00 + m1 * c01 + m2 * c02;
            if (std::fabs(det) > 1e-12f) {
                const float inv_det = 1.0f / det;
                norm_mat[0] = c00 * inv_det; norm_mat[1] = c01 * inv_det; norm_mat[2] = c02 * inv_det;
                norm_mat[4] = c10 * inv_det; norm_mat[5] = c11 * inv_det; norm_mat[6] = c12 * inv_det;
                norm_mat[8] = c20 * inv_det; norm_mat[9] = c21 * inv_det; norm_mat[10] = c22 * inv_det;
            }
            transform_normals_chunk(
                chunk.normals,
                out_n + (static_cast<size_t>(info.v_off) * 3),
                chunk.vertex_count,
                norm_mat
            );
        } else if (!wireframe && out_n) {
            const size_t n_base = static_cast<size_t>(info.v_off) * 3;
            for (int k = 0; k < chunk.vertex_count; ++k) {
                out_n[n_base + static_cast<size_t>(k) * 3 + 0] = 0.0f;
                out_n[n_base + static_cast<size_t>(k) * 3 + 1] = 0.0f;
                out_n[n_base + static_cast<size_t>(k) * 3 + 2] = 1.0f;
            }
        }

        offset_indices_chunk(
            chunk.indices,
            out_idx + info.idx_off,
            chunk.index_count,
            info.v_off
        );
    };

    if (total_verts >= 4096 && valid_infos.size() > 1) {
        onion_internal::FastThreadPool::instance().parallel_for_chunks(
            static_cast<int>(valid_infos.size()),
            [&](int c) { process_valid_chunk(static_cast<size_t>(c)); }
        );
    } else if (total_verts >= 4096 && valid_infos.size() == 1) {
        const auto& info = valid_infos[0];
        const auto& chunk = chunks[info.chunk_idx];
        const int hw = onion_internal::FastThreadPool::instance().hardware_threads();
        const int num_slices = std::min(hw, std::max(1, chunk.vertex_count / 2048));
        const int32_t v_step = (chunk.vertex_count + num_slices - 1) / num_slices;
        const int active_slices = (chunk.vertex_count + v_step - 1) / v_step;
        float norm_mat[16];
        if (!wireframe && chunk.normals) {
            std::memcpy(norm_mat, chunk.matrix, 16 * sizeof(float));
            const float m0 = chunk.matrix[0], m1 = chunk.matrix[1], m2 = chunk.matrix[2];
            const float m4 = chunk.matrix[4], m5 = chunk.matrix[5], m6 = chunk.matrix[6];
            const float m8 = chunk.matrix[8], m9 = chunk.matrix[9], m10 = chunk.matrix[10];
            const float c00 = m5 * m10 - m6 * m9;
            const float c01 = -(m4 * m10 - m6 * m8);
            const float c02 = m4 * m9 - m5 * m8;
            const float c10 = -(m1 * m10 - m2 * m9);
            const float c11 = m0 * m10 - m2 * m8;
            const float c12 = -(m0 * m9 - m1 * m8);
            const float c20 = m1 * m6 - m2 * m5;
            const float c21 = -(m0 * m6 - m2 * m4);
            const float c22 = m0 * m5 - m1 * m4;
            const float det = m0 * c00 + m1 * c01 + m2 * c02;
            if (std::fabs(det) > 1e-12f) {
                const float inv_det = 1.0f / det;
                norm_mat[0] = c00 * inv_det; norm_mat[1] = c01 * inv_det; norm_mat[2] = c02 * inv_det;
                norm_mat[4] = c10 * inv_det; norm_mat[5] = c11 * inv_det; norm_mat[6] = c12 * inv_det;
                norm_mat[8] = c20 * inv_det; norm_mat[9] = c21 * inv_det; norm_mat[10] = c22 * inv_det;
            }
        }
        struct SliceStats { float bmin[3], bmax[3]; double bsum[3]; };
        std::vector<SliceStats> stats(static_cast<size_t>(active_slices));
        const int32_t i_step = (chunk.index_count + active_slices - 1) / active_slices;
        onion_internal::FastThreadPool::instance().parallel_for_chunks(active_slices, [&](int s) {
            const int32_t v_s = s * v_step;
            const int32_t v_count_s = std::min(chunk.vertex_count, v_s + v_step) - v_s;
            auto& st = stats[static_cast<size_t>(s)];
            if (v_count_s > 0) {
                const size_t f_off = static_cast<size_t>(v_s) * 3;
                transform_vertices_chunk(chunk.vertices + f_off, out_v + f_off, v_count_s, chunk.matrix, st.bmin, st.bmax, st.bsum);
                if (!wireframe && chunk.normals) {
                    transform_normals_chunk(chunk.normals + f_off, out_n + f_off, v_count_s, norm_mat);
                } else if (!wireframe && out_n) {
                    for (int32_t k = 0; k < v_count_s; ++k) {
                        out_n[f_off + static_cast<size_t>(k) * 3 + 0] = 0.0f;
                        out_n[f_off + static_cast<size_t>(k) * 3 + 1] = 0.0f;
                        out_n[f_off + static_cast<size_t>(k) * 3 + 2] = 1.0f;
                    }
                }
            } else {
                st.bmin[0] = st.bmin[1] = st.bmin[2] = 1e30f;
                st.bmax[0] = st.bmax[1] = st.bmax[2] = -1e30f;
                st.bsum[0] = st.bsum[1] = st.bsum[2] = 0.0;
            }
            const int32_t i_s = s * i_step;
            const int32_t i_count_s = std::min(chunk.index_count, i_s + i_step) - i_s;
            if (i_count_s > 0) {
                offset_indices_chunk(chunk.indices + i_s, out_idx + i_s, i_count_s, 0);
            }
        });
        for (int s = 0; s < active_slices; ++s) {
            const auto& st = stats[static_cast<size_t>(s)];
            for (int k = 0; k < 3; ++k) {
                global_min[k] = std::min(global_min[k], st.bmin[k]);
                global_max[k] = std::max(global_max[k], st.bmax[k]);
                global_sum[k] += st.bsum[k];
            }
        }
        valid_infos.clear();
    } else {
        for (size_t vi = 0; vi < valid_infos.size(); ++vi) {
            process_valid_chunk(vi);
        }
    }

    for (const auto& info : valid_infos) {
        for (int k = 0; k < 3; ++k) {
            global_min[k] = std::min(global_min[k], info.c_min[k]);
            global_max[k] = std::max(global_max[k], info.c_max[k]);
            global_sum[k] += info.c_sum[k];
        }
    }

    // 4. Finalize AABB and Centroid
    std::memcpy(g_last_merged_result.aabb_min, global_min, 3 * sizeof(float));
    std::memcpy(g_last_merged_result.aabb_max, global_max, 3 * sizeof(float));

    if (total_verts > 0) {
        g_last_merged_result.centroid[0] = static_cast<float>(global_sum[0] / total_verts);
        g_last_merged_result.centroid[1] = static_cast<float>(global_sum[1] / total_verts);
        g_last_merged_result.centroid[2] = static_cast<float>(global_sum[2] / total_verts);
    } else {
        std::memset(g_last_merged_result.centroid, 0, 3 * sizeof(float));
    }

    g_last_merged_result.to_result(out_result);
    return 1;
}

// ---------------------------------------------------------------------------
// Hardware Accelerated View Frustum Culling
// ---------------------------------------------------------------------------

int onion_core_is_box_visible(
    const float* mvp,
    const float* aabb_min,
    const float* aabb_max
) {
    if (!mvp || !aabb_min || !aabb_max) return 1;

    // MVP matrix is row-major: 4 rows x 4 cols
    // Planes: Left, Right, Bottom, Top, Near, Far
    float planes[6][4];

    // Left: row 3 + row 0
    planes[0][0] = mvp[12] + mvp[0];
    planes[0][1] = mvp[13] + mvp[1];
    planes[0][2] = mvp[14] + mvp[2];
    planes[0][3] = mvp[15] + mvp[3];

    // Right: row 3 - row 0
    planes[1][0] = mvp[12] - mvp[0];
    planes[1][1] = mvp[13] - mvp[1];
    planes[1][2] = mvp[14] - mvp[2];
    planes[1][3] = mvp[15] - mvp[3];

    // Bottom: row 3 + row 1
    planes[2][0] = mvp[12] + mvp[4];
    planes[2][1] = mvp[13] + mvp[5];
    planes[2][2] = mvp[14] + mvp[6];
    planes[2][3] = mvp[15] + mvp[7];

    // Top: row 3 - row 1
    planes[3][0] = mvp[12] - mvp[4];
    planes[3][1] = mvp[13] - mvp[5];
    planes[3][2] = mvp[14] - mvp[6];
    planes[3][3] = mvp[15] - mvp[7];

    // Near: row 3 + row 2
    planes[4][0] = mvp[12] + mvp[8];
    planes[4][1] = mvp[13] + mvp[9];
    planes[4][2] = mvp[14] + mvp[10];
    planes[4][3] = mvp[15] + mvp[11];

    // Far: row 3 - row 2
    planes[5][0] = mvp[12] - mvp[8];
    planes[5][1] = mvp[13] - mvp[9];
    planes[5][2] = mvp[14] - mvp[10];
    planes[5][3] = mvp[15] - mvp[11];

    for (int p = 0; p < 6; ++p) {
        float px = (planes[p][0] > 0.0f) ? aabb_max[0] : aabb_min[0];
        float py = (planes[p][1] > 0.0f) ? aabb_max[1] : aabb_min[1];
        float pz = (planes[p][2] > 0.0f) ? aabb_max[2] : aabb_min[2];

        // Dot product of plane normal and p-vertex + plane distance
        float dist = planes[p][0] * px + planes[p][1] * py + planes[p][2] * pz + planes[p][3];
        if (dist < 0.0f) {
            // Box is completely outside this frustum plane
            return 0;
        }
    }

    return 1; // Box is inside or intersecting the frustum
}

// ---------------------------------------------------------------------------
// Vectorized Alpha Falloff
// ---------------------------------------------------------------------------

void onion_core_calc_alphas(
    const float* t_values,
    float* out_alphas,
    int count,
    float base_alpha,
    float falloff,
    int curve_type
) {
    if (!t_values || !out_alphas || count <= 0) return;

    for (int i = 0; i < count; ++i) {
        float t = t_values[i];
        float factor = 1.0f;

        if (curve_type == ONION_CURVE_LINEAR) {
            factor = 1.0f - t * falloff;
        } else if (curve_type == ONION_CURVE_EXPONENTIAL) {
            factor = std::pow(std::max(0.0f, 1.0f - t), 1.0f + falloff * 2.0f);
        } else { // Smooth (default)
            float ts = t * falloff;
            if (ts >= 1.0f) {
                factor = 0.1f;
            } else if (ts <= 0.0f) {
                factor = 1.0f;
            } else {
                factor = 1.0f - ts * ts * (3.0f - 2.0f * ts);
            }
        }

        out_alphas[i] = base_alpha * std::max(0.1f, factor);
    }
}

// ---------------------------------------------------------------------------
// Geometry Frame Cache
// ---------------------------------------------------------------------------

// Invalidate cache entries for a specific object
std::unordered_map<uint64_t, std::unordered_set<uint64_t>> g_obj_keys_map;

int onion_cache_has_frame(uint64_t object_id, int32_t frame) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    uint64_t key = make_cache_key(object_id, frame);
    return g_frame_cache.find(key) != g_frame_cache.end() ? 1 : 0;
}

int onion_cache_put_frame(uint64_t object_id, int32_t frame, const OnionMergedMeshResult* mesh_data) {
    if (!mesh_data) return 0;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    uint64_t key = make_cache_key(object_id, frame);

    auto entry = std::make_shared<MeshData>();
    entry->object_id = object_id;
    entry->vertex_count = mesh_data->vertex_count;
    entry->index_count = mesh_data->index_count;
    entry->prim_type = mesh_data->prim_type;

    if (mesh_data->vertices && mesh_data->vertex_count > 0) {
        entry->vertices.assign(mesh_data->vertices, mesh_data->vertices + (mesh_data->vertex_count * 3));
    }
    if (mesh_data->normals && mesh_data->vertex_count > 0) {
        entry->normals.assign(mesh_data->normals, mesh_data->normals + (mesh_data->vertex_count * 3));
    }
    if (mesh_data->indices && mesh_data->index_count > 0) {
        entry->indices.assign(mesh_data->indices, mesh_data->indices + mesh_data->index_count);
    }

    std::memcpy(entry->aabb_min, mesh_data->aabb_min, 3 * sizeof(float));
    std::memcpy(entry->aabb_max, mesh_data->aabb_max, 3 * sizeof(float));
    std::memcpy(entry->centroid, mesh_data->centroid, 3 * sizeof(float));

    // LRU management
    auto it = g_cache_lru_map.find(key);
    if (it != g_cache_lru_map.end()) {
        g_cache_lru.erase(it->second);
    }
    g_cache_lru.push_front(key);
    g_cache_lru_map[key] = g_cache_lru.begin();
    g_frame_cache[key] = entry;

    g_obj_keys_map[object_id].insert(key);

    // Prune if over max capacity
    while (g_cache_lru.size() > g_max_cache_entries) {
        uint64_t old_key = g_cache_lru.back();
        g_cache_lru.pop_back();
        g_cache_lru_map.erase(old_key);
        auto fit = g_frame_cache.find(old_key);
        if (fit != g_frame_cache.end()) {
            auto oit = g_obj_keys_map.find(fit->second->object_id);
            if (oit != g_obj_keys_map.end()) {
                oit->second.erase(old_key);
            }
            g_frame_cache.erase(fit);
        }
    }

    return 1;
}

int onion_cache_get_frame(uint64_t object_id, int32_t frame, OnionMergedMeshResult* out_result) {
    if (!out_result) return 0;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    uint64_t key = make_cache_key(object_id, frame);
    auto it = g_frame_cache.find(key);
    if (it == g_frame_cache.end()) {
        return 0;
    }

    // Move to front of LRU
    auto lru_it = g_cache_lru_map.find(key);
    if (lru_it != g_cache_lru_map.end()) {
        g_cache_lru.erase(lru_it->second);
        g_cache_lru.push_front(key);
        g_cache_lru_map[key] = g_cache_lru.begin();
    }

    it->second->to_result(out_result);
    return 1;
}

void onion_cache_invalidate_object(uint64_t object_id) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    auto it = g_obj_keys_map.find(object_id);
    if (it == g_obj_keys_map.end()) return;

    for (uint64_t key : it->second) {
        auto lru_it = g_cache_lru_map.find(key);
        if (lru_it != g_cache_lru_map.end()) {
            g_cache_lru.erase(lru_it->second);
            g_cache_lru_map.erase(lru_it);
        }
        g_frame_cache.erase(key);
    }
    g_obj_keys_map.erase(it);
}

void onion_cache_clear(void) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_frame_cache.clear();
    g_cache_lru.clear();
    g_cache_lru_map.clear();
    g_obj_keys_map.clear();
}

void onion_cache_set_max_entries(int max_entries) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_max_cache_entries = max_entries > 10 ? static_cast<size_t>(max_entries) : 10;
}

int onion_cache_entry_count(void) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return static_cast<int>(g_frame_cache.size());
}

// ---------------------------------------------------------------------------
// Ghost Marker System
// ---------------------------------------------------------------------------

int onion_marker_add(int32_t frame, const char* tag, const OnionMergedMeshResult* mesh_data) {
    if (!mesh_data) return 0;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    GhostMarker marker;
    marker.frame = frame;
    marker.tag = tag ? tag : "";
    marker.mesh.vertex_count = mesh_data->vertex_count;
    marker.mesh.index_count = mesh_data->index_count;
    marker.mesh.prim_type = mesh_data->prim_type;

    if (mesh_data->vertices && mesh_data->vertex_count > 0) {
        marker.mesh.vertices.assign(mesh_data->vertices, mesh_data->vertices + (mesh_data->vertex_count * 3));
    }
    if (mesh_data->normals && mesh_data->vertex_count > 0) {
        marker.mesh.normals.assign(mesh_data->normals, mesh_data->normals + (mesh_data->vertex_count * 3));
    }
    if (mesh_data->indices && mesh_data->index_count > 0) {
        marker.mesh.indices.assign(mesh_data->indices, mesh_data->indices + mesh_data->index_count);
    }

    std::memcpy(marker.mesh.aabb_min, mesh_data->aabb_min, 3 * sizeof(float));
    std::memcpy(marker.mesh.aabb_max, mesh_data->aabb_max, 3 * sizeof(float));
    std::memcpy(marker.mesh.centroid, mesh_data->centroid, 3 * sizeof(float));

    g_ghost_markers.push_back(std::move(marker));
    return static_cast<int>(g_ghost_markers.size());
}

int onion_marker_count(void) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return static_cast<int>(g_ghost_markers.size());
}

int onion_marker_get(int index, OnionMergedMeshResult* out_result, int* out_frame) {
    if (!out_result || index < 0) return 0;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (static_cast<size_t>(index) >= g_ghost_markers.size()) {
        return 0;
    }

    const auto& m = g_ghost_markers[index];
    m.mesh.to_result(out_result);
    if (out_frame) *out_frame = m.frame;
    return 1;
}

int onion_marker_get_centroid(int index, float* cx, float* cy, float* cz) {
    if (index < 0 || !cx || !cy || !cz) return 0;
    std::lock_guard<std::mutex> lock(g_state_mutex);

    if (static_cast<size_t>(index) >= g_ghost_markers.size()) {
        return 0;
    }

    const auto& m = g_ghost_markers[index];
    *cx = m.mesh.centroid[0];
    *cy = m.mesh.centroid[1];
    *cz = m.mesh.centroid[2];
    return 1;
}

static bool tags_intersect(const std::string& a, const std::string& b) {
    if (b.empty()) return true;
    size_t b_start = 0;
    while (b_start <= b.size()) {
        size_t b_end = b.find(',', b_start);
        if (b_end == std::string::npos) b_end = b.size();
        size_t b_len = b_end - b_start;
        if (b_len > 0) {
            size_t a_start = 0;
            while (a_start <= a.size()) {
                size_t a_end = a.find(',', a_start);
                if (a_end == std::string::npos) a_end = a.size();
                size_t a_len = a_end - a_start;
                if (a_len == b_len && a.compare(a_start, a_len, b, b_start, b_len) == 0) {
                    return true;
                }
                if (a_end == a.size()) break;
                a_start = a_end + 1;
            }
        }
        if (b_end == b.size()) break;
        b_start = b_end + 1;
    }
    return false;
}

int onion_marker_remove_for(int32_t frame, const char* tag) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    int removed = 0;
    std::string tag_str = tag ? tag : "";

    auto it = g_ghost_markers.begin();
    while (it != g_ghost_markers.end()) {
        bool match_frame = (frame == -2147483648) || (it->frame == frame);
        bool match_tag = tag_str.empty() || tags_intersect(it->tag, tag_str);

        if (match_frame && match_tag) {
            it = g_ghost_markers.erase(it);
            removed++;
        } else {
            ++it;
        }
    }

    return removed;
}

void onion_marker_clear(void) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_ghost_markers.clear();
}

} // extern "C"
