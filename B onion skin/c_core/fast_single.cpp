#include "onion_skin_core.h"
#include "thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using onion_internal::FastThreadPool;

constexpr int32_t PARALLEL_THRESHOLD = 4096;

struct RangeBounds {
    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
};

std::mutex g_normal_scratch_mtx;
std::vector<std::vector<float>> g_normal_scratch;

static inline bool is_identity_4x4(const float* m) {
    if (!m) return true;
    const float eps = 1e-6f;
    return (fabsf(m[0] - 1.0f) < eps && fabsf(m[1]) < eps && fabsf(m[2]) < eps && fabsf(m[3]) < eps &&
            fabsf(m[4]) < eps && fabsf(m[5] - 1.0f) < eps && fabsf(m[6]) < eps && fabsf(m[7]) < eps &&
            fabsf(m[8]) < eps && fabsf(m[9]) < eps && fabsf(m[10] - 1.0f) < eps && fabsf(m[11]) < eps &&
            fabsf(m[12]) < eps && fabsf(m[13]) < eps && fabsf(m[14]) < eps && fabsf(m[15] - 1.0f) < eps);
}

static void process_bounds_only_range(
    const float* in_verts,
    int32_t start,
    int32_t end,
    RangeBounds& out
) {
    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

    int32_t i = start;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 vmin_x = _mm256_set1_ps(1e30f);
    __m256 vmin_y = _mm256_set1_ps(1e30f);
    __m256 vmin_z = _mm256_set1_ps(1e30f);
    __m256 vmax_x = _mm256_set1_ps(-1e30f);
    __m256 vmax_y = _mm256_set1_ps(-1e30f);
    __m256 vmax_z = _mm256_set1_ps(-1e30f);
    __m256d vsum_x0 = _mm256_setzero_pd(), vsum_x1 = _mm256_setzero_pd();
    __m256d vsum_y0 = _mm256_setzero_pd(), vsum_y1 = _mm256_setzero_pd();
    __m256d vsum_z0 = _mm256_setzero_pd(), vsum_z1 = _mm256_setzero_pd();

    for (; i <= end - 8; i += 8) {
        alignas(32) float vx_arr[8], vy_arr[8], vz_arr[8];
        const float* p = in_verts + static_cast<size_t>(i) * 3;
        for (int k = 0; k < 8; ++k) {
            vx_arr[k] = p[k * 3];
            vy_arr[k] = p[k * 3 + 1];
            vz_arr[k] = p[k * 3 + 2];
        }
        __m256 vx = _mm256_load_ps(vx_arr);
        __m256 vy = _mm256_load_ps(vy_arr);
        __m256 vz = _mm256_load_ps(vz_arr);
        vmin_x = _mm256_min_ps(vmin_x, vx);
        vmin_y = _mm256_min_ps(vmin_y, vy);
        vmin_z = _mm256_min_ps(vmin_z, vz);
        vmax_x = _mm256_max_ps(vmax_x, vx);
        vmax_y = _mm256_max_ps(vmax_y, vy);
        vmax_z = _mm256_max_ps(vmax_z, vz);
        vsum_x0 = _mm256_add_pd(vsum_x0, _mm256_cvtps_pd(_mm256_castps256_ps128(vx)));
        vsum_x1 = _mm256_add_pd(vsum_x1, _mm256_cvtps_pd(_mm256_extractf128_ps(vx, 1)));
        vsum_y0 = _mm256_add_pd(vsum_y0, _mm256_cvtps_pd(_mm256_castps256_ps128(vy)));
        vsum_y1 = _mm256_add_pd(vsum_y1, _mm256_cvtps_pd(_mm256_extractf128_ps(vy, 1)));
        vsum_z0 = _mm256_add_pd(vsum_z0, _mm256_cvtps_pd(_mm256_castps256_ps128(vz)));
        vsum_z1 = _mm256_add_pd(vsum_z1, _mm256_cvtps_pd(_mm256_extractf128_ps(vz, 1)));
    }
    alignas(32) float rmin[8], rmax[8];
    alignas(32) double rsum[4];
    _mm256_store_ps(rmin, vmin_x); for (int k = 0; k < 8; ++k) if (rmin[k] < min_x) min_x = rmin[k];
    _mm256_store_ps(rmin, vmin_y); for (int k = 0; k < 8; ++k) if (rmin[k] < min_y) min_y = rmin[k];
    _mm256_store_ps(rmin, vmin_z); for (int k = 0; k < 8; ++k) if (rmin[k] < min_z) min_z = rmin[k];
    _mm256_store_ps(rmax, vmax_x); for (int k = 0; k < 8; ++k) if (rmax[k] > max_x) max_x = rmax[k];
    _mm256_store_ps(rmax, vmax_y); for (int k = 0; k < 8; ++k) if (rmax[k] > max_y) max_y = rmax[k];
    _mm256_store_ps(rmax, vmax_z); for (int k = 0; k < 8; ++k) if (rmax[k] > max_z) max_z = rmax[k];
    _mm256_store_pd(rsum, _mm256_add_pd(vsum_x0, vsum_x1)); for (int k = 0; k < 4; ++k) sum_x += rsum[k];
    _mm256_store_pd(rsum, _mm256_add_pd(vsum_y0, vsum_y1)); for (int k = 0; k < 4; ++k) sum_y += rsum[k];
    _mm256_store_pd(rsum, _mm256_add_pd(vsum_z0, vsum_z1)); for (int k = 0; k < 4; ++k) sum_z += rsum[k];
#endif

    for (; i < end; ++i) {
        const size_t base = static_cast<size_t>(i) * 3;
        const float x = in_verts[base];
        const float y = in_verts[base + 1];
        const float z = in_verts[base + 2];
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
        if (z < min_z) min_z = z;
        if (z > max_z) max_z = z;
        sum_x += static_cast<double>(x);
        sum_y += static_cast<double>(y);
        sum_z += static_cast<double>(z);
    }

    out.min_x = min_x; out.min_y = min_y; out.min_z = min_z;
    out.max_x = max_x; out.max_y = max_y; out.max_z = max_z;
    out.sum_x = sum_x; out.sum_y = sum_y; out.sum_z = sum_z;
}

static void process_transform_range(
    const float* in_verts,
    const float* in_normals,
    int32_t start,
    int32_t end,
    const float m[12],
    const float nm[9],
    float* out_verts,
    float* out_normals,
    RangeBounds& out
) {
    const float m0 = m[0], m1 = m[1], m2 = m[2], m3 = m[3];
    const float m4 = m[4], m5 = m[5], m6 = m[6], m7 = m[7];
    const float m8 = m[8], m9 = m[9], m10 = m[10], m11 = m[11];

    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

    int32_t i = start;
    const bool do_normals = (out_normals != nullptr && in_normals != nullptr);

#if defined(__AVX2__) && defined(__FMA__)
    __m256 vm0 = _mm256_set1_ps(m0), vm1 = _mm256_set1_ps(m1), vm2 = _mm256_set1_ps(m2), vm3 = _mm256_set1_ps(m3);
    __m256 vm4 = _mm256_set1_ps(m4), vm5 = _mm256_set1_ps(m5), vm6 = _mm256_set1_ps(m6), vm7 = _mm256_set1_ps(m7);
    __m256 vm8 = _mm256_set1_ps(m8), vm9 = _mm256_set1_ps(m9), vm10 = _mm256_set1_ps(m10), vm11 = _mm256_set1_ps(m11);

    __m256 vmin_x = _mm256_set1_ps(1e30f), vmin_y = _mm256_set1_ps(1e30f), vmin_z = _mm256_set1_ps(1e30f);
    __m256 vmax_x = _mm256_set1_ps(-1e30f), vmax_y = _mm256_set1_ps(-1e30f), vmax_z = _mm256_set1_ps(-1e30f);
    __m256d vsum_x0 = _mm256_setzero_pd(), vsum_x1 = _mm256_setzero_pd();
    __m256d vsum_y0 = _mm256_setzero_pd(), vsum_y1 = _mm256_setzero_pd();
    __m256d vsum_z0 = _mm256_setzero_pd(), vsum_z1 = _mm256_setzero_pd();

    __m256 vn0 = _mm256_set1_ps(nm[0]), vn1 = _mm256_set1_ps(nm[1]), vn2 = _mm256_set1_ps(nm[2]);
    __m256 vn3 = _mm256_set1_ps(nm[3]), vn4 = _mm256_set1_ps(nm[4]), vn5 = _mm256_set1_ps(nm[5]);
    __m256 vn6 = _mm256_set1_ps(nm[6]), vn7 = _mm256_set1_ps(nm[7]), vn8 = _mm256_set1_ps(nm[8]);
    __m256 veps = _mm256_set1_ps(1e-12f);
    __m256 vthree = _mm256_set1_ps(3.0f);
    __m256 vhalf = _mm256_set1_ps(0.5f);

    for (; i <= end - 8; i += 8) {
        alignas(32) float vx_arr[8], vy_arr[8], vz_arr[8];
        const size_t base = static_cast<size_t>(i) * 3;
        for (int k = 0; k < 8; ++k) {
            vx_arr[k] = in_verts[base + k * 3];
            vy_arr[k] = in_verts[base + k * 3 + 1];
            vz_arr[k] = in_verts[base + k * 3 + 2];
        }
        __m256 vx = _mm256_load_ps(vx_arr);
        __m256 vy = _mm256_load_ps(vy_arr);
        __m256 vz = _mm256_load_ps(vz_arr);

        __m256 tx = _mm256_fmadd_ps(vm0, vx, _mm256_fmadd_ps(vm1, vy, _mm256_fmadd_ps(vm2, vz, vm3)));
        __m256 ty = _mm256_fmadd_ps(vm4, vx, _mm256_fmadd_ps(vm5, vy, _mm256_fmadd_ps(vm6, vz, vm7)));
        __m256 tz = _mm256_fmadd_ps(vm8, vx, _mm256_fmadd_ps(vm9, vy, _mm256_fmadd_ps(vm10, vz, vm11)));

        vmin_x = _mm256_min_ps(vmin_x, tx);
        vmin_y = _mm256_min_ps(vmin_y, ty);
        vmin_z = _mm256_min_ps(vmin_z, tz);
        vmax_x = _mm256_max_ps(vmax_x, tx);
        vmax_y = _mm256_max_ps(vmax_y, ty);
        vmax_z = _mm256_max_ps(vmax_z, tz);

        vsum_x0 = _mm256_add_pd(vsum_x0, _mm256_cvtps_pd(_mm256_castps256_ps128(tx)));
        vsum_x1 = _mm256_add_pd(vsum_x1, _mm256_cvtps_pd(_mm256_extractf128_ps(tx, 1)));
        vsum_y0 = _mm256_add_pd(vsum_y0, _mm256_cvtps_pd(_mm256_castps256_ps128(ty)));
        vsum_y1 = _mm256_add_pd(vsum_y1, _mm256_cvtps_pd(_mm256_extractf128_ps(ty, 1)));
        vsum_z0 = _mm256_add_pd(vsum_z0, _mm256_cvtps_pd(_mm256_castps256_ps128(tz)));
        vsum_z1 = _mm256_add_pd(vsum_z1, _mm256_cvtps_pd(_mm256_extractf128_ps(tz, 1)));

        _mm256_store_ps(vx_arr, tx);
        _mm256_store_ps(vy_arr, ty);
        _mm256_store_ps(vz_arr, tz);
        for (int k = 0; k < 8; ++k) {
            out_verts[base + k * 3] = vx_arr[k];
            out_verts[base + k * 3 + 1] = vy_arr[k];
            out_verts[base + k * 3 + 2] = vz_arr[k];
        }

        if (do_normals) {
            alignas(32) float nx_arr[8], ny_arr[8], nz_arr[8];
            for (int k = 0; k < 8; ++k) {
                nx_arr[k] = in_normals[base + k * 3];
                ny_arr[k] = in_normals[base + k * 3 + 1];
                nz_arr[k] = in_normals[base + k * 3 + 2];
            }
            __m256 nx = _mm256_load_ps(nx_arr);
            __m256 ny = _mm256_load_ps(ny_arr);
            __m256 nz = _mm256_load_ps(nz_arr);

            __m256 tnx = _mm256_fmadd_ps(vn0, nx, _mm256_fmadd_ps(vn1, ny, _mm256_mul_ps(vn2, nz)));
            __m256 tny = _mm256_fmadd_ps(vn3, nx, _mm256_fmadd_ps(vn4, ny, _mm256_mul_ps(vn5, nz)));
            __m256 tnz = _mm256_fmadd_ps(vn6, nx, _mm256_fmadd_ps(vn7, ny, _mm256_mul_ps(vn8, nz)));

            __m256 len2 = _mm256_add_ps(_mm256_fmadd_ps(tnx, tnx, _mm256_mul_ps(tny, tny)), _mm256_mul_ps(tnz, tnz));
            __m256 valid_mask = _mm256_cmp_ps(len2, veps, _CMP_GT_OQ);
            __m256 safe_len2 = _mm256_max_ps(len2, veps);
            __m256 nr0 = _mm256_rsqrt_ps(safe_len2);
            __m256 nr1 = _mm256_mul_ps(_mm256_mul_ps(vhalf, nr0), _mm256_fnmadd_ps(_mm256_mul_ps(safe_len2, nr0), nr0, vthree));
            nr1 = _mm256_and_ps(nr1, valid_mask);

            _mm256_store_ps(nx_arr, _mm256_mul_ps(tnx, nr1));
            _mm256_store_ps(ny_arr, _mm256_mul_ps(tny, nr1));
            _mm256_store_ps(nz_arr, _mm256_mul_ps(tnz, nr1));
            for (int k = 0; k < 8; ++k) {
                out_normals[base + k * 3] = nx_arr[k];
                out_normals[base + k * 3 + 1] = ny_arr[k];
                out_normals[base + k * 3 + 2] = nz_arr[k];
            }
        }
    }

    alignas(32) float rmin[8], rmax[8];
    alignas(32) double rsum[4];
    _mm256_store_ps(rmin, vmin_x); for (int k = 0; k < 8; ++k) if (rmin[k] < min_x) min_x = rmin[k];
    _mm256_store_ps(rmin, vmin_y); for (int k = 0; k < 8; ++k) if (rmin[k] < min_y) min_y = rmin[k];
    _mm256_store_ps(rmin, vmin_z); for (int k = 0; k < 8; ++k) if (rmin[k] < min_z) min_z = rmin[k];
    _mm256_store_ps(rmax, vmax_x); for (int k = 0; k < 8; ++k) if (rmax[k] > max_x) max_x = rmax[k];
    _mm256_store_ps(rmax, vmax_y); for (int k = 0; k < 8; ++k) if (rmax[k] > max_y) max_y = rmax[k];
    _mm256_store_ps(rmax, vmax_z); for (int k = 0; k < 8; ++k) if (rmax[k] > max_z) max_z = rmax[k];
    _mm256_store_pd(rsum, _mm256_add_pd(vsum_x0, vsum_x1)); for (int k = 0; k < 4; ++k) sum_x += rsum[k];
    _mm256_store_pd(rsum, _mm256_add_pd(vsum_y0, vsum_y1)); for (int k = 0; k < 4; ++k) sum_y += rsum[k];
    _mm256_store_pd(rsum, _mm256_add_pd(vsum_z0, vsum_z1)); for (int k = 0; k < 4; ++k) sum_z += rsum[k];
#endif

    if (do_normals) {
        const float n0 = nm[0], n1 = nm[1], n2 = nm[2];
        const float n3 = nm[3], n4 = nm[4], n5 = nm[5];
        const float n6 = nm[6], n7 = nm[7], n8 = nm[8];

        for (; i < end; ++i) {
            const size_t base = static_cast<size_t>(i) * 3;
            const float x = in_verts[base];
            const float y = in_verts[base + 1];
            const float z = in_verts[base + 2];

            const float tx = m0 * x + m1 * y + m2 * z + m3;
            const float ty = m4 * x + m5 * y + m6 * z + m7;
            const float tz = m8 * x + m9 * y + m10 * z + m11;

            out_verts[base] = tx;
            out_verts[base + 1] = ty;
            out_verts[base + 2] = tz;

            if (tx < min_x) min_x = tx;
            if (tx > max_x) max_x = tx;
            if (ty < min_y) min_y = ty;
            if (ty > max_y) max_y = ty;
            if (tz < min_z) min_z = tz;
            if (tz > max_z) max_z = tz;
            sum_x += static_cast<double>(tx);
            sum_y += static_cast<double>(ty);
            sum_z += static_cast<double>(tz);

            const float nx = in_normals[base];
            const float ny = in_normals[base + 1];
            const float nz = in_normals[base + 2];

            float tnx = n0 * nx + n1 * ny + n2 * nz;
            float tny = n3 * nx + n4 * ny + n5 * nz;
            float tnz = n6 * nx + n7 * ny + n8 * nz;
            const float lsq = tnx * tnx + tny * tny + tnz * tnz;
            if (lsq > 1e-12f) {
                const float inv_l = 1.0f / sqrtf(lsq);
                tnx *= inv_l;
                tny *= inv_l;
                tnz *= inv_l;
            }
            out_normals[base] = tnx;
            out_normals[base + 1] = tny;
            out_normals[base + 2] = tnz;
        }
    } else {
        for (; i < end; ++i) {
            const size_t base = static_cast<size_t>(i) * 3;
            const float x = in_verts[base];
            const float y = in_verts[base + 1];
            const float z = in_verts[base + 2];

            const float tx = m0 * x + m1 * y + m2 * z + m3;
            const float ty = m4 * x + m5 * y + m6 * z + m7;
            const float tz = m8 * x + m9 * y + m10 * z + m11;

            out_verts[base] = tx;
            out_verts[base + 1] = ty;
            out_verts[base + 2] = tz;

            if (tx < min_x) min_x = tx;
            if (tx > max_x) max_x = tx;
            if (ty < min_y) min_y = ty;
            if (ty > max_y) max_y = ty;
            if (tz < min_z) min_z = tz;
            if (tz > max_z) max_z = tz;
            sum_x += static_cast<double>(tx);
            sum_y += static_cast<double>(ty);
            sum_z += static_cast<double>(tz);
        }
    }

    out.min_x = min_x; out.min_y = min_y; out.min_z = min_z;
    out.max_x = max_x; out.max_y = max_y; out.max_z = max_z;
    out.sum_x = sum_x; out.sum_y = sum_y; out.sum_z = sum_z;
}

}  // namespace

extern "C" void onion_fast_pool_shutdown(void) {
    FastThreadPool::instance().shutdown();
    std::lock_guard<std::mutex> lock(g_normal_scratch_mtx);
    std::vector<std::vector<float>>().swap(g_normal_scratch);
}

extern "C" ONION_API int onion_core_get_thread_count(void) {
    return FastThreadPool::instance().hardware_threads();
}

extern "C" ONION_API int onion_core_transform_single_mesh(
    const float* in_verts,
    const float* in_normals,
    int32_t vertex_count,
    const float* matrix_row_major,
    float* out_verts,
    float* out_normals,
    float* out_aabb_min,
    float* out_aabb_max,
    float* out_centroid
) {
    if (!in_verts || vertex_count <= 0 || !out_aabb_min || !out_aabb_max || !out_centroid) {
        return 0;
    }

    const bool ident = is_identity_4x4(matrix_row_major);
    const bool bounds_only = (!out_verts) || (ident && out_verts == in_verts && (!out_normals || out_normals == in_normals));

    float m[12];
    float nm[9];
    if (!bounds_only) {
        if (ident) {
            m[0] = 1.0f; m[1] = 0.0f; m[2] = 0.0f; m[3] = 0.0f;
            m[4] = 0.0f; m[5] = 1.0f; m[6] = 0.0f; m[7] = 0.0f;
            m[8] = 0.0f; m[9] = 0.0f; m[10] = 1.0f; m[11] = 0.0f;
            nm[0] = 1.0f; nm[1] = 0.0f; nm[2] = 0.0f;
            nm[3] = 0.0f; nm[4] = 1.0f; nm[5] = 0.0f;
            nm[6] = 0.0f; nm[7] = 0.0f; nm[8] = 1.0f;
        } else {
            for (int i = 0; i < 12; ++i) m[i] = matrix_row_major[i];
            const float m0 = m[0], m1 = m[1], m2 = m[2];
            const float m4 = m[4], m5 = m[5], m6 = m[6];
            const float m8 = m[8], m9 = m[9], m10 = m[10];
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
            if (fabsf(det) > 1e-12f) {
                const float inv_det = 1.0f / det;
                nm[0] = c00 * inv_det; nm[1] = c01 * inv_det; nm[2] = c02 * inv_det;
                nm[3] = c10 * inv_det; nm[4] = c11 * inv_det; nm[5] = c12 * inv_det;
                nm[6] = c20 * inv_det; nm[7] = c21 * inv_det; nm[8] = c22 * inv_det;
            } else {
                nm[0] = m0; nm[1] = m1; nm[2] = m2;
                nm[3] = m4; nm[4] = m5; nm[5] = m6;
                nm[6] = m8; nm[7] = m9; nm[8] = m10;
            }
        }
    }

    int num_chunks = 1;
    if (vertex_count >= PARALLEL_THRESHOLD) {
        num_chunks = FastThreadPool::instance().hardware_threads();
        const int max_by_size = vertex_count / 2048;
        if (num_chunks > max_by_size) num_chunks = std::max(1, max_by_size);
    }

    RangeBounds total;
    if (num_chunks == 1) {
        if (bounds_only) {
            process_bounds_only_range(in_verts, 0, vertex_count, total);
        } else {
            process_transform_range(in_verts, in_normals, 0, vertex_count, m, nm, out_verts, out_normals, total);
        }
    } else {
        std::vector<RangeBounds> partials(static_cast<size_t>(num_chunks));
        const int32_t chunk = (vertex_count + num_chunks - 1) / num_chunks;
        const int active_chunks = (vertex_count + chunk - 1) / chunk;

        FastThreadPool::instance().parallel_for_chunks(active_chunks, [&](int c) {
            const int32_t s = c * chunk;
            const int32_t e = std::min(vertex_count, s + chunk);
            if (s >= e) return;
            if (bounds_only) {
                process_bounds_only_range(in_verts, s, e, partials[static_cast<size_t>(c)]);
            } else {
                process_transform_range(in_verts, in_normals, s, e, m, nm, out_verts, out_normals, partials[static_cast<size_t>(c)]);
            }
        });

        for (int c = 0; c < active_chunks; ++c) {
            const auto& p = partials[static_cast<size_t>(c)];
            if (p.min_x < total.min_x) total.min_x = p.min_x;
            if (p.min_y < total.min_y) total.min_y = p.min_y;
            if (p.min_z < total.min_z) total.min_z = p.min_z;
            if (p.max_x > total.max_x) total.max_x = p.max_x;
            if (p.max_y > total.max_y) total.max_y = p.max_y;
            if (p.max_z > total.max_z) total.max_z = p.max_z;
            total.sum_x += p.sum_x;
            total.sum_y += p.sum_y;
            total.sum_z += p.sum_z;
        }
    }

    out_aabb_min[0] = total.min_x;
    out_aabb_min[1] = total.min_y;
    out_aabb_min[2] = total.min_z;
    out_aabb_max[0] = total.max_x;
    out_aabb_max[1] = total.max_y;
    out_aabb_max[2] = total.max_z;
    const double inv_n = 1.0 / static_cast<double>(vertex_count);
    out_centroid[0] = static_cast<float>(total.sum_x * inv_n);
    out_centroid[1] = static_cast<float>(total.sum_y * inv_n);
    out_centroid[2] = static_cast<float>(total.sum_z * inv_n);
    return 1;
}

static inline bool project_point_to_screen(
    const float* mvp,
    float vp_w,
    float vp_h,
    float x,
    float y,
    float z,
    float& out_px,
    float& out_py
) {
    const float cx = mvp[0] * x + mvp[1] * y + mvp[2] * z + mvp[3];
    const float cy = mvp[4] * x + mvp[5] * y + mvp[6] * z + mvp[7];
    const float cw = mvp[12] * x + mvp[13] * y + mvp[14] * z + mvp[15];
    if (fabsf(cw) < 1e-6f) {
        out_px = 0.0f;
        out_py = 0.0f;
        return false;
    }
    const float inv_w = 1.0f / cw;
    const float ndc_x = cx * inv_w;
    const float ndc_y = cy * inv_w;
    out_px = (ndc_x * 0.5f + 0.5f) * vp_w;
    out_py = (ndc_y * 0.5f + 0.5f) * vp_h;
    return (cw > -1e-4f);
}

extern "C" ONION_API int onion_core_filter_screen_space(
    const float* mvp_matrix_row_major,
    float vp_width,
    float vp_height,
    const float* ref_centroid,
    const float* candidate_centroids,
    int32_t count,
    float min_pixel_dist,
    int32_t* out_keep_mask,
    float* out_screen_coords
) {
    if (!mvp_matrix_row_major || !candidate_centroids || count <= 0 || !out_keep_mask) {
        return 0;
    }
    if (vp_width <= 1.0f) vp_width = 1920.0f;
    if (vp_height <= 1.0f) vp_height = 1080.0f;

    const float min_dist = (min_pixel_dist > 0.0f) ? min_pixel_dist : 0.0f;
    const float min_dist_sq = min_dist * min_dist;

    float last_px = 0.0f;
    float last_py = 0.0f;
    bool has_anchor = false;

    if (ref_centroid) {
        has_anchor = project_point_to_screen(
            mvp_matrix_row_major, vp_width, vp_height,
            ref_centroid[0], ref_centroid[1], ref_centroid[2],
            last_px, last_py
        );
    }

    int kept_count = 0;
    int best_idx = -1;
    float best_dist_sq = -1.0f;

    for (int32_t i = 0; i < count; ++i) {
        const size_t base = static_cast<size_t>(i) * 3;
        float px = 0.0f, py = 0.0f;
        const bool ok = project_point_to_screen(
            mvp_matrix_row_major, vp_width, vp_height,
            candidate_centroids[base],
            candidate_centroids[base + 1],
            candidate_centroids[base + 2],
            px, py
        );
        if (out_screen_coords) {
            out_screen_coords[i * 2] = px;
            out_screen_coords[i * 2 + 1] = py;
        }
        if (!ok) {
            out_keep_mask[i] = 0;
            continue;
        }
        if (!has_anchor) {
            out_keep_mask[i] = 1;
            last_px = px;
            last_py = py;
            has_anchor = true;
            ++kept_count;
            continue;
        }
        const float dx = px - last_px;
        const float dy = py - last_py;
        const float dsq = dx * dx + dy * dy;
        if (dsq > best_dist_sq) {
            best_dist_sq = dsq;
            best_idx = i;
        }
        if (dsq >= min_dist_sq) {
            out_keep_mask[i] = 1;
            last_px = px;
            last_py = py;
            ++kept_count;
        } else {
            out_keep_mask[i] = 0;
        }
    }

    if (kept_count == 0 && best_idx >= 0 && best_dist_sq >= 0.0625f) {
        out_keep_mask[best_idx] = 1;
        kept_count = 1;
    }

    return kept_count;
}

static inline void accumulate_tri_normals_range(
    const float* verts,
    int32_t vertex_count,
    const int32_t* tri_indices,
    int32_t t_start,
    int32_t t_end,
    float* accum
) {
    for (int32_t t = t_start; t < t_end; ++t) {
        const size_t tb = static_cast<size_t>(t) * 3;
        const int32_t i0 = tri_indices[tb];
        const int32_t i1 = tri_indices[tb + 1];
        const int32_t i2 = tri_indices[tb + 2];
        if (static_cast<uint32_t>(i0) >= static_cast<uint32_t>(vertex_count) ||
            static_cast<uint32_t>(i1) >= static_cast<uint32_t>(vertex_count) ||
            static_cast<uint32_t>(i2) >= static_cast<uint32_t>(vertex_count)) {
            continue;
        }
        const float* p0 = verts + static_cast<size_t>(i0) * 3;
        const float* p1 = verts + static_cast<size_t>(i1) * 3;
        const float* p2 = verts + static_cast<size_t>(i2) * 3;
        const float e1x = p1[0] - p0[0], e1y = p1[1] - p0[1], e1z = p1[2] - p0[2];
        const float e2x = p2[0] - p0[0], e2y = p2[1] - p0[1], e2z = p2[2] - p0[2];
        const float fnx = e1y * e2z - e1z * e2y;
        const float fny = e1z * e2x - e1x * e2z;
        const float fnz = e1x * e2y - e1y * e2x;

        float* o0 = accum + static_cast<size_t>(i0) * 3;
        float* o1 = accum + static_cast<size_t>(i1) * 3;
        float* o2 = accum + static_cast<size_t>(i2) * 3;
        o0[0] += fnx; o0[1] += fny; o0[2] += fnz;
        o1[0] += fnx; o1[1] += fny; o1[2] += fnz;
        o2[0] += fnx; o2[1] += fny; o2[2] += fnz;
    }
}

static inline void normalize_normals_range(float* normals, int32_t v_start, int32_t v_end) {
    int32_t i = v_start;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 veps = _mm256_set1_ps(1e-12f);
    const __m256 vthree = _mm256_set1_ps(3.0f);
    const __m256 vhalf = _mm256_set1_ps(0.5f);
    for (; i <= v_end - 8; i += 8) {
        alignas(32) float nx_arr[8], ny_arr[8], nz_arr[8];
        const size_t base = static_cast<size_t>(i) * 3;
        for (int k = 0; k < 8; ++k) {
            nx_arr[k] = normals[base + k * 3];
            ny_arr[k] = normals[base + k * 3 + 1];
            nz_arr[k] = normals[base + k * 3 + 2];
        }
        __m256 nx = _mm256_load_ps(nx_arr);
        __m256 ny = _mm256_load_ps(ny_arr);
        __m256 nz = _mm256_load_ps(nz_arr);
        __m256 len2 = _mm256_add_ps(_mm256_fmadd_ps(nx, nx, _mm256_mul_ps(ny, ny)), _mm256_mul_ps(nz, nz));
        __m256 valid_mask = _mm256_cmp_ps(len2, veps, _CMP_GT_OQ);
        __m256 safe_len2 = _mm256_max_ps(len2, veps);
        __m256 nr0 = _mm256_rsqrt_ps(safe_len2);
        __m256 nr1 = _mm256_mul_ps(_mm256_mul_ps(vhalf, nr0), _mm256_fnmadd_ps(_mm256_mul_ps(safe_len2, nr0), nr0, vthree));
        nr1 = _mm256_and_ps(nr1, valid_mask);

        __m256 onx = _mm256_mul_ps(nx, nr1);
        __m256 ony = _mm256_mul_ps(ny, nr1);
        __m256 onz = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(nz, nr1), valid_mask);

        _mm256_store_ps(nx_arr, onx);
        _mm256_store_ps(ny_arr, ony);
        _mm256_store_ps(nz_arr, onz);
        for (int k = 0; k < 8; ++k) {
            normals[base + k * 3] = nx_arr[k];
            normals[base + k * 3 + 1] = ny_arr[k];
            normals[base + k * 3 + 2] = nz_arr[k];
        }
    }
#endif
    for (; i < v_end; ++i) {
        const size_t base = static_cast<size_t>(i) * 3;
        float nx = normals[base];
        float ny = normals[base + 1];
        float nz = normals[base + 2];
        const float lsq = nx * nx + ny * ny + nz * nz;
        if (lsq > 1e-12f) {
            const float inv = 1.0f / sqrtf(lsq);
            normals[base] = nx * inv;
            normals[base + 1] = ny * inv;
            normals[base + 2] = nz * inv;
        } else {
            normals[base] = 0.0f;
            normals[base + 1] = 0.0f;
            normals[base + 2] = 1.0f;
        }
    }
}

extern "C" ONION_API int onion_core_compute_normals(
    const float* verts,
    int32_t vertex_count,
    const int32_t* tri_indices,
    int32_t tri_count,
    float* out_normals
) {
    if (!verts || !tri_indices || !out_normals || vertex_count <= 0 || tri_count <= 0) {
        return 0;
    }

    const size_t total_floats = static_cast<size_t>(vertex_count) * 3;
    if (tri_count < 8192) {
        std::memset(out_normals, 0, total_floats * sizeof(float));
        accumulate_tri_normals_range(verts, vertex_count, tri_indices, 0, tri_count, out_normals);
        normalize_normals_range(out_normals, 0, vertex_count);
        return 1;
    }

    std::lock_guard<std::mutex> lock(g_normal_scratch_mtx);
    const int hw = FastThreadPool::instance().hardware_threads();
    const int num_scratch = std::clamp(hw / 2, 2, 6);
    if (static_cast<int>(g_normal_scratch.size()) < num_scratch) {
        g_normal_scratch.resize(static_cast<size_t>(num_scratch));
    }
    for (int k = 0; k < num_scratch; ++k) {
        if (g_normal_scratch[static_cast<size_t>(k)].size() < total_floats) {
            g_normal_scratch[static_cast<size_t>(k)].resize(total_floats);
        }
    }

    // Pass 1: Lock-free parallel face-normal accumulation across K thread-local buffers
    const int32_t tri_chunk = (tri_count + num_scratch - 1) / num_scratch;
    const int active_pass1 = (tri_count + tri_chunk - 1) / tri_chunk;
    FastThreadPool::instance().parallel_for_chunks(active_pass1, [&](int c) {
        float* buf = g_normal_scratch[static_cast<size_t>(c)].data();
        std::memset(buf, 0, total_floats * sizeof(float));
        const int32_t s = c * tri_chunk;
        const int32_t e = std::min(tri_count, s + tri_chunk);
        accumulate_tri_normals_range(verts, vertex_count, tri_indices, s, e, buf);
    });

    // Pass 2: Parallel sum of K buffers + AVX2 rsqrt normal normalization across all hardware threads
    const int num_norm_chunks = std::min(hw, std::max(1, vertex_count / 2048));
    const int32_t v_chunk = (vertex_count + num_norm_chunks - 1) / num_norm_chunks;
    const int active_pass2 = (vertex_count + v_chunk - 1) / v_chunk;
    FastThreadPool::instance().parallel_for_chunks(active_pass2, [&](int c) {
        const int32_t v_s = c * v_chunk;
        const int32_t v_e = std::min(vertex_count, v_s + v_chunk);
        if (v_s >= v_e) return;
        const size_t f_s = static_cast<size_t>(v_s) * 3;
        const size_t f_e = static_cast<size_t>(v_e) * 3;
        const float* src0 = g_normal_scratch[0].data();
        std::memcpy(out_normals + f_s, src0 + f_s, (f_e - f_s) * sizeof(float));
        for (int k = 1; k < active_pass1; ++k) {
            const float* sk = g_normal_scratch[static_cast<size_t>(k)].data();
            size_t j = f_s;
#if defined(__AVX2__)
            for (; j + 8 <= f_e; j += 8) {
                __m256 a = _mm256_loadu_ps(out_normals + j);
                __m256 b = _mm256_loadu_ps(sk + j);
                _mm256_storeu_ps(out_normals + j, _mm256_add_ps(a, b));
            }
#endif
            for (; j < f_e; ++j) {
                out_normals[j] += sk[j];
            }
        }
        normalize_normals_range(out_normals, v_s, v_e);
    });
    return 1;
}

extern "C" ONION_API int onion_core_triangulate_polys(
    const int32_t* loop_verts,
    const int32_t* loop_totals,
    int32_t poly_count,
    int32_t* out_tri_indices
) {
    if (!loop_verts || !loop_totals || !out_tri_indices || poly_count <= 0) {
        return 0;
    }

    auto triangulate_slice = [&](int32_t p_start, int32_t p_end, int32_t lv_off, int32_t tri_off) {
        size_t cur_lv = static_cast<size_t>(lv_off);
        size_t cur_out = static_cast<size_t>(tri_off);
        for (int32_t p = p_start; p < p_end; ++p) {
            const int32_t lt = loop_totals[p];
            if (lt == 3) {
                out_tri_indices[cur_out] = loop_verts[cur_lv];
                out_tri_indices[cur_out + 1] = loop_verts[cur_lv + 1];
                out_tri_indices[cur_out + 2] = loop_verts[cur_lv + 2];
                cur_out += 3;
            } else if (lt == 4) {
                const int32_t v0 = loop_verts[cur_lv];
                const int32_t v1 = loop_verts[cur_lv + 1];
                const int32_t v2 = loop_verts[cur_lv + 2];
                const int32_t v3 = loop_verts[cur_lv + 3];
                out_tri_indices[cur_out] = v0;
                out_tri_indices[cur_out + 1] = v1;
                out_tri_indices[cur_out + 2] = v2;
                out_tri_indices[cur_out + 3] = v0;
                out_tri_indices[cur_out + 4] = v2;
                out_tri_indices[cur_out + 5] = v3;
                cur_out += 6;
            } else if (lt > 4) {
                const int32_t v0 = loop_verts[cur_lv];
                for (int32_t k = 1; k < lt - 1; ++k) {
                    out_tri_indices[cur_out] = v0;
                    out_tri_indices[cur_out + 1] = loop_verts[cur_lv + k];
                    out_tri_indices[cur_out + 2] = loop_verts[cur_lv + k + 1];
                    cur_out += 3;
                }
            }
            cur_lv += static_cast<size_t>(lt > 0 ? lt : 0);
        }
    };

    if (poly_count < PARALLEL_THRESHOLD) {
        triangulate_slice(0, poly_count, 0, 0);
        return 1;
    }

    const int hw = FastThreadPool::instance().hardware_threads();
    const int num_chunks = std::min(hw, std::max(1, poly_count / 2048));
    const int32_t chunk = (poly_count + num_chunks - 1) / num_chunks;
    const int active_chunks = (poly_count + chunk - 1) / chunk;

    std::vector<int32_t> chunk_lv_off(static_cast<size_t>(active_chunks), 0);
    std::vector<int32_t> chunk_tri_off(static_cast<size_t>(active_chunks), 0);
    int32_t acc_lv = 0;
    int32_t acc_tri = 0;
    for (int c = 0; c < active_chunks; ++c) {
        chunk_lv_off[static_cast<size_t>(c)] = acc_lv;
        chunk_tri_off[static_cast<size_t>(c)] = acc_tri;
        const int32_t p_s = c * chunk;
        const int32_t p_e = std::min(poly_count, p_s + chunk);
        for (int32_t p = p_s; p < p_e; ++p) {
            const int32_t lt = loop_totals[p];
            if (lt >= 3) {
                acc_lv += lt;
                acc_tri += (lt - 2) * 3;
            } else if (lt > 0) {
                acc_lv += lt;
            }
        }
    }

    FastThreadPool::instance().parallel_for_chunks(active_chunks, [&](int c) {
        const int32_t p_s = c * chunk;
        const int32_t p_e = std::min(poly_count, p_s + chunk);
        triangulate_slice(p_s, p_e, chunk_lv_off[static_cast<size_t>(c)], chunk_tri_off[static_cast<size_t>(c)]);
    });
    return 1;
}

static inline int box_visible_mvp(const float mvp[16], const float aabb_min[3], const float aabb_max[3]) {
    float planes[6][4];
    planes[0][0] = mvp[12] + mvp[0]; planes[0][1] = mvp[13] + mvp[1]; planes[0][2] = mvp[14] + mvp[2]; planes[0][3] = mvp[15] + mvp[3];
    planes[1][0] = mvp[12] - mvp[0]; planes[1][1] = mvp[13] - mvp[1]; planes[1][2] = mvp[14] - mvp[2]; planes[1][3] = mvp[15] - mvp[3];
    planes[2][0] = mvp[12] + mvp[4]; planes[2][1] = mvp[13] + mvp[5]; planes[2][2] = mvp[14] + mvp[6]; planes[2][3] = mvp[15] + mvp[7];
    planes[3][0] = mvp[12] - mvp[4]; planes[3][1] = mvp[13] - mvp[5]; planes[3][2] = mvp[14] - mvp[6]; planes[3][3] = mvp[15] - mvp[7];
    planes[4][0] = mvp[12] + mvp[8]; planes[4][1] = mvp[13] + mvp[9]; planes[4][2] = mvp[14] + mvp[10]; planes[4][3] = mvp[15] + mvp[11];
    planes[5][0] = mvp[12] - mvp[8]; planes[5][1] = mvp[13] - mvp[9]; planes[5][2] = mvp[14] - mvp[10]; planes[5][3] = mvp[15] - mvp[11];
    for (int p = 0; p < 6; ++p) {
        const float px = (planes[p][0] > 0.0f) ? aabb_max[0] : aabb_min[0];
        const float py = (planes[p][1] > 0.0f) ? aabb_max[1] : aabb_min[1];
        const float pz = (planes[p][2] > 0.0f) ? aabb_max[2] : aabb_min[2];
        if (planes[p][0] * px + planes[p][1] * py + planes[p][2] * pz + planes[p][3] < 0.0f) {
            return 0;
        }
    }
    return 1;
}

extern "C" ONION_API int onion_core_pack_instances(
    const float* base_mvp_row_major,
    const float* cam_pos_or_dir,
    int32_t is_ortho,
    int32_t count,
    const float* world_mats_row_major,
    const float* world_offsets,
    const float* aabb_mins,
    const float* aabb_maxs,
    const float* colors,
    const float* fresnel_params,
    const float* cam_ws,
    float* out_ubo_floats
) {
    if (!base_mvp_row_major || !cam_pos_or_dir || !world_mats_row_major || !colors || !fresnel_params || !cam_ws || !out_ubo_floats || count <= 0) {
        return 0;
    }

    int32_t vis = 0;
    const float cx = cam_pos_or_dir[0], cy = cam_pos_or_dir[1], cz = cam_pos_or_dir[2];

    for (int32_t i = 0; i < count; ++i) {
        const float* w = world_mats_row_major + static_cast<size_t>(i) * 16;
        float me[16];
        if (world_offsets) {
            const float ox = world_offsets[i * 3];
            const float oy = world_offsets[i * 3 + 1];
            const float oz = world_offsets[i * 3 + 2];
            for (int c = 0; c < 4; ++c) {
                const float r3 = w[12 + c];
                me[c]      = w[c]      + ox * r3;
                me[4 + c]  = w[4 + c]  + oy * r3;
                me[8 + c]  = w[8 + c]  + oz * r3;
                me[12 + c] = r3;
            }
        } else {
            std::memcpy(me, w, 16 * sizeof(float));
        }

        // inst_mvp = base_mvp * me
        float inst_mvp[16];
        for (int r = 0; r < 4; ++r) {
            const float b0 = base_mvp_row_major[r * 4];
            const float b1 = base_mvp_row_major[r * 4 + 1];
            const float b2 = base_mvp_row_major[r * 4 + 2];
            const float b3 = base_mvp_row_major[r * 4 + 3];
            for (int c = 0; c < 4; ++c) {
                inst_mvp[r * 4 + c] = b0 * me[c] + b1 * me[4 + c] + b2 * me[8 + c] + b3 * me[12 + c];
            }
        }

        if (aabb_mins && aabb_maxs) {
            if (!box_visible_mvp(inst_mvp, aabb_mins + static_cast<size_t>(i) * 3, aabb_maxs + static_cast<size_t>(i) * 3)) {
                continue;
            }
        }

        // Transform camera pos/dir into local space using inverse of unshifted world matrix w
        float clx = cx, cly = cy, clz = cz;
        const float w0 = w[0], w1 = w[1], w2 = w[2];
        const float w4 = w[4], w5 = w[5], w6 = w[6];
        const float w8 = w[8], w9 = w[9], w10 = w[10];
        const float c00 = w5 * w10 - w6 * w9;
        const float c01 = -(w4 * w10 - w6 * w8);
        const float c02 = w4 * w9 - w5 * w8;
        const float c10 = -(w1 * w10 - w2 * w9);
        const float c11 = w0 * w10 - w2 * w8;
        const float c12 = -(w0 * w9 - w1 * w8);
        const float c20 = w1 * w6 - w2 * w5;
        const float c21 = -(w0 * w6 - w2 * w4);
        const float c22 = w0 * w5 - w1 * w4;
        const float det = w0 * c00 + w1 * c01 + w2 * c02;
        if (fabsf(det) > 1e-12f) {
            const float inv_det = 1.0f / det;
            // Inverse upper 3x3 is transpose(cofactors) * inv_det
            const float ix00 = c00 * inv_det, ix01 = c10 * inv_det, ix02 = c20 * inv_det;
            const float ix10 = c01 * inv_det, ix11 = c11 * inv_det, ix12 = c21 * inv_det;
            const float ix20 = c02 * inv_det, ix21 = c12 * inv_det, ix22 = c22 * inv_det;
            if (is_ortho) {
                clx = ix00 * cx + ix01 * cy + ix02 * cz;
                cly = ix10 * cx + ix11 * cy + ix12 * cz;
                clz = ix20 * cx + ix21 * cy + ix22 * cz;
            } else {
                const float dx = cx - w[3], dy = cy - w[7], dz = cz - w[11];
                clx = ix00 * dx + ix01 * dy + ix02 * dz;
                cly = ix10 * dx + ix11 * dy + ix12 * dz;
                clz = ix20 * dx + ix21 * dy + ix22 * dz;
            }
        }

        float* dst = out_ubo_floats + static_cast<size_t>(vis) * 28;
        // GLSL std140 mat4 column-major transpose of row-major inst_mvp
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                dst[c * 4 + r] = inst_mvp[r * 4 + c];
            }
        }
        dst[16] = colors[i * 4];
        dst[17] = colors[i * 4 + 1];
        dst[18] = colors[i * 4 + 2];
        dst[19] = colors[i * 4 + 3];

        dst[20] = clx;
        dst[21] = cly;
        dst[22] = clz;
        dst[23] = cam_ws[i];

        dst[24] = fresnel_params[i * 4];
        dst[25] = fresnel_params[i * 4 + 1];
        dst[26] = fresnel_params[i * 4 + 2];
        dst[27] = fresnel_params[i * 4 + 3];
        ++vis;
    }
    return vis;
}
