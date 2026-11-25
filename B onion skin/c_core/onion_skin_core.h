#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #define ONION_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
  #define ONION_API __attribute__((visibility("default")))
#else
  #define ONION_API
#endif

/* Primitive types */
#define ONION_PRIM_TRIS  0
#define ONION_PRIM_LINES 1

/* Falloff curve types */
#define ONION_CURVE_LINEAR      0
#define ONION_CURVE_SMOOTH      1
#define ONION_CURVE_EXPONENTIAL 2

/**
 * Input descriptor for a single submesh chunk to be merged.
 */
typedef struct {
    const float* vertices;     /* Flat array of local coords: [x0, y0, z0, x1, ...] */
    const float* normals;      /* Flat array of local normals: [nx0, ny0, nz0, ...] (or NULL) */
    const int32_t* indices;    /* Flat index buffer: [i0, i1, i2, ...] */
    int32_t vertex_count;      /* Number of vertices (vertices length / 3) */
    int32_t index_count;       /* Number of indices (triangles * 3 or edges * 2) */
    float matrix[16];          /* 4x4 row-major world transform matrix */
} OnionMeshChunkInput;

/**
 * Output descriptor for merged mesh geometry and spatial bounds.
 * Note: Pointers remain valid until the next call to merge_meshes or until cleared.
 */
typedef struct {
    const float* vertices;     /* Flat array of world coords: [x0, y0, z0, ...] */
    const float* normals;      /* Flat array of world normals (or NULL for wireframe) */
    const int32_t* indices;    /* Flat array of merged indices */
    int32_t vertex_count;      /* Total vertices */
    int32_t index_count;       /* Total indices */
    int32_t prim_type;         /* ONION_PRIM_TRIS or ONION_PRIM_LINES */
    float aabb_min[3];         /* Min bounding box [min_x, min_y, min_z] */
    float aabb_max[3];         /* Max bounding box [max_x, max_y, max_z] */
    float centroid[3];         /* Centroid [cx, cy, cz] for instant label placement */
} OnionMergedMeshResult;

/* System lifecycle */
ONION_API int onion_core_init(int thread_count);
ONION_API void onion_core_shutdown(void);
void onion_fast_pool_shutdown(void);
ONION_API int onion_core_get_thread_count(void);

/* String hashing utility (64-bit FNV-1a) */
ONION_API uint64_t onion_hash_name(const char* name);

/* SIMD Accelerated Mesh Merging & Transformation */
ONION_API int onion_core_merge_meshes(
    const OnionMeshChunkInput* chunks,
    int chunk_count,
    int wireframe,
    OnionMergedMeshResult* out_result
);

/* SIMD Accelerated Single-Chunk Zero-Index-Copy Transformation & Bounds */
ONION_API int onion_core_transform_single_mesh(
    const float* in_verts,
    const float* in_normals,
    int32_t vertex_count,
    const float* matrix_row_major,
    float* out_verts,
    float* out_normals,
    float* out_aabb_min,
    float* out_aabb_max,
    float* out_centroid
);

/* Multi-threaded AVX2/FMA area-weighted smooth vertex normal computation */
ONION_API int onion_core_compute_normals(
    const float* verts,
    int32_t vertex_count,
    const int32_t* tri_indices,
    int32_t tri_count,
    float* out_normals
);

/* Multi-threaded polygon fan triangulation (triangles/quads/ngons) */
ONION_API int onion_core_triangulate_polys(
    const int32_t* loop_verts,
    const int32_t* loop_totals,
    int32_t poly_count,
    int32_t* out_tri_indices
);

/* C++ frustum culling + matrix inverse + std140 UBO packing for GPU draw_instanced */
ONION_API int onion_core_pack_instances(
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
);

/* Screen-Space Distance Frame Filtering & Projection */
ONION_API int onion_core_filter_screen_space(
    const float* mvp_matrix_row_major,
    float vp_width,
    float vp_height,
    const float* ref_centroid,
    const float* candidate_centroids,
    int32_t count,
    float min_pixel_dist,
    int32_t* out_keep_mask,
    float* out_screen_coords
);

/* Hardware accelerated frustum culling */
ONION_API int onion_core_is_box_visible(
    const float* mvp_matrix_row_major,
    const float* aabb_min,
    const float* aabb_max
);

/* Vectorized Alpha Falloff calculation */
ONION_API void onion_core_calc_alphas(
    const float* t_values,
    float* out_alphas,
    int count,
    float base_alpha,
    float falloff,
    int curve_type
);

/* Geometry Frame Cache in C++ */
ONION_API int onion_cache_has_frame(uint64_t object_id, int32_t frame);
ONION_API int onion_cache_put_frame(uint64_t object_id, int32_t frame, const OnionMergedMeshResult* mesh_data);
ONION_API int onion_cache_get_frame(uint64_t object_id, int32_t frame, OnionMergedMeshResult* out_result);
ONION_API void onion_cache_invalidate_object(uint64_t object_id);
ONION_API void onion_cache_clear(void);
ONION_API void onion_cache_set_max_entries(int max_entries);
ONION_API int onion_cache_entry_count(void);

/* Persistent Ghost Marker System in C++ */
ONION_API int onion_marker_add(int32_t frame, const char* tag, const OnionMergedMeshResult* mesh_data);
ONION_API int onion_marker_count(void);
ONION_API int onion_marker_get(int index, OnionMergedMeshResult* out_result, int* out_frame);
ONION_API int onion_marker_get_centroid(int index, float* cx, float* cy, float* cz);
ONION_API int onion_marker_remove_for(int32_t frame, const char* tag);
ONION_API void onion_marker_clear(void);

#ifdef __cplusplus
}
#endif
