# SPDX-License-Identifier: GPL-3.0-or-later
"""Python ctypes bridge for the C++ Onion Skin Core backend."""

import os
import sys
import atexit
import shutil
import ctypes
import numpy as np

_lib = None
_is_available = False


def _atexit_shutdown():
    if _lib is not None:
        try:
            _lib.onion_core_shutdown()
        except Exception:
            pass


atexit.register(_atexit_shutdown)

class OnionMeshChunkInput(ctypes.Structure):
    _fields_ = [
        ("vertices", ctypes.c_void_p),
        ("normals", ctypes.c_void_p),
        ("indices", ctypes.c_void_p),
        ("vertex_count", ctypes.c_int32),
        ("index_count", ctypes.c_int32),
        ("matrix", ctypes.c_float * 16),
    ]

class OnionMergedMeshResult(ctypes.Structure):
    _fields_ = [
        ("vertices", ctypes.POINTER(ctypes.c_float)),
        ("normals", ctypes.POINTER(ctypes.c_float)),
        ("indices", ctypes.POINTER(ctypes.c_int32)),
        ("vertex_count", ctypes.c_int32),
        ("index_count", ctypes.c_int32),
        ("prim_type", ctypes.c_int32),
        ("aabb_min", ctypes.c_float * 3),
        ("aabb_max", ctypes.c_float * 3),
        ("centroid", ctypes.c_float * 3),
    ]


def _resolve_platform_lib_candidates(base_dir):
    plat = sys.platform
    if plat.startswith("win") or plat == "cygwin":
        ext = ".dll"
        names = ("libonion_skin_core.dll", "onion_skin_core.dll")
    elif plat == "darwin":
        ext = ".dylib"
        names = ("libonion_skin_core.dylib",)
    else:
        ext = ".so"
        names = ("libonion_skin_core.so",)
    for name in names:
        candidate = os.path.join(base_dir, name)
        if os.path.exists(candidate):
            return candidate, ext
    return os.path.join(base_dir, names[0]), ext


def _get_active_lib():
    global _lib, _is_available
    if _lib is not None:
        return _lib

    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_path, ext = _resolve_platform_lib_candidates(base_dir)
    if not os.path.exists(src_path):
        _is_available = False
        return None

    try:
        mtime = int(os.path.getmtime(src_path))
        active_path = os.path.join(base_dir, f"libonion_skin_core_v{mtime}{ext}")
        if not os.path.exists(active_path):
            for f in os.listdir(base_dir):
                if f.startswith("libonion_skin_core_v") and f.endswith(ext):
                    try:
                        os.remove(os.path.join(base_dir, f))
                    except Exception:
                        pass
            shutil.copy2(src_path, active_path)
        _lib = ctypes.CDLL(active_path)
    except Exception:
        try:
            _lib = ctypes.CDLL(src_path)
        except Exception:
            _is_available = False
            return None

    _setup_function_signatures(_lib)
    _lib.onion_core_init(0)
    _is_available = True
    return _lib


def _setup_function_signatures(lib):
    # Lifecycle
    lib.onion_core_init.argtypes = [ctypes.c_int]
    lib.onion_core_init.restype = ctypes.c_int

    lib.onion_core_shutdown.argtypes = []
    lib.onion_core_shutdown.restype = None

    if hasattr(lib, "onion_core_get_thread_count"):
        lib.onion_core_get_thread_count.argtypes = []
        lib.onion_core_get_thread_count.restype = ctypes.c_int

    if hasattr(lib, "onion_core_compute_normals"):
        lib.onion_core_compute_normals.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.onion_core_compute_normals.restype = ctypes.c_int

    if hasattr(lib, "onion_core_triangulate_polys"):
        lib.onion_core_triangulate_polys.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.onion_core_triangulate_polys.restype = ctypes.c_int

    if hasattr(lib, "onion_core_pack_instances"):
        lib.onion_core_pack_instances.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.onion_core_pack_instances.restype = ctypes.c_int

    lib.onion_hash_name.argtypes = [ctypes.c_char_p]
    lib.onion_hash_name.restype = ctypes.c_uint64

    # Mesh Merging
    lib.onion_core_merge_meshes.argtypes = [
        ctypes.POINTER(OnionMeshChunkInput),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(OnionMergedMeshResult)
    ]
    lib.onion_core_merge_meshes.restype = ctypes.c_int

    lib.onion_core_transform_single_mesh.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.onion_core_transform_single_mesh.restype = ctypes.c_int

    if hasattr(lib, "onion_core_filter_screen_space"):
        lib.onion_core_filter_screen_space.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.onion_core_filter_screen_space.restype = ctypes.c_int

    # Frustum Culling
    lib.onion_core_is_box_visible.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.onion_core_is_box_visible.restype = ctypes.c_int

    # Alpha falloff
    lib.onion_core_calc_alphas.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int
    ]
    lib.onion_core_calc_alphas.restype = None

    # Cache
    lib.onion_cache_has_frame.argtypes = [ctypes.c_uint64, ctypes.c_int32]
    lib.onion_cache_has_frame.restype = ctypes.c_int

    lib.onion_cache_put_frame.argtypes = [ctypes.c_uint64, ctypes.c_int32, ctypes.POINTER(OnionMergedMeshResult)]
    lib.onion_cache_put_frame.restype = ctypes.c_int

    lib.onion_cache_get_frame.argtypes = [ctypes.c_uint64, ctypes.c_int32, ctypes.POINTER(OnionMergedMeshResult)]
    lib.onion_cache_get_frame.restype = ctypes.c_int

    lib.onion_cache_invalidate_object.argtypes = [ctypes.c_uint64]
    lib.onion_cache_invalidate_object.restype = None

    lib.onion_cache_clear.argtypes = []
    lib.onion_cache_clear.restype = None

    lib.onion_cache_set_max_entries.argtypes = [ctypes.c_int]
    lib.onion_cache_set_max_entries.restype = None

    lib.onion_cache_entry_count.argtypes = []
    lib.onion_cache_entry_count.restype = ctypes.c_int

    # Markers
    lib.onion_marker_add.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.POINTER(OnionMergedMeshResult)]
    lib.onion_marker_add.restype = ctypes.c_int

    lib.onion_marker_count.argtypes = []
    lib.onion_marker_count.restype = ctypes.c_int

    lib.onion_marker_get.argtypes = [ctypes.c_int, ctypes.POINTER(OnionMergedMeshResult), ctypes.POINTER(ctypes.c_int32)]
    lib.onion_marker_get.restype = ctypes.c_int

    lib.onion_marker_get_centroid.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.onion_marker_get_centroid.restype = ctypes.c_int

    lib.onion_marker_remove_for.argtypes = [ctypes.c_int32, ctypes.c_char_p]
    lib.onion_marker_remove_for.restype = ctypes.c_int

    lib.onion_marker_clear.argtypes = []
    lib.onion_marker_clear.restype = None


def is_available():
    lib = _get_active_lib()
    return lib is not None


def ensure_initialized():
    """Ensure C++ native library is loaded and initialized (safe across disable/re-enable)."""
    lib = _get_active_lib()
    if lib:
        lib.onion_core_init(0)
        return True
    return False


def hash_name(name):
    lib = _get_active_lib()
    if not lib:
        # Fallback Python 64-bit FNV-1a
        h = 14695981039346656037
        for b in name.encode('utf-8'):
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h
    return lib.onion_hash_name(name.encode('utf-8'))


def _mesh_result_to_arrays(res):
    """Convert C++ OnionMergedMeshResult struct to numpy array views and bounds."""
    v_count = res.vertex_count
    i_count = res.index_count
    prim = 'LINES' if res.prim_type == 1 else 'TRIS'

    if v_count <= 0 or i_count <= 0 or not res.vertices or not res.indices:
        return None, None, None, prim, (0, 0, 0), (0, 0, 0), (0, 0, 0)

    # Owned copies to prevent in-place buffer mutation across multiple objects/frames
    verts = np.copy(np.ctypeslib.as_array(res.vertices, shape=(v_count, 3)))
    normals = np.copy(np.ctypeslib.as_array(res.normals, shape=(v_count, 3))) if res.normals else None

    elem_size = 2 if prim == 'LINES' else 3
    indices = np.copy(np.ctypeslib.as_array(res.indices, shape=(i_count // elem_size, elem_size)))

    aabb_min = tuple(res.aabb_min)
    aabb_max = tuple(res.aabb_max)
    centroid = tuple(res.centroid)

    return verts, normals, indices, prim, aabb_min, aabb_max, centroid


def merge_mesh_chunks(chunks_data, wireframe=False):
    """Merge mesh chunks using AVX2/SIMD hardware acceleration in C++.

    chunks_data: list of tuples:
      (verts_co_1d_float32, verts_no_1d_float32, indices_1d_int32, matrix_4x4_row_major)
    """
    lib = _get_active_lib()
    if not lib or not chunks_data:
        return None, None, None, 'TRIS', (0, 0, 0), (0, 0, 0), (0, 0, 0)

    chunk_count = len(chunks_data)
    c_chunks = (OnionMeshChunkInput * chunk_count)()

    # Keep references to prevent GC during call
    _refs = []

    for i, c_data in enumerate(chunks_data):
        verts_co, verts_no, indices, mat = c_data
        v_count = len(verts_co) // 3
        idx_count = len(indices)

        c_chunks[i].vertex_count = v_count
        c_chunks[i].index_count = idx_count
        c_chunks[i].vertices = verts_co.ctypes.data_as(ctypes.c_void_p)
        c_chunks[i].normals = verts_no.ctypes.data_as(ctypes.c_void_p) if verts_no is not None else None
        c_chunks[i].indices = indices.ctypes.data_as(ctypes.c_void_p)

        mat_flat = [float(x) for row in mat for x in row] if hasattr(mat[0], '__len__') else list(mat)
        c_chunks[i].matrix = (ctypes.c_float * 16)(*mat_flat)
        _refs.append((verts_co, verts_no, indices))

    out_res = OnionMergedMeshResult()
    wire_flag = 1 if wireframe else 0

    success = lib.onion_core_merge_meshes(c_chunks, chunk_count, wire_flag, ctypes.byref(out_res))
    if not success:
        return None, None, None, 'TRIS', (0, 0, 0), (0, 0, 0), (0, 0, 0)

    return _mesh_result_to_arrays(out_res)


def _is_identity_mat(mat):
    for i in range(4):
        row = mat[i]
        for j in range(4):
            if abs(float(row[j]) - (1.0 if i == j else 0.0)) > 1e-6:
                return False
    return True


def transform_single_mesh(verts_co, verts_no, mat, out_verts=None, out_normals=None, return_arrays=True):
    """Transform one mesh chunk in C++ AVX2/multi-thread directly into NumPy arrays without copying indices."""
    lib = _get_active_lib()
    if not lib:
        return None

    in_v = np.ascontiguousarray(verts_co, dtype=np.float32)
    v_count = in_v.size // 3
    if v_count <= 0:
        return None

    in_n = np.ascontiguousarray(verts_no, dtype=np.float32) if verts_no is not None else None
    is_ident = _is_identity_mat(mat)

    if out_verts is None:
        if is_ident:
            out_v_ptr = None
            res_v = in_v.reshape(-1, 3)
        else:
            res_v = np.empty((v_count, 3), dtype=np.float32)
            out_v_ptr = res_v.ctypes.data_as(ctypes.c_void_p)
    else:
        res_v = out_verts
        out_v_ptr = res_v.ctypes.data_as(ctypes.c_void_p)

    if in_n is not None:
        if out_normals is None:
            if is_ident and out_verts is None:
                out_n_ptr = None
                res_n = in_n.reshape(-1, 3)
            else:
                res_n = np.empty((v_count, 3), dtype=np.float32)
                out_n_ptr = res_n.ctypes.data_as(ctypes.c_void_p)
        else:
            res_n = out_normals
            out_n_ptr = res_n.ctypes.data_as(ctypes.c_void_p)
    else:
        res_n = None
        out_n_ptr = None

    if is_ident and out_v_ptr is None:
        c_mat = None
    else:
        mat_flat = [float(x) for row in mat for x in row] if hasattr(mat[0], '__len__') else [float(x) for x in mat]
        c_mat = (ctypes.c_float * 16)(*mat_flat)

    c_min = (ctypes.c_float * 3)()
    c_max = (ctypes.c_float * 3)()
    c_ctr = (ctypes.c_float * 3)()

    ok = lib.onion_core_transform_single_mesh(
        in_v.ctypes.data_as(ctypes.c_void_p),
        in_n.ctypes.data_as(ctypes.c_void_p) if in_n is not None else None,
        ctypes.c_int32(v_count),
        c_mat,
        out_v_ptr,
        out_n_ptr,
        c_min,
        c_max,
        c_ctr,
    )
    if not ok:
        return None

    b_min = (float(c_min[0]), float(c_min[1]), float(c_min[2]))
    b_max = (float(c_max[0]), float(c_max[1]), float(c_max[2]))
    centroid = (float(c_ctr[0]), float(c_ctr[1]), float(c_ctr[2]))
    if not return_arrays:
        return b_min, b_max, centroid
    return res_v, res_n, b_min, b_max, centroid


def is_box_visible(mvp_matrix, aabb_min, aabb_max):
    """Test 3D AABB against camera view frustum planes extracted from MVP matrix."""
    lib = _get_active_lib()
    if not lib:
        return True

    # Flatten MVP matrix (4x4 row-major)
    if hasattr(mvp_matrix[0], '__len__'):
        mvp_flat = [float(x) for row in mvp_matrix for x in row]
    else:
        mvp_flat = [float(x) for x in mvp_matrix]

    c_mvp = (ctypes.c_float * 16)(*mvp_flat)
    c_min = (ctypes.c_float * 3)(*aabb_min)
    c_max = (ctypes.c_float * 3)(*aabb_max)

    return bool(lib.onion_core_is_box_visible(c_mvp, c_min, c_max))


def filter_screen_space_frames(mvp_matrix, vp_width, vp_height, ref_centroid, candidate_centroids, min_pixel_dist):
    """Filter candidate frames by minimum screen-space distance (in pixels) using C++ acceleration.

    Returns (keep_mask_list_of_bools, screen_coords_list_of_xy_tuples).
    """
    count = len(candidate_centroids)
    if count == 0:
        return [], []

    if hasattr(mvp_matrix[0], '__len__'):
        mvp_flat = [float(x) for row in mvp_matrix for x in row]
    else:
        mvp_flat = [float(x) for x in mvp_matrix]

    lib = _get_active_lib()
    if lib and hasattr(lib, "onion_core_filter_screen_space"):
        c_mvp = (ctypes.c_float * 16)(*mvp_flat)
        c_ref = (ctypes.c_float * 3)(*ref_centroid) if ref_centroid is not None else None
        flat_cand = [float(c) for pt in candidate_centroids for c in pt]
        c_cand = (ctypes.c_float * (count * 3))(*flat_cand)
        c_mask = (ctypes.c_int32 * count)()
        c_scr = (ctypes.c_float * (count * 2))()
        lib.onion_core_filter_screen_space(
            c_mvp,
            ctypes.c_float(float(vp_width)),
            ctypes.c_float(float(vp_height)),
            c_ref,
            c_cand,
            ctypes.c_int32(count),
            ctypes.c_float(float(min_pixel_dist)),
            c_mask,
            c_scr,
        )
        keep_mask = [bool(c_mask[i]) for i in range(count)]
        screen_coords = [(float(c_scr[2 * i]), float(c_scr[2 * i + 1])) for i in range(count)]
        return keep_mask, screen_coords

    # Python fallback
    vp_w = max(1.0, float(vp_width))
    vp_h = max(1.0, float(vp_height))
    min_d2 = max(0.0, float(min_pixel_dist)) ** 2

    def _proj(pt):
        x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
        cx = mvp_flat[0] * x + mvp_flat[1] * y + mvp_flat[2] * z + mvp_flat[3]
        cy = mvp_flat[4] * x + mvp_flat[5] * y + mvp_flat[6] * z + mvp_flat[7]
        cw = mvp_flat[12] * x + mvp_flat[13] * y + mvp_flat[14] * z + mvp_flat[15]
        if abs(cw) < 1e-6:
            return 0.0, 0.0, False
        inv_w = 1.0 / cw
        px = (cx * inv_w * 0.5 + 0.5) * vp_w
        py = (cy * inv_w * 0.5 + 0.5) * vp_h
        return px, py, (cw > -1e-4)

    has_anchor = False
    last_px, last_py = 0.0, 0.0
    if ref_centroid is not None:
        last_px, last_py, has_anchor = _proj(ref_centroid)

    keep_mask = [False] * count
    screen_coords = [(0.0, 0.0)] * count
    best_idx = -1
    best_d2 = -1.0
    kept = 0

    for i, pt in enumerate(candidate_centroids):
        px, py, ok = _proj(pt)
        screen_coords[i] = (px, py)
        if not ok:
            continue
        if not has_anchor:
            keep_mask[i] = True
            last_px, last_py = px, py
            has_anchor = True
            kept += 1
            continue
        d2 = (px - last_px) ** 2 + (py - last_py) ** 2
        if d2 > best_d2:
            best_d2 = d2
            best_idx = i
        if d2 >= min_d2:
            keep_mask[i] = True
            last_px, last_py = px, py
            kept += 1

    if kept == 0 and best_idx >= 0 and best_d2 >= 0.0625:
        keep_mask[best_idx] = True

    return keep_mask, screen_coords


# ---------------------------------------------------------------------------
# C++ Frame Cache Interface
# ---------------------------------------------------------------------------

def cache_has_frame(object_name, frame):
    lib = _get_active_lib()
    if not lib:
        return False
    obj_id = hash_name(object_name)
    return bool(lib.onion_cache_has_frame(obj_id, frame))


def cache_put_frame(object_name, frame, verts, normals, indices, prim_type, aabb_min, aabb_max, centroid):
    lib = _get_active_lib()
    if not lib:
        return False
    obj_id = hash_name(object_name)

    res = OnionMergedMeshResult()
    v_arr = np.ascontiguousarray(np.asarray(verts, dtype=np.float32).reshape(-1, 3))
    res.vertex_count = len(v_arr)
    res.vertices = v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if normals is not None:
        n_arr = np.ascontiguousarray(np.asarray(normals, dtype=np.float32).reshape(-1, 3))
        res.normals = n_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        res.normals = None

    idx_flat = np.ascontiguousarray(np.asarray(indices, dtype=np.int32).reshape(-1))
    res.index_count = len(idx_flat)
    res.indices = idx_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    res.prim_type = 1 if prim_type == 'LINES' else 0
    res.aabb_min = (ctypes.c_float * 3)(*aabb_min)
    res.aabb_max = (ctypes.c_float * 3)(*aabb_max)
    res.centroid = (ctypes.c_float * 3)(*centroid)

    return bool(lib.onion_cache_put_frame(obj_id, frame, ctypes.byref(res)))


def cache_get_frame(object_name, frame):
    lib = _get_active_lib()
    if not lib:
        return None
    obj_id = hash_name(object_name)

    out_res = OnionMergedMeshResult()
    if not lib.onion_cache_get_frame(obj_id, frame, ctypes.byref(out_res)):
        return None

    return _mesh_result_to_arrays(out_res)


def cache_invalidate_object(object_name):
    lib = _get_active_lib()
    if lib:
        obj_id = hash_name(object_name)
        lib.onion_cache_invalidate_object(obj_id)


def cache_clear():
    lib = _get_active_lib()
    if lib:
        lib.onion_cache_clear()


# ---------------------------------------------------------------------------
# C++ Ghost Marker Interface
# ---------------------------------------------------------------------------

def marker_add(frame, verts, normals, indices, prim_type, aabb_min, aabb_max, centroid, tag=""):
    lib = _get_active_lib()
    if not lib:
        return 0

    res = OnionMergedMeshResult()
    v_arr = np.ascontiguousarray(np.asarray(verts, dtype=np.float32).reshape(-1, 3))
    res.vertex_count = len(v_arr)
    res.vertices = v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if normals is not None:
        n_arr = np.ascontiguousarray(np.asarray(normals, dtype=np.float32).reshape(-1, 3))
        res.normals = n_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        res.normals = None

    idx_flat = np.ascontiguousarray(np.asarray(indices, dtype=np.int32).reshape(-1))
    res.index_count = len(idx_flat)
    res.indices = idx_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    res.prim_type = 1 if prim_type == 'LINES' else 0
    res.aabb_min = (ctypes.c_float * 3)(*aabb_min)
    res.aabb_max = (ctypes.c_float * 3)(*aabb_max)
    res.centroid = (ctypes.c_float * 3)(*centroid)

    c_tag = tag.encode('utf-8')
    return lib.onion_marker_add(frame, c_tag, ctypes.byref(res))


def marker_count():
    lib = _get_active_lib()
    return lib.onion_marker_count() if lib else 0


def marker_get(index):
    lib = _get_active_lib()
    if not lib:
        return None, 0

    out_res = OnionMergedMeshResult()
    out_frame = ctypes.c_int32(0)

    if not lib.onion_marker_get(index, ctypes.byref(out_res), ctypes.byref(out_frame)):
        return None, 0

    mesh_data = _mesh_result_to_arrays(out_res)
    return mesh_data, out_frame.value


def marker_get_centroid(index):
    lib = _get_active_lib()
    if not lib:
        return (0.0, 0.0, 0.0)

    cx = ctypes.c_float(0.0)
    cy = ctypes.c_float(0.0)
    cz = ctypes.c_float(0.0)

    if lib.onion_marker_get_centroid(index, ctypes.byref(cx), ctypes.byref(cy), ctypes.byref(cz)):
        return (cx.value, cy.value, cz.value)
    return (0.0, 0.0, 0.0)


def marker_remove_for(frame, tag=""):
    lib = _get_active_lib()
    if not lib:
        return 0
    return lib.onion_marker_remove_for(frame, tag.encode('utf-8'))


def marker_clear():
    lib = _get_active_lib()
    if lib:
        lib.onion_marker_clear()


# ---------------------------------------------------------------------------
# Vectorized Alpha Falloff
# ---------------------------------------------------------------------------

_CURVE_MAP = {
    'linear': 0,
    'smooth': 1,
    'exponential': 2,
}

def calc_alphas(t_values, base_alpha, falloff, curve_name='smooth'):
    lib = _get_active_lib()
    count = len(t_values)
    if count == 0:
        return []

    c_type = _CURVE_MAP.get(curve_name, 1)

    if not lib:
        # Fallback
        res = []
        for t in t_values:
            if c_type == 0:
                factor = 1.0 - t * falloff
            elif c_type == 2:
                factor = max(0.0, 1.0 - t) ** (1.0 + falloff * 2.0)
            else:
                ts = t * falloff
                factor = 0.1 if ts >= 1.0 else (1.0 if ts <= 0.0 else 1.0 - ts * ts * (3.0 - 2.0 * ts))
            res.append(base_alpha * max(0.1, factor))
        return res

    t_arr = (ctypes.c_float * count)(*t_values)
    out_arr = (ctypes.c_float * count)()

    lib.onion_core_calc_alphas(t_arr, out_arr, count, ctypes.c_float(base_alpha), ctypes.c_float(falloff), c_type)
    return list(out_arr)


# ---------------------------------------------------------------------------
# Multi-Core Geometry Normals, Triangulation & GPU Instance Packing
# ---------------------------------------------------------------------------

_PACK_CAM3 = np.zeros(3, dtype=np.float32)
_PACK_MVP16 = np.zeros(16, dtype=np.float32)


def get_thread_count():
    lib = _get_active_lib()
    if lib and hasattr(lib, "onion_core_get_thread_count"):
        return int(lib.onion_core_get_thread_count())
    return os.cpu_count() or 4


def compute_normals(verts_co_1d, tri_indices_1d):
    """Compute area-weighted smooth vertex normals via multi-threaded AVX2/FMA C++."""
    lib = _get_active_lib()
    if not lib or not hasattr(lib, "onion_core_compute_normals"):
        return None
    v_arr = np.ascontiguousarray(verts_co_1d, dtype=np.float32)
    t_arr = np.ascontiguousarray(tri_indices_1d, dtype=np.int32)
    v_count = v_arr.size // 3
    t_count = t_arr.size // 3
    if v_count <= 0 or t_count <= 0:
        return None
    out = np.empty(v_count * 3, dtype=np.float32)
    ok = lib.onion_core_compute_normals(
        v_arr.ctypes.data_as(ctypes.c_void_p),
        v_count,
        t_arr.ctypes.data_as(ctypes.c_void_p),
        t_count,
        out.ctypes.data_as(ctypes.c_void_p),
    )
    return out if ok else None


def triangulate_polys(loop_verts_1d, loop_totals_1d, total_tris):
    """Fan-triangulate polygon loops via multi-threaded C++."""
    lib = _get_active_lib()
    if not lib or not hasattr(lib, "onion_core_triangulate_polys") or total_tris <= 0:
        return None
    lv = np.ascontiguousarray(loop_verts_1d, dtype=np.int32)
    lt = np.ascontiguousarray(loop_totals_1d, dtype=np.int32)
    poly_count = lt.size
    if poly_count <= 0:
        return None
    out = np.empty(total_tris * 3, dtype=np.int32)
    ok = lib.onion_core_triangulate_polys(
        lv.ctypes.data_as(ctypes.c_void_p),
        lt.ctypes.data_as(ctypes.c_void_p),
        poly_count,
        out.ctypes.data_as(ctypes.c_void_p),
    )
    return out if ok else None


def pack_instances(
    base_mvp,
    cam_pos_or_dir,
    is_ortho,
    world_mats_f32,
    world_offsets_f32,
    aabb_mins_f32,
    aabb_maxs_f32,
    colors_f32,
    fresnel_params_f32,
    cam_ws_f32,
    out_ubo_buf,
    count=None,
):
    """Frustum-cull and pack up to 64 GPU instances into std140 UBO buffer in C++."""
    lib = _get_active_lib()
    if not lib or not hasattr(lib, "onion_core_pack_instances"):
        return -1
    if count is None:
        count = len(world_mats_f32)
    if count <= 0:
        return 0
    _PACK_MVP16[:] = (
        base_mvp[0][0], base_mvp[0][1], base_mvp[0][2], base_mvp[0][3],
        base_mvp[1][0], base_mvp[1][1], base_mvp[1][2], base_mvp[1][3],
        base_mvp[2][0], base_mvp[2][1], base_mvp[2][2], base_mvp[2][3],
        base_mvp[3][0], base_mvp[3][1], base_mvp[3][2], base_mvp[3][3],
    )
    _PACK_CAM3[0] = cam_pos_or_dir[0]
    _PACK_CAM3[1] = cam_pos_or_dir[1]
    _PACK_CAM3[2] = cam_pos_or_dir[2]
    return int(lib.onion_core_pack_instances(
        _PACK_MVP16.ctypes.data_as(ctypes.c_void_p),
        _PACK_CAM3.ctypes.data_as(ctypes.c_void_p),
        1 if is_ortho else 0,
        int(count),
        world_mats_f32.ctypes.data_as(ctypes.c_void_p),
        world_offsets_f32.ctypes.data_as(ctypes.c_void_p) if world_offsets_f32 is not None else None,
        aabb_mins_f32.ctypes.data_as(ctypes.c_void_p) if aabb_mins_f32 is not None else None,
        aabb_maxs_f32.ctypes.data_as(ctypes.c_void_p) if aabb_maxs_f32 is not None else None,
        colors_f32.ctypes.data_as(ctypes.c_void_p),
        fresnel_params_f32.ctypes.data_as(ctypes.c_void_p),
        cam_ws_f32.ctypes.data_as(ctypes.c_void_p),
        out_ubo_buf.ctypes.data_as(ctypes.c_void_p),
    ))
