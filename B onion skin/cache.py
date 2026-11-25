# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU batch storage and caching for onion skin overlays.

Provides a multi-tier frame cache to avoid evaluating the scene multiple times:
1. Native C++ SIMD geometry cache for zero-copy memory storage.
2. GPU VRAM batch pooling to reuse GPUBatch objects across frames.
3. Hardware accelerated custom shaders (Vulkan/OpenGL optimized).
"""

import collections
import gpu
from gpu_extras.batch import batch_for_shader

from .c_core import core_bridge as cb

_shader = None
_fresnel_shader = None
_inst_fresnel_shader = None
_inst_flat_shader = None
_inst_ubo = None
_shader_fmt_cache = {}
MAX_GPU_INSTANCES = 64
_shared_ibo_cache = collections.OrderedDict()
MAX_SHARED_IBOS = 256

# Current ghost frame batches — rebuilt on frame change.
# List of (object_name, frame, direction, t, dist, gpu_batch, aabb_min, aabb_max)
_onion_batches = []

# Frame cache to avoid re-evaluating scene for evaluated frames
# Key: (object_name, frame) -> list of entries:
# [ext_name, verts, normals, indices, prim_type, batch_map, aabb_min, aabb_max, centroid]
_frame_cache = collections.OrderedDict()
MAX_CACHE_SIZE = 4096

# Per-object index of valid cached frame numbers and invalidated C++ frames
_frame_cached_indices = {}
_invalidated_frames = {}

# Ghost marker cache — persists until user removes them.
# Each entry: (frame, verts, indices, prim_type, obj_names, centroid, normals, aabb_min, aabb_max)
_ghost_markers = []
_ghost_batches = {}    # {index: gpu_batch}
_ghost_undo_bank = []  # Bounded snapshots to support Undo of marker removals


# ---------------------------------------------------------------------------
# Frame Cache Management
# ---------------------------------------------------------------------------

def get_frame(object_name, frame):
    inv = _invalidated_frames.get(object_name)
    if inv and frame in inv:
        return None

    key = (object_name, frame)
    if key in _frame_cache:
        _frame_cache.move_to_end(key)
        return _frame_cache[key]

    if cb.is_available():
        res = cb.cache_get_frame(object_name, frame)
        if res and res[0] is not None:
            verts, normals, indices, prim, b_min, b_max, center = res
            entry = [object_name, verts, normals, indices, prim, {}, b_min, b_max, center]
            put_frame(object_name, frame, [entry])
            return _frame_cache[key]
    return None


def has_frame(object_name, frame):
    inv = _invalidated_frames.get(object_name)
    if inv and frame in inv:
        return False
    if (object_name, frame) in _frame_cache:
        return True
    if cb.is_available():
        return cb.cache_has_frame(object_name, frame)
    return False


def put_frame(object_name, frame, batch):
    inv = _invalidated_frames.get(object_name)
    if inv:
        inv.discard(frame)

    key = (object_name, frame)
    if key in _frame_cache:
        _frame_cache.move_to_end(key)
    _frame_cache[key] = batch

    if batch is not None:
        _frame_cached_indices.setdefault(object_name, set()).add(frame)
    else:
        idx_set = _frame_cached_indices.get(object_name)
        if idx_set:
            idx_set.discard(frame)

    if len(_frame_cache) > MAX_CACHE_SIZE:
        ev_key, _ = _frame_cache.popitem(last=False)
        ev_set = _frame_cached_indices.get(ev_key[0])
        if ev_set:
            ev_set.discard(ev_key[1])


def get_cached_frames(object_name):
    """Return sorted list of valid cached frame numbers for object_name."""
    idx_set = _frame_cached_indices.get(object_name)
    return sorted(idx_set) if idx_set else []


def invalidate_frames(object_name, frames):
    """Selectively invalidate specific frame numbers for object_name without clearing all frames."""
    if not frames:
        return
    inv = _invalidated_frames.setdefault(object_name, set())
    idx_set = _frame_cached_indices.get(object_name)
    for f in frames:
        _frame_cache.pop((object_name, f), None)
        inv.add(f)
        if idx_set is not None:
            idx_set.discard(f)


def clear_frame_cache(object_name=None):
    """Clear frame cache (optionally for a specific object)."""
    if object_name is None:
        _frame_cache.clear()
        _frame_cached_indices.clear()
        _invalidated_frames.clear()
        if cb.is_available():
            cb.cache_clear()
    else:
        keys_to_del = [k for k in _frame_cache if k[0] == object_name]
        for k in keys_to_del:
            del _frame_cache[k]
        _frame_cached_indices.pop(object_name, None)
        _invalidated_frames.pop(object_name, None)
        if cb.is_available():
            cb.cache_invalidate_object(object_name)


# ---------------------------------------------------------------------------
# GPU Shaders
# ---------------------------------------------------------------------------

def get_shader():
    """Get or create the high-performance GPU flat shader."""
    global _shader
    if _shader is not None:
        return _shader

    try:
        s_info = gpu.types.GPUShaderCreateInfo()
        s_info.vertex_in(0, 'VEC3', 'pos')
        s_info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        s_info.push_constant('VEC4', 'color')
        s_info.fragment_out(0, 'VEC4', 'fragColor')
        s_info.vertex_source('''
            void main() {
                gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
            }
        ''')
        s_info.fragment_source('''
            void main() {
                fragColor = color;
            }
        ''')
        _shader = gpu.shader.create_from_info(s_info)
    except Exception:
        try:
            _shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        except Exception:
            try:
                _shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            except Exception:
                try:
                    _shader = gpu.shader.from_builtin('FLAT_COLOR')
                except Exception:
                    _shader = None
    return _shader


def get_fresnel_shader():
    """Get or create the GPU 3D Shaded (Specular + Shadow + Contour) & Fresnel shader (112-byte push constants)."""
    global _fresnel_shader
    if _fresnel_shader is not None:
        return _fresnel_shader

    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("fresnel_interface")
        vert_out.smooth('VEC3', "v_normal")
        vert_out.smooth('VEC3', "v_view_dir")
        vert_out.smooth('FLOAT', "v_fresnel")

        f_info = gpu.types.GPUShaderCreateInfo()
        f_info.vertex_in(0, 'VEC3', "pos")
        f_info.vertex_in(1, 'VEC3', "normal")
        f_info.vertex_out(vert_out)

        # 112 bytes total push constants (strictly within 128-byte Vulkan hardware budget)
        f_info.push_constant('MAT4', "ModelViewProjectionMatrix")   # 64 bytes
        f_info.push_constant('VEC4', "color")                        # 16 bytes
        f_info.push_constant('VEC4', "camera_pos_or_dir")           # 16 bytes (.xyz = cam pos/dir, .w = +/-shininess or 0 for legacy)
        f_info.push_constant('VEC4', "fresnel_params")              # 16 bytes (power, fade, shadow_or_ortho, specular)

        f_info.fragment_out(0, 'VEC4', "fragColor")

        f_info.vertex_source('''
            void main() {
                gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
                bool legacy = (abs(camera_pos_or_dir.w) < 0.5);
                bool is_ortho = legacy ? (fresnel_params.z > 0.5) : (camera_pos_or_dir.w > 0.0);
                vec3 raw_v = is_ortho ? camera_pos_or_dir.xyz : (camera_pos_or_dir.xyz - pos);
                float v_len = length(raw_v);
                vec3 view_dir = (v_len > 1e-6) ? (raw_v / v_len) : vec3(0.0, 0.0, 1.0);
                float n_len = length(normal);
                vec3 view_normal = (n_len > 1e-6) ? (normal / n_len) : view_dir;
                v_normal = view_normal;
                v_view_dir = view_dir;
                float f = clamp(1.0 - abs(dot(view_normal, view_dir)), 0.0, 1.0);
                v_fresnel = pow(f, max(fresnel_params.x, 0.1));
            }
        ''')

        f_info.fragment_source('''
            void main() {
                float n_len = length(v_normal);
                float v_len = length(v_view_dir);
                vec3 v = (v_len > 1e-6) ? (v_view_dir / v_len) : vec3(0.0, 0.0, 1.0);
                vec3 n = (n_len > 1e-6) ? (v_normal / n_len) : v;
                float ndotv = clamp(abs(dot(n, v)), 0.0, 1.0);

                bool legacy = (abs(camera_pos_or_dir.w) < 0.5);
                float shadow_str = legacy ? 0.0 : clamp(fresnel_params.z, 0.0, 1.0);
                float spec_str = legacy ? 0.0 : clamp(fresnel_params.w, 0.0, 1.0);
                float shininess = legacy ? 24.0 : max(abs(camera_pos_or_dir.w), 2.0);

                vec3 rgb = color.rgb;
                float alpha = color.a * mix(1.0, v_fresnel, clamp(fresnel_params.y, 0.0, 1.0));

                if (shadow_str > 0.001 || spec_str > 0.001) {
                    vec3 up_ref = (abs(v.z) < 0.95) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
                    vec3 right_dir = normalize(cross(v, up_ref));
                    vec3 up_dir = cross(right_dir, v);
                    vec3 light_dir = normalize(v * 0.80 + right_dir * 0.35 + up_dir * 0.48);
                    vec3 half_dir = normalize(v + light_dir);

                    vec3 nf = (dot(n, v) >= 0.0) ? n : -n;
                    float ndotl = clamp(dot(nf, light_dir) * 0.5 + 0.5, 0.0, 1.0);
                    float hemi = mix(ndotv, ndotl, 0.58);

                    // Studio diffuse shadow + crisp dark silhouette contour rim
                    float diffuse_shade = smoothstep(0.04, 0.88, hemi);
                    float contour_rim = smoothstep(0.0, 0.36, ndotv);
                    float combined_light = diffuse_shade * (0.42 + 0.58 * contour_rim);
                    float dark_factor = mix(1.0, combined_light, shadow_str * 0.85);
                    rgb *= dark_factor;

                    // Blinn-Phong specular highlight on curved forms
                    float ndoth = clamp(dot(nf, half_dir), 0.0, 1.0);
                    float spec = pow(ndoth, shininess) * spec_str * smoothstep(0.05, 0.35, ndotv);
                    rgb = clamp(rgb + vec3(spec * 0.85), 0.0, 1.0);

                    // Subtle opacity boost at contour rim and specular peak for crisp 3D readability
                    float rim_boost = (1.0 - contour_rim) * shadow_str * 0.24;
                    alpha = clamp(alpha + spec * 0.25 + rim_boost * color.a, 0.0, 1.0);
                }

                fragColor = vec4(rgb, alpha);
            }
        ''')

        _fresnel_shader = gpu.shader.create_from_info(f_info)
    except Exception:
        _fresnel_shader = None
    return _fresnel_shader


_INST_UBO_TYPEDEF = '''
    struct OnionInstItem {
        mat4 mvp;
        vec4 color;
        vec4 cam_pos_or_dir;
        vec4 fresnel_params;
    };
    struct OnionInstBlock {
        OnionInstItem items[64];
    };
'''


def get_instanced_flat_shader():
    """Get or create GPU instanced flat shader (up to 64 instances per draw call)."""
    global _inst_flat_shader
    if _inst_flat_shader is not None:
        return _inst_flat_shader
    try:
        s_info = gpu.types.GPUShaderCreateInfo()
        s_info.typedef_source(_INST_UBO_TYPEDEF)
        s_info.uniform_buf(0, 'OnionInstBlock', 'u_inst')
        s_info.vertex_in(0, 'VEC3', 'pos')
        v_out = gpu.types.GPUStageInterfaceInfo("inst_flat_interface")
        v_out.flat('VEC4', 'v_color')
        s_info.vertex_out(v_out)
        s_info.fragment_out(0, 'VEC4', 'fragColor')
        s_info.vertex_source('''
            void main() {
                OnionInstItem it = u_inst.items[gl_InstanceID];
                gl_Position = it.mvp * vec4(pos, 1.0);
                v_color = it.color;
            }
        ''')
        s_info.fragment_source('''
            void main() {
                fragColor = v_color;
            }
        ''')
        _inst_flat_shader = gpu.shader.create_from_info(s_info)
    except Exception:
        _inst_flat_shader = None
    return _inst_flat_shader


def get_instanced_fresnel_shader():
    """Get or create GPU instanced 3D Shaded + Fresnel shader (up to 64 instances per draw call)."""
    global _inst_fresnel_shader
    if _inst_fresnel_shader is not None:
        return _inst_fresnel_shader
    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("inst_fresnel_interface")
        vert_out.smooth('VEC3', "v_normal")
        vert_out.smooth('VEC3', "v_view_dir")
        vert_out.smooth('FLOAT', "v_fresnel")
        vert_out.flat('VEC4', "v_color")
        vert_out.flat('VEC4', "v_cam")
        vert_out.flat('VEC4', "v_params")

        f_info = gpu.types.GPUShaderCreateInfo()
        f_info.typedef_source(_INST_UBO_TYPEDEF)
        f_info.uniform_buf(0, 'OnionInstBlock', 'u_inst')
        f_info.vertex_in(0, 'VEC3', "pos")
        f_info.vertex_in(1, 'VEC3', "normal")
        f_info.vertex_out(vert_out)
        f_info.fragment_out(0, 'VEC4', "fragColor")

        f_info.vertex_source('''
            void main() {
                OnionInstItem it = u_inst.items[gl_InstanceID];
                gl_Position = it.mvp * vec4(pos, 1.0);
                bool legacy = (abs(it.cam_pos_or_dir.w) < 0.5);
                bool is_ortho = legacy ? (it.fresnel_params.z > 0.5) : (it.cam_pos_or_dir.w > 0.0);
                vec3 raw_v = is_ortho ? it.cam_pos_or_dir.xyz : (it.cam_pos_or_dir.xyz - pos);
                float v_len = length(raw_v);
                vec3 view_dir = (v_len > 1e-6) ? (raw_v / v_len) : vec3(0.0, 0.0, 1.0);
                float n_len = length(normal);
                vec3 view_normal = (n_len > 1e-6) ? (normal / n_len) : view_dir;
                v_normal = view_normal;
                v_view_dir = view_dir;
                float f = clamp(1.0 - abs(dot(view_normal, view_dir)), 0.0, 1.0);
                v_fresnel = pow(f, max(it.fresnel_params.x, 0.1));
                v_color = it.color;
                v_cam = it.cam_pos_or_dir;
                v_params = it.fresnel_params;
            }
        ''')

        f_info.fragment_source('''
            void main() {
                float n_len = length(v_normal);
                float v_len = length(v_view_dir);
                vec3 v = (v_len > 1e-6) ? (v_view_dir / v_len) : vec3(0.0, 0.0, 1.0);
                vec3 n = (n_len > 1e-6) ? (v_normal / n_len) : v;
                float ndotv = clamp(abs(dot(n, v)), 0.0, 1.0);

                bool legacy = (abs(v_cam.w) < 0.5);
                float shadow_str = legacy ? 0.0 : clamp(v_params.z, 0.0, 1.0);
                float spec_str = legacy ? 0.0 : clamp(v_params.w, 0.0, 1.0);
                float shininess = legacy ? 24.0 : max(abs(v_cam.w), 2.0);

                vec3 rgb = v_color.rgb;
                float alpha = v_color.a * mix(1.0, v_fresnel, clamp(v_params.y, 0.0, 1.0));

                if (shadow_str > 0.001 || spec_str > 0.001) {
                    vec3 up_ref = (abs(v.z) < 0.95) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
                    vec3 right_dir = normalize(cross(v, up_ref));
                    vec3 up_dir = cross(right_dir, v);
                    vec3 light_dir = normalize(v * 0.80 + right_dir * 0.35 + up_dir * 0.48);
                    vec3 half_dir = normalize(v + light_dir);

                    vec3 nf = (dot(n, v) >= 0.0) ? n : -n;
                    float ndotl = clamp(dot(nf, light_dir) * 0.5 + 0.5, 0.0, 1.0);
                    float hemi = mix(ndotv, ndotl, 0.58);

                    float diffuse_shade = smoothstep(0.04, 0.88, hemi);
                    float contour_rim = smoothstep(0.0, 0.36, ndotv);
                    float combined_light = diffuse_shade * (0.42 + 0.58 * contour_rim);
                    float dark_factor = mix(1.0, combined_light, shadow_str * 0.85);
                    rgb *= dark_factor;

                    float ndoth = clamp(dot(nf, half_dir), 0.0, 1.0);
                    float spec = pow(ndoth, shininess) * spec_str * smoothstep(0.05, 0.35, ndotv);
                    rgb = clamp(rgb + vec3(spec * 0.85), 0.0, 1.0);

                    float rim_boost = (1.0 - contour_rim) * shadow_str * 0.24;
                    alpha = clamp(alpha + spec * 0.25 + rim_boost * v_color.a, 0.0, 1.0);
                }

                fragColor = vec4(rgb, alpha);
            }
        ''')

        _inst_fresnel_shader = gpu.shader.create_from_info(f_info)
    except Exception:
        _inst_fresnel_shader = None
    return _inst_fresnel_shader


def get_instance_ubo():
    """Get or create reusable GPUUniformBuf for instanced drawing."""
    global _inst_ubo
    if _inst_ubo is None:
        try:
            _inst_ubo = gpu.types.GPUUniformBuf(bytes(MAX_GPU_INSTANCES * 28 * 4))
        except Exception:
            _inst_ubo = None
    return _inst_ubo


def update_instance_ubo(data_bytes):
    """Update reusable instancing UBO contents and return it."""
    ubo = get_instance_ubo()
    if ubo is not None:
        ubo.update(data_bytes)
    return ubo


def clear_shared_ibos():
    """Clear shared GPUIndexBuf and shader format caches."""
    global _inst_ubo
    _shared_ibo_cache.clear()
    _shader_fmt_cache.clear()
    _inst_ubo = None


def _get_shader_vbo_format(shader):
    """Return cached GPUVertFormat and expected attribute names for a shader."""
    cached = _shader_fmt_cache.get(shader)
    if cached is not None:
        return cached
    fmt = gpu.types.GPUVertFormat()
    attrs = []
    for name, attr_type in shader.attrs_info_get():
        if name is None:
            continue
        if attr_type == 'VEC3':
            fmt.attr_add(id=name, comp_type='F32', len=3, fetch_mode='FLOAT')
        elif attr_type == 'VEC2':
            fmt.attr_add(id=name, comp_type='F32', len=2, fetch_mode='FLOAT')
        elif attr_type == 'VEC4':
            fmt.attr_add(id=name, comp_type='F32', len=4, fetch_mode='FLOAT')
        elif attr_type == 'FLOAT':
            fmt.attr_add(id=name, comp_type='F32', len=1, fetch_mode='FLOAT')
        else:
            return None
        attrs.append(name)
    res = (fmt, tuple(attrs))
    _shader_fmt_cache[shader] = res
    return res


def _get_shared_ibo(indices, prim_type):
    """Return a cached GPUIndexBuf when invariant topology arrays are reused across frames."""
    stride = 2 if prim_type == 'LINES' else 3
    if not hasattr(indices, 'ctypes') or not hasattr(indices, 'size'):
        return None
    elem_len = indices.size // stride
    if elem_len <= 0:
        return None
    seq = indices.reshape(elem_len, stride) if indices.ndim != 2 or indices.shape[1] != stride else indices
    if hasattr(seq, 'flags') and (not seq.flags['C_CONTIGUOUS'] or seq.dtype.name != 'int32'):
        return None
    owner = indices.base if getattr(indices, 'base', None) is not None else indices
    ibo_key = (int(seq.ctypes.data), prim_type, int(indices.size))
    cached = _shared_ibo_cache.get(ibo_key)
    if cached is not None and cached[0] is owner:
        _shared_ibo_cache.move_to_end(ibo_key)
        return cached[1]
    ibo = gpu.types.GPUIndexBuf(type=prim_type, seq=seq)
    if len(_shared_ibo_cache) >= MAX_SHARED_IBOS:
        _shared_ibo_cache.popitem(last=False)
    _shared_ibo_cache[ibo_key] = (owner, ibo)
    return ibo


def _build_fast_batch(shader, prim_type, verts, normals, indices):
    """Construct GPUBatch directly via GPUVertBuf + shared GPUIndexBuf for ndarray inputs."""
    if not (hasattr(verts, 'ctypes') and hasattr(indices, 'ctypes')):
        return None
    if not verts.flags['C_CONTIGUOUS'] or verts.dtype.name != 'float32':
        return None
    v_count = verts.shape[0] if verts.ndim == 2 else (verts.size // 3)
    if v_count <= 0:
        return None
    fmt_info = _get_shader_vbo_format(shader)
    if fmt_info is None:
        return None
    fmt, attrs = fmt_info
    v_seq = verts.reshape(v_count, 3) if verts.ndim != 2 else verts
    if 'normal' in attrs:
        if normals is None or not hasattr(normals, 'ctypes') or not normals.flags['C_CONTIGUOUS'] or normals.dtype.name != 'float32':
            return None
        n_seq = normals.reshape(v_count, 3) if normals.ndim != 2 else normals
    else:
        n_seq = None

    ibo = _get_shared_ibo(indices, prim_type)
    if ibo is None:
        return None

    vbo = gpu.types.GPUVertBuf(fmt, v_count)
    vbo.attr_fill("pos", v_seq)
    if n_seq is not None:
        vbo.attr_fill("normal", n_seq)
    return gpu.types.GPUBatch(type=prim_type, buf=vbo, elem=ibo)


def get_entry_gpu_batch(entry, use_wireframe=False, use_fresnel=False, use_shading=False):
    """Lazily construct and return GPU batch for a frame cache entry.

    entry format: [ext_obj_name, verts, normals, indices, prim_type, batch_map, ...]
    """
    if not isinstance(entry, (list, tuple)) or len(entry) < 6:
        return None

    verts = entry[1]
    normals = entry[2]
    indices = entry[3]
    prim_type = entry[4]
    batch_map = entry[5]

    key = 'fresnel' if ((use_fresnel or use_shading) and not use_wireframe and normals is not None) else 'standard'

    if key in batch_map:
        return batch_map[key]

    if key == 'fresnel':
        shader = get_fresnel_shader()
        if shader is None:
            shader = get_shader()
        if shader is None:
            return None
        try:
            batch = _build_fast_batch(shader, prim_type, verts, normals, indices)
            if batch is None:
                data = {"pos": verts, "normal": normals} if normals is not None else {"pos": verts}
                batch = batch_for_shader(shader, prim_type, data, indices=indices)
            batch_map[key] = batch
            return batch
        except Exception:
            return None
    else:
        shader = get_shader()
        if shader is None:
            return None
        try:
            batch = _build_fast_batch(shader, prim_type, verts, None, indices)
            if batch is None:
                batch = batch_for_shader(shader, prim_type, {"pos": verts}, indices=indices)
            batch_map[key] = batch
            return batch
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Rigid base mesh cache (one GPUBatch in local space per object)
# ---------------------------------------------------------------------------

_rigid_mesh_cache = {}


def get_rigid_batch(object_name, use_wireframe=False, use_fresnel=False, signature=None):
    """Get cached local-space GPU batch for a rigid object."""
    key = (object_name, bool(use_wireframe), bool(use_fresnel))
    cached = _rigid_mesh_cache.get(key)
    if cached is None:
        return None
    if signature is not None:
        cached_sig = cached[5] if len(cached) > 5 else None
        if cached_sig != signature:
            del _rigid_mesh_cache[key]
            return None
    return cached[:5]


def put_rigid_batch(object_name, use_wireframe, use_fresnel, gpu_batch, aabb_min, aabb_max, centroid, prim_type, signature=None):
    """Store local-space GPU batch for a rigid object."""
    key = (object_name, bool(use_wireframe), bool(use_fresnel))
    _rigid_mesh_cache[key] = (gpu_batch, aabb_min, aabb_max, centroid, prim_type, signature)


def clear_rigid_cache(object_name=None):
    """Clear rigid batch cache for all or a specific object."""
    if object_name is None:
        _rigid_mesh_cache.clear()
    else:
        keys_to_del = [k for k in _rigid_mesh_cache if k[0] == object_name]
        for k in keys_to_del:
            del _rigid_mesh_cache[k]


# ---------------------------------------------------------------------------
# Rigid matrix cache ((object_name, frame) -> Matrix)
# ---------------------------------------------------------------------------

_rigid_matrix_cache = {}
_rigid_cached_indices = {}


def get_rigid_matrix(object_name, frame):
    return _rigid_matrix_cache.get((object_name, frame))


def put_rigid_matrix(object_name, frame, matrix):
    _rigid_matrix_cache[(object_name, frame)] = matrix
    _rigid_cached_indices.setdefault(object_name, set()).add(frame)


def get_rigid_cached_frames(object_name):
    idx_set = _rigid_cached_indices.get(object_name)
    return sorted(idx_set) if idx_set else []


def invalidate_rigid_matrices(object_name, frames):
    if not frames:
        return
    idx_set = _rigid_cached_indices.get(object_name)
    for f in frames:
        _rigid_matrix_cache.pop((object_name, f), None)
        if idx_set is not None:
            idx_set.discard(f)


def clear_rigid_matrix_cache(object_name=None):
    if object_name is None:
        _rigid_matrix_cache.clear()
        _rigid_cached_indices.clear()
    else:
        keys_to_del = [k for k in _rigid_matrix_cache if k[0] == object_name]
        for k in keys_to_del:
            del _rigid_matrix_cache[k]
        _rigid_cached_indices.pop(object_name, None)


# ---------------------------------------------------------------------------
# Live onion batches (current frame set)
# ---------------------------------------------------------------------------

def set_onion_batches(batches):
    global _onion_batches
    _onion_batches = batches


def get_onion_batches():
    return _onion_batches


def clear_onion_batches():
    global _onion_batches
    _onion_batches = []


# ---------------------------------------------------------------------------
# Ghost markers (user-pinned persistent frames)
# ---------------------------------------------------------------------------

def _resync_cpp_markers():
    """Rebuild C++ ghost marker vector so it matches _ghost_markers 1:1."""
    if not cb.is_available():
        return
    cb.marker_clear()
    for m in _ghost_markers:
        frame, verts, indices, prim_type, obj_names, centroid = m[:6]
        normals = m[6] if len(m) > 6 else None
        aabb_min = m[7] if len(m) > 7 else (0.0, 0.0, 0.0)
        aabb_max = m[8] if len(m) > 8 else (0.0, 0.0, 0.0)
        tag = ",".join(obj_names)
        cb.marker_add(
            frame, verts, normals, indices, prim_type,
            aabb_min, aabb_max, centroid, tag=tag
        )


def push_ghost_undo_snapshot():
    """Save a shallow snapshot of _ghost_markers before destructive removals/clears."""
    snap = [
        (m[0], m[1], m[2], m[3], list(m[4]), m[5], m[6], m[7], m[8])
        if len(m) >= 9 else (m[0], m[1], m[2], m[3], list(m[4]), m[5], None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for m in _ghost_markers
    ]
    _ghost_undo_bank.append(snap)
    if len(_ghost_undo_bank) > 16:
        del _ghost_undo_bank[0]


def sync_ghost_markers_to_count(target_count):
    """Synchronize runtime _ghost_markers with scene.onion_skin_settings.marker_count after Undo/Redo."""
    cur_count = len(_ghost_markers)
    if target_count <= 0:
        if cur_count > 0:
            push_ghost_undo_snapshot()
            clear_ghost_markers(keep_undo=True)
        return 0
    if target_count == cur_count:
        return cur_count
    if target_count < cur_count:
        push_ghost_undo_snapshot()
        del _ghost_markers[target_count:]
        _ghost_batches.clear()
        _resync_cpp_markers()
        return len(_ghost_markers)

    # target_count > cur_count (e.g., user pressed Undo after Remove/Clear Ghosts)
    for idx in range(len(_ghost_undo_bank) - 1, -1, -1):
        snap = _ghost_undo_bank[idx]
        if len(snap) == target_count:
            _ghost_markers[:] = [
                (m[0], m[1], m[2], m[3], list(m[4]), m[5], m[6], m[7], m[8])
                for m in snap
            ]
            _ghost_batches.clear()
            _resync_cpp_markers()
            return len(_ghost_markers)
    return cur_count


def add_ghost_marker(frame, verts, indices, prim_type, obj_names=None, normals=None, aabb_min=None, aabb_max=None, centroid=None):
    """Add a persistent ghost marker into C++ engine and local cache."""
    clean_names = list(dict.fromkeys(str(n) for n in (obj_names or []) if n))
    tag = ",".join(clean_names)

    if aabb_min is None:
        aabb_min = (0.0, 0.0, 0.0)
    if aabb_max is None:
        aabb_max = (0.0, 0.0, 0.0)
    if centroid is None:
        if len(verts) > 0:
            c = verts.mean(axis=0) if hasattr(verts, 'mean') else (0.0, 0.0, 0.0)
            centroid = (float(c[0]), float(c[1]), float(c[2]))
        else:
            centroid = (0.0, 0.0, 0.0)

    if cb.is_available():
        cb.marker_add(frame, verts, normals, indices, prim_type, aabb_min, aabb_max, centroid, tag=tag)

    _ghost_markers.append((frame, verts, indices, prim_type, clean_names, centroid, normals, aabb_min, aabb_max))
    _ghost_batches.clear()


def clear_ghost_markers(keep_undo=False):
    """Clear all ghost markers."""
    if cb.is_available():
        cb.marker_clear()
    _ghost_markers.clear()
    _ghost_batches.clear()
    if not keep_undo:
        _ghost_undo_bank.clear()


def remove_ghost_markers_for(frame, obj_names):
    """Remove ghost markers that match frame (or None for any frame) and obj_names."""
    obj_set = set(obj_names) if obj_names else set()
    to_remove = []
    for i, marker in enumerate(_ghost_markers):
        m_frame = marker[0]
        m_objs = marker[4] if len(marker) > 4 else []
        if frame is not None and m_frame != frame:
            continue
        if not obj_set or (m_objs and bool(obj_set & set(m_objs))):
            to_remove.append(i)

    if not to_remove:
        return 0

    for i in reversed(to_remove):
        _ghost_markers.pop(i)

    _ghost_batches.clear()
    if cb.is_available():
        c_frame = frame if frame is not None else -2147483648
        tag = ",".join(sorted(obj_set)) if obj_set else ""
        cb.marker_remove_for(c_frame, tag)
        if cb.marker_count() != len(_ghost_markers):
            _resync_cpp_markers()
    return len(to_remove)


def remove_ghost_markers_for_object(obj_keys):
    """Drop markers belonging solely to obj_keys, or strip obj_keys from multi-object tags."""
    rem_set = set(obj_keys) if obj_keys else set()
    if not rem_set or not _ghost_markers:
        return 0

    removed = 0
    changed = False
    new_markers = []
    for m in _ghost_markers:
        m_objs = m[4] if len(m) > 4 else []
        if not m_objs or not (rem_set & set(m_objs)):
            new_markers.append(m)
            continue
        remaining = [k for k in m_objs if k not in rem_set]
        if not remaining:
            removed += 1
            changed = True
        else:
            changed = True
            new_markers.append((
                m[0], m[1], m[2], m[3], remaining, m[5],
                m[6] if len(m) > 6 else None,
                m[7] if len(m) > 7 else (0.0, 0.0, 0.0),
                m[8] if len(m) > 8 else (0.0, 0.0, 0.0),
            ))

    if changed:
        _ghost_markers[:] = new_markers
        _ghost_batches.clear()
        _resync_cpp_markers()
    return removed


def rename_ghost_marker_keys(old_key, new_key, old_name=None, new_name=None):
    """Update object keys/names inside existing ghost markers after an ID datablock rename."""
    if not _ghost_markers:
        return False
    changed = False
    for m in _ghost_markers:
        m_objs = m[4] if len(m) > 4 else None
        if not m_objs:
            continue
        for idx_k, k in enumerate(m_objs):
            if old_key and k == old_key and new_key and k != new_key:
                m_objs[idx_k] = new_key
                changed = True
            elif old_name and k == old_name and new_name and k != new_name:
                m_objs[idx_k] = new_name
                changed = True
    if changed:
        _resync_cpp_markers()
    return changed


def get_ghost_markers():
    return _ghost_markers


def get_ghost_batch(index):
    """Get GPU batch for ghost marker at index, building lazily."""
    if index in _ghost_batches:
        return _ghost_batches[index]

    if index < 0 or index >= len(_ghost_markers):
        return None

    marker = _ghost_markers[index]
    verts = marker[1]
    indices = marker[2]
    prim_type = marker[3]
    shader = get_shader()
    if shader is None:
        return None

    try:
        batch = batch_for_shader(shader, prim_type, {"pos": verts}, indices=indices)
        _ghost_batches[index] = batch
        return batch
    except Exception:
        return None


def get_ghost_centroid(index):
    """Return (cx, cy, cz) centroid for ghost marker index in O(1) time."""
    if 0 <= index < len(_ghost_markers):
        marker = _ghost_markers[index]
        if len(marker) > 5 and marker[5] is not None:
            return marker[5]
        verts = marker[1]
        n = len(verts)
        if n > 0:
            cx = float(sum(v[0] for v in verts)) / n
            cy = float(sum(v[1] for v in verts)) / n
            cz = float(sum(v[2] for v in verts)) / n
            return (cx, cy, cz)

    if cb.is_available() and 0 <= index < cb.marker_count():
        return cb.marker_get_centroid(index)
    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup():
    global _shader, _fresnel_shader, _inst_fresnel_shader, _inst_flat_shader, _inst_ubo
    _shader = None
    _fresnel_shader = None
    _inst_fresnel_shader = None
    _inst_flat_shader = None
    _inst_ubo = None
    clear_shared_ibos()
    clear_onion_batches()
    clear_ghost_markers()
    clear_frame_cache()
    clear_rigid_cache()
    clear_rigid_matrix_cache()
    if cb.is_available():
        lib = cb._get_active_lib()
        if lib:
            lib.onion_core_shutdown()
