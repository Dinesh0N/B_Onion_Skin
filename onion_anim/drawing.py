# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU drawing for onion skin - Ultra Performance."""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from . import cache

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# -----------------------------------------------------------------------------
# Mesh Collection
# -----------------------------------------------------------------------------

def get_mesh_objects(context):
    """Get all mesh objects (deduplicated)."""
    settings = context.scene.onion_skin_settings
    meshes = []
    seen = set()
    
    for item in context.scene.onion_skin_objects:
        obj = item.object
        if not obj or obj.name in seen:
            continue
        
        if obj.type == 'MESH':
            meshes.append(obj)
            seen.add(obj.name)
        elif obj.type == 'ARMATURE' and settings.include_children:
            for child in obj.children:
                if child.type == 'MESH' and child.name not in seen:
                    meshes.append(child)
                    seen.add(child.name)
    
    return meshes


def extract_all_meshes_merged(objects, depsgraph, wireframe=False):
    """Merge ALL objects into single vertex/index arrays."""
    if not HAS_NUMPY:
        return extract_all_meshes_simple(objects, depsgraph, wireframe)
    
    all_verts = []
    all_indices = []
    vertex_offset = 0
    
    for obj in objects:
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            
            if not mesh or len(mesh.vertices) == 0:
                continue
            
            num_verts = len(mesh.vertices)
            mat = eval_obj.matrix_world
            
            verts_co = np.empty(num_verts * 3, dtype=np.float32)
            mesh.vertices.foreach_get('co', verts_co)
            verts_co = verts_co.reshape(-1, 3)
            
            mat_np = np.array(mat.to_3x3(), dtype=np.float32)
            loc_np = np.array(mat.translation, dtype=np.float32)
            verts_world = verts_co @ mat_np.T + loc_np
            
            all_verts.append(verts_world)
            
            if wireframe:
                num_edges = len(mesh.edges)
                if num_edges > 0:
                    edge_data = np.empty(num_edges * 2, dtype=np.int32)
                    mesh.edges.foreach_get('vertices', edge_data)
                    edge_data += vertex_offset
                    all_indices.append(edge_data.reshape(-1, 2))
            else:
                mesh.calc_loop_triangles()
                num_tris = len(mesh.loop_triangles)
                if num_tris > 0:
                    tri_data = np.empty(num_tris * 3, dtype=np.int32)
                    mesh.loop_triangles.foreach_get('vertices', tri_data)
                    tri_data += vertex_offset
                    all_indices.append(tri_data.reshape(-1, 3))
            
            vertex_offset += num_verts
            eval_obj.to_mesh_clear()
            
        except Exception:
            continue
    
    if not all_verts or not all_indices:
        return None, None, None
    
    merged_verts = np.vstack(all_verts)
    merged_indices = np.vstack(all_indices)
    
    verts_list = [tuple(v) for v in merged_verts]
    indices_list = [tuple(idx) for idx in merged_indices]
    
    prim_type = 'LINES' if wireframe else 'TRIS'
    
    return verts_list, indices_list, prim_type


def extract_all_meshes_simple(objects, depsgraph, wireframe=False):
    """Fallback without numpy."""
    all_verts = []
    all_indices = []
    vertex_offset = 0
    
    for obj in objects:
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            
            if not mesh or len(mesh.vertices) == 0:
                continue
            
            mat = eval_obj.matrix_world
            
            for v in mesh.vertices:
                all_verts.append((mat @ v.co)[:])
            
            if wireframe:
                for e in mesh.edges:
                    all_indices.append((e.vertices[0] + vertex_offset, 
                                       e.vertices[1] + vertex_offset))
            else:
                mesh.calc_loop_triangles()
                for t in mesh.loop_triangles:
                    all_indices.append((t.vertices[0] + vertex_offset,
                                       t.vertices[1] + vertex_offset,
                                       t.vertices[2] + vertex_offset))
            
            vertex_offset += len(mesh.vertices)
            eval_obj.to_mesh_clear()
            
        except Exception:
            continue
    
    if not all_verts or not all_indices:
        return None, None, None
    
    prim_type = 'LINES' if wireframe else 'TRIS'
    return all_verts, all_indices, prim_type


def cache_frame(context, frame):
    """Cache a frame's merged mesh data."""
    if cache.is_frame_cached(frame):
        return True
    
    scene = context.scene
    settings = scene.onion_skin_settings
    
    objects = get_mesh_objects(context)
    if not objects:
        return False
    
    scene.frame_set(frame)
    depsgraph = context.evaluated_depsgraph_get()
    
    verts, indices, prim_type = extract_all_meshes_merged(
        objects, depsgraph, settings.use_wireframe
    )
    
    if verts and indices:
        cache.add_to_cache(frame, verts, indices, prim_type)
        return True
    return False


# -----------------------------------------------------------------------------
# Frame Calculation
# -----------------------------------------------------------------------------

def get_frame_range(settings, scene):
    if settings.use_frame_range:
        return settings.frame_range_start, settings.frame_range_end
    return scene.frame_start, scene.frame_end


def get_needed_frames(current, settings, scene):
    range_start, range_end = get_frame_range(settings, scene)
    
    if settings.use_frame_range:
        if current < range_start or current > range_end:
            return []
    
    frames = []
    step = settings.frame_step
    
    if settings.show_before:
        max_before = settings.frames_before
        inv_max = 1.0 / max_before if max_before > 0 else 0
        for i in range(step, max_before + 1, step):
            f = current - i
            if f >= range_start:
                frames.append((f, 'before', i * inv_max, i))
    
    if settings.show_after:
        max_after = settings.frames_after
        inv_max = 1.0 / max_after if max_after > 0 else 0
        for i in range(step, max_after + 1, step):
            f = current + i
            if f <= range_end:
                frames.append((f, 'after', i * inv_max, i))
    
    return frames


def ensure_frames_cached(context):
    scene = context.scene
    settings = scene.onion_skin_settings
    current = scene.frame_current
    
    if current == cache.get_last_frame():
        return
    
    cache.set_last_frame(current)
    
    original_frame = current
    needed = get_needed_frames(current, settings, scene)
    
    for frame, _, _, _ in needed:
        cache_frame(context, frame)
    
    if scene.frame_current != original_frame:
        scene.frame_set(original_frame)


# -----------------------------------------------------------------------------
# Alpha Calculation
# -----------------------------------------------------------------------------

def calculate_alpha(base_alpha, t, falloff, curve):
    if curve == 'linear':
        factor = 1.0 - t * falloff
    elif curve == 'exponential':
        factor = (1.0 - t) ** (1.0 + falloff * 2.0)
    else:
        ts = t * falloff
        if ts >= 1.0:
            factor = 0.1
        elif ts <= 0.0:
            factor = 1.0
        else:
            factor = 1.0 - ts * ts * (3.0 - 2.0 * ts)
    
    return base_alpha * max(0.1, factor)


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------

def draw_onion_skins():
    context = bpy.context
    
    if not context.area or context.area.type != 'VIEW_3D':
        return
    
    scene = context.scene
    if not hasattr(scene, 'onion_skin_settings'):
        return
    
    settings = scene.onion_skin_settings
    if not settings.enabled or not scene.onion_skin_objects:
        return
    
    ensure_frames_cached(context)
    
    frames_to_draw = get_needed_frames(scene.frame_current, settings, scene)
    if not frames_to_draw:
        return
    
    frames_to_draw.sort(key=lambda x: -x[3])
    
    shader = cache.get_shader()
    
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE' if settings.use_xray else 'LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    
    is_wire = settings.use_wireframe
    if is_wire:
        gpu.state.line_width_set(1.5)
    
    shader.bind()
    
    color_before = tuple(settings.color_before)
    color_after = tuple(settings.color_after)
    falloff = settings.opacity_falloff
    curve = getattr(settings, 'falloff_curve', 'smooth')
    
    for frame, direction, t, dist in frames_to_draw:
        batch = cache.get_batch(frame)
        if batch is None:
            continue
        
        base = color_before if direction == 'before' else color_after
        alpha = calculate_alpha(base[3], t, falloff, curve)
        
        shader.uniform_float("color", (base[0], base[1], base[2], alpha))
        batch.draw(shader)
    
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    
    if is_wire:
        gpu.state.line_width_set(1.0)


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

_draw_handler = None


def register():
    global _draw_handler
    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_onion_skins, (), 'WINDOW', 'POST_VIEW'
    )


def unregister():
    global _draw_handler
    if _draw_handler:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None
