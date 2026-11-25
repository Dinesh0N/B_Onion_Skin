# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU drawing and C++ backend integration for onion skin overlays.

Optimized with:
- C++ SIMD (AVX2/FMA) vertex/normal transformations and mesh merging.
- Hardware-accelerated camera view frustum culling.
- GPU VRAM batch pooling and 112-byte push constant shaders.
- Smart dependency graph tracking to avoid unnecessary cache flushes.
"""

import re
import time
import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras.view3d_utils import location_3d_to_region_2d
from bpy.app.handlers import persistent
from mathutils import Matrix, Euler, Vector, Quaternion

from . import cache, properties
from .c_core import core_bridge as cb

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


_vis_cache_rev = 0
_vis_info_cache = {}
_playback_build_cache = None
_rigid_analytic_cache = {}
_dg_dep_cache = None
_delta_cap_cache = {}
_stale_offscreen_action_roots = set()
_last_modal_eval_end = 0.0

_INST_MAX = cache.MAX_GPU_INSTANCES
if HAS_NUMPY:
    _INST_MATS = np.zeros((_INST_MAX, 16), dtype=np.float32)
    _INST_OFFS = np.zeros((_INST_MAX, 3), dtype=np.float32)
    _INST_MINS = np.zeros((_INST_MAX, 3), dtype=np.float32)
    _INST_MAXS = np.zeros((_INST_MAX, 3), dtype=np.float32)
    _INST_COLS = np.zeros((_INST_MAX, 4), dtype=np.float32)
    _INST_PARAMS = np.zeros((_INST_MAX, 4), dtype=np.float32)
    _INST_CAMWS = np.zeros(_INST_MAX, dtype=np.float32)
    _INST_UBO_OUT = np.zeros((_INST_MAX, 28), dtype=np.float32)
else:
    _INST_MATS = _INST_OFFS = _INST_MINS = _INST_MAXS = None
    _INST_COLS = _INST_PARAMS = _INST_CAMWS = _INST_UBO_OUT = None


def bump_vis_cache_rev():
    """Invalidate lightweight per-revision visibility and playback lookup caches."""
    global _vis_cache_rev, _playback_build_cache, _dg_dep_cache
    _vis_cache_rev += 1
    _vis_info_cache.clear()
    _playback_build_cache = None
    _rigid_analytic_cache.clear()
    _dg_dep_cache = None
    _delta_cap_cache.clear()


# ---------------------------------------------------------------------------
# Helpers & Identity
# ---------------------------------------------------------------------------

_GEO_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}

_ACTIVE_MODAL_PREFIXES = (
    'TRANSFORM_OT_',
    'GRAPH_OT_',
    'ACTION_OT_',
    'NLA_OT_',
    'POSE_OT_',
)

_DEFORMING_MODIFIER_TYPES = {
    'ARMATURE', 'CAST', 'CURVE', 'DISPLACE', 'HOOK', 'LAPLACIANDEFORM',
    'LATTICE', 'MESH_CACHE', 'MESH_DEFORM', 'MESH_SEQUENCE_CACHE',
    'SHRINKWRAP', 'SIMPLE_DEFORM', 'SMOOTH', 'CORRECTIVE_SMOOTH',
    'SURFACE_DEFORM', 'WARP', 'WAVE', 'CLOTH', 'COLLISION',
    'DYNAMIC_PAINT', 'FLUID', 'OCEAN', 'PARTICLE_SYSTEM', 'SOFT_BODY',
    'SURFACE', 'NODES', 'BUILD', 'EXPLODE',
}

_OBJECT_REF_MODIFIER_ATTRS = (
    'object', 'mirror_object', 'offset_object', 'target',
    'start_cap', 'end_cap', 'curve', 'origin',
)


def get_object_key(obj):
    """Return a unique, stable cache key distinguishing local, override, and linked datablocks."""
    if obj is None:
        return ""
    try:
        lib = getattr(obj, 'library', None)
        if lib is not None:
            return f"{obj.name} [{lib.filepath}]"
        return obj.name
    except Exception:
        return ""


def _get_data_key(data):
    """Return a unique cache key for a Mesh or other ID datablock."""
    if data is None:
        return ""
    try:
        lib = getattr(data, 'library', None)
        if lib is not None:
            return f"{data.name} [{lib.filepath}]"
        return data.name
    except Exception:
        return ""


def screen_animation_playing(context):
    """Check if viewport animation is currently playing across context screen or any window."""
    try:
        scr = getattr(context, 'screen', None) if context is not None else None
        if scr is not None and scr.is_animation_playing:
            return True
        wm = getattr(context, 'window_manager', None) if context is not None else None
        if wm is None:
            wm = getattr(bpy.context, 'window_manager', None)
        if wm is not None:
            for win in wm.windows:
                wscr = getattr(win, 'screen', None)
                if wscr is not None and wscr.is_animation_playing:
                    return True
    except (AttributeError, ReferenceError):
        pass
    return False


def _empty_has_gn_geometry(obj):
    """Return True if obj is an EMPTY with active Geometry Nodes modifiers."""
    if obj is None or obj.type != 'EMPTY':
        return False
    return any(
        getattr(m, 'type', None) == 'NODES' and getattr(m, 'show_viewport', True)
        for m in obj.modifiers
    )


def _empty_is_collection_instance(obj):
    """Return True if obj is an EMPTY that instances a valid collection."""
    if obj is None or obj.type != 'EMPTY':
        return False
    coll = getattr(obj, 'instance_collection', None)
    if obj.instance_type != 'COLLECTION' or coll is None:
        return False
    if getattr(coll, 'is_missing', False):
        return False
    return True


def _is_valid_object(obj):
    """Return True if obj is valid and not a missing linked library datablock."""
    if obj is None:
        return False
    try:
        if getattr(obj, 'is_missing', False):
            return False
        if obj.type in _GEO_TYPES or obj.type == 'ARMATURE':
            d = getattr(obj, 'data', None)
            if d is None or getattr(d, 'is_missing', False):
                return False
    except Exception:
        return False
    return True


def can_add_to_onion(obj):
    """Return True if obj is a type the onion skin system can handle."""
    if not _is_valid_object(obj):
        return False
    if obj.name.startswith('WGT-'):
        return False
    if obj.type in _GEO_TYPES or obj.type == 'ARMATURE':
        return True
    if obj.type == 'EMPTY':
        return _empty_is_collection_instance(obj) or _empty_has_gn_geometry(obj)
    return False


def _get_metaball_base_name(name):
    """Return Blender metaball family base name (strip trailing .digits suffix)."""
    base, sep, ext = name.rpartition('.')
    if sep and ext.isdigit():
        return base
    return name


def _get_metaball_family(obj, scene=None, metaball_map=None):
    """Return all valid META objects sharing obj's metaball family."""
    if not obj or obj.type != 'META':
        return []
    base = _get_metaball_base_name(obj.name)
    lib = getattr(obj, 'library', None)
    if metaball_map is not None:
        family = list(metaball_map.get((lib, base), ()))
        if obj not in family:
            family.append(obj)
        return family
    sc = scene if scene is not None else getattr(bpy.context, 'scene', None)
    pool = sc.objects if sc is not None else bpy.data.objects
    family = [
        o for o in pool
        if o.type == 'META'
        and getattr(o, 'library', None) == lib
        and _get_metaball_base_name(o.name) == base
        and _is_valid_object(o)
    ]
    if obj not in family:
        family.append(obj)
    return family


def _get_metaball_mother(obj, scene=None, metaball_map=None):
    """Return the basis (mother) metaball object that holds the family's polygonized mesh."""
    if not obj or obj.type != 'META':
        return obj
    base = _get_metaball_base_name(obj.name)
    family = _get_metaball_family(obj, scene, metaball_map=metaball_map)
    for o in family:
        if o.name == base:
            return o
    return min(family, key=lambda o: o.name) if family else obj


def _iter_modifier_targets(obj):
    """Yield external Object datablocks and NodeTrees referenced by obj.modifiers."""
    objs = []
    trees = []
    for mod in getattr(obj, 'modifiers', ()):
        for attr in _OBJECT_REF_MODIFIER_ATTRS:
            tgt = getattr(mod, attr, None)
            if isinstance(tgt, bpy.types.Object) and _is_valid_object(tgt):
                objs.append(tgt)
        if mod.type == 'NODES':
            ng = getattr(mod, 'node_group', None)
            if ng is not None:
                trees.append(ng)
                try:
                    for item in ng.interface.items_tree:
                        if getattr(item, 'item_type', None) == 'SOCKET' and getattr(item, 'socket_type', '') == 'NodeSocketObject':
                            val = mod.get(item.identifier)
                            if isinstance(val, bpy.types.Object) and _is_valid_object(val):
                                objs.append(val)
                except Exception:
                    pass
    return objs, trees


def _collect_nested_node_trees(tree, visited_trees=None, ref_objs=None):
    """Collect nested node groups and object references inside a GeometryNodeTree."""
    if visited_trees is None:
        visited_trees = set()
    if ref_objs is None:
        ref_objs = set()
    if tree is None or tree in visited_trees:
        return visited_trees, ref_objs
    visited_trees.add(tree)
    for node in getattr(tree, 'nodes', ()):
        sub_tree = getattr(node, 'node_tree', None)
        if sub_tree is not None:
            _collect_nested_node_trees(sub_tree, visited_trees, ref_objs)
        for attr in ('object', 'collection'):
            val = getattr(node, attr, None)
            if isinstance(val, bpy.types.Object) and _is_valid_object(val):
                ref_objs.add(val)
        for sock in getattr(node, 'inputs', ()):
            if getattr(sock, 'type', None) == 'OBJECT':
                val = getattr(sock, 'default_value', None)
                if isinstance(val, bpy.types.Object) and _is_valid_object(val):
                    ref_objs.add(val)
    return visited_trees, ref_objs


def _build_scene_lookup_maps(scene):
    """Build (dep_arm_map, vis_arm_map, children_map, meta_map, scene_objs) in a single pass over scene.objects."""
    dep_arm_map = {}
    vis_arm_map = {}
    children_map = {}
    meta_map = {}
    scene_objs = set()
    if scene is None:
        return dep_arm_map, vis_arm_map, children_map, meta_map, scene_objs
    for sobj in scene.objects:
        if not _is_valid_object(sobj):
            continue
        scene_objs.add(sobj)
        parent = sobj.parent
        if parent is not None:
            children_map.setdefault(parent, []).append(sobj)
        otype = sobj.type
        if otype == 'META':
            mkey = (getattr(sobj, 'library', None), _get_metaball_base_name(sobj.name))
            meta_map.setdefault(mkey, []).append(sobj)
        if otype in _GEO_TYPES and not sobj.name.startswith('WGT-'):
            for m in sobj.modifiers:
                if m.type == 'ARMATURE':
                    arm = getattr(m, 'object', None)
                    if arm is not None:
                        dep_arm_map.setdefault(arm, []).append(sobj)
                        if m.show_viewport:
                            vis_arm_map.setdefault(arm, []).append(sobj)
    return dep_arm_map, vis_arm_map, children_map, meta_map, scene_objs


def _build_armature_children_map(scene):
    """Map each Armature object to scene geometry objects deformed by an Armature modifier."""
    return _build_scene_lookup_maps(scene)[0]


def _gather_object_dependencies(
    obj,
    scene=None,
    armature_children_map=None,
    children_map=None,
    metaball_map=None,
    scene_objs=None,
):
    """Collect all Objects, Collections, and NodeTrees affecting obj's evaluated geometry/pose."""
    dep_objs = set()
    dep_colls = set()
    dep_trees = set()

    if not _is_valid_object(obj):
        return dep_objs, dep_colls, dep_trees

    queue = [obj]
    sc = scene if scene is not None else getattr(bpy.context, 'scene', None)

    while queue:
        curr = queue.pop()
        if curr in dep_objs or not _is_valid_object(curr):
            continue
        dep_objs.add(curr)

        if children_map is not None and (scene_objs is None or curr in scene_objs):
            curr_children = children_map.get(curr, ())
        else:
            curr_children = getattr(curr, 'children', ())
        for child in curr_children:
            if not child.name.startswith('WGT-') and _is_valid_object(child):
                queue.append(child)

        if curr.type == 'ARMATURE':
            if armature_children_map is not None:
                for sobj in armature_children_map.get(curr, ()):
                    queue.append(sobj)
            elif sc is not None:
                for sobj in sc.objects:
                    if sobj.type in _GEO_TYPES and not sobj.name.startswith('WGT-') and _is_valid_object(sobj):
                        for m in sobj.modifiers:
                            if m.type == 'ARMATURE' and getattr(m, 'object', None) == curr:
                                queue.append(sobj)
                                break

        if curr.parent and _is_valid_object(curr.parent):
            queue.append(curr.parent)

        for con in getattr(curr, 'constraints', ()):
            tgt = getattr(con, 'target', None)
            if isinstance(tgt, bpy.types.Object) and _is_valid_object(tgt):
                queue.append(tgt)

        mod_objs, mod_trees = _iter_modifier_targets(curr)
        for mo in mod_objs:
            queue.append(mo)
        for mt in mod_trees:
            trees, node_ref_objs = _collect_nested_node_trees(mt)
            dep_trees.update(trees)
            for no in node_ref_objs:
                queue.append(no)

        for mod in getattr(curr, 'modifiers', ()):
            mcoll = getattr(mod, 'collection', None)
            if mcoll is not None and not getattr(mcoll, 'is_missing', False):
                dep_colls.add(mcoll)
                for cobj in getattr(mcoll, 'all_objects', ()):
                    if not cobj.name.startswith('WGT-') and _is_valid_object(cobj):
                        queue.append(cobj)

        if curr.type == 'EMPTY' and _empty_is_collection_instance(curr):
            coll = curr.instance_collection
            dep_colls.add(coll)
            for cobj in coll.all_objects:
                if not cobj.name.startswith('WGT-') and _is_valid_object(cobj):
                    queue.append(cobj)

        if curr.type == 'META':
            use_mmap = metaball_map if (metaball_map is not None and (scene_objs is None or curr in scene_objs)) else None
            for sib in _get_metaball_family(curr, sc, metaball_map=use_mmap):
                queue.append(sib)

    return dep_objs, dep_colls, dep_trees


def _get_armature_candidate_meshes(obj, settings, scene=None, children_map=None, arm_map=None):
    """Return all geometry child or Armature-deformed objects of an Armature regardless of Outliner visibility."""
    result = []
    seen = set()

    def _collect(parent):
        p_children = children_map.get(parent, ()) if children_map is not None else parent.children
        for child in p_children:
            if _is_valid_object(child) and child.type in _GEO_TYPES:
                if not child.name.startswith('WGT-') and child not in seen:
                    seen.add(child)
                    result.append(child)
            if (children_map.get(child) if children_map is not None else child.children):
                _collect(child)

    _collect(obj)
    if arm_map is not None:
        for sobj in arm_map.get(obj, ()):
            if sobj not in seen:
                seen.add(sobj)
                result.append(sobj)
    else:
        sc = scene if scene is not None else getattr(bpy.context, 'scene', None)
        if sc is not None:
            for sobj in sc.objects:
                if sobj not in seen and sobj.type in _GEO_TYPES and not sobj.name.startswith('WGT-') and _is_valid_object(sobj):
                    for m in sobj.modifiers:
                        if m.show_viewport and m.type == 'ARMATURE' and getattr(m, 'object', None) == obj:
                            seen.add(sobj)
                            result.append(sobj)
                            break
    return result


def resolve_mesh_objects(
    obj,
    settings,
    scene=None,
    children_map=None,
    arm_map=None,
    metaball_map=None,
    view_layer=None,
    vis_cols=None,
    include_hidden=False,
    v3d=None,
):
    """Return a list of geometry source objects for obj."""
    if not _is_valid_object(obj):
        return []

    if obj.type == 'META':
        mother = _get_metaball_mother(obj, scene, metaball_map=metaball_map)
        return [mother] if _is_valid_object(mother) else [obj]

    if obj.type in _GEO_TYPES:
        return [obj]

    if obj.type == 'ARMATURE' and settings.include_children:
        result = _get_armature_candidate_meshes(
            obj, settings, scene=scene, children_map=children_map, arm_map=arm_map
        )
        if not include_hidden:
            vl = view_layer if view_layer is not None else getattr(bpy.context, 'view_layer', None)
            if vl is not None:
                if vis_cols is None:
                    vis_cols = _get_visible_collections(vl)
                has_lv = v3d is not None and getattr(v3d, 'local_view', None) is not None
                filtered = []
                for m in result:
                    if not _is_outliner_visible(m, vl, vis_cols=vis_cols):
                        continue
                    if has_lv and not _safe_visible_get(m, vl, v3d):
                        continue
                    filtered.append(m)
                return filtered
        return result

    if obj.type == 'EMPTY':
        if _empty_is_collection_instance(obj) and settings.include_children:
            result = []
            for cobj in obj.instance_collection.all_objects:
                if _is_valid_object(cobj) and cobj.type in _GEO_TYPES:
                    if not cobj.name.startswith('WGT-') and not getattr(cobj, 'hide_viewport', False):
                        result.append(cobj)
            return result

        if _empty_has_gn_geometry(obj):
            return [obj]

    return []


# ---------------------------------------------------------------------------
# Action, Rigid Classification & Topology Cache
# ---------------------------------------------------------------------------

_mesh_topology_cache = {}


_merged_topology_cache = {}


def clear_topology_cache(target_key=None):
    """Clear cached topology indices for all meshes or keys matching target_key."""
    if target_key is None:
        _mesh_topology_cache.clear()
        _merged_topology_cache.clear()
        cache.clear_shared_ibos()
    else:
        keys_to_del = [
            k for k in _mesh_topology_cache
            if k[0] == target_key or (len(k) > 1 and k[1] == target_key)
        ]
        for k in keys_to_del:
            del _mesh_topology_cache[k]
        if keys_to_del:
            _merged_topology_cache.clear()


def collect_action_fcurves(act, anim_data=None, include_muted=False):
    """Retrieve active fcurves from Blender 5.2 layered actions (filtered by slot) or legacy actions."""
    res = []
    if not act:
        return res

    slot_handle = None
    if anim_data is not None:
        slot = getattr(anim_data, 'action_slot', None)
        if slot is not None:
            slot_handle = getattr(slot, 'handle', None)

    if hasattr(act, 'layers'):
        for layer in act.layers:
            if not include_muted and getattr(layer, 'mute', False):
                continue
            for strip in layer.strips:
                for bag in getattr(strip, 'channelbags', []):
                    if slot_handle is not None and getattr(bag, 'slot_handle', None) != slot_handle:
                        continue
                    for fc in bag.fcurves:
                        if not include_muted and getattr(fc, 'mute', False):
                            continue
                        res.append(fc)
    elif hasattr(act, 'fcurves'):
        for fc in act.fcurves:
            if not include_muted and getattr(fc, 'mute', False):
                continue
            res.append(fc)
    return res


def _has_active_nla(anim):
    """Return True if animation_data has at least one unmuted NLA track with strips."""
    if not anim:
        return False
    for track in getattr(anim, 'nla_tracks', ()):
        if not getattr(track, 'mute', False) and len(getattr(track, 'strips', ())) > 0:
            return True
    return False


def _modifier_param_signature(m):
    """Extract key structural/parametric attributes of a modifier for change detection."""
    ng = getattr(m, 'node_group', None)
    return (
        m.name,
        m.type,
        m.show_viewport,
        getattr(m, 'levels', None),
        getattr(m, 'render_levels', None),
        getattr(m, 'subdivision_type', None),
        getattr(m, 'width', None),
        getattr(m, 'segments', None),
        getattr(m, 'thickness', None),
        getattr(m, 'offset', None),
        getattr(m, 'count', None),
        getattr(m, 'factor', None),
        getattr(m, 'strength', None),
        getattr(m, 'operation', None),
        tuple(getattr(m, 'use_axis', ())) if hasattr(m, 'use_axis') else None,
        _get_data_key(ng) if ng else None,
    )


def get_object_geometry_signature(obj):
    """Compute a lightweight signature of an object's modifiers and mesh data."""
    if not obj:
        return ()
    mods = tuple(_modifier_param_signature(m) for m in getattr(obj, 'modifiers', ()))
    mesh_info = ()
    if getattr(obj, 'data', None):
        d = obj.data
        if hasattr(d, 'vertices'):
            mesh_info = (len(d.vertices), len(getattr(d, 'edges', ())), len(getattr(d, 'polygons', ())))
        elif hasattr(d, 'splines'):
            mesh_info = tuple(
                len(s.bezier_points) if hasattr(s, 'bezier_points') else len(s.points)
                for s in d.splines
            )
        elif hasattr(d, 'body'):
            mesh_info = (d.body,)
    return (mods, mesh_info)


def is_object_rigid(obj):
    """Determine whether obj is a rigid object (transform-only animation, no vertex deforms)."""
    if not obj or obj.type not in _GEO_TYPES or obj.type == 'META':
        return False

    for mod in obj.modifiers:
        if not mod.show_viewport:
            continue
        if mod.type in _DEFORMING_MODIFIER_TYPES:
            return False
        for attr in _OBJECT_REF_MODIFIER_ATTRS:
            if getattr(mod, attr, None) is not None:
                return False
        if getattr(mod, 'collection', None) is not None:
            return False

    data = getattr(obj, 'data', None)
    if data is not None:
        sk = getattr(data, 'shape_keys', None)
        if sk is not None:
            sk_anim = getattr(sk, 'animation_data', None)
            if sk_anim and (
                len(collect_action_fcurves(sk_anim.action, sk_anim)) > 0
                or getattr(sk_anim, 'drivers', None)
                or _has_active_nla(sk_anim)
            ):
                return False
        d_anim = getattr(data, 'animation_data', None)
        if d_anim and (
            len(collect_action_fcurves(d_anim.action, d_anim)) > 0
            or getattr(d_anim, 'drivers', None)
            or _has_active_nla(d_anim)
        ):
            return False

    anim = getattr(obj, 'animation_data', None)
    if anim:
        if getattr(anim, 'drivers', None):
            for drv in anim.drivers:
                if getattr(drv, 'data_path', '').startswith(('modifiers', 'data')):
                    return False
        if anim.action:
            for fc in collect_action_fcurves(anim.action, anim):
                if fc.data_path.startswith(('modifiers', 'data')):
                    return False
        if _has_active_nla(anim):
            for track in anim.nla_tracks:
                if not track.mute:
                    for strip in track.strips:
                        if getattr(strip, 'action', None):
                            for fc in collect_action_fcurves(strip.action):
                                if fc.data_path.startswith(('modifiers', 'data')):
                                    return False
    return True


# ---------------------------------------------------------------------------
# Fast Mesh Data & Analytical Transform Evaluation
# ---------------------------------------------------------------------------

def _extract_mesh_data(mesh, obj_key, xform, wireframe, need_normals, allow_topo_cache=True, allow_fast_cpp_normals=False):
    """Read vertices, optional normals, and indices for a Blender Mesh datablock."""
    num_verts = len(mesh.vertices)
    if num_verts == 0:
        return None

    topo_key = (
        obj_key,
        _get_data_key(mesh),
        wireframe,
        num_verts,
        len(mesh.edges),
        len(mesh.polygons),
    )

    if allow_topo_cache and topo_key in _mesh_topology_cache:
        indices = _mesh_topology_cache[topo_key]
    else:
        if wireframe:
            num_edges = len(mesh.edges)
            if num_edges == 0:
                return None
            indices = np.empty(num_edges * 2, dtype=np.int32)
            ev_attr = mesh.attributes.get('.edge_verts')
            if (
                ev_attr is not None
                and ev_attr.data_type == 'INT32_2D'
                and len(ev_attr.data) == num_edges
            ):
                ev_attr.data.foreach_get('value', indices)
            else:
                mesh.edges.foreach_get('vertices', indices)
        else:
            indices = None
            num_polys = len(mesh.polygons)
            cv_attr = mesh.attributes.get('.corner_vert')
            if cv_attr is not None and cv_attr.data_type == 'INT' and num_polys > 0:
                nl = len(cv_attr.data)
                if nl == num_polys * 3:
                    indices = np.empty(nl, dtype=np.int32)
                    cv_attr.data.foreach_get('value', indices)
                elif num_polys >= 2048 and nl > 0 and cb.is_available():
                    lt = np.empty(num_polys, dtype=np.int32)
                    mesh.polygons.foreach_get('loop_total', lt)
                    if int(lt.min()) >= 3 and int(lt.max()) <= 4:
                        lv = np.empty(nl, dtype=np.int32)
                        cv_attr.data.foreach_get('value', lv)
                        total_tris = int(lt.sum()) - 2 * num_polys
                        indices = cb.triangulate_polys(lv, lt, total_tris)
            if indices is None:
                mesh.calc_loop_triangles()
                num_tris = len(mesh.loop_triangles)
                if num_tris == 0:
                    return None
                indices = np.empty(num_tris * 3, dtype=np.int32)
                mesh.loop_triangles.foreach_get('vertices', indices)

        if allow_topo_cache:
            _mesh_topology_cache[topo_key] = indices

    verts_co = np.empty(num_verts * 3, dtype=np.float32)
    pos_attr = mesh.attributes.get('position')
    if (
        pos_attr is not None
        and pos_attr.data_type == 'FLOAT_VECTOR'
        and pos_attr.domain == 'POINT'
        and len(pos_attr.data) == num_verts
    ):
        pos_attr.data.foreach_get('vector', verts_co)
    else:
        mesh.vertices.foreach_get('co', verts_co)

    verts_no = None
    if not wireframe and need_normals:
        if allow_fast_cpp_normals and num_verts >= 4096 and cb.is_available():
            verts_no = cb.compute_normals(verts_co, indices)
        if verts_no is None:
            verts_no = np.empty(num_verts * 3, dtype=np.float32)
            v_norms = mesh.vertex_normals
            if len(v_norms) == num_verts:
                v_norms.foreach_get('vector', verts_no)
            else:
                mesh.vertices.foreach_get('normal', verts_no)

    return (verts_co, verts_no, indices, xform)


def _extract_object_to_mesh_chunks(obj, depsgraph, wireframe, need_normals, xform_override=None):
    """Extract chunk data for an evaluated geometry object using numpy/topology cache."""
    if not HAS_NUMPY:
        return None, None
    try:
        eval_obj = obj.evaluated_get(depsgraph)
    except Exception:
        return None, None

    mesh = None
    needs_clear = False
    if eval_obj.type == 'MESH':
        mesh = eval_obj.data
    elif eval_obj.type in _GEO_TYPES:
        try:
            mesh = eval_obj.to_mesh(preserve_all_data_layers=False)
            needs_clear = True
        except Exception:
            return None, None
    else:
        return None, None

    if not mesh or len(mesh.vertices) == 0:
        if mesh and needs_clear:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass
        return None, None

    xform = xform_override if xform_override is not None else eval_obj.matrix_world.copy()
    obj_key = get_object_key(obj)
    if obj_key not in _cached_signatures:
        _cached_signatures[obj_key] = get_object_geometry_signature(obj)
    chunk = _extract_mesh_data(
        mesh, obj_key, xform, wireframe, need_normals,
        allow_topo_cache=(not needs_clear), allow_fast_cpp_normals=True,
    )
    if chunk is None and needs_clear:
        try:
            eval_obj.to_mesh_clear()
        except Exception:
            pass
        return None, None

    return chunk, (eval_obj if needs_clear else None)


def extract_rigid_mesh(obj, wireframe=False, need_normals=False, depsgraph=None):
    """Extract mesh data in local object space once for instant rigid instancing."""
    empty_ret = (None, None, None, 'LINES' if wireframe else 'TRIS', (0, 0, 0), (0, 0, 0), (0, 0, 0))
    dg = depsgraph if depsgraph is not None else bpy.context.evaluated_depsgraph_get()
    mesh = None
    eval_obj = None
    needs_clear = False

    if obj.type == 'MESH':
        if any(m.show_viewport for m in obj.modifiers):
            try:
                eval_obj = obj.evaluated_get(dg)
                mesh = eval_obj.to_mesh(preserve_all_data_layers=False)
                needs_clear = True
            except Exception:
                mesh = obj.data
        else:
            mesh = obj.data
    elif obj.type in _GEO_TYPES:
        try:
            eval_obj = obj.evaluated_get(dg)
            mesh = eval_obj.to_mesh(preserve_all_data_layers=False)
            needs_clear = True
        except Exception:
            return empty_ret
    else:
        return empty_ret

    if not mesh or len(mesh.vertices) == 0:
        if needs_clear and eval_obj:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass
        return empty_ret

    try:
        obj_key = get_object_key(obj)
        if obj_key not in _cached_signatures:
            _cached_signatures[obj_key] = get_object_geometry_signature(obj)
        chunk = _extract_mesh_data(
            mesh,
            obj_key,
            Matrix.Identity(4),
            wireframe,
            need_normals,
            allow_topo_cache=(not needs_clear),
        )
        if chunk is None:
            return empty_ret
        verts_co, verts_no, indices, _ = chunk
        verts_local = verts_co.reshape(-1, 3)
        normals_local = verts_no.reshape(-1, 3) if verts_no is not None else None
        prim_type = 'LINES' if wireframe else 'TRIS'
        b_min = tuple(float(x) for x in verts_local.min(axis=0))
        b_max = tuple(float(x) for x in verts_local.max(axis=0))
        centroid = tuple(float(x) for x in verts_local.mean(axis=0))
        return verts_local, normals_local, indices, prim_type, b_min, b_max, centroid
    finally:
        if needs_clear and eval_obj:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass


_TRANSFORM_DATA_PATHS = (
    'location', 'rotation_euler', 'rotation_quaternion', 'rotation_axis_angle', 'scale',
    'delta_location', 'delta_rotation_euler', 'delta_rotation_quaternion', 'delta_scale',
)


def _build_transform_fc_map(obj):
    """Build fcurve lookup map for an object's active action slot and detect unsupported paths."""
    fc_map = {dp: {} for dp in _TRANSFORM_DATA_PATHS}
    anim = getattr(obj, 'animation_data', None)
    has_unsupported = False
    if anim and anim.action:
        for fc in collect_action_fcurves(anim.action, anim):
            dp = fc.data_path
            if dp in fc_map:
                fc_map[dp][fc.array_index] = fc
            elif dp.startswith(('constraints', 'rotation_mode')):
                has_unsupported = True
    return fc_map, has_unsupported


def _eval_local_matrix(obj, fc_map, f):
    """Evaluate local transform matrix matching Blender object_to_mat4 (including deltas)."""
    def _vec(dp, fallback, n):
        d = fc_map[dp]
        return [d[i].evaluate(f) if i in d else fallback[i] for i in range(n)]

    loc = Vector(_vec('location', obj.location, 3))
    dloc = Vector(_vec('delta_location', obj.delta_location, 3))
    sc = Vector(_vec('scale', obj.scale, 3))
    dsc = Vector(_vec('delta_scale', obj.delta_scale, 3))
    tot_loc = loc + dloc
    tot_sc = Vector((sc[i] * dsc[i] for i in range(3)))

    mode = obj.rotation_mode
    if mode == 'QUATERNION':
        q = Quaternion(_vec('rotation_quaternion', obj.rotation_quaternion, 4))
        dq = Quaternion(_vec('delta_rotation_quaternion', obj.delta_rotation_quaternion, 4))
        rot_mat = (dq @ q).to_matrix().to_4x4()
    elif mode == 'AXIS_ANGLE':
        aa = _vec('rotation_axis_angle', obj.rotation_axis_angle, 4)
        axis = Vector((aa[1], aa[2], aa[3]))
        rot_mat = Matrix.Rotation(aa[0], 4, axis) if axis.length > 1e-6 else Matrix.Identity(4)
    else:
        r = Euler(_vec('rotation_euler', obj.rotation_euler, 3), mode).to_matrix()
        dr = Euler(_vec('delta_rotation_euler', obj.delta_rotation_euler, 3), 'XYZ').to_matrix()
        rot_mat = (dr @ r).to_4x4()

    return Matrix.Translation(tot_loc) @ rot_mat @ Matrix.Diagonal(tot_sc).to_4x4()


_ANALYTIC_XFORM_DRV_PREFIXES = ('location', 'rotation_', 'scale', 'delta_')


def _can_analytically_eval_obj(o):
    """Return True if an object's local matrix can be evaluated via _eval_local_matrix without frame_set."""
    if o.parent is not None and getattr(o, 'parent_type', 'OBJECT') != 'OBJECT':
        return False
    anim = getattr(o, 'animation_data', None)
    if anim:
        if _has_active_nla(anim):
            return False
        drivers = getattr(anim, 'drivers', None)
        if drivers and any(
            getattr(d, 'data_path', '').startswith(_ANALYTIC_XFORM_DRV_PREFIXES)
            for d in drivers
        ):
            return False
    return True


def _build_analytic_world_chain(obj):
    """Return (True, chain) if obj's world matrix can be evaluated analytically via F-curves."""
    chain = []
    curr = obj
    while curr is not None:
        if len(getattr(curr, 'constraints', ())) > 0 or not _can_analytically_eval_obj(curr):
            return False, None
        c_map, has_unsup = _build_transform_fc_map(curr)
        if has_unsup:
            return False, None
        chain.append((curr, c_map))
        curr = curr.parent
    chain.reverse()
    return True, chain


def _eval_world_matrix_analytic(chain, frame):
    """Evaluate world matrix at frame using prebuilt analytic parent chain."""
    world_mat = Matrix.Identity(4)
    for idx_c, (c_obj, c_map) in enumerate(chain):
        local_mat = _eval_local_matrix(c_obj, c_map, frame)
        if idx_c == 0:
            world_mat = local_mat
        else:
            world_mat = world_mat @ c_obj.matrix_parent_inverse @ local_mat
    return world_mat


def _can_analytically_eval_rigid(obj):
    """Return True if a rigid object can be evaluated analytically (parent chain or single full CHILD_OF)."""
    if obj is None:
        return False
    ok_chain, _ = _build_analytic_world_chain(obj)
    if ok_chain:
        return True
    if obj.parent is not None or not _can_analytically_eval_obj(obj):
        return False
    active_con = [
        c for c in obj.constraints
        if not getattr(c, 'mute', False) and getattr(c, 'influence', 1.0) > 0.0
    ]
    if len(active_con) != 1:
        return False
    con = active_con[0]
    tgt = getattr(con, 'target', None)
    if (
        con.type != 'CHILD_OF'
        or tgt is None
        or getattr(con, 'subtarget', '')
        or abs(getattr(con, 'influence', 1.0) - 1.0) >= 1e-4
    ):
        return False
    all_channels = all(
        getattr(con, attr, True) for attr in (
            'use_location_x', 'use_location_y', 'use_location_z',
            'use_rotation_x', 'use_rotation_y', 'use_rotation_z',
            'use_scale_x', 'use_scale_y', 'use_scale_z',
        )
    )
    if not all_channels or tgt.parent is not None or len(tgt.constraints) != 0 or not _can_analytically_eval_obj(tgt):
        return False
    _, t_unsup = _build_transform_fc_map(tgt)
    _, o_unsup = _build_transform_fc_map(obj)
    return (not t_unsup) and (not o_unsup)


def _find_nearest_cached_frame(cached_frames, target_frame, current, direction, max_dist, used_frames):
    """Pick the nearest unused cached frame on the matching side of current within max_dist."""
    if not cached_frames:
        return None
    best_cf = None
    best_dist = max_dist + 1
    for cf in cached_frames:
        if cf in used_frames:
            continue
        if direction == 'before':
            if cf >= current:
                continue
        else:
            if cf <= current:
                continue
        d = abs(cf - target_frame)
        if d <= max_dist and (d < best_dist or (d == best_dist and best_cf is None)):
            best_dist = d
            best_cf = cf
    return best_cf


def evaluate_rigid_matrices(scene, obj, frames, allow_frame_set=True, prev_matrices=None,
                            needed_info=None, current_frame=None, frame_step=2):
    """Evaluate matrix_world across requested frames in sub-millisecond time."""
    if not frames:
        return {}

    obj_key = get_object_key(obj)
    matrices = {}
    missing_frames = []

    for f in frames:
        cached_mat = cache.get_rigid_matrix(obj_key, f)
        if cached_mat is not None:
            matrices[f] = cached_mat
        else:
            missing_frames.append(f)

    if not missing_frames:
        return matrices

    anim = getattr(obj, 'animation_data', None)
    act = anim.action if anim else None
    fcs = collect_action_fcurves(act, anim) if act else ()
    fc_sig = (act.as_pointer(), len(fcs), tuple(getattr(fc, 'mute', False) for fc in fcs)) if act else (0, 0, ())
    p_obj = obj.parent
    if p_obj is not None:
        p_anim = getattr(p_obj, 'animation_data', None)
        p_act = p_anim.action if p_anim else None
        p_fcs = collect_action_fcurves(p_act, p_anim) if p_act else ()
        p_sig = (p_obj.as_pointer(), p_act.as_pointer() if p_act else 0, len(p_fcs))
    else:
        p_sig = 0
    if len(obj.constraints) > 0:
        c_list = []
        for c in obj.constraints:
            t_obj = getattr(c, 'target', None)
            t_anim = getattr(t_obj, 'animation_data', None) if t_obj is not None else None
            t_act = t_anim.action if t_anim else None
            t_fcs = collect_action_fcurves(t_act, t_anim) if t_act else ()
            c_list.append((
                c.type,
                bool(getattr(c, 'mute', False)),
                round(float(getattr(c, 'influence', 1.0)), 4),
                t_obj.as_pointer() if t_obj is not None else 0,
                t_act.as_pointer() if t_act is not None else 0,
                len(t_fcs),
            ))
        c_sig = tuple(c_list)
    else:
        c_sig = ()
    spec_sig = (p_sig, c_sig, fc_sig)
    cached_plan = _rigid_analytic_cache.get(obj_key)
    if cached_plan is not None and cached_plan[0] == spec_sig:
        kind = cached_plan[1]
        if kind == 'DIRECT':
            fc_map = cached_plan[2]
            for f in missing_frames:
                mat = _eval_local_matrix(obj, fc_map, f)
                matrices[f] = mat
                cache.put_rigid_matrix(obj_key, f, mat)
            return matrices
        elif kind == 'CHAIN':
            chain = cached_plan[2]
            for f in missing_frames:
                world_mat = _eval_world_matrix_analytic(chain, f)
                matrices[f] = world_mat
                cache.put_rigid_matrix(obj_key, f, world_mat)
            return matrices
        elif kind == 'CHILD_OF':
            tgt, t_map, o_map, inv_mat = cached_plan[2:]
            for f in missing_frames:
                world_mat = _eval_local_matrix(tgt, t_map, f) @ inv_mat @ _eval_local_matrix(obj, o_map, f)
                matrices[f] = world_mat
                cache.put_rigid_matrix(obj_key, f, world_mat)
            return matrices
    else:
        # Fast-Path 1: Standalone object with direct F-curves or static transform (no constraints/NLA/drivers)
        if obj.parent is None and len(obj.constraints) == 0 and _can_analytically_eval_obj(obj):
            fc_map, has_unsupported = _build_transform_fc_map(obj)
            if not has_unsupported:
                _rigid_analytic_cache[obj_key] = (spec_sig, 'DIRECT', fc_map)
                for f in missing_frames:
                    mat = _eval_local_matrix(obj, fc_map, f)
                    matrices[f] = mat
                    cache.put_rigid_matrix(obj_key, f, mat)
                return matrices

        # Fast-Path 2: Parented object chain with no constraints/drivers/NLA
        if obj.parent is not None and len(obj.constraints) == 0 and _can_analytically_eval_obj(obj):
            chain = []
            curr = obj
            can_eval_chain = True
            while curr is not None:
                if len(curr.constraints) > 0 or not _can_analytically_eval_obj(curr):
                    can_eval_chain = False
                    break
                c_map, has_unsup = _build_transform_fc_map(curr)
                if has_unsup:
                    can_eval_chain = False
                    break
                chain.append((curr, c_map))
                curr = curr.parent

            if can_eval_chain:
                chain.reverse()
                _rigid_analytic_cache[obj_key] = (spec_sig, 'CHAIN', chain)
                for f in missing_frames:
                    world_mat = _eval_world_matrix_analytic(chain, f)
                    matrices[f] = world_mat
                    cache.put_rigid_matrix(obj_key, f, world_mat)
                return matrices

        # Fast-Path 3: Object with a single full-channel CHILD_OF constraint targeting an unconstrained F-curve object
        active_con = [
            c for c in obj.constraints
            if not getattr(c, 'mute', False) and getattr(c, 'influence', 1.0) > 0.0
        ]
        if (
            obj.parent is None
            and len(active_con) == 1
            and active_con[0].type == 'CHILD_OF'
            and getattr(active_con[0], 'target', None) is not None
            and not getattr(active_con[0], 'subtarget', '')
            and abs(getattr(active_con[0], 'influence', 1.0) - 1.0) < 1e-4
            and _can_analytically_eval_obj(obj)
        ):
            con = active_con[0]
            tgt = con.target
            all_channels = all(
                getattr(con, attr, True) for attr in (
                    'use_location_x', 'use_location_y', 'use_location_z',
                    'use_rotation_x', 'use_rotation_y', 'use_rotation_z',
                    'use_scale_x', 'use_scale_y', 'use_scale_z',
                )
            )
            if all_channels and tgt.parent is None and len(tgt.constraints) == 0 and _can_analytically_eval_obj(tgt):
                t_map, t_unsup = _build_transform_fc_map(tgt)
                o_map, o_unsup = _build_transform_fc_map(obj)
                if not t_unsup and not o_unsup:
                    inv_mat = con.inverse_matrix.copy()
                    _rigid_analytic_cache[obj_key] = (spec_sig, 'CHILD_OF', tgt, t_map, o_map, inv_mat)
                    for f in missing_frames:
                        tgt_mat = _eval_local_matrix(tgt, t_map, f)
                        local_mat = _eval_local_matrix(obj, o_map, f)
                        world_mat = tgt_mat @ inv_mat @ local_mat
                        matrices[f] = world_mat
                        cache.put_rigid_matrix(obj_key, f, world_mat)
                    return matrices

        _rigid_analytic_cache[obj_key] = (spec_sig, 'COMPLEX')

    # Fallback for complex constraints / NLA / drivers: avoid frame_set during playback
    is_playing_now = screen_animation_playing(bpy.context)
    if not allow_frame_set or is_playing_now:
        dir_by_frame = {item[0]: item[1] for item in needed_info} if needed_info else {}
        cached_frames = cache.get_rigid_cached_frames(obj_key)
        used = {'before': set(f for f in matrices if current_frame is None or f < current_frame),
                'after': set(f for f in matrices if current_frame is None or f > current_frame)}
        max_span = 100000 if is_playing_now else max(frame_step * 4, 16)
        for f in missing_frames:
            d = dir_by_frame.get(f, 'before' if (current_frame is not None and f < current_frame) else 'after')
            if current_frame is not None:
                nf = _find_nearest_cached_frame(
                    cached_frames, f, current_frame, d, max_span, used[d]
                )
                if nf is not None:
                    mat = cache.get_rigid_matrix(obj_key, nf)
                    if mat is not None:
                        matrices[f] = mat
                        used[d].add(nf)
                        continue
            if prev_matrices and (obj_key, f) in prev_matrices:
                if current_frame is None or (f < current_frame if d == 'before' else f > current_frame):
                    matrices[f] = prev_matrices[(obj_key, f)]
                    used[d].add(f)
                    continue
            if is_playing_now:
                if current_frame is not None and cached_frames:
                    nf_any_side = _find_nearest_cached_frame(
                        cached_frames, f, current_frame, d, max_span, ()
                    )
                    if nf_any_side is None:
                        nf_any_side = min(cached_frames, key=lambda cf: abs(cf - f))
                    mat = cache.get_rigid_matrix(obj_key, nf_any_side)
                    if mat is not None:
                        matrices[f] = mat
                        continue
                matrices[f] = obj.matrix_world.copy()
        return matrices

    global _is_evaluating
    was_eval = _is_evaluating
    _is_evaluating = True
    orig_frame = scene.frame_current
    dg = bpy.context.evaluated_depsgraph_get()
    try:
        for f in missing_frames:
            scene.frame_set(f)
            eval_obj = obj.evaluated_get(dg)
            mat = eval_obj.matrix_world.copy()
            matrices[f] = mat
            cache.put_rigid_matrix(obj_key, f, mat)
        scene.frame_set(orig_frame)
    finally:
        _is_evaluating = was_eval
    return matrices


# ---------------------------------------------------------------------------
# High-Performance Mesh Extraction (C++ SIMD + Fallback + GeoNodes / Instances)
# ---------------------------------------------------------------------------

def _collect_instance_chunks_from_depsgraph(wanted_orig_objs, depsgraph, wireframe, need_normals):
    """Single-pass collection of evaluated Geometry Nodes & collection instances from depsgraph."""
    result_chunks = {o: [] for o in wanted_orig_objs}
    meshes_to_clear = []
    if not wanted_orig_objs or not HAS_NUMPY:
        return result_chunks, meshes_to_clear

    for inst in depsgraph.object_instances:
        if not inst.is_instance:
            continue
        parent = getattr(inst, 'parent', None)
        if parent is None:
            continue
        orig_parent = parent.original
        if orig_parent not in wanted_orig_objs:
            continue

        inst_obj = inst.object
        if inst_obj is None or inst_obj.type not in _GEO_TYPES or inst_obj.name.startswith('WGT-'):
            continue

        xform = inst.matrix_world.copy()
        mesh = None
        needs_clear = False
        if inst_obj.type == 'MESH':
            mesh = inst_obj.data
        else:
            try:
                mesh = inst_obj.to_mesh(preserve_all_data_layers=False)
                needs_clear = True
            except Exception:
                continue

        if not mesh or len(mesh.vertices) == 0:
            if mesh and needs_clear:
                try:
                    inst_obj.to_mesh_clear()
                except Exception:
                    pass
            continue

        inst_key = f"{get_object_key(orig_parent)}::inst::{len(result_chunks[orig_parent])}"
        chunk = _extract_mesh_data(
            mesh, inst_key, xform, wireframe, need_normals,
            allow_topo_cache=False, allow_fast_cpp_normals=True,
        )
        if needs_clear:
            try:
                inst_obj.to_mesh_clear()
            except Exception:
                pass
        if chunk is not None:
            result_chunks[orig_parent].append(chunk)

    return result_chunks, meshes_to_clear


def _merge_multi_chunks_fast(chunks, wireframe, need_normals):
    """Transform and merge multiple chunks in C++ while reusing invariant merged index buffers."""
    stride = 2 if wireframe else 3
    prim_type = 'LINES' if wireframe else 'TRIS'

    topo_sig = (wireframe, tuple((int(c[2].ctypes.data), int(c[2].size), len(c[0]) // 3) for c in chunks))
    cached = _merged_topology_cache.get(topo_sig)
    merged_indices = None
    if cached is not None:
        cand_idx, refs = cached
        if len(refs) == len(chunks) and all((c[2].base if c[2].base is not None else c[2]) is ref for c, ref in zip(chunks, refs)):
            merged_indices = cand_idx

    total_verts = sum(len(c[0]) // 3 for c in chunks)
    if total_verts <= 0:
        return None

    if merged_indices is None:
        idx_parts = []
        v_off = 0
        refs = []
        for c in chunks:
            nv = len(c[0]) // 3
            idx_2d = np.ascontiguousarray(c[2], dtype=np.int32).reshape(-1, stride)
            idx_parts.append(idx_2d + v_off if v_off > 0 else idx_2d)
            refs.append(c[2].base if c[2].base is not None else c[2])
            v_off += nv
        merged_indices = np.ascontiguousarray(np.vstack(idx_parts) if len(idx_parts) > 1 else idx_parts[0], dtype=np.int32)
        if merged_indices.size <= 0:
            return None
        if len(_merged_topology_cache) >= 128:
            _merged_topology_cache.pop(next(iter(_merged_topology_cache)), None)
        _merged_topology_cache[topo_sig] = (merged_indices, refs)

    has_normals = (not wireframe) and need_normals and any(c[1] is not None for c in chunks)
    merged_verts = np.empty((total_verts, 3), dtype=np.float32)
    if has_normals:
        all_have_n = all(c[1] is not None for c in chunks)
        merged_normals = np.empty((total_verts, 3), dtype=np.float32) if all_have_n else np.zeros((total_verts, 3), dtype=np.float32)
    else:
        merged_normals = None

    bmin_x = bmin_y = bmin_z = float('inf')
    bmax_x = bmax_y = bmax_z = float('-inf')
    sum_x = sum_y = sum_z = 0.0
    v_off = 0

    for verts_co, verts_no, _indices, xform in chunks:
        nv = len(verts_co) // 3
        if nv <= 0:
            continue
        out_v_slice = merged_verts[v_off:v_off + nv]
        out_n_slice = merged_normals[v_off:v_off + nv] if (merged_normals is not None and verts_no is not None) else None
        res = cb.transform_single_mesh(
            verts_co,
            verts_no if out_n_slice is not None else None,
            xform,
            out_verts=out_v_slice,
            out_normals=out_n_slice,
            return_arrays=False,
        )
        if res is None:
            return None
        c_min, c_max, c_ctr = res
        if c_min[0] < bmin_x: bmin_x = c_min[0]
        if c_min[1] < bmin_y: bmin_y = c_min[1]
        if c_min[2] < bmin_z: bmin_z = c_min[2]
        if c_max[0] > bmax_x: bmax_x = c_max[0]
        if c_max[1] > bmax_y: bmax_y = c_max[1]
        if c_max[2] > bmax_z: bmax_z = c_max[2]
        sum_x += c_ctr[0] * nv
        sum_y += c_ctr[1] * nv
        sum_z += c_ctr[2] * nv
        v_off += nv

    inv_n = 1.0 / total_verts
    return (
        merged_verts,
        merged_normals,
        merged_indices,
        prim_type,
        (bmin_x, bmin_y, bmin_z),
        (bmax_x, bmax_y, bmax_z),
        (sum_x * inv_n, sum_y * inv_n, sum_z * inv_n),
    )


def _merge_chunks(chunks, meshes_to_clear, wireframe, need_normals):
    """Merge extracted mesh chunks via C++ AVX2/FMA SIMD with numpy fallback."""
    empty_ret = (None, None, None, 'LINES' if wireframe else 'TRIS', (0, 0, 0), (0, 0, 0), (0, 0, 0))
    try:
        if not chunks:
            return empty_ret

        if cb.is_available() and HAS_NUMPY:
            stride = 2 if wireframe else 3
            prim_type = 'LINES' if wireframe else 'TRIS'
            if len(chunks) == 1:
                verts_co, verts_no, indices, xform = chunks[0]
                if len(indices) > 0:
                    res = cb.transform_single_mesh(
                        verts_co,
                        verts_no if (not wireframe and need_normals) else None,
                        xform,
                    )
                    if res is not None:
                        verts, normals, b_min, b_max, centroid = res
                        idx_arr = indices if hasattr(indices, 'ctypes') else np.ascontiguousarray(indices, dtype=np.int32)
                        idx_2d = idx_arr.reshape(-1, stride)
                        return verts, normals, idx_2d, prim_type, b_min, b_max, centroid
            else:
                multi_res = _merge_multi_chunks_fast(chunks, wireframe, need_normals)
                if multi_res is not None:
                    return multi_res

            res = cb.merge_mesh_chunks(chunks, wireframe=wireframe)
            if res[0] is not None:
                return res

        if HAS_NUMPY:
            all_verts = []
            all_normals = []
            all_indices = []
            vertex_offset = 0
            for verts_co, verts_no, indices, xform in chunks:
                num_verts = len(verts_co) // 3
                verts_local = verts_co.reshape(-1, 3)
                xform_np = np.asarray(xform, dtype=np.float32)
                if np.allclose(xform_np, np.eye(4, dtype=np.float32), atol=1e-6):
                    verts_world = verts_local
                else:
                    verts_world = verts_local @ xform_np[:3, :3].T + xform_np[:3, 3]
                all_verts.append(verts_world)

                if not wireframe and need_normals and verts_no is not None:
                    normals_local = verts_no.reshape(-1, 3)
                    rot_mat = xform_np[:3, :3]
                    if np.allclose(rot_mat, np.eye(3, dtype=np.float32), atol=1e-6):
                        all_normals.append(normals_local)
                    else:
                        try:
                            n_mat = np.linalg.inv(rot_mat)
                        except np.linalg.LinAlgError:
                            n_mat = rot_mat.T
                        tn = normals_local @ n_mat
                        lens = np.linalg.norm(tn, axis=1, keepdims=True)
                        lens = np.where(lens > 1e-6, lens, 1.0)
                        all_normals.append(tn / lens)

                idx_arr = np.asarray(indices, dtype=np.int32).reshape(-1, 2 if wireframe else 3)
                all_indices.append(idx_arr + vertex_offset if vertex_offset > 0 else idx_arr)
                vertex_offset += num_verts

            merged_verts = all_verts[0] if len(all_verts) == 1 else np.ascontiguousarray(np.vstack(all_verts), dtype=np.float32)
            merged_indices = all_indices[0] if len(all_indices) == 1 else np.ascontiguousarray(np.vstack(all_indices), dtype=np.int32)
            merged_normals = (all_normals[0] if len(all_normals) == 1 else np.ascontiguousarray(np.vstack(all_normals), dtype=np.float32)) if all_normals else None
            prim_type = 'LINES' if wireframe else 'TRIS'
            b_min = tuple(float(x) for x in merged_verts.min(axis=0))
            b_max = tuple(float(x) for x in merged_verts.max(axis=0))
            centroid = tuple(float(x) for x in merged_verts.mean(axis=0))
            return merged_verts, merged_normals, merged_indices, prim_type, b_min, b_max, centroid

        return empty_ret
    finally:
        for ev_obj in meshes_to_clear:
            try:
                ev_obj.to_mesh_clear()
            except Exception:
                pass


def _extract_simple(objects, depsgraph, wireframe, instance_matrix=None):
    """Pure-Python fallback mesh extractor when numpy is unavailable."""
    all_verts = []
    all_normals = []
    all_indices = []
    vertex_offset = 0

    for obj in objects:
        if not _is_valid_object(obj):
            continue
        eval_obj = None
        needs_clear = False
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            if eval_obj.type == 'MESH':
                mesh = eval_obj.data
            elif eval_obj.type in _GEO_TYPES:
                mesh = eval_obj.to_mesh(preserve_all_data_layers=False)
                needs_clear = True
            elif eval_obj.type == 'EMPTY' and _empty_has_gn_geometry(obj):
                gs = eval_obj.evaluated_geometry()
                mesh = gs.mesh if gs else None
            else:
                continue

            if not mesh or len(mesh.vertices) == 0:
                continue

            xform = instance_matrix if instance_matrix is not None else eval_obj.matrix_world
            rot_mat = xform.to_3x3()

            all_verts.extend(tuple(xform @ v.co) for v in mesh.vertices)
            if not wireframe:
                all_normals.extend(tuple(rot_mat @ v.normal) for v in mesh.vertices)

            if wireframe:
                all_indices.extend((e.vertices[0] + vertex_offset, e.vertices[1] + vertex_offset) for e in mesh.edges)
            else:
                mesh.calc_loop_triangles()
                all_indices.extend(
                    (t.vertices[0] + vertex_offset, t.vertices[1] + vertex_offset, t.vertices[2] + vertex_offset)
                    for t in mesh.loop_triangles
                )
            vertex_offset += len(mesh.vertices)
        except Exception:
            continue
        finally:
            if needs_clear and eval_obj:
                try:
                    eval_obj.to_mesh_clear()
                except Exception:
                    pass

    if not all_verts or not all_indices:
        return None, None, None, None, (0, 0, 0), (0, 0, 0), (0, 0, 0)

    prim_type = 'LINES' if wireframe else 'TRIS'
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    zs = [v[2] for v in all_verts]
    b_min = (min(xs), min(ys), min(zs))
    b_max = (max(xs), max(ys), max(zs))
    n = len(all_verts)
    centroid = (sum(xs) / n, sum(ys) / n, sum(zs) / n)
    return all_verts, all_normals if all_normals else None, all_indices, prim_type, b_min, b_max, centroid


def _compute_instance_matrix(eval_instancer, eval_child):
    """Compute world matrix for a collection instance member using evaluated transforms."""
    coll = eval_instancer.instance_collection
    offset = coll.instance_offset if coll else (0, 0, 0)
    offset_inv = Matrix.Translation((-offset[0], -offset[1], -offset[2]))
    return eval_instancer.matrix_world @ offset_inv @ eval_child.matrix_world


def extract_all_meshes_merged(objects, depsgraph, wireframe=False, need_normals=False,
                               instance_matrix=None, instance_matrix_map=None):
    """Merge geometry from a list of objects into a single vertex/normal/index set.

    Accelerated using C++ AVX2/FMA SIMD vectorization with static topology caching.
    Returns (verts, normals, indices, prim_type, aabb_min, aabb_max, centroid).
    """
    if not HAS_NUMPY:
        return _extract_simple(objects, depsgraph, wireframe, instance_matrix)

    chunks = []
    meshes_to_clear = []
    gn_parents = {
        o for o in objects
        if _is_valid_object(o) and (
            o.type == 'EMPTY' and _empty_has_gn_geometry(o)
            or any(getattr(m, 'type', None) == 'NODES' and m.show_viewport for m in o.modifiers)
        )
    }
    inst_chunks_by_parent = {}
    if gn_parents:
        inst_chunks_by_parent, _ = _collect_instance_chunks_from_depsgraph(
            gn_parents, depsgraph, wireframe, need_normals
        )

    for obj in objects:
        if not _is_valid_object(obj):
            continue

        if obj.type == 'EMPTY':
            p_chunks = inst_chunks_by_parent.get(obj, [])
            if p_chunks:
                if instance_matrix is not None:
                    # Apply instance override matrix if called via an outer container
                    chunks.extend((vc, vn, idx, instance_matrix @ xm) for (vc, vn, idx, xm) in p_chunks)
                else:
                    chunks.extend(p_chunks)
            elif _empty_has_gn_geometry(obj):
                try:
                    ev = obj.evaluated_get(depsgraph)
                    gs = ev.evaluated_geometry()
                    mesh = gs.mesh if gs else None
                    if mesh and len(mesh.vertices) > 0:
                        xform = instance_matrix if instance_matrix is not None else ev.matrix_world.copy()
                        c = _extract_mesh_data(
                            mesh, get_object_key(obj), xform, wireframe, need_normals, allow_topo_cache=False
                        )
                        if c is not None:
                            chunks.append(c)
                except Exception:
                    pass
            continue

        xform_ov = None
        if instance_matrix_map is not None:
            ok = get_object_key(obj)
            if ok in instance_matrix_map:
                xform_ov = instance_matrix_map[ok]
            elif obj.name in instance_matrix_map:
                xform_ov = instance_matrix_map[obj.name]
        elif instance_matrix is not None:
            xform_ov = instance_matrix

        chunk, ev_clear = _extract_object_to_mesh_chunks(
            obj, depsgraph, wireframe, need_normals, xform_override=xform_ov
        )
        if chunk is not None:
            chunks.append(chunk)
            if ev_clear is not None:
                meshes_to_clear.append(ev_clear)

        if obj in inst_chunks_by_parent and inst_chunks_by_parent[obj]:
            chunks.extend(inst_chunks_by_parent[obj])

    return _merge_chunks(chunks, meshes_to_clear, wireframe, need_normals)


def _collect_member_chunks(mobj, depsgraph, wireframe, need_normals, precollected_inst_chunks=None):
    """Extract evaluated mesh chunks and any GeoNodes instance chunks for a single member object."""
    chunks = []
    meshes_to_clear = []
    chunk, ev_clear = _extract_object_to_mesh_chunks(mobj, depsgraph, wireframe, need_normals)
    if chunk is not None:
        chunks.append(chunk)
        if ev_clear is not None:
            meshes_to_clear.append(ev_clear)
    if precollected_inst_chunks and mobj in precollected_inst_chunks:
        chunks.extend(precollected_inst_chunks[mobj])
    elif any(getattr(m, 'type', None) == 'NODES' and m.show_viewport for m in mobj.modifiers):
        res_map, _ = _collect_instance_chunks_from_depsgraph({mobj}, depsgraph, wireframe, need_normals)
        chunks.extend(res_map.get(mobj, []))
    return chunks, meshes_to_clear


def _extract_tracked_single(obj, settings, depsgraph, wireframe, need_normals,
                            precollected_inst_chunks=None, scene=None,
                            view_layer=None, vis_cols=None, v3d=None):
    """Extract merged world-space mesh for any tracked root object (Geo, Rig, Metaball, GN Empty, Collection)."""
    if not _is_valid_object(obj):
        return None, None, None, 'LINES' if wireframe else 'TRIS', (0, 0, 0), (0, 0, 0), (0, 0, 0)

    if obj.type == 'EMPTY' and (_empty_is_collection_instance(obj) or _empty_has_gn_geometry(obj)):
        if obj.type == 'EMPTY' and _empty_is_collection_instance(obj) and not settings.include_children:
            return None, None, None, 'LINES' if wireframe else 'TRIS', (0, 0, 0), (0, 0, 0), (0, 0, 0)

        p_chunks = None
        if precollected_inst_chunks is not None:
            p_chunks = precollected_inst_chunks.get(obj)
        if p_chunks is None and HAS_NUMPY:
            res_map, _ = _collect_instance_chunks_from_depsgraph({obj}, depsgraph, wireframe, need_normals)
            p_chunks = res_map.get(obj, [])

        if p_chunks:
            return _merge_chunks(list(p_chunks), [], wireframe, need_normals)

        # Fallback for unlinked collection instancers (not in scene.objects) or unlinked GN Empties
        if _empty_is_collection_instance(obj):
            members = resolve_mesh_objects(
                obj, settings, scene=scene, view_layer=view_layer, vis_cols=vis_cols, v3d=v3d
            )
            if not members:
                return None, None, None, 'LINES' if wireframe else 'TRIS', (0, 0, 0), (0, 0, 0), (0, 0, 0)
            try:
                eval_inst = obj.evaluated_get(depsgraph)
            except Exception:
                eval_inst = obj
            inst_map = {}
            for cobj in members:
                try:
                    eval_c = cobj.evaluated_get(depsgraph)
                except Exception:
                    eval_c = cobj
                inst_map[get_object_key(cobj)] = _compute_instance_matrix(eval_inst, eval_c)
            return extract_all_meshes_merged(
                members, depsgraph, wireframe=wireframe, need_normals=need_normals,
                instance_matrix_map=inst_map
            )

        return extract_all_meshes_merged(
            [obj], depsgraph, wireframe=wireframe, need_normals=need_normals
        )

    members = resolve_mesh_objects(
        obj, settings, scene=scene, view_layer=view_layer, vis_cols=vis_cols, v3d=v3d
    )
    if not members:
        return None, None, None, 'LINES' if wireframe else 'TRIS', (0, 0, 0), (0, 0, 0), (0, 0, 0)

    if not HAS_NUMPY:
        return _extract_simple(members, depsgraph, wireframe)

    chunks = []
    meshes_to_clear = []
    for mobj in members:
        m_chunks, m_clear = _collect_member_chunks(
            mobj, depsgraph, wireframe, need_normals, precollected_inst_chunks=precollected_inst_chunks
        )
        chunks.extend(m_chunks)
        meshes_to_clear.extend(m_clear)

    return _merge_chunks(chunks, meshes_to_clear, wireframe, need_normals)


def _extract_tracked_entries(obj, settings, depsgraph, wireframe, need_normals,
                             precollected_inst_chunks=None, scene=None,
                             view_layer=None, vis_cols=None, v3d=None):
    """Extract per-part GPU entry lists for Armatures (enabling per-mesh Outliner/Local View filtering) or a 1-entry list."""
    if not _is_valid_object(obj):
        return []

    if obj.type == 'ARMATURE' and getattr(settings, 'include_children', True):
        members = resolve_mesh_objects(
            obj, settings, scene=scene, view_layer=view_layer, vis_cols=vis_cols, v3d=v3d
        )
        if not members:
            return []
        entries = []
        if not HAS_NUMPY:
            for mobj in members:
                verts, normals, indices, prim_type, b_min, b_max, center = _extract_simple(
                    [mobj], depsgraph, wireframe
                )
                if verts is not None and indices is not None:
                    entries.append([get_object_key(mobj), verts, normals, indices, prim_type, {}, b_min, b_max, center])
            return entries
        for mobj in members:
            m_chunks, m_clear = _collect_member_chunks(
                mobj, depsgraph, wireframe, need_normals, precollected_inst_chunks=precollected_inst_chunks
            )
            verts, normals, indices, prim_type, b_min, b_max, center = _merge_chunks(
                m_chunks, m_clear, wireframe, need_normals
            )
            if verts is not None and indices is not None:
                entries.append([get_object_key(mobj), verts, normals, indices, prim_type, {}, b_min, b_max, center])
        return entries

    verts, normals, indices, prim_type, b_min, b_max, center = _extract_tracked_single(
        obj, settings, depsgraph, wireframe, need_normals,
        precollected_inst_chunks=precollected_inst_chunks, scene=scene,
        view_layer=view_layer, vis_cols=vis_cols, v3d=v3d,
    )
    if verts is not None and indices is not None:
        return [[get_object_key(obj), verts, normals, indices, prim_type, {}, b_min, b_max, center]]
    return []


def extract_tracked_objects_merged(targets, depsgraph, wireframe=False, need_normals=False,
                                   view_layer=None, vis_cols=None, v3d=None):
    """Extract and merge geometry for a list of tracked objects (used for ghost markers)."""
    all_verts = []
    all_normals = []
    all_indices = []
    v_offset = 0
    prim_type = 'LINES' if wireframe else 'TRIS'

    if not HAS_NUMPY:
        objs = []
        for t in targets:
            objs.extend(resolve_mesh_objects(
                t, properties.get_object_settings(t),
                view_layer=view_layer, vis_cols=vis_cols, v3d=v3d,
            ))
        return _extract_simple(objs, depsgraph, wireframe)

    valid_results = []
    for obj in targets:
        res = _extract_tracked_single(
            obj, properties.get_object_settings(obj), depsgraph, wireframe, need_normals,
            view_layer=view_layer, vis_cols=vis_cols, v3d=v3d,
        )
        if res[0] is not None and res[2] is not None:
            valid_results.append(res)

    if not valid_results:
        return None, None, None, prim_type, (0, 0, 0), (0, 0, 0), (0, 0, 0)

    if len(valid_results) == 1:
        return valid_results[0]

    for v, n, idx, pt, _, _, _ in valid_results:
        all_verts.append(v)
        if n is not None and not wireframe and need_normals:
            all_normals.append(n)
        all_indices.append(idx if v_offset == 0 else (idx + v_offset))
        v_offset += len(v)
        prim_type = pt

    mv = np.vstack(all_verts)
    mi = np.vstack(all_indices)
    mn = np.vstack(all_normals) if (all_normals and len(all_normals) == len(all_verts)) else None
    b_min = (
        min(r[4][0] for r in valid_results),
        min(r[4][1] for r in valid_results),
        min(r[4][2] for r in valid_results),
    )
    b_max = (
        max(r[5][0] for r in valid_results),
        max(r[5][1] for r in valid_results),
        max(r[5][2] for r in valid_results),
    )
    total_v = sum(len(r[0]) for r in valid_results)
    if total_v > 0:
        cx = sum(r[6][0] * len(r[0]) for r in valid_results) / total_v
        cy = sum(r[6][1] * len(r[0]) for r in valid_results) / total_v
        cz = sum(r[6][2] * len(r[0]) for r in valid_results) / total_v
        centroid = (cx, cy, cz)
    else:
        centroid = (0.0, 0.0, 0.0)
    return mv, mn, mi, prim_type, b_min, b_max, centroid


# ---------------------------------------------------------------------------
# Frame Range & Keyframe Helpers
# ---------------------------------------------------------------------------

def get_frame_range(settings, scene):
    if getattr(settings, 'use_frame_range', False):
        return getattr(settings, 'frame_range_start', scene.frame_start), getattr(settings, 'frame_range_end', scene.frame_end)
    return scene.frame_start, scene.frame_end


def _update_kf_range_from_action(act, anim, min_frame, max_frame, unbounded):
    """Inspect active F-curves, keyframes, and F-curve modifiers to determine frame bounds."""
    if not act:
        return min_frame, max_frame, unbounded
    fcs = collect_action_fcurves(act, anim)
    if not fcs:
        return min_frame, max_frame, unbounded

    for fc in fcs:
        kps = getattr(fc, 'keyframe_points', ())
        if len(kps) > 0:
            min_frame = min(min_frame, kps[0].co.x)
            max_frame = max(max_frame, kps[-1].co.x)

        for fm in getattr(fc, 'modifiers', ()):
            if getattr(fm, 'mute', False):
                continue
            if getattr(fm, 'use_restricted_range', False):
                min_frame = min(min_frame, getattr(fm, 'frame_start', min_frame))
                max_frame = max(max_frame, getattr(fm, 'frame_end', max_frame))
            elif fm.type in ('CYCLES', 'GENERATOR', 'FNGENERATOR', 'NOISE', 'LIMITS', 'ENVELOPE'):
                if fm.type == 'CYCLES':
                    if getattr(fm, 'cycles_before', 0) == 0 or getattr(fm, 'cycles_after', 0) == 0:
                        unbounded = True
                    elif len(kps) > 0:
                        span = max(0.0, kps[-1].co.x - kps[0].co.x)
                        min_frame = min(min_frame, kps[0].co.x - span * getattr(fm, 'cycles_before', 0))
                        max_frame = max(max_frame, kps[-1].co.x + span * getattr(fm, 'cycles_after', 0))
                elif fm.type in ('GENERATOR', 'FNGENERATOR', 'NOISE'):
                    unbounded = True
    return min_frame, max_frame, unbounded


def _update_kf_range_from_anim(holder, min_frame, max_frame, unbounded=False):
    if holder is None:
        return min_frame, max_frame, unbounded

    anim = getattr(holder, 'animation_data', None)
    had_local_anim = False

    if anim:
        if anim.action:
            fcs = collect_action_fcurves(anim.action, anim)
            if fcs:
                had_local_anim = True
                min_frame, max_frame, unbounded = _update_kf_range_from_action(
                    anim.action, anim, min_frame, max_frame, unbounded
                )
        for track in getattr(anim, 'nla_tracks', ()):
            if getattr(track, 'mute', False):
                continue
            for strip in getattr(track, 'strips', ()):
                had_local_anim = True
                min_frame = min(min_frame, strip.frame_start)
                max_frame = max(max_frame, strip.frame_end)
        if getattr(anim, 'drivers', None) and len(anim.drivers) > 0:
            # Drivers can animate dynamically at any frame
            pass

    if not had_local_anim and getattr(holder, 'override_library', None) and holder.override_library.reference:
        ref = holder.override_library.reference
        ref_anim = getattr(ref, 'animation_data', None)
        if ref_anim and ref_anim.action:
            min_frame, max_frame, unbounded = _update_kf_range_from_action(
                ref_anim.action, ref_anim, min_frame, max_frame, unbounded
            )

    return min_frame, max_frame, unbounded


def get_keyframe_range_for_object(obj, scene=None):
    min_frame = float('inf')
    max_frame = float('-inf')
    unbounded = False

    min_frame, max_frame, unbounded = _update_kf_range_from_anim(obj, min_frame, max_frame, unbounded)

    if getattr(obj, 'data', None):
        min_frame, max_frame, unbounded = _update_kf_range_from_anim(obj.data, min_frame, max_frame, unbounded)
        sk = getattr(obj.data, 'shape_keys', None)
        if sk is not None:
            min_frame, max_frame, unbounded = _update_kf_range_from_anim(sk, min_frame, max_frame, unbounded)

    if obj.type == 'ARMATURE':
        for child in obj.children_recursive:
            min_frame, max_frame, unbounded = _update_kf_range_from_anim(child, min_frame, max_frame, unbounded)

    if obj.type == 'EMPTY' and _empty_is_collection_instance(obj):
        for cobj in obj.instance_collection.all_objects:
            min_frame, max_frame, unbounded = _update_kf_range_from_anim(cobj, min_frame, max_frame, unbounded)

    if obj.type == 'META':
        for sib in _get_metaball_family(obj, scene):
            if sib is not obj:
                min_frame, max_frame, unbounded = _update_kf_range_from_anim(sib, min_frame, max_frame, unbounded)

    # Walk upstream dependencies (parent chain, Armature/modifier targets, constraint targets)
    visited = {obj}
    up_queue = []
    if obj.parent and obj.parent not in visited:
        up_queue.append(obj.parent)
    for con in getattr(obj, 'constraints', ()):
        if not getattr(con, 'mute', False) and getattr(con, 'influence', 1.0) > 0.0:
            tgt = getattr(con, 'target', None)
            if isinstance(tgt, bpy.types.Object) and tgt not in visited:
                up_queue.append(tgt)
    mod_objs, _ = _iter_modifier_targets(obj)
    for mo in mod_objs:
        if mo not in visited:
            up_queue.append(mo)

    while up_queue:
        u = up_queue.pop()
        if u in visited or not _is_valid_object(u):
            continue
        visited.add(u)
        min_frame, max_frame, unbounded = _update_kf_range_from_anim(u, min_frame, max_frame, unbounded)
        if u.parent and u.parent not in visited:
            up_queue.append(u.parent)
        for con in getattr(u, 'constraints', ()):
            if not getattr(con, 'mute', False) and getattr(con, 'influence', 1.0) > 0.0:
                tgt = getattr(con, 'target', None)
                if isinstance(tgt, bpy.types.Object) and tgt not in visited:
                    up_queue.append(tgt)

    if unbounded or min_frame == float('inf'):
        return None, None
    return min_frame, max_frame


def get_needed_frames(current, settings, scene, onion_skin_objects=None, obj=None, use_kf_cache=False):
    range_start, range_end = get_frame_range(settings, scene)

    use_fr = getattr(settings, 'use_frame_range', False)
    if use_fr and (current < range_start or current > range_end):
        return []

    if getattr(settings, 'limit_to_keyframes', False) and obj is not None:
        ok = get_object_key(obj)
        kf_pair = _cached_kf_ranges.get(ok) if use_kf_cache else None
        if kf_pair is None:
            kf_pair = get_keyframe_range_for_object(obj, scene=scene)
            _cached_kf_ranges[ok] = kf_pair
        kf_start, kf_end = kf_pair
        if kf_start is not None and kf_end is not None:
            if use_fr:
                range_start = max(range_start, int(kf_start))
                range_end = min(range_end, int(kf_end))
            else:
                range_start = int(kf_start)
                range_end = int(kf_end)

    frames = []
    step = max(1, int(getattr(settings, 'frame_step', 2)))

    show_before = getattr(settings, 'show_before', True)
    if show_before:
        max_before = getattr(settings, 'frames_before', 3)
        inv_max = 1.0 / max_before if max_before > 0 else 0
        for i in range(step, max_before + 1, step):
            f = current - i
            if range_start <= f <= range_end:
                frames.append((f, 'before', i * inv_max, i))

    show_after = getattr(settings, 'show_after', True)
    if show_after:
        max_after = getattr(settings, 'frames_after', 3)
        inv_max = 1.0 / max_after if max_after > 0 else 0
        for i in range(step, max_after + 1, step):
            f = current + i
            if range_start <= f <= range_end:
                frames.append((f, 'after', i * inv_max, i))

    return frames


def calculate_alpha(base_alpha, t, falloff, curve):
    if curve == 'linear':
        factor = 1.0 - t * falloff
    elif curve == 'exponential':
        factor = (max(0.0, 1.0 - t) ** (1.0 + falloff * 2.0))
    else:
        ts = t * falloff
        factor = 0.1 if ts >= 1.0 else (1.0 if ts <= 0.0 else 1.0 - ts * ts * (3.0 - 2.0 * ts))
    return base_alpha * (0.1 if factor < 0.1 else factor)


def _safe_redraw(context):
    try:
        if hasattr(context, "window_manager") and context.window_manager:
            for window in context.window_manager.windows:
                screen = window.screen
                if screen:
                    for area in screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
        elif hasattr(context, "screen") and context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except (AttributeError, ReferenceError):
        pass


# ---------------------------------------------------------------------------
# Snapshot / Restore & Active Modal Transform Detection
# ---------------------------------------------------------------------------

def _snapshot_object_state(obj, include_pose_bones=True):
    state = {
        'location': obj.location.copy(),
        'rotation_euler': obj.rotation_euler.copy(),
        'rotation_quaternion': obj.rotation_quaternion.copy(),
        'rotation_axis_angle': list(obj.rotation_axis_angle),
        'rotation_mode': obj.rotation_mode,
        'scale': obj.scale.copy(),
        'delta_location': obj.delta_location.copy(),
        'delta_rotation_euler': obj.delta_rotation_euler.copy(),
        'delta_rotation_quaternion': obj.delta_rotation_quaternion.copy(),
        'delta_scale': obj.delta_scale.copy(),
    }

    if include_pose_bones and obj.type == 'ARMATURE' and obj.pose:
        bone_states = {}
        for pbone in obj.pose.bones:
            bone_states[pbone.name] = {
                'location': pbone.location.copy(),
                'rotation_euler': pbone.rotation_euler.copy(),
                'rotation_quaternion': pbone.rotation_quaternion.copy(),
                'rotation_axis_angle': list(pbone.rotation_axis_angle),
                'rotation_mode': pbone.rotation_mode,
                'scale': pbone.scale.copy(),
            }
        state['pose_bones'] = bone_states

    if hasattr(obj, 'data') and getattr(obj.data, 'shape_keys', None):
        sk = obj.data.shape_keys
        if sk and hasattr(sk, 'key_blocks'):
            state['shape_keys'] = {kb.name: kb.value for kb in sk.key_blocks}

    return state


def _restore_object_state(obj, state):
    for attr in (
        'rotation_mode', 'location', 'rotation_euler', 'rotation_quaternion',
        'rotation_axis_angle', 'scale', 'delta_location',
        'delta_rotation_euler', 'delta_rotation_quaternion', 'delta_scale',
    ):
        try:
            cur = list(getattr(obj, attr)) if attr == 'rotation_axis_angle' else getattr(obj, attr)
            if cur != state[attr]:
                setattr(obj, attr, state[attr])
        except Exception:
            pass

    if 'pose_bones' in state and obj.type == 'ARMATURE' and obj.pose:
        for pbone_name, bstate in state['pose_bones'].items():
            pbone = obj.pose.bones.get(pbone_name)
            if not pbone:
                continue
            try:
                if pbone.rotation_mode != bstate['rotation_mode']:
                    pbone.rotation_mode = bstate['rotation_mode']
            except Exception:
                pass
            for attr in ('location', 'rotation_euler', 'rotation_quaternion', 'rotation_axis_angle', 'scale'):
                try:
                    cur = list(getattr(pbone, attr)) if attr == 'rotation_axis_angle' else getattr(pbone, attr)
                    if cur != bstate[attr]:
                        setattr(pbone, attr, bstate[attr])
                except Exception:
                    pass

    if 'shape_keys' in state and hasattr(obj, 'data') and getattr(obj.data, 'shape_keys', None):
        try:
            sk = obj.data.shape_keys
            if sk and hasattr(sk, 'key_blocks'):
                anim_data = getattr(sk, 'animation_data', None)
                if not (anim_data and getattr(anim_data, 'action', None)):
                    for kb_name, val in state['shape_keys'].items():
                        kb = sk.key_blocks.get(kb_name)
                        if kb and kb.value != val:
                            kb.value = val
        except Exception:
            pass


def _is_transforming(context):
    """Detect if an interactive modal transform, pose, or F-Curve manipulation operator is currently running."""
    try:
        wm = getattr(context, 'window_manager', None)
        if wm:
            for win in getattr(wm, 'windows', ()):
                for op in getattr(win, 'modal_operators', ()):
                    idname = getattr(op, 'bl_idname', '') or ''
                    if idname.startswith(_ACTIVE_MODAL_PREFIXES):
                        return True
        win = getattr(context, 'window', None) if context else None
        if win:
            for op in getattr(win, 'modal_operators', ()):
                idname = getattr(op, 'bl_idname', '') or ''
                if idname.startswith(_ACTIVE_MODAL_PREFIXES):
                    return True
    except (AttributeError, ReferenceError):
        pass
    return False


def _is_scrubbing_modal(context):
    """Detect if an interactive timeline/dopesheet scrub operator is currently active."""
    try:
        wm = getattr(context, 'window_manager', None)
        if wm:
            for win in getattr(wm, 'windows', ()):
                for op in getattr(win, 'modal_operators', ()):
                    idname = getattr(op, 'bl_idname', '') or ''
                    if idname.startswith('ANIM_OT_'):
                        return True
        win = getattr(context, 'window', None) if context else None
        if win:
            for op in getattr(win, 'modal_operators', ()):
                idname = getattr(op, 'bl_idname', '') or ''
                if idname.startswith('ANIM_OT_'):
                    return True
    except (AttributeError, ReferenceError):
        pass
    return False


# ---------------------------------------------------------------------------
# Per-Frame Action Signatures & Playback Capture Budgets
# ---------------------------------------------------------------------------

_PHYSICS_MOD_TYPES = {'CLOTH', 'SOFT_BODY', 'FLUID', 'PARTICLE_SYSTEM', 'DYNAMIC_PAINT'}
_TIME_DEPENDENT_CONSTRAINTS = {'ACTION', 'FOLLOW_PATH'}

_action_frame_sigs = {}
_playback_eval_meta = {}
_cached_kf_ranges = {}
_passive_extract_costs = {}
_last_heavy_passive_capture = {}
_last_ghost_slots = {}
_last_nonplay_fc_time = 0.0
_was_playing = False


def _clear_ghost_slots(obj_key=None):
    """Clear slot fallback memory for all or a single object key."""
    if obj_key is None:
        _last_ghost_slots.clear()
    else:
        keys = [k for k in _last_ghost_slots if k[0] == obj_key]
        for k in keys:
            del _last_ghost_slots[k]


def _clear_object_action_sigs(obj_key=None):
    """Clear stored per-frame action signatures and delta transforms for all or a single object key."""
    if obj_key is None:
        _action_frame_sigs.clear()
        _playback_eval_meta.clear()
        _cached_kf_ranges.clear()
        _deforming_base_mats.clear()
        _deforming_delta_mats.clear()
        _frame_last_eval_seq.clear()
        _stale_offscreen_action_roots.clear()
    else:
        keys = [k for k in _action_frame_sigs if k[0] == obj_key]
        for k in keys:
            del _action_frame_sigs[k]
        seq_keys = [k for k in _frame_last_eval_seq if k[0] == obj_key]
        for k in seq_keys:
            del _frame_last_eval_seq[k]
        _playback_eval_meta.pop(obj_key, None)
        _cached_kf_ranges.pop(obj_key, None)
        _delta_cap_cache.pop(obj_key, None)
        _stale_offscreen_action_roots.discard(obj_key)
        _clear_deforming_deltas(obj_key)


def _can_use_per_frame_action_inval(dep_objs, dep_trees=()):
    """Return True if frame F depends purely on evaluated dependency F-curves at frame F."""
    for dobj in dep_objs:
        for mod in getattr(dobj, 'modifiers', ()):
            if getattr(mod, 'show_viewport', True) and getattr(mod, 'type', '') in _PHYSICS_MOD_TYPES:
                return False
        for con in getattr(dobj, 'constraints', ()):
            if not getattr(con, 'mute', False) and getattr(con, 'type', '') in _TIME_DEPENDENT_CONSTRAINTS:
                return False
        holders = [dobj, getattr(dobj, 'data', None)]
        sk = getattr(holders[1], 'shape_keys', None) if holders[1] is not None else None
        if sk is not None:
            holders.append(sk)
        for h in holders:
            if h is None:
                continue
            anim = getattr(h, 'animation_data', None)
            if anim and _has_active_nla(anim):
                return False
    for tree in dep_trees:
        tanim = getattr(tree, 'animation_data', None)
        if tanim and _has_active_nla(tanim):
            return False
    return True


def _collect_dep_fcurves_sorted(dep_objs, dep_trees=()):
    """Collect active F-curves across dependency objects and node trees in deterministic order."""
    entries = []
    for dobj in sorted(dep_objs, key=get_object_key):
        okey = get_object_key(dobj)
        anim = getattr(dobj, 'animation_data', None)
        if anim and anim.action:
            for fc in collect_action_fcurves(anim.action, anim):
                entries.append((('O', okey, fc.data_path, fc.array_index), fc))
        data = getattr(dobj, 'data', None)
        if data is not None:
            danim = getattr(data, 'animation_data', None)
            if danim and danim.action:
                dkey = _get_data_key(data)
                for fc in collect_action_fcurves(danim.action, danim):
                    entries.append((('D', dkey, fc.data_path, fc.array_index), fc))
            sk = getattr(data, 'shape_keys', None)
            if sk is not None:
                sanim = getattr(sk, 'animation_data', None)
                if sanim and sanim.action:
                    skey = _get_data_key(sk)
                    for fc in collect_action_fcurves(sanim.action, sanim):
                        entries.append((('K', skey, fc.data_path, fc.array_index), fc))
    for tree in sorted(dep_trees, key=_get_data_key):
        tanim = getattr(tree, 'animation_data', None)
        if tanim and tanim.action:
            tkey = _get_data_key(tree)
            for fc in collect_action_fcurves(tanim.action, tanim):
                entries.append((('T', tkey, fc.data_path, fc.array_index), fc))
    return entries


def _eval_dep_actions_at_frame(fc_entries, frame):
    """Evaluate all dependency F-curves at frame into a compact comparable signature tuple."""
    return tuple((tag, fc.evaluate(frame)) for tag, fc in fc_entries)


_TIME_DEPENDENT_MOD_TYPES = {'WAVE', 'BUILD', 'MESH_CACHE', 'MESH_SEQUENCE_CACHE', 'OCEAN'}
_DELTA_SAFE_MOD_TYPES = {
    'ARMATURE', 'SUBSURF', 'MULTIRES', 'SOLIDIFY', 'BEVEL', 'MIRROR',
    'WEIGHTED_NORMAL', 'TRIANGULATE', 'WELD', 'EDGE_SPLIT', 'MASK',
    'CORRECTIVE_SMOOTH', 'SMOOTH',
}

_deforming_base_mats = {}
_deforming_delta_mats = {}
_eval_seq_counter = 0
_frame_last_eval_seq = {}
_cached_outliner_vis = {}
_cached_coll_sigs = {}
_last_modal_edit_time = 0.0
_was_transforming = False

_FRAME_IDENT_RE = re.compile(r'\bframe\b')
_BPY_IDENT_RE = re.compile(r'\b(?:bpy|scene|context)\b')


def _clear_deforming_deltas(obj_key=None):
    """Clear stored base/delta world matrices for all or a single object key."""
    if obj_key is None:
        _deforming_base_mats.clear()
        _deforming_delta_mats.clear()
    else:
        for d in (_deforming_base_mats, _deforming_delta_mats):
            for k in [k for k in d if k[0] == obj_key]:
                del d[k]


def _get_collection_membership_signature(coll):
    """Return a compact membership signature so child transforms don't look like collection edits."""
    try:
        return (
            tuple(o.name for o in getattr(coll, 'objects', ())),
            tuple(c.name for c in getattr(coll, 'children', ())),
        )
    except Exception:
        return None


def _seed_collection_sigs(coll, visited=None):
    """Seed _cached_coll_sigs for coll and nested child collections before edits occur."""
    if coll is None:
        return
    if visited is None:
        visited = set()
    if coll in visited:
        return
    visited.add(coll)
    ckey = _get_data_key(coll)
    if ckey not in _cached_coll_sigs:
        _cached_coll_sigs[ckey] = _get_collection_membership_signature(coll)
    for child in getattr(coll, 'children', ()):
        _seed_collection_sigs(child, visited)


def _get_visible_collections(view_layer):
    """Return set of Collections visible and included in view_layer (Outliner eye/monitor/exclude)."""
    vis_cols = set()
    if view_layer is None:
        return vis_cols
    root_lc = getattr(view_layer, 'layer_collection', None)
    if root_lc is None:
        return vis_cols
    stack = [root_lc]
    while stack:
        lc = stack.pop()
        if getattr(lc, 'exclude', False) or getattr(lc, 'hide_viewport', False):
            continue
        col = getattr(lc, 'collection', None)
        if col is None or getattr(col, 'hide_viewport', False):
            continue
        vis_cols.add(col)
        stack.extend(getattr(lc, 'children', ()))
    return vis_cols


def _is_outliner_visible(obj, view_layer, vis_cols=None):
    """Return True if obj is visible in Outliner (eye, monitor, collection hide/exclude), ignoring Local View."""
    if not _is_valid_object(obj):
        return False
    if getattr(obj, 'hide_viewport', False):
        return False
    if view_layer is not None:
        try:
            if obj.hide_get(view_layer=view_layer):
                return False
        except Exception:
            pass
        if vis_cols is None:
            vis_cols = _get_visible_collections(view_layer)
        try:
            cols = getattr(obj, 'users_collection', ())
            if not cols or not any(c in vis_cols for c in cols):
                return False
        except Exception:
            pass
    return True


def _safe_visible_get(obj, view_layer, v3d=None):
    """Safely check Blender viewport visibility (Outliner + optional Local View)."""
    try:
        return bool(obj.visible_get(view_layer=view_layer, viewport=v3d))
    except Exception:
        return False


def _compute_outliner_vis_sig(obj, settings, view_layer, scene=None, vis_cols=None, children_map=None, arm_map=None, metaball_map=None):
    """Return (obj_vis, child_vis_tuple) Outliner visibility signature for tracked object."""
    if view_layer is not None and vis_cols is None:
        vis_cols = _get_visible_collections(view_layer)
    obj_vis = _is_outliner_visible(obj, view_layer, vis_cols=vis_cols)
    child_vis = ()
    if obj.type == 'ARMATURE' and getattr(settings, 'include_children', True):
        child_vis = tuple(
            (get_object_key(m), _is_outliner_visible(m, view_layer, vis_cols=vis_cols))
            for m in _get_armature_candidate_meshes(
                obj, settings, scene=scene, children_map=children_map, arm_map=arm_map,
            )
        )
    return (obj_vis, child_vis)


def _get_visible_viewport_part_meshes(
    obj, settings, view_layer, v3d=None, scene=None, vis_cols=None, children_map=None, arm_map=None
):
    """Return (is_visible, visible_part_meshes_or_none) for obj in Outliner and optional Local View."""
    if not _is_valid_object(obj):
        return False, None
    if view_layer is not None:
        if vis_cols is None:
            vis_cols = _get_visible_collections(view_layer)
        if not _is_outliner_visible(obj, view_layer, vis_cols=vis_cols):
            return False, None

    has_lv = v3d is not None and getattr(v3d, 'local_view', None) is not None
    if obj.type == 'ARMATURE' and getattr(settings, 'include_children', True):
        cand = _get_armature_candidate_meshes(
            obj, settings, scene=scene, children_map=children_map, arm_map=arm_map,
        )
        vis_m = [m for m in cand if _is_outliner_visible(m, view_layer, vis_cols=vis_cols)] if view_layer is not None else list(cand)
        if not vis_m:
            return False, []
        if has_lv:
            lv_m = [m for m in vis_m if _safe_visible_get(m, view_layer, v3d)]
            return bool(lv_m), lv_m
        return True, vis_m

    if has_lv:
        return _safe_visible_get(obj, view_layer, v3d), None
    return True, None


def _is_object_visible_in_viewport(obj, settings, view_layer, v3d=None, scene=None, vis_cols=None, children_map=None, arm_map=None, metaball_map=None):
    """Return True if tracked obj is visible in Outliner (view_layer) and optional Local View (v3d)."""
    vis, _ = _get_visible_viewport_part_meshes(
        obj, settings, view_layer, v3d=v3d, scene=scene, vis_cols=vis_cols,
        children_map=children_map, arm_map=arm_map,
    )
    return vis


def _collect_viewport_visible_info(context, out_part_keys=None, out_part_objs=None):
    """Collect visible tracked object settings/keys/names for the current Outliner & 3D Viewport."""
    scene = context.scene
    view_layer = getattr(context, 'view_layer', None)
    space = getattr(context, 'space_data', None)
    v3d = space if (space is not None and getattr(space, 'type', None) == 'VIEW_3D') else None
    has_lv = v3d is not None and getattr(v3d, 'local_view', None) is not None
    vkey = None
    if not has_lv:
        vkey = (
            scene.as_pointer(),
            view_layer.as_pointer() if view_layer is not None else 0,
            _vis_cache_rev,
            len(scene.onion_skin_objects),
        )
        cached_v = _vis_info_cache.get(vkey)
        if cached_v is not None:
            c_settings, c_objs, c_vis_keys, c_vis_names, c_all_keys, c_pkeys, c_pobjs = cached_v
            if out_part_keys is not None and c_pkeys:
                out_part_keys.update(c_pkeys)
            if out_part_objs is not None and c_pobjs:
                out_part_objs.update(c_pobjs)
            return c_settings, c_objs, c_vis_keys, c_vis_names, c_all_keys

    vis_cols = _get_visible_collections(view_layer) if view_layer is not None else None
    _, vis_arm_map, children_map, meta_map, _ = _build_scene_lookup_maps(scene)
    obj_settings_by_key = {}
    obj_by_key = {}
    vis_keys = set()
    vis_names = set()
    all_keys = set()
    cached_pkeys = {}
    cached_pobjs = {}
    for item in scene.onion_skin_objects:
        obj = item.object
        if not obj or not _is_valid_object(obj):
            continue
        ok = get_object_key(obj)
        all_keys.add(ok)
        if not item.visible:
            continue
        s_cfg = properties.get_object_settings(obj, item)
        vis, part_meshes = _get_visible_viewport_part_meshes(
            obj, s_cfg, view_layer, v3d=v3d, scene=scene, vis_cols=vis_cols,
            children_map=children_map, arm_map=vis_arm_map,
        )
        if not vis:
            continue
        vis_keys.add(ok)
        vis_names.add(obj.name)
        obj_settings_by_key[ok] = s_cfg
        obj_by_key[ok] = obj
        if part_meshes is not None:
            pk = {get_object_key(m) for m in part_meshes}
            cached_pkeys[ok] = pk
            cached_pobjs[ok] = part_meshes
            if out_part_keys is not None:
                out_part_keys[ok] = pk
            if out_part_objs is not None:
                out_part_objs[ok] = part_meshes
    if vkey is not None:
        _vis_info_cache[vkey] = (
            obj_settings_by_key, obj_by_key, vis_keys, vis_names, all_keys, cached_pkeys, cached_pobjs
        )
    return obj_settings_by_key, obj_by_key, vis_keys, vis_names, all_keys


def _is_marker_visible_in_viewport(marker, vis_keys, vis_names, all_keys, vis_part_keys=None):
    """Return True if a ghost marker's owning object(s) or isolated part(s) are visible in this viewport."""
    m_objs = marker[4] if len(marker) > 4 else ()
    if not m_objs:
        return True
    vis_parts = set()
    if vis_part_keys:
        for pset in vis_part_keys.values():
            vis_parts.update(pset)
    if any(m in all_keys for m in m_objs):
        return any((m in vis_keys) or (m in vis_parts) for m in m_objs)
    if vis_parts and any(m in vis_parts for m in m_objs):
        return True
    return any(m in vis_names for m in m_objs)


def _has_scene_time_node(node_tree, visited=None):
    """Return True if node_tree or any nested group reads Scene Time."""
    if node_tree is None:
        return False
    if visited is None:
        visited = set()
    if node_tree in visited:
        return False
    visited.add(node_tree)
    for node in getattr(node_tree, 'nodes', ()):
        if getattr(node, 'bl_idname', '') == 'GeometryNodeInputSceneTime':
            return True
        sub = getattr(node, 'node_tree', None)
        if sub is not None and _has_scene_time_node(sub, visited):
            return True
    return False


def _is_gn_tree_local_space_safe(node_tree, visited=None):
    """Return True if Geometry Nodes evaluation only depends on local space (safe for GPU delta xform)."""
    if node_tree is None:
        return True
    if visited is None:
        visited = set()
    if node_tree in visited:
        return False
    visited.add(node_tree)
    for node in getattr(node_tree, 'nodes', ()):
        bid = getattr(node, 'bl_idname', '')
        if bid in ('GeometryNodeInputSceneTime', 'GeometryNodeSelfObject', 'GeometryNodeInputActiveCamera'):
            return False
        if bid in ('GeometryNodeObjectInfo', 'GeometryNodeCollectionInfo'):
            if getattr(node, 'transform_space', 'ORIGINAL') == 'RELATIVE':
                return False
        sub = getattr(node, 'node_tree', None)
        if sub is not None and not _is_gn_tree_local_space_safe(sub, visited):
            return False
    return True


def _collect_allowed_driver_ids(dep_objs, dep_trees):
    """Collect original IDs evaluated directly during direct F-curve stepping."""
    allowed = set()
    for dobj in dep_objs:
        allowed.add(getattr(dobj, 'original', dobj))
        d = getattr(dobj, 'data', None)
        if d is not None:
            allowed.add(getattr(d, 'original', d))
            sk = getattr(d, 'shape_keys', None)
            if sk is not None:
                allowed.add(getattr(sk, 'original', sk))
    for tree in dep_trees:
        allowed.add(getattr(tree, 'original', tree))
    return allowed


def _has_unsafe_driver_for_direct_eval(holder, allowed_ids):
    """Return True if holder has any driver that depends on scene.frame_set or external IDs."""
    anim = getattr(holder, 'animation_data', None)
    if not anim:
        return False
    drivers = anim.drivers
    if not drivers:
        return False
    for fcu_drv in drivers:
        drv = fcu_drv.driver
        if drv is None:
            return True
        if not drv.is_valid:
            continue
        variables = drv.variables
        if len(variables) == 0 and (
            len(fcu_drv.keyframe_points) > 0 or len(fcu_drv.modifiers) > 0
        ):
            return True
        if drv.type == 'SCRIPTED':
            expr = drv.expression or ''
            if 'frame' in expr or 'bpy' in expr:
                var_names = {v.name for v in variables}
                if 'frame' not in var_names and _FRAME_IDENT_RE.search(expr):
                    return True
                if 'bpy' not in var_names and _BPY_IDENT_RE.search(expr):
                    return True
        for var in variables:
            for targ in var.targets:
                tid = targ.id
                if tid is not None:
                    if targ.id_type == 'SCENE':
                        return True
                    if tid not in allowed_ids and getattr(tid, 'original', tid) not in allowed_ids:
                        return True
    return False


_XFORM_DELTA_DRIVER_PATHS = ('location', 'rotation_', 'scale', 'matrix_')


def _analyze_drivers_for_xform_delta(root_obj, dep_objs, dep_trees):
    """Return (delta_allowed, scale_sensitive) based on whether internal drivers read root Object transforms."""
    root_orig = getattr(root_obj, 'original', root_obj)
    scale_sensitive = False
    holders = []
    for dobj in dep_objs:
        holders.append(dobj)
        d = getattr(dobj, 'data', None)
        if d is not None:
            holders.append(d)
            sk = getattr(d, 'shape_keys', None)
            if sk is not None:
                holders.append(sk)
    holders.extend(dep_trees)
    for h in holders:
        anim = getattr(h, 'animation_data', None)
        if not anim:
            continue
        drivers = anim.drivers
        if not drivers:
            continue
        for fcu_drv in drivers:
            drv = fcu_drv.driver
            if drv is None or not drv.is_valid:
                continue
            for var in drv.variables:
                vtype = var.type
                if vtype == 'SINGLE_PROP':
                    for targ in var.targets:
                        dp = targ.data_path or ''
                        if not dp or dp.startswith('pose.bones') or not dp.startswith(_XFORM_DELTA_DRIVER_PATHS):
                            continue
                        tid = targ.id
                        if tid is not None and (tid == root_orig or getattr(tid, 'original', tid) == root_orig):
                            return False, True
                elif vtype in ('TRANSFORMS', 'LOC_DIFF', 'ROTATION_DIFF'):
                    for targ in var.targets:
                        tid = targ.id
                        if tid is None:
                            continue
                        if not targ.bone_target:
                            return False, True
                        if targ.transform_space == 'WORLD_SPACE':
                            if vtype == 'TRANSFORMS':
                                return False, True
                            if vtype == 'LOC_DIFF':
                                scale_sensitive = True
    return True, scale_sensitive


def _can_use_direct_fcurve_eval(dep_objs, dep_trees=()):
    """Return True if frame F can be evaluated directly via F-curves + depsgraph.update() without frame_set."""
    if not _can_use_per_frame_action_inval(dep_objs, dep_trees):
        return False
    allowed_ids = _collect_allowed_driver_ids(dep_objs, dep_trees)
    for dobj in dep_objs:
        for mod in getattr(dobj, 'modifiers', ()):
            if getattr(mod, 'show_viewport', True) and getattr(mod, 'type', '') in _TIME_DEPENDENT_MOD_TYPES:
                return False
        holders = [dobj, getattr(dobj, 'data', None)]
        if holders[1] is not None:
            holders.append(getattr(holders[1], 'shape_keys', None))
        for h in holders:
            if h is not None and _has_unsafe_driver_for_direct_eval(h, allowed_ids):
                return False
    for tree in dep_trees:
        if _has_unsafe_driver_for_direct_eval(tree, allowed_ids):
            return False
        if _has_scene_time_node(tree):
            return False
    return True


def _eval_xform_delta_capability(root_obj, dep_objs, dep_trees=()):
    """Return (delta_ok, scale_sensitive) once without re-scanning Rigify drivers twice."""
    if root_obj is None or root_obj.type not in ('ARMATURE', 'MESH', 'CURVE', 'FONT', 'SURFACE', 'EMPTY'):
        return False, False
    inst_members = None
    if root_obj.type == 'EMPTY':
        if _empty_is_collection_instance(root_obj):
            inst_coll = getattr(root_obj, 'instance_collection', None)
            inst_members = set(inst_coll.all_objects) if inst_coll else set()
        elif _empty_has_gn_geometry(root_obj):
            if len(dep_objs) != 1 or any(not _is_gn_tree_local_space_safe(t) for t in dep_trees):
                return False, False
        else:
            return False, False
    elif dep_trees and any(not _is_gn_tree_local_space_safe(t) for t in dep_trees):
        return False, False
    if not _can_use_direct_fcurve_eval(dep_objs, dep_trees):
        return False, False
    can_analytic, _ = _build_analytic_world_chain(root_obj)
    if not can_analytic:
        return False, False
    for dobj in dep_objs:
        for mod in getattr(dobj, 'modifiers', ()):
            if not getattr(mod, 'show_viewport', True):
                continue
            mtype = getattr(mod, 'type', '')
            if mtype == 'NODES':
                if _is_gn_tree_local_space_safe(getattr(mod, 'node_group', None)):
                    continue
                return False, False
            if mtype not in _DELTA_SAFE_MOD_TYPES:
                return False, False
        if dobj != root_obj:
            if inst_members is not None and dobj in inst_members:
                if len(getattr(dobj, 'constraints', ())) > 0:
                    return False, False
                continue
            if getattr(dobj, 'parent', None) != root_obj or getattr(dobj, 'parent_type', 'OBJECT') == 'BONE':
                return False, False
            if len(getattr(dobj, 'constraints', ())) > 0:
                return False, False
    return _analyze_drivers_for_xform_delta(root_obj, dep_objs, dep_trees)


def _get_cached_delta_capability(root_key, root_obj, dep_objs, dep_trees, fc_entries=None):
    """Return cached (can_delta, scale_sensitive, ok_chain, xform_chain) for pure action edits."""
    if root_obj is None:
        return False, False, False, None
    act_ptrs = tuple(
        (
            d.as_pointer(),
            d.animation_data.action.as_pointer()
            if (getattr(d, 'animation_data', None) and d.animation_data.action)
            else 0,
        )
        for d in dep_objs
    )
    fc_sig = (
        len(fc_entries),
        sum(1 for _, fc in fc_entries if getattr(fc, 'mute', False)),
    ) if fc_entries is not None else None
    sig = (root_obj.as_pointer(), act_ptrs, len(dep_trees), fc_sig, _vis_cache_rev)
    cached = _delta_cap_cache.get(root_key)
    if cached is not None and cached[0] == sig:
        return cached[1], cached[2], cached[3], cached[4]
    can_delta, scale_sens = _eval_xform_delta_capability(root_obj, dep_objs, dep_trees)
    if can_delta:
        ok_chain, chain = _build_analytic_world_chain(root_obj)
    else:
        ok_chain, chain = False, None
    if fc_sig is not None:
        _delta_cap_cache[root_key] = (sig, can_delta, scale_sens, ok_chain, chain)
    return can_delta, scale_sens, ok_chain, chain


def _can_use_xform_delta_shortcut(root_obj, dep_objs, dep_trees=()):
    """Return True if pure Object-Transform edits on cached root_obj frames can reuse deformed geometry via GPU matrix."""
    return _eval_xform_delta_capability(root_obj, dep_objs, dep_trees)[0]


def _split_fc_entries_for_root(fc_entries, root_key):
    """Split dependency F-curves into (non_xform_fcs, root_obj_xform_fcs)."""
    non_xform = []
    xform = []
    for tag, fc in fc_entries:
        if tag[0] == 'O' and tag[1] == root_key and tag[2] in _TRANSFORM_DATA_PATHS:
            xform.append((tag, fc))
        else:
            non_xform.append((tag, fc))
    return non_xform, xform


def _eval_split_action_sigs_presplit(non_xform, xform, frame):
    """Evaluate pre-split (non_xform_fcs, root_xform_fcs) at frame."""
    return _eval_dep_actions_at_frame(non_xform, frame), _eval_dep_actions_at_frame(xform, frame)


def _eval_split_action_sigs(fc_entries, root_key, frame):
    """Return (non_xform_sig, root_xform_sig) tuple for per-frame change & transform-delta detection."""
    non_xform, xform = _split_fc_entries_for_root(fc_entries, root_key)
    return _eval_dep_actions_at_frame(non_xform, frame), _eval_dep_actions_at_frame(xform, frame)


def _split_rna_data_path(data_path):
    """Split an RNA data_path into (parent_path, prop_or_key, is_bracket) respecting quotes."""
    in_sq = False
    in_dq = False
    escaped = False
    depth = 0
    last_dot = -1
    last_top_bracket = -1
    last_top_close = -1
    for i, ch in enumerate(data_path):
        if escaped:
            escaped = False
            continue
        if ch == '\\' and (in_sq or in_dq):
            escaped = True
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == '[':
                if depth == 0:
                    last_top_bracket = i
                depth += 1
            elif ch == ']':
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        last_top_close = i
            elif ch == '.' and depth == 0:
                last_dot = i
    if last_top_close == len(data_path) - 1 and last_top_bracket > last_dot and last_top_bracket != -1:
        import ast
        key_val = ast.literal_eval(data_path[last_top_bracket + 1:-1])
        return data_path[:last_top_bracket], key_val, True
    if last_dot != -1:
        return data_path[:last_dot], data_path[last_dot + 1:], False
    return "", data_path, False


def _resolve_fcurve_setter(holder, fc):
    """Resolve a direct RNA property setter tuple for fc on holder, or None to trigger frame_set fallback."""
    if getattr(holder, 'library', None) is not None and not getattr(holder, 'override_library', None):
        return None
    try:
        base_path, prop_or_key, is_bracket = _split_rna_data_path(fc.data_path)
        target = holder.path_resolve(base_path) if base_path else holder
        cur = target[prop_or_key] if is_bracket else getattr(target, prop_or_key)
    except Exception:
        return None

    is_seq = hasattr(cur, '__len__') and not isinstance(cur, (str, bytes))
    idx = fc.array_index
    if is_seq:
        if idx < 0 or idx >= len(cur):
            return None
        elem = cur[idx]
        if isinstance(elem, bool):
            val_kind = 'bool'
            orig_val = bool(elem)
        elif isinstance(elem, int):
            val_kind = 'int'
            orig_val = int(elem)
        elif isinstance(elem, float):
            val_kind = 'float'
            orig_val = float(elem)
        else:
            return None
    else:
        if isinstance(cur, bool):
            val_kind = 'bool'
            orig_val = bool(cur)
        elif isinstance(cur, int):
            val_kind = 'int'
            orig_val = int(cur)
        elif isinstance(cur, float):
            val_kind = 'float'
            orig_val = float(cur)
        else:
            return None
    return (holder, target, prop_or_key, is_bracket, is_seq, idx, val_kind, orig_val, fc)


def _build_direct_setters(dep_objs, dep_trees=()):
    """Build direct property setter list and minimal refresh tag specs across dep_objs and dep_trees."""
    setters = []
    tag_ids = []
    seen_tags = set()
    full_tag_ptrs = set()

    def _add_tag(id_block):
        if id_block is not None and id_block not in seen_tags:
            seen_tags.add(id_block)
            tag_ids.append(id_block)

    def _gather_from(holder, tag_owner):
        if holder is None:
            return True
        anim = getattr(holder, 'animation_data', None)
        if not anim or not anim.action:
            return True
        fcs = collect_action_fcurves(anim.action, anim)
        if not fcs:
            return True
        _add_tag(holder)
        if tag_owner is not None:
            _add_tag(tag_owner)
        holder_is_obj = isinstance(holder, bpy.types.Object)
        for fc in fcs:
            if getattr(fc, 'mute', False):
                continue
            s = _resolve_fcurve_setter(holder, fc)
            if s is None:
                return False
            is_obj_xform = holder_is_obj and (holder is tag_owner or tag_owner is None) and (fc.data_path in _TRANSFORM_DATA_PATHS)
            if not is_obj_xform:
                full_tag_ptrs.add(holder.as_pointer())
                if tag_owner is not None:
                    full_tag_ptrs.add(tag_owner.as_pointer())
            setters.append(s + (tag_owner,))
        return True

    for dobj in dep_objs:
        if not _gather_from(dobj, dobj):
            return None, None
        data = getattr(dobj, 'data', None)
        if data is not None:
            if not _gather_from(data, dobj):
                return None, None
            sk = getattr(data, 'shape_keys', None)
            if sk is not None and not _gather_from(sk, dobj):
                return None, None
    for tree in dep_trees:
        if not _gather_from(tree, tree):
            return None, None
    tag_specs = [
        (ib, None if (not isinstance(ib, bpy.types.Object) or ib.as_pointer() in full_tag_ptrs) else {'OBJECT'})
        for ib in tag_ids
    ]
    return setters, tag_specs


def _apply_direct_setters(setters, tag_ids, frame):
    """Evaluate and write changed F-curve values for frame directly onto targets and minimal-refresh tag datablocks."""
    changed_ptrs = set()
    for holder, target, prop_or_key, is_bracket, is_seq, idx, val_kind, _, fc, tag_owner in setters:
        val = fc.evaluate(frame)
        if val_kind == 'bool':
            cval = bool(round(val))
        elif val_kind == 'int':
            cval = int(round(val))
        else:
            cval = float(val)
        if is_seq:
            cur = target[prop_or_key] if is_bracket else getattr(target, prop_or_key)
            if cur[idx] != cval:
                cur[idx] = cval
                changed_ptrs.add(holder.as_pointer())
                if tag_owner is not None:
                    changed_ptrs.add(tag_owner.as_pointer())
        elif is_bracket:
            if target[prop_or_key] != cval:
                target[prop_or_key] = cval
                changed_ptrs.add(holder.as_pointer())
                if tag_owner is not None:
                    changed_ptrs.add(tag_owner.as_pointer())
        else:
            if getattr(target, prop_or_key) != cval:
                setattr(target, prop_or_key, cval)
                changed_ptrs.add(holder.as_pointer())
                if tag_owner is not None:
                    changed_ptrs.add(tag_owner.as_pointer())
    for item in tag_ids:
        id_block, ref_flags = item if isinstance(item, tuple) else (item, None)
        try:
            if changed_ptrs and id_block.as_pointer() not in changed_ptrs:
                continue
            if ref_flags is not None:
                id_block.update_tag(refresh=ref_flags)
            else:
                id_block.update_tag()
        except Exception:
            pass


def _restore_direct_setters(setters, tag_ids):
    """Restore original pre-evaluation values written by direct setters."""
    changed_ptrs = set()
    for holder, target, prop_or_key, is_bracket, is_seq, idx, _, orig_val, _, tag_owner in reversed(setters):
        try:
            if is_seq:
                cur = target[prop_or_key] if is_bracket else getattr(target, prop_or_key)
                if cur[idx] != orig_val:
                    cur[idx] = orig_val
                    changed_ptrs.add(holder.as_pointer())
                    if tag_owner is not None:
                        changed_ptrs.add(tag_owner.as_pointer())
            elif is_bracket:
                if target[prop_or_key] != orig_val:
                    target[prop_or_key] = orig_val
                    changed_ptrs.add(holder.as_pointer())
                    if tag_owner is not None:
                        changed_ptrs.add(tag_owner.as_pointer())
            else:
                if getattr(target, prop_or_key) != orig_val:
                    setattr(target, prop_or_key, orig_val)
                    changed_ptrs.add(holder.as_pointer())
                    if tag_owner is not None:
                        changed_ptrs.add(tag_owner.as_pointer())
        except Exception:
            pass
    for item in tag_ids:
        id_block, ref_flags = item if isinstance(item, tuple) else (item, None)
        try:
            if changed_ptrs and id_block.as_pointer() not in changed_ptrs:
                continue
            if ref_flags is not None:
                id_block.update_tag(refresh=ref_flags)
            else:
                id_block.update_tag()
        except Exception:
            pass


def _should_debounce_heavy_modal(scene):
    """Non-starving modal pacing guard: never block updates indefinitely during continuous mouse movement."""
    return (time.perf_counter() - _last_modal_eval_end) < 0.012


# ---------------------------------------------------------------------------
# Batch Building (Double-Buffered, Zero-Flicker, Real-Time Modal Budgeted)
# ---------------------------------------------------------------------------

_is_building = False
_is_evaluating = False


def _is_identity_matrix(mat, eps=1e-6):
    """Return True if a 4x4 matrix is within eps of Identity."""
    for r in range(4):
        for c in range(4):
            target = 1.0 if r == c else 0.0
            if abs(mat[r][c] - target) >= eps:
                return False
    return True


def _wrap_deforming_delta_batch(obj_key, frame, settings, batch):
    """Wrap a cached deforming frame batch as an instant GPU-transformed payload when delta_mat is active."""
    if not isinstance(batch, list) or not batch:
        return batch
    d_mat = _deforming_delta_mats.get((obj_key, frame))
    if d_mat is None or (obj_key, frame) not in _deforming_base_mats or not cache.has_frame(obj_key, frame):
        return batch
    if _is_identity_matrix(d_mat):
        return batch
    wf = bool(settings.use_wireframe)
    use_fn = bool(settings.use_fresnel)
    use_sh = bool(getattr(settings, 'use_shading', True))
    want_fn = True if not wf else use_fn

    if len(batch) == 1:
        ent = batch[0]
        if not wf and ent[2] is None:
            return batch
        gpu_b = cache.get_entry_gpu_batch(
            ent, use_wireframe=wf, use_fresnel=want_fn, use_shading=use_sh,
        )
        if gpu_b is None:
            return batch
        return ('RIGID', gpu_b, d_mat, ent[6], ent[7], ent[8])

    sub_items = []
    for ent in batch:
        if not wf and ent[2] is None:
            return batch
        gpu_b = cache.get_entry_gpu_batch(
            ent, use_wireframe=wf, use_fresnel=want_fn, use_shading=use_sh,
        )
        if gpu_b is None:
            return batch
        sub_items.append((ent[0], gpu_b, ent[6], ent[7], ent[8]))
    return ('RIGID_LIST', d_mat, sub_items)


def _try_reuse_same_pose_cached_frame(obj_key, frame, fc_entries, xform_chain, scale_sensitive=False):
    """Reuse an already-cached same-pose GPU batch with an analytic world delta matrix when possible."""
    global _eval_seq_counter
    if fc_entries is None or xform_chain is None:
        return False
    cur_non_xform, cur_xform = _eval_split_action_sigs(fc_entries, obj_key, frame)
    for cf_ref in cache.get_cached_frames(obj_key):
        if cf_ref == frame:
            continue
        ref_sig = _action_frame_sigs.get((obj_key, cf_ref))
        if (
            ref_sig is None
            or not isinstance(ref_sig, tuple)
            or len(ref_sig) != 2
            or ref_sig[0] != cur_non_xform
        ):
            continue
        base_mat_ref = _deforming_base_mats.get((obj_key, cf_ref))
        ref_batch = cache.get_frame(obj_key, cf_ref)
        if base_mat_ref is None or not isinstance(ref_batch, list) or not ref_batch:
            continue
        try:
            new_mat = _eval_world_matrix_analytic(xform_chain, frame)
            if scale_sensitive:
                if max(abs(a - b) for a, b in zip(new_mat.to_scale(), base_mat_ref.to_scale())) >= 1e-4:
                    continue
            d_mat = new_mat @ base_mat_ref.inverted()
        except Exception:
            continue
        cache.put_frame(obj_key, frame, ref_batch)
        _deforming_base_mats[(obj_key, frame)] = base_mat_ref
        _deforming_delta_mats[(obj_key, frame)] = d_mat
        _action_frame_sigs[(obj_key, frame)] = (cur_non_xform, cur_xform)
        _eval_seq_counter += 1
        _frame_last_eval_seq[(obj_key, frame)] = _eval_seq_counter
        return True
    return False


def _get_or_build_playback_meta(obj, obj_key, scene, arm_map=None):
    """Return cached (dep_objs, dep_trees, can_direct, fc_entries, delta_chain, scale_sens) for obj_key."""
    meta = _playback_eval_meta.get(obj_key)
    if meta is not None:
        return meta
    if arm_map is None:
        arm_map = _build_armature_children_map(scene)
    dep_objs, _, dep_trees = _gather_object_dependencies(obj, scene, armature_children_map=arm_map)
    fc_entries = _collect_dep_fcurves_sorted(dep_objs, dep_trees) if _can_use_per_frame_action_inval(dep_objs, dep_trees) else None
    can_direct = _can_use_direct_fcurve_eval(dep_objs, dep_trees)
    can_delta, scale_sens, ok_chain, chain = _get_cached_delta_capability(
        obj_key, obj, dep_objs, dep_trees, fc_entries=fc_entries
    )
    delta_chain = chain if (can_delta and ok_chain) else None
    meta = (dep_objs, dep_trees, can_direct, fc_entries, delta_chain, scale_sens)
    _playback_eval_meta[obj_key] = meta
    return meta


def _collect_inst_parent_objs(items, scene, view_layer=None, vis_cols=None):
    """Collect tracked or child objects needing depsgraph.object_instances extraction."""
    inst_parent_objs = set()
    for _, obj, _, settings, _ in items:
        if obj.type == 'EMPTY' and (_empty_has_gn_geometry(obj) or _empty_is_collection_instance(obj)):
            inst_parent_objs.add(obj)
        elif obj.type in _GEO_TYPES and any(getattr(m, 'type', None) == 'NODES' and m.show_viewport for m in obj.modifiers):
            inst_parent_objs.add(obj)
        elif obj.type == 'ARMATURE':
            for mobj in resolve_mesh_objects(
                obj, settings, scene=scene, view_layer=view_layer, vis_cols=vis_cols
            ):
                if any(getattr(m, 'type', None) == 'NODES' and m.show_viewport for m in mobj.modifiers):
                    inst_parent_objs.add(mobj)
    return inst_parent_objs


def _cache_extracted_entries(obj_key, frame, entries, settings, build_gpu=True):
    """Store extracted geometry entries in the frame cache and optionally build GPU batches eagerly."""
    if entries:
        if build_gpu:
            wf = bool(settings.use_wireframe)
            use_fn = bool(settings.use_fresnel)
            use_sh = bool(getattr(settings, 'use_shading', True))
            for entry in entries:
                cache.get_entry_gpu_batch(
                    entry,
                    use_wireframe=wf,
                    use_fresnel=use_fn,
                    use_shading=use_sh,
                )
        cache.put_frame(obj_key, frame, entries)
    else:
        cache.put_frame(obj_key, frame, None)


def _eval_and_cache_deforming_at_frame(
    frame, items_at_frame, dg, inst_parent_objs, dirty_fc_map, dirty_delta_chains, scene, dg_step_cost=0.0,
    view_layer=None, vis_cols=None, build_gpu=True,
):
    """Extract and cache geometry for all items_at_frame missing frame at the already-evaluated depsgraph."""
    global _eval_seq_counter
    inst_chunks_cache = {}
    needed_inst_parents = {
        obj for _, obj, obj_key, _, needed in items_at_frame
        if obj in inst_parent_objs
        and not cache.has_frame(obj_key, frame)
        and any((x[0] if isinstance(x, (tuple, list)) else x) == frame for x in needed)
    }
    n_extracted = max(1, len(items_at_frame))
    per_item_dg_cost = dg_step_cost / n_extracted
    for _item, obj, obj_key, settings, needed in items_at_frame:
        if cache.has_frame(obj_key, frame) or not any(
            (x[0] if isinstance(x, (tuple, list)) else x) == frame for x in needed
        ):
            continue
        wf = bool(settings.use_wireframe)
        need_fn = not wf
        flag_key = (wf, need_fn)
        if needed_inst_parents and flag_key not in inst_chunks_cache and HAS_NUMPY:
            inst_chunks_cache[flag_key], _ = _collect_instance_chunks_from_depsgraph(
                needed_inst_parents, dg, wf, need_fn
            )
        pre_inst = inst_chunks_cache.get(flag_key)

        t_e0 = time.perf_counter()
        entries = _extract_tracked_entries(
            obj, settings, dg, wf, need_fn,
            precollected_inst_chunks=pre_inst, scene=scene,
            view_layer=view_layer, vis_cols=vis_cols,
        )
        _passive_extract_costs[obj_key] = (time.perf_counter() - t_e0) + per_item_dg_cost

        fc_entries = dirty_fc_map.get(obj_key)
        if fc_entries is not None:
            _action_frame_sigs[(obj_key, frame)] = _eval_split_action_sigs(fc_entries, obj_key, frame)

        _deforming_delta_mats.pop((obj_key, frame), None)
        chain = dirty_delta_chains.get(obj_key)
        if chain is not None:
            try:
                eval_mat = obj.evaluated_get(dg).matrix_world.copy()
                analytic_mat = _eval_world_matrix_analytic(chain, frame)
                max_diff = max(abs(eval_mat[r][c] - analytic_mat[r][c]) for r in range(4) for c in range(4))
                if max_diff < 1e-3:
                    _deforming_base_mats[(obj_key, frame)] = eval_mat
                else:
                    _deforming_base_mats.pop((obj_key, frame), None)
            except Exception:
                _deforming_base_mats.pop((obj_key, frame), None)
        else:
            _deforming_base_mats.pop((obj_key, frame), None)

        _cache_extracted_entries(obj_key, frame, entries, settings, build_gpu=build_gpu)
        _eval_seq_counter += 1
        _frame_last_eval_seq[(obj_key, frame)] = _eval_seq_counter


def _build_onion_batches(context, current, depsgraph, allow_deforming_eval=True, eval_time_budget=None, fast_modal_reuse=False):
    """Evaluate ghost frames and build GPU batches with C++ SIMD acceleration, rigid instancing, and double buffering.

    Returns True if evaluation paused early due to eval_time_budget and still has remaining uncached frames.
    """
    global _is_building, _is_evaluating, _was_playing, _eval_seq_counter, _playback_build_cache
    if _is_building:
        return False

    scene = context.scene
    global_settings = scene.onion_skin_settings

    if not global_settings.enabled or not scene.onion_skin_objects:
        cache.clear_onion_batches()
        return False

    is_playing = screen_animation_playing(context)
    if is_playing:
        _was_playing = True
    if _stale_offscreen_action_roots and not _is_transforming(context):
        _reconcile_offscreen_action_roots(scene)

    # Index previous batches so interactive drags / playback never wipe out ghosts (zero flicker)
    prev_deforming = {}
    prev_rigid_mats = {}
    for b_item in cache.get_onion_batches():
        if len(b_item) >= 6:
            ok, fr, _, _, _, payload = b_item
            if isinstance(payload, tuple) and len(payload) >= 3 and payload[0] == 'RIGID':
                prev_rigid_mats[(ok, fr)] = payload[2]
                prev_deforming[(ok, fr)] = payload
            else:
                prev_deforming[(ok, fr)] = payload

    new_batches = []
    incomplete = False
    _is_building = True
    _is_evaluating = True

    try:
        view_layer = getattr(context, 'view_layer', None)
        pb_key = (
            scene.as_pointer(),
            view_layer.as_pointer() if view_layer is not None else 0,
            _vis_cache_rev,
            len(scene.onion_skin_objects),
        )
        cached_pb = _playback_build_cache
        if (is_playing or fast_modal_reuse) and cached_pb is not None and cached_pb[0] == pb_key:
            visible_playback_entries, play_arm_map, vis_cols = cached_pb[1], cached_pb[2], cached_pb[3]
        else:
            vis_cols = _get_visible_collections(view_layer) if view_layer is not None else None
            _, vis_arm_map, children_map, meta_map, _ = _build_scene_lookup_maps(scene)
            visible_keys = set()
            visible_item_settings = {}
            for idx, item in enumerate(scene.onion_skin_objects):
                obj = item.object
                if not obj or not item.visible or not _is_valid_object(obj):
                    continue
                _track_object_identity(obj)
                s_cfg = properties.get_object_settings(obj, item)
                ok = get_object_key(obj)
                if view_layer is not None:
                    _cached_outliner_vis[ok] = _compute_outliner_vis_sig(
                        obj, s_cfg, view_layer, scene=scene, vis_cols=vis_cols,
                        children_map=children_map, arm_map=vis_arm_map, metaball_map=meta_map,
                    )
                if not _is_object_visible_in_viewport(
                    obj, s_cfg, view_layer, v3d=None, scene=scene, vis_cols=vis_cols,
                    children_map=children_map, arm_map=vis_arm_map, metaball_map=meta_map,
                ):
                    continue
                if ok not in _cached_signatures:
                    _cached_signatures[ok] = get_object_geometry_signature(obj)
                if obj.type == 'EMPTY' and getattr(obj, 'instance_type', None) == 'COLLECTION':
                    _seed_collection_sigs(getattr(obj, 'instance_collection', None))
                visible_keys.add(ok)
                visible_item_settings[idx] = s_cfg

            visible_playback_entries = []
            for idx, item in enumerate(scene.onion_skin_objects):
                settings = visible_item_settings.get(idx)
                if settings is None:
                    continue
                obj = item.object
                if obj.type == 'META':
                    mother = _get_metaball_mother(obj, scene, metaball_map=meta_map)
                    if mother is not None and mother is not obj and get_object_key(mother) in visible_keys:
                        continue
                obj_key = get_object_key(obj)
                visible_playback_entries.append((
                    item, obj, obj_key, settings, is_object_rigid(obj), get_object_geometry_signature(obj)
                ))
            play_arm_map = vis_arm_map
            _playback_build_cache = (pb_key, visible_playback_entries, play_arm_map, vis_cols)
            if not is_playing and not fast_modal_reuse:
                _vis_info_cache.clear()

        # 1. Process RIGID objects (instant matrix evaluation, 1 shared GPUBatch in VRAM)
        deforming_items = []
        use_kf_fast = bool(is_playing or fast_modal_reuse)
        for item, obj, obj_key, settings, obj_is_rigid, sig in visible_playback_entries:
            needed = get_needed_frames(current, settings, scene, obj=obj, use_kf_cache=use_kf_fast)
            if not needed:
                continue

            if obj_is_rigid:
                use_wf = bool(settings.use_wireframe)
                use_lit = not use_wf
                rigid_info = cache.get_rigid_batch(obj_key, use_wireframe=use_wf, use_fresnel=use_lit, signature=sig)
                if rigid_info is None:
                    verts, normals, indices, prim, b_min, b_max, center = extract_rigid_mesh(
                        obj, wireframe=use_wf, need_normals=use_lit, depsgraph=depsgraph
                    )
                    if verts is not None and indices is not None:
                        shader = cache.get_fresnel_shader() if (use_lit and normals is not None) else cache.get_shader()
                        if shader is None:
                            shader = cache.get_shader()
                        gpu_batch = None
                        if shader:
                            try:
                                if HAS_NUMPY:
                                    gpu_batch = cache._build_fast_batch(
                                        shader, prim, verts, normals if (use_lit and shader == cache.get_fresnel_shader()) else None, indices
                                    )
                                if gpu_batch is None:
                                    attrs = {"pos": verts}
                                    if use_lit and normals is not None and shader == cache.get_fresnel_shader():
                                        attrs["normal"] = normals
                                    gpu_batch = batch_for_shader(shader, prim, attrs, indices=indices)
                            except Exception:
                                gpu_batch = None
                        cache.put_rigid_batch(obj_key, use_wf, use_lit, gpu_batch, b_min, b_max, center, prim, signature=sig)
                        rigid_info = (gpu_batch, b_min, b_max, center, prim)

                if rigid_info is not None:
                    if is_playing and depsgraph is not None and cache.get_rigid_matrix(obj_key, current) is None:
                        try:
                            cache.put_rigid_matrix(obj_key, current, obj.evaluated_get(depsgraph).matrix_world.copy())
                        except Exception:
                            pass
                    gpu_batch, b_min, b_max, center, prim = rigid_info
                    frames_needed = [f for f, _, _, _ in needed]
                    matrices = evaluate_rigid_matrices(
                        scene, obj, frames_needed,
                        allow_frame_set=allow_deforming_eval,
                        prev_matrices=prev_rigid_mats,
                        needed_info=needed,
                        current_frame=current,
                        frame_step=settings.frame_step,
                    )
                    for f, direction, t, dist in needed:
                        if f in matrices:
                            new_batches.append((
                                obj_key, f, direction, t, dist,
                                ('RIGID', gpu_batch, matrices[f], b_min, b_max, center)
                            ))
            else:
                deforming_items.append((item, obj, obj_key, settings, needed))

        # 2. Process DEFORMING objects (Rigs, GeoNodes, Metaballs, Collection Instances, etc.)
        if deforming_items:
            if is_playing or not allow_deforming_eval:
                # Passively capture the already-evaluated current frame during playback (0 frame_set calls)
                if is_playing and depsgraph is not None:
                    now_t = time.perf_counter()
                    capture_candidates = []
                    if play_arm_map is None:
                        play_arm_map = _build_armature_children_map(scene)
                    for item, obj, obj_key, settings, needed in deforming_items:
                        if not cache.has_frame(obj_key, current):
                            _dep_o, _dep_t, _can_dir, fc_entries, delta_chain, scale_sens = _get_or_build_playback_meta(
                                obj, obj_key, scene, play_arm_map
                            )
                            if _try_reuse_same_pose_cached_frame(
                                obj_key, current, fc_entries, delta_chain, scale_sens
                            ):
                                continue
                            has_any = len(cache.get_cached_frames(obj_key)) > 0
                            est_cost = _passive_extract_costs.get(obj_key, 0.0)
                            last_cap = _last_heavy_passive_capture.get(obj_key, 0.0)
                            if has_any and est_cost > 0.005 and (now_t - last_cap) < 0.25:
                                continue
                            capture_candidates.append(
                                (has_any, est_cost > 0.005, last_cap, est_cost, obj, obj_key, settings, fc_entries, delta_chain)
                            )
                    capture_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
                    passive_start = now_t
                    heavy_captured = 0
                    for has_any, is_heavy, _last_cap, est_cost, obj, obj_key, settings, fc_entries, delta_chain in capture_candidates:
                        elapsed_passive = time.perf_counter() - passive_start
                        if has_any and is_heavy:
                            if heavy_captured >= 1 and (elapsed_passive + est_cost) > 0.004:
                                continue
                            if elapsed_passive > 0.004:
                                continue
                        elif has_any and elapsed_passive > 0.004:
                            continue

                        wf = bool(settings.use_wireframe)
                        need_fn = not wf
                        t_c0 = time.perf_counter()
                        entries = _extract_tracked_entries(
                            obj, settings, depsgraph, wf, need_fn, scene=scene,
                            view_layer=view_layer, vis_cols=vis_cols,
                        )
                        t_c1 = time.perf_counter()
                        cost_now = t_c1 - t_c0
                        _passive_extract_costs[obj_key] = cost_now
                        if cost_now > 0.005:
                            _last_heavy_passive_capture[obj_key] = t_c1
                            heavy_captured += 1
                        if fc_entries is not None:
                            _action_frame_sigs[(obj_key, current)] = _eval_split_action_sigs(fc_entries, obj_key, current)
                        _deforming_delta_mats.pop((obj_key, current), None)
                        if delta_chain is not None:
                            try:
                                eval_mat = obj.evaluated_get(depsgraph).matrix_world.copy()
                                analytic_mat = _eval_world_matrix_analytic(delta_chain, current)
                                max_diff = max(abs(eval_mat[r][c] - analytic_mat[r][c]) for r in range(4) for c in range(4))
                                if max_diff < 1e-3:
                                    _deforming_base_mats[(obj_key, current)] = eval_mat
                                else:
                                    _deforming_base_mats.pop((obj_key, current), None)
                            except Exception:
                                _deforming_base_mats.pop((obj_key, current), None)
                        else:
                            _deforming_base_mats.pop((obj_key, current), None)
                        _cache_extracted_entries(obj_key, current, entries, settings, build_gpu=False)
                        _eval_seq_counter += 1
                        _frame_last_eval_seq[(obj_key, current)] = _eval_seq_counter

                # Stage 1 & Stage 2 playback acceleration (GPU matrix instancing + direct F-curve micro-fill)
                if is_playing and allow_deforming_eval:
                    if play_arm_map is None:
                        play_arm_map = _build_armature_children_map(scene)
                    micro_items = []
                    micro_missing = {}
                    micro_setters = {}
                    micro_deps = {}
                    play_fc_map = {}
                    play_delta_chains = {}
                    play_delta_scale = {}

                    for d_entry in deforming_items:
                        _it, obj, obj_key, settings, needed = d_entry
                        dep_objs, dep_trees, can_direct, fc_entries, delta_chain, scale_sens = _get_or_build_playback_meta(
                            obj, obj_key, scene, play_arm_map
                        )
                        if fc_entries is not None:
                            play_fc_map[obj_key] = fc_entries
                        if delta_chain is not None:
                            play_delta_chains[obj_key] = delta_chain
                            play_delta_scale[obj_key] = scale_sens

                        still_missing = set()
                        for f, _, _, _ in needed:
                            if cache.has_frame(obj_key, f):
                                continue
                            if (
                                fc_entries is not None
                                and delta_chain is not None
                                and _try_reuse_same_pose_cached_frame(obj_key, f, fc_entries, delta_chain, scale_sens)
                            ):
                                continue
                            still_missing.add(f)

                        if still_missing and can_direct and _passive_extract_costs.get(obj_key, 0.0) <= 0.004:
                            setters, tag_ids = _build_direct_setters(dep_objs, dep_trees)
                            if setters is not None:
                                micro_items.append(d_entry)
                                micro_missing[obj_key] = still_missing
                                micro_setters[obj_key] = (setters, tag_ids)
                                micro_deps[obj_key] = dep_objs

                    if micro_items:
                        micro_inst_parents = _collect_inst_parent_objs(micro_items, scene, view_layer, vis_cols)
                        frame_prios = {}
                        for _it, _o, ok, _s, _n in micro_items:
                            est_c = _passive_extract_costs.get(ok, 0.001)
                            for f in micro_missing[ok]:
                                key_p = (est_c, abs(f - current), f)
                                if f not in frame_prios or key_p < frame_prios[f]:
                                    frame_prios[f] = key_p
                        micro_frames = sorted(frame_prios.keys(), key=lambda f: frame_prios[f])

                        micro_snap_objs = set()
                        for ok in micro_deps:
                            micro_snap_objs.update(micro_deps[ok])
                        micro_snaps = {}
                        for dep_obj in micro_snap_objs:
                            try:
                                micro_snaps[dep_obj] = _snapshot_object_state(dep_obj)
                            except Exception:
                                pass

                        micro_start = time.perf_counter()
                        micro_evals = 0
                        micro_touched = []
                        micro_seen = set()
                        try:
                            for frame in micro_frames:
                                raw_f = [
                                    e for e in micro_items
                                    if frame in micro_missing[e[2]] and not cache.has_frame(e[2], frame)
                                ]
                                items_at_f = []
                                for e in raw_f:
                                    ok_e = e[2]
                                    if ok_e in play_fc_map and ok_e in play_delta_chains:
                                        if _try_reuse_same_pose_cached_frame(
                                            ok_e, frame, play_fc_map[ok_e], play_delta_chains[ok_e],
                                            play_delta_scale.get(ok_e, False),
                                        ):
                                            continue
                                    items_at_f.append(e)
                                if not items_at_f:
                                    continue
                                if micro_evals >= 1 and (time.perf_counter() - micro_start) >= 0.003:
                                    break
                                t_dg0 = time.perf_counter()
                                for _, _, ok_e, _, _ in items_at_f:
                                    if ok_e not in micro_seen:
                                        micro_seen.add(ok_e)
                                        micro_touched.append(ok_e)
                                    setters, tag_ids = micro_setters[ok_e]
                                    _apply_direct_setters(setters, tag_ids, frame)
                                dg_play = context.evaluated_depsgraph_get()
                                dg_play.update()
                                dg_cost = time.perf_counter() - t_dg0
                                micro_evals += 1
                                _eval_and_cache_deforming_at_frame(
                                    frame, items_at_f, dg_play, micro_inst_parents,
                                    play_fc_map, play_delta_chains, scene, dg_cost,
                                    view_layer=view_layer, vis_cols=vis_cols, build_gpu=False,
                                )
                        finally:
                            if micro_evals > 0:
                                for ok_e in reversed(micro_touched):
                                    setters, tag_ids = micro_setters[ok_e]
                                    _restore_direct_setters(setters, tag_ids)
                                for snap_obj, state in micro_snaps.items():
                                    if _is_valid_object(snap_obj):
                                        _restore_object_state(snap_obj, state)
                                try:
                                    context.evaluated_depsgraph_get().update()
                                except Exception:
                                    pass

                for item, obj, obj_key, settings, needed in deforming_items:
                    cached_frames = cache.get_cached_frames(obj_key) if is_playing else ()
                    used_dir = {'before': set(), 'after': set()}
                    slot_batches = [None] * len(needed)
                    slot_dir_idx = [None] * len(needed)

                    dir_slots = {'before': [], 'after': []}
                    for idx_n, (frame, direction, _t, _dist) in enumerate(needed):
                        dir_slots[direction].append((idx_n, frame))
                    for d in ('before', 'after'):
                        dir_slots[d].sort(key=lambda pair: abs(pair[1] - current))
                        for s_idx, (idx_n, _fr) in enumerate(dir_slots[d]):
                            slot_dir_idx[idx_n] = (d, s_idx)

                    # Reserve exact cache hits first so nearest-frame fallback never duplicates a frame
                    for idx_n, (frame, direction, t, dist) in enumerate(needed):
                        batch = cache.get_frame(obj_key, frame)
                        if batch is not None:
                            batch = _wrap_deforming_delta_batch(obj_key, frame, settings, batch)
                            slot_batches[idx_n] = batch
                            used_dir[direction].add(frame)
                            d, s_idx = slot_dir_idx[idx_n]
                            _last_ghost_slots[(obj_key, d, s_idx)] = (frame, batch)

                    search_dist = max(abs(scene.frame_end - scene.frame_start) + 100, 10000) if is_playing else max(settings.frame_step * 4, 16)
                    slot_window = search_dist if is_playing else max(search_dist * 2, 48)
                    for idx_n, (frame, direction, t, dist) in enumerate(needed):
                        batch = slot_batches[idx_n]
                        if batch is not None:
                            new_batches.append((obj_key, frame, direction, t, dist, batch))
                            continue

                        d, s_idx = slot_dir_idx[idx_n]
                        actual_f = None
                        if is_playing:
                            nf = _find_nearest_cached_frame(
                                cached_frames, frame, current, direction, search_dist, used_dir[direction]
                            )
                            if nf is not None:
                                batch = cache.get_frame(obj_key, nf)
                                if batch is not None:
                                    batch = _wrap_deforming_delta_batch(obj_key, nf, settings, batch)
                                    actual_f = nf
                                    used_dir[direction].add(nf)
                        if batch is None:
                            pf = prev_deforming.get((obj_key, frame))
                            if pf is not None and (
                                not is_playing or (frame < current if direction == 'before' else frame > current)
                            ):
                                if frame not in used_dir[direction]:
                                    batch = pf
                                    actual_f = frame
                                    used_dir[direction].add(frame)
                        if batch is None:
                            slot_prev = _last_ghost_slots.get((obj_key, d, s_idx))
                            if slot_prev is not None:
                                prev_f, prev_b = slot_prev
                                valid_side = (prev_f < current) if d == 'before' else (prev_f > current)
                                if (
                                    prev_b is not None
                                    and valid_side
                                    and prev_f not in used_dir[direction]
                                    and abs(frame - prev_f) <= slot_window
                                ):
                                    batch = prev_b
                                    actual_f = prev_f
                                    used_dir[direction].add(prev_f)
                        if batch is None and is_playing and cached_frames:
                            nf_hold = _find_nearest_cached_frame(
                                cached_frames, frame, current, direction, search_dist, ()
                            )
                            if nf_hold is None:
                                nf_hold = min(cached_frames, key=lambda cf: abs(cf - frame))
                            raw_hold = cache.get_frame(obj_key, nf_hold)
                            if raw_hold is not None:
                                batch = _wrap_deforming_delta_batch(obj_key, nf_hold, settings, raw_hold)
                                actual_f = nf_hold
                        if batch is not None:
                            if actual_f is not None:
                                _last_ghost_slots[(obj_key, d, s_idx)] = (actual_f, batch)
                            new_batches.append((obj_key, frame, direction, t, dist, batch))
            else:
                dirty_items = []
                dirty_frames_by_key = {}
                for d_entry in deforming_items:
                    _, obj, obj_key, _, needed = d_entry
                    missing_set = {f for f, _, _, _ in needed if not cache.has_frame(obj_key, f)}
                    if missing_set:
                        dirty_items.append(d_entry)
                        dirty_frames_by_key[obj_key] = missing_set

                dirty_dep_cache = {}
                dirty_fc_map = {}
                dirty_delta_chains = {}
                dirty_delta_scale_sens = {}
                direct_dirty_items = []
                direct_setters_by_key = {}
                frameset_dirty_items = []
                arm_map = _build_armature_children_map(scene) if dirty_items else None

                for d_entry in dirty_items:
                    _, obj, obj_key, _, _ = d_entry
                    dep_objs, _, dep_trees = _gather_object_dependencies(obj, scene, armature_children_map=arm_map)
                    dirty_dep_cache[obj_key] = (dep_objs, dep_trees)
                    _cached_kf_ranges[obj_key] = get_keyframe_range_for_object(obj, scene=scene)
                    if _can_use_per_frame_action_inval(dep_objs, dep_trees):
                        dirty_fc_map[obj_key] = _collect_dep_fcurves_sorted(dep_objs, dep_trees)
                    can_delta, scale_sens, ok_chain, chain = _get_cached_delta_capability(
                        obj_key, obj, dep_objs, dep_trees, fc_entries=dirty_fc_map.get(obj_key)
                    )
                    if can_delta and ok_chain:
                        dirty_delta_chains[obj_key] = chain
                        dirty_delta_scale_sens[obj_key] = scale_sens
                    if _can_use_direct_fcurve_eval(dep_objs, dep_trees):
                        setters, tag_ids = _build_direct_setters(dep_objs, dep_trees)
                        if setters is not None:
                            direct_dirty_items.append(d_entry)
                            direct_setters_by_key[obj_key] = (setters, tag_ids)
                            continue
                    frameset_dirty_items.append(d_entry)

                # Collect objects needing depsgraph.object_instances extraction (GN & Collection Instances)
                inst_parent_objs = set()
                for _, obj, _, settings, _ in dirty_items:
                    if obj.type == 'EMPTY' and (_empty_has_gn_geometry(obj) or _empty_is_collection_instance(obj)):
                        inst_parent_objs.add(obj)
                    elif obj.type in _GEO_TYPES and any(getattr(m, 'type', None) == 'NODES' and m.show_viewport for m in obj.modifiers):
                        inst_parent_objs.add(obj)
                    elif obj.type == 'ARMATURE':
                        for mobj in resolve_mesh_objects(
                            obj, settings, scene=scene, view_layer=view_layer, vis_cols=vis_cols
                        ):
                            if any(getattr(m, 'type', None) == 'NODES' and m.show_viewport for m in mobj.modifiers):
                                inst_parent_objs.add(mobj)

                eval_start = time.perf_counter()
                evaluated_frames = 0
                max_frame_cost = 0.0

                # Pass 1: Direct F-curve evaluation via depsgraph.update() (zero scene.frame_set / handler spikes)
                if direct_dirty_items:
                    direct_frame_dist = {}
                    direct_frame_seq = {}
                    for _, _, obj_key, _, _ in direct_dirty_items:
                        for f in dirty_frames_by_key[obj_key]:
                            d = abs(f - current)
                            if f not in direct_frame_dist or d < direct_frame_dist[f]:
                                direct_frame_dist[f] = d
                            seq = _frame_last_eval_seq.get((obj_key, f), 0)
                            if f not in direct_frame_seq or seq < direct_frame_seq[f]:
                                direct_frame_seq[f] = seq
                    direct_frames = sorted(
                        direct_frame_dist.keys(),
                        key=lambda f: (direct_frame_seq[f], direct_frame_dist[f], f),
                    )

                    direct_objs_snap = set()
                    for _, _, obj_key, _, _ in direct_dirty_items:
                        direct_objs_snap.update(dirty_dep_cache[obj_key][0])
                    direct_snapshots = {}
                    for dep_obj in direct_objs_snap:
                        try:
                            direct_snapshots[dep_obj] = _snapshot_object_state(dep_obj, include_pose_bones=False)
                        except Exception:
                            pass

                    touched_setter_keys = []
                    seen_touched_keys = set()
                    try:
                        for frame in direct_frames:
                            raw_items_at_f = [
                                e for e in direct_dirty_items
                                if frame in dirty_frames_by_key[e[2]] and not cache.has_frame(e[2], frame)
                            ]
                            items_at_f = []
                            for e in raw_items_at_f:
                                ok_e = e[2]
                                if ok_e in dirty_fc_map and ok_e in dirty_delta_chains and (is_playing or e[1].type != 'EMPTY'):
                                    if _try_reuse_same_pose_cached_frame(
                                        ok_e, frame, dirty_fc_map[ok_e], dirty_delta_chains[ok_e],
                                        dirty_delta_scale_sens.get(ok_e, False),
                                    ):
                                        continue
                                items_at_f.append(e)
                            if not items_at_f:
                                continue
                            if eval_time_budget is not None and evaluated_frames >= 1:
                                elapsed = time.perf_counter() - eval_start
                                if elapsed >= eval_time_budget or (elapsed + max_frame_cost * 0.75) >= (eval_time_budget * 1.25):
                                    incomplete = True
                                    break
                            t_step0 = time.perf_counter()
                            for _, _, obj_key, _, _ in items_at_f:
                                if obj_key not in seen_touched_keys:
                                    seen_touched_keys.add(obj_key)
                                    touched_setter_keys.append(obj_key)
                                setters, tag_ids = direct_setters_by_key[obj_key]
                                _apply_direct_setters(setters, tag_ids, frame)
                            dg = context.evaluated_depsgraph_get()
                            dg.update()
                            dg_cost = time.perf_counter() - t_step0
                            evaluated_frames += 1
                            _eval_and_cache_deforming_at_frame(
                                frame, items_at_f, dg, inst_parent_objs, dirty_fc_map, dirty_delta_chains, scene, dg_cost,
                                view_layer=view_layer, vis_cols=vis_cols,
                            )
                            step_cost = time.perf_counter() - t_step0
                            if step_cost > max_frame_cost:
                                max_frame_cost = step_cost
                    finally:
                        if evaluated_frames > 0:
                            for obj_key in reversed(touched_setter_keys):
                                setters, tag_ids = direct_setters_by_key[obj_key]
                                _restore_direct_setters(setters, tag_ids)
                            for snap_obj, state in direct_snapshots.items():
                                if _is_valid_object(snap_obj):
                                    _restore_object_state(snap_obj, state)
                            try:
                                context.evaluated_depsgraph_get().update()
                            except Exception:
                                pass

                # Pass 2: Full scene.frame_set evaluation only when physics / NLA / drivers / scene-time require it
                if frameset_dirty_items:
                    elapsed_p2 = time.perf_counter() - eval_start
                    if incomplete or (
                        eval_time_budget is not None
                        and evaluated_frames >= 1
                        and (elapsed_p2 >= eval_time_budget or (elapsed_p2 + max_frame_cost * 0.75) >= (eval_time_budget * 1.25))
                    ):
                        incomplete = True
                    else:
                        fs_frame_dist = {}
                        fs_frame_seq = {}
                        for _, _, obj_key, _, _ in frameset_dirty_items:
                            for f in dirty_frames_by_key[obj_key]:
                                d = abs(f - current)
                                if f not in fs_frame_dist or d < fs_frame_dist[f]:
                                    fs_frame_dist[f] = d
                                seq = _frame_last_eval_seq.get((obj_key, f), 0)
                                if f not in fs_frame_seq or seq < fs_frame_seq[f]:
                                    fs_frame_seq[f] = seq
                        frames_to_eval = sorted(
                            fs_frame_dist.keys(),
                            key=lambda f: (fs_frame_seq[f], fs_frame_dist[f], f),
                        )

                        objs_to_snapshot = set()
                        for _, _, obj_key, _, _ in frameset_dirty_items:
                            objs_to_snapshot.update(dirty_dep_cache[obj_key][0])
                        snapshots = {}
                        for dep_obj in objs_to_snapshot:
                            try:
                                snapshots[dep_obj] = _snapshot_object_state(dep_obj)
                            except Exception:
                                pass

                        original_frame = scene.frame_current
                        fs_evaluated = 0
                        try:
                            for frame in frames_to_eval:
                                if eval_time_budget is not None and evaluated_frames >= 1:
                                    elapsed = time.perf_counter() - eval_start
                                    if elapsed >= eval_time_budget or (elapsed + max_frame_cost * 0.75) >= (eval_time_budget * 1.25):
                                        incomplete = True
                                        break
                                t_fs0 = time.perf_counter()
                                scene.frame_set(frame)
                                dg = context.evaluated_depsgraph_get()
                                fs_cost = time.perf_counter() - t_fs0
                                evaluated_frames += 1
                                fs_evaluated += 1
                                items_at_f = [
                                    e for e in frameset_dirty_items
                                    if frame in dirty_frames_by_key[e[2]] and not cache.has_frame(e[2], frame)
                                ]
                                _eval_and_cache_deforming_at_frame(
                                    frame, items_at_f, dg, inst_parent_objs, dirty_fc_map, dirty_delta_chains, scene, fs_cost,
                                    view_layer=view_layer, vis_cols=vis_cols,
                                )
                                step_cost = time.perf_counter() - t_fs0
                                if step_cost > max_frame_cost:
                                    max_frame_cost = step_cost
                        finally:
                            if fs_evaluated > 0:
                                scene.frame_set(original_frame)
                                for snap_obj, state in snapshots.items():
                                    if _is_valid_object(snap_obj):
                                        _restore_object_state(snap_obj, state)
                                try:
                                    context.evaluated_depsgraph_get().update()
                                except Exception:
                                    pass

                for item, obj, obj_key, settings, needed in deforming_items:
                    dir_slots = {'before': [], 'after': []}
                    for idx_n, (frame, direction, _t, _dist) in enumerate(needed):
                        dir_slots[direction].append((idx_n, frame))
                    slot_dir_idx = [None] * len(needed)
                    for d in ('before', 'after'):
                        dir_slots[d].sort(key=lambda pair: abs(pair[1] - current))
                        for s_idx, (idx_n, _fr) in enumerate(dir_slots[d]):
                            slot_dir_idx[idx_n] = (d, s_idx)

                    cached_frames = cache.get_cached_frames(obj_key) if incomplete else ()
                    used_dir = {'before': set(), 'after': set()}
                    slot_batches = [None] * len(needed)
                    for idx_n, (frame, direction, t, dist) in enumerate(needed):
                        batch = cache.get_frame(obj_key, frame)
                        if batch is not None:
                            batch = _wrap_deforming_delta_batch(obj_key, frame, settings, batch)
                            slot_batches[idx_n] = batch
                            used_dir[direction].add(frame)
                            d, s_idx = slot_dir_idx[idx_n]
                            _last_ghost_slots[(obj_key, d, s_idx)] = (frame, batch)

                    search_dist = max(settings.frame_step * 4, 16)
                    slot_window = max(search_dist * 2, 48)
                    for idx_n, (frame, direction, t, dist) in enumerate(needed):
                        batch = slot_batches[idx_n]
                        if batch is not None:
                            new_batches.append((obj_key, frame, direction, t, dist, batch))
                            continue
                        if incomplete:
                            d, s_idx = slot_dir_idx[idx_n]
                            actual_f = None
                            batch = prev_deforming.get((obj_key, frame))
                            if batch is not None and frame not in used_dir[direction]:
                                actual_f = frame
                                used_dir[direction].add(frame)
                            else:
                                batch = None
                            if batch is None and cached_frames:
                                nf = _find_nearest_cached_frame(
                                    cached_frames, frame, current, direction, search_dist, used_dir[direction]
                                )
                                if nf is not None:
                                    batch = cache.get_frame(obj_key, nf)
                                    if batch is not None:
                                        actual_f = nf
                                        used_dir[direction].add(nf)
                            if batch is None:
                                slot_prev = _last_ghost_slots.get((obj_key, d, s_idx))
                                if slot_prev is not None:
                                    prev_f, prev_b = slot_prev
                                    valid_side = (prev_f < current) if d == 'before' else (prev_f > current)
                                    if (
                                        prev_b is not None
                                        and valid_side
                                        and prev_f not in used_dir[direction]
                                        and abs(frame - prev_f) <= slot_window
                                    ):
                                        batch = prev_b
                                        actual_f = prev_f
                                        used_dir[direction].add(prev_f)
                            if batch is not None:
                                if actual_f is not None:
                                    _last_ghost_slots[(obj_key, d, s_idx)] = (actual_f, batch)
                                new_batches.append((obj_key, frame, direction, t, dist, batch))

        cache.set_onion_batches(new_batches)
    finally:
        _is_evaluating = False
        _is_building = False

    return incomplete


# ---------------------------------------------------------------------------
# Viewport Drawing Callbacks
# ---------------------------------------------------------------------------

def _box_corners(b_min, b_max):
    """Return the 8 Vector corners of an axis-aligned bounding box."""
    return [
        Vector((b_min[0], b_min[1], b_min[2])),
        Vector((b_min[0], b_min[1], b_max[2])),
        Vector((b_min[0], b_max[1], b_min[2])),
        Vector((b_min[0], b_max[1], b_max[2])),
        Vector((b_max[0], b_min[1], b_min[2])),
        Vector((b_max[0], b_min[1], b_max[2])),
        Vector((b_max[0], b_max[1], b_min[2])),
        Vector((b_max[0], b_max[1], b_max[2])),
    ]


def _payload_world_corners(b_payload):
    """Extract world-space bounding box corners from a filtered batch payload."""
    if not b_payload:
        return []
    if isinstance(b_payload, tuple) and len(b_payload) >= 6 and b_payload[0] == 'RIGID':
        mw = b_payload[2]
        return [mw @ c for c in _box_corners(b_payload[3], b_payload[4])]
    if isinstance(b_payload, tuple) and len(b_payload) >= 3 and b_payload[0] == 'RIGID_LIST':
        mw, sub_items = b_payload[1], b_payload[2]
        return [
            mw @ c
            for s in sub_items
            if len(s) >= 4 and s[2] is not None and s[3] is not None
            for c in _box_corners(s[2], s[3])
        ]
    entries = b_payload if isinstance(b_payload, list) else [b_payload]
    return [
        c
        for ent in entries
        if isinstance(ent, (list, tuple)) and len(ent) >= 8 and ent[6] is not None and ent[7] is not None
        for c in _box_corners(ent[6], ent[7])
    ]


def _object_world_corners(obj, part_objs=None):
    """Collect world-space bounding box corners for an object or its visible child meshes."""
    if part_objs:
        targets = part_objs
    elif obj and obj.type in ('MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'VOLUME', 'POINTCLOUD'):
        targets = [obj]
    else:
        return []
    corners = []
    for t in targets:
        if not _is_valid_object(t):
            continue
        try:
            mw = t.matrix_world
            corners.extend(mw @ Vector(c) for c in t.bound_box)
        except Exception:
            pass
    return corners


def _bounds_center_and_span(corners, axis_vec):
    """Compute bounding center and effective horizontal extent along axis_vec in world space."""
    if not corners:
        return None, 0.0
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    center = (
        float(0.5 * (min(xs) + max(xs))),
        float(0.5 * (min(ys) + max(ys))),
        float(0.5 * (min(zs) + max(zs))),
    )
    dots = [c.dot(axis_vec) for c in corners]
    proj_span = max(dots) - min(dots)
    max_dim = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return center, max(float(proj_span), 0.35 * float(max_dim))


def draw_onion_skins():
    """Main draw callback for all onion skin overlays."""
    global _was_playing
    context = bpy.context

    if not context.area or context.area.type != 'VIEW_3D':
        return

    space = getattr(context, 'space_data', None)
    rv3d = getattr(space, 'region_3d', None) if space else None
    if rv3d is None:
        return

    scene = context.scene
    if not hasattr(scene, 'onion_skin_settings'):
        return

    global_settings = scene.onion_skin_settings
    if not global_settings.enabled or not scene.onion_skin_objects:
        return

    if _was_playing and not screen_animation_playing(context):
        _was_playing = False
        _trigger_build(needs_full_eval=True, progressive=True)

    onion_batches = cache.get_onion_batches()
    ghost_markers = cache.get_ghost_markers()

    if not onion_batches and not ghost_markers:
        return

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_mask_set(False)

    is_ortho = 0.0 if rv3d.is_perspective else 1.0

    # Camera View-Projection matrices
    mv_matrix = gpu.matrix.get_model_view_matrix()
    proj_matrix = gpu.matrix.get_projection_matrix()
    mvp_matrix = proj_matrix @ mv_matrix

    cam_pos = rv3d.view_matrix.inverted().translation if rv3d.is_perspective else rv3d.view_matrix.to_3x3()[2]
    cam_pos_or_dir = (cam_pos.x, cam_pos.y, cam_pos.z, 0.0)

    vis_part_keys = {}
    vis_part_objs = {}
    vis_obj_settings, vis_obj_by_key, vis_keys, vis_names, all_keys = _collect_viewport_visible_info(
        context, out_part_keys=vis_part_keys, out_part_objs=vis_part_objs
    )

    def _filter_batch_parts(obj_key, b_payload):
        allowed = vis_part_keys.get(obj_key)
        if allowed is None:
            return b_payload
        if isinstance(b_payload, tuple) and len(b_payload) >= 3 and b_payload[0] == 'RIGID_LIST':
            kept_sub = [s for s in b_payload[2] if s[0] in allowed]
            return ('RIGID_LIST', b_payload[1], kept_sub) if kept_sub else None
        if isinstance(b_payload, list):
            kept = [e for e in b_payload if e[0] in allowed]
            return kept if kept else None
        return b_payload

    def _bind_onion_shader(mvp_mat, mat_world, color_vec, has_normals, use_wf, use_fn, use_sh, eff_power, eff_fade, eff_shadow, eff_spec, cam_w):
        fresnel_sh = cache.get_fresnel_shader() if (not use_wf and has_normals) else None
        if fresnel_sh is not None:
            sh = fresnel_sh
            sh.bind()
            sh.uniform_float("ModelViewProjectionMatrix", mvp_mat)
            sh.uniform_float("color", color_vec)
            if mat_world is not None:
                try:
                    mat_inv = mat_world.inverted()
                    if rv3d.is_perspective:
                        cam_local = mat_inv @ Vector(cam_pos_or_dir[:3])
                    else:
                        cam_local = mat_inv.to_3x3() @ Vector(cam_pos_or_dir[:3])
                    cam_param = (cam_local.x, cam_local.y, cam_local.z, cam_w)
                except Exception:
                    cam_param = (cam_pos_or_dir[0], cam_pos_or_dir[1], cam_pos_or_dir[2], cam_w)
            else:
                cam_param = (cam_pos_or_dir[0], cam_pos_or_dir[1], cam_pos_or_dir[2], cam_w)
            sh.uniform_float("camera_pos_or_dir", cam_param)
            sh.uniform_float("fresnel_params", (eff_power, eff_fade, eff_shadow, eff_spec))
            return sh
        sh = cache.get_shader()
        if not sh:
            return None
        sh.bind()
        sh.uniform_float("ModelViewProjectionMatrix", mvp_mat)
        sh.uniform_float("color", color_vec)
        return sh

    # Draw onion frames (before/after)
    if onion_batches and vis_obj_settings:
        obj_settings_by_key = vis_obj_settings
        obj_by_key = vis_obj_by_key

        region = context.region
        vp_w = max(1.0, float(region.width)) if region else 1920.0
        vp_h = max(1.0, float(region.height)) if region else 1080.0

        def _batch_world_centroid(obj_key, b_payload):
            b_payload = _filter_batch_parts(obj_key, b_payload)
            if not b_payload:
                return (0.0, 0.0, 0.0)
            if isinstance(b_payload, tuple) and len(b_payload) >= 6 and b_payload[0] == 'RIGID':
                mw = b_payload[2]
                ctr = b_payload[5]
                try:
                    wc = mw @ Vector(ctr)
                    return (float(wc.x), float(wc.y), float(wc.z))
                except Exception:
                    return (0.0, 0.0, 0.0)
            if isinstance(b_payload, tuple) and len(b_payload) >= 3 and b_payload[0] == 'RIGID_LIST':
                mw, sub_items = b_payload[1], b_payload[2]
                pts = []
                for s in sub_items:
                    if len(s) >= 5 and s[4] is not None:
                        try:
                            pts.append(mw @ Vector(s[4]))
                        except Exception:
                            pass
                if pts:
                    inv = 1.0 / len(pts)
                    return (
                        float(sum(p.x for p in pts) * inv),
                        float(sum(p.y for p in pts) * inv),
                        float(sum(p.z for p in pts) * inv),
                    )
                return (0.0, 0.0, 0.0)
            entries = b_payload if isinstance(b_payload, list) else [b_payload]
            ctrs = [
                ent[8] for ent in entries
                if isinstance(ent, (list, tuple)) and len(ent) >= 9 and ent[8] is not None
            ]
            if ctrs:
                inv = 1.0 / len(ctrs)
                return (
                    float(sum(c[0] for c in ctrs) * inv),
                    float(sum(c[1] for c in ctrs) * inv),
                    float(sum(c[2] for c in ctrs) * inv),
                )
            return (0.0, 0.0, 0.0)

        try:
            view_inv = rv3d.view_matrix.inverted()
            cam_right = view_inv.col[0].to_3d()
            if cam_right.length_squared > 1e-10:
                cam_right.normalize()
            else:
                cam_right = Vector((1.0, 0.0, 0.0))
        except Exception:
            cam_right = Vector((1.0, 0.0, 0.0))

        # Apply optional Screen-Space Spacing (Adaptive Distance Filter + Camera-Aligned Spread)
        draw_items = []
        any_screen_spaced = any(
            getattr(s, 'use_screen_space_spacing', False)
            for s in obj_settings_by_key.values()
        )
        if any_screen_spaced:
            grouped = {}
            for b_item in onion_batches:
                ok, fr, direction, t_val, dist_val, b_payload = b_item
                s_cfg = obj_settings_by_key.get(ok)
                if not s_cfg:
                    continue
                if not getattr(s_cfg, 'use_screen_space_spacing', False):
                    draw_items.append((ok, fr, direction, t_val, dist_val, b_payload, None))
                else:
                    grouped.setdefault((ok, direction), []).append(b_item)

            # Use exact scene-unit distance (meters) per tracked object so walk-cycle pose width
            # changes never cause ghosts to pulsate/accordion while still scaling 1:1 with camera zoom.
            spread_step_by_key = {}
            for (ok, _direction) in grouped:
                if ok in spread_step_by_key:
                    continue
                s_cfg = obj_settings_by_key[ok]
                spread_step_by_key[ok] = max(
                    0.0, float(getattr(s_cfg, 'screen_space_spread_distance', 1.5))
                )

            for (ok, direction), grp in grouped.items():
                s_cfg = obj_settings_by_key[ok]
                grp.sort(key=lambda x: x[4])  # Closest to current frame first
                mode = getattr(s_cfg, 'screen_space_mode', 'SPREAD')
                ss_dist = max(0.0, float(getattr(s_cfg, 'screen_space_distance', 85.0)))

                if mode == 'FILTER' and ss_dist > 0.0:
                    ref_ctr = None
                    cur_cached = cache.get_frame(ok, scene.frame_current)
                    if cur_cached:
                        ref_ctr = _batch_world_centroid(ok, cur_cached)
                    elif ok in obj_by_key:
                        c_obj = obj_by_key[ok]
                        if is_object_rigid(c_obj):
                            use_wf_c = bool(s_cfg.use_wireframe)
                            r_info = cache.get_rigid_batch(ok, use_wireframe=use_wf_c, use_fresnel=(not use_wf_c))
                            if r_info is not None and r_info[3] is not None:
                                try:
                                    cur_mat = cache.get_rigid_matrix(ok, scene.frame_current)
                                    if cur_mat is None:
                                        ok_ch, ch = _build_analytic_world_chain(c_obj)
                                        if ok_ch:
                                            cur_mat = _eval_world_matrix_analytic(ch, scene.frame_current)
                                        else:
                                            cur_mat = c_obj.matrix_world
                                    wc = cur_mat @ Vector(r_info[3])
                                    ref_ctr = (float(wc.x), float(wc.y), float(wc.z))
                                except Exception:
                                    ref_ctr = None
                        if ref_ctr is None:
                            ref_ctr, _ = _bounds_center_and_span(
                                _object_world_corners(c_obj, vis_part_objs.get(ok)), cam_right
                            )
                        if ref_ctr is None:
                            try:
                                tr = c_obj.matrix_world.translation
                                ref_ctr = (float(tr.x), float(tr.y), float(tr.z))
                            except Exception:
                                ref_ctr = None
                    cand_ctrs = [_batch_world_centroid(ok, x[5]) for x in grp]
                    keep_mask, _ = cb.filter_screen_space_frames(
                        mvp_matrix, vp_w, vp_h, ref_ctr, cand_ctrs, ss_dist
                    )
                    kept_grp = [x for x, keep in zip(grp, keep_mask) if keep]
                else:
                    kept_grp = grp

                dir_sign = -1.0 if direction == 'before' else 1.0
                world_step = spread_step_by_key.get(ok, 0.0)
                for rank_idx, (_, fr, dir_val, orig_t, dist_val, b_payload) in enumerate(kept_grp):
                    rank_1 = float(rank_idx + 1)
                    eff_t = orig_t
                    if mode == 'SPREAD' and world_step > 1e-5:
                        world_offset = cam_right * (dir_sign * rank_1 * world_step)
                    else:
                        world_offset = None
                    draw_items.append((ok, fr, dir_val, eff_t, dist_val, b_payload, world_offset))
        else:
            for ok, fr, direction, t_val, dist_val, b_payload in onion_batches:
                draw_items.append((ok, fr, direction, t_val, dist_val, b_payload, None))

        def _preupload_cached_vbos_budgeted(vis_settings, budget_sec=0.0025):
            """Upload cached CPU geometry into GPU VBOs during idle redraws to avoid playback spikes."""
            if not vis_settings:
                return
            t_up0 = time.perf_counter()
            for (c_ok, _fr), val in list(cache._frame_cache.items()):
                s_obj = vis_settings.get(c_ok)
                if s_obj is None or not isinstance(val, list) or not val:
                    continue
                use_wf_u = bool(s_obj.use_wireframe)
                use_fn_u = bool(s_obj.use_fresnel)
                use_sh_u = bool(getattr(s_obj, 'use_shading', True))
                want_key = 'fresnel' if ((not use_wf_u) and (use_fn_u or use_sh_u)) else 'standard'
                for ent in val:
                    if not isinstance(ent, list) or len(ent) < 6:
                        continue
                    bmap = ent[5]
                    if want_key in bmap or (ent[2] is None and 'standard' in bmap):
                        continue
                    cache.get_entry_gpu_batch(
                        ent, use_wireframe=use_wf_u, use_fresnel=(not use_wf_u), use_shading=use_sh_u,
                    )
                    if (time.perf_counter() - t_up0) >= budget_sec:
                        return

        last_depth_mode = None
        last_line_width = None

        def _set_draw_state(dm, lw):
            nonlocal last_depth_mode, last_line_width
            if dm != last_depth_mode:
                gpu.state.depth_test_set(dm)
                last_depth_mode = dm
            if lw != last_line_width:
                gpu.state.line_width_set(lw)
                last_line_width = lw

        fresnel_available = (cache.get_fresnel_shader() is not None)
        inst_groups = {}
        deforming_draws = []

        for obj_key, frame, direction, t, dist, batch, world_offset in sorted(draw_items, key=lambda x: -x[4]):
            settings = obj_settings_by_key.get(obj_key)
            if not settings:
                continue
            batch = _filter_batch_parts(obj_key, batch)
            if not batch:
                continue

            base = settings.color_before if direction == 'before' else settings.color_after
            alpha = calculate_alpha(base[3], t, settings.opacity_falloff, settings.falloff_curve)
            color_vec = (base[0], base[1], base[2], alpha)

            use_wf = bool(settings.use_wireframe)
            use_fn = bool(settings.use_fresnel)
            use_sh = bool(getattr(settings, 'use_shading', True))
            eff_fade = float(settings.fresnel_fade) if use_fn else 0.0
            eff_power = float(settings.fresnel_power) if use_fn else 2.0
            eff_shadow = float(getattr(settings, 'shadow_strength', 0.65)) if use_sh else 0.0
            eff_spec = float(getattr(settings, 'specular_strength', 0.35)) if use_sh else 0.0
            shininess = max(2.0, float(getattr(settings, 'specular_shininess', 24.0)))
            cam_w = shininess if is_ortho > 0.5 else -shininess
            depth_mode = 'NONE' if settings.use_xray else ('LESS_EQUAL' if use_wf else 'LESS')
            line_width = 1.5 if use_wf else 1.0
            fparams = (eff_power, eff_fade, eff_shadow, eff_spec)

            # Collect RIGID and RIGID_LIST payloads for Vulkan/OpenGL GPU hardware instancing
            if isinstance(batch, tuple) and len(batch) >= 6 and batch[0] == 'RIGID':
                _, gpu_batch, mat_world, b_min, b_max, _center = batch
                if not gpu_batch:
                    continue
                use_lit_real = (not use_wf) and fresnel_available
                gkey = (id(gpu_batch), use_lit_real, depth_mode, line_width)
                grp_entry = inst_groups.get(gkey)
                if grp_entry is None:
                    grp_entry = [gpu_batch, use_lit_real, use_wf, use_fn, use_sh, depth_mode, line_width, []]
                    inst_groups[gkey] = grp_entry
                grp_entry[7].append((mat_world, world_offset, b_min, b_max, color_vec, fparams, cam_w))
                continue

            if isinstance(batch, tuple) and len(batch) >= 3 and batch[0] == 'RIGID_LIST':
                _, d_mat, sub_items = batch
                if not sub_items:
                    continue
                use_lit_real = (not use_wf) and fresnel_available
                for _, sub_gpu, sub_bmin, sub_bmax, _sub_ctr in sub_items:
                    if not sub_gpu:
                        continue
                    gkey = (id(sub_gpu), use_lit_real, depth_mode, line_width)
                    grp_entry = inst_groups.get(gkey)
                    if grp_entry is None:
                        grp_entry = [sub_gpu, use_lit_real, use_wf, use_fn, use_sh, depth_mode, line_width, []]
                        inst_groups[gkey] = grp_entry
                    grp_entry[7].append((d_mat, world_offset, sub_bmin, sub_bmax, color_vec, fparams, cam_w))
                continue

            deforming_draws.append((
                batch, world_offset, color_vec, use_wf, use_fn, use_sh,
                eff_power, eff_fade, eff_shadow, eff_spec, cam_w, depth_mode, line_width,
            ))

        # 1. Execute hardware instanced draw calls (1 UBO + draw_instanced per shared mesh/chunk)
        for gpu_batch, use_lit_real, g_wf, g_fn, g_sh, dm, lw, grp_items in inst_groups.values():
            inst_sh = (
                cache.get_instanced_fresnel_shader()
                if use_lit_real else cache.get_instanced_flat_shader()
            )
            use_hw_inst = (HAS_NUMPY and inst_sh is not None and cache.get_instance_ubo() is not None)
            _set_draw_state(dm, lw)
            if use_hw_inst:
                total_it = len(grp_items)
                is_ortho_flag = (is_ortho > 0.5)
                for ofs in range(0, total_it, _INST_MAX):
                    chunk = grp_items[ofs:ofs + _INST_MAX]
                    n = len(chunk)
                    any_offset = False
                    all_bounds = True
                    for idx_i, (mw, w_off, b_min, b_max, col_v, fp_v, cw_v) in enumerate(chunk):
                        _INST_MATS[idx_i] = (
                            mw[0][0], mw[0][1], mw[0][2], mw[0][3],
                            mw[1][0], mw[1][1], mw[1][2], mw[1][3],
                            mw[2][0], mw[2][1], mw[2][2], mw[2][3],
                            mw[3][0], mw[3][1], mw[3][2], mw[3][3],
                        )
                        if w_off is not None and w_off.length_squared > 1e-10:
                            any_offset = True
                            _INST_OFFS[idx_i] = (w_off.x, w_off.y, w_off.z)
                        else:
                            _INST_OFFS[idx_i] = (0.0, 0.0, 0.0)
                        if b_min is not None and b_max is not None and len(b_min) >= 3 and len(b_max) >= 3:
                            _INST_MINS[idx_i] = (b_min[0], b_min[1], b_min[2])
                            _INST_MAXS[idx_i] = (b_max[0], b_max[1], b_max[2])
                        else:
                            all_bounds = False
                        _INST_COLS[idx_i] = col_v
                        _INST_PARAMS[idx_i] = fp_v
                        _INST_CAMWS[idx_i] = cw_v
                    vis_n = cb.pack_instances(
                        mvp_matrix, cam_pos_or_dir, is_ortho_flag,
                        _INST_MATS,
                        _INST_OFFS if any_offset else None,
                        _INST_MINS if all_bounds else None,
                        _INST_MAXS if all_bounds else None,
                        _INST_COLS, _INST_PARAMS, _INST_CAMWS, _INST_UBO_OUT,
                        count=n,
                    )
                    if vis_n == 0:
                        continue
                    if vis_n > 0:
                        ubo = cache.update_instance_ubo(_INST_UBO_OUT.tobytes())
                        inst_sh.bind()
                        inst_sh.uniform_block("u_inst", ubo)
                        gpu_batch.draw_instanced(inst_sh, instance_count=vis_n)
                        continue
                    for mw, w_off, _bmin, _bmax, col_v, fp_v, cw_v in chunk:
                        mvp_e = (mvp_matrix @ Matrix.Translation(w_off)) if (w_off is not None and w_off.length_squared > 1e-10) else mvp_matrix
                        sh_f = _bind_onion_shader(
                            mvp_e @ mw, mw, col_v, True, g_wf, g_fn, g_sh,
                            fp_v[0], fp_v[1], fp_v[2], fp_v[3], cw_v,
                        )
                        if sh_f:
                            gpu_batch.draw(sh_f)
            else:
                for mw, w_off, _bmin, _bmax, col_v, fp_v, cw_v in grp_items:
                    mvp_e = (mvp_matrix @ Matrix.Translation(w_off)) if (w_off is not None and w_off.length_squared > 1e-10) else mvp_matrix
                    sh_f = _bind_onion_shader(
                        mvp_e @ mw, mw, col_v, True, g_wf, g_fn, g_sh,
                        fp_v[0], fp_v[1], fp_v[2], fp_v[3], cw_v,
                    )
                    if sh_f:
                        gpu_batch.draw(sh_f)

        # 2. Execute standard DEFORMING batch path with frustum culling
        for (
            batch, world_offset, color_vec, use_wf, use_fn, use_sh,
            eff_power, eff_fade, eff_shadow, eff_spec, cam_w, depth_mode, line_width,
        ) in deforming_draws:
            visible_entries = batch if isinstance(batch, list) else [batch]
            if not visible_entries:
                continue
            mvp_effective = (
                mvp_matrix @ Matrix.Translation(world_offset)
                if (world_offset is not None and world_offset.length_squared > 1e-10)
                else mvp_matrix
            )
            culled_entries = []
            for entry in visible_entries:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) >= 8
                    and entry[6] is not None
                    and entry[7] is not None
                    and not cb.is_box_visible(mvp_effective, entry[6], entry[7])
                ):
                    continue
                culled_entries.append(entry)
            if not culled_entries:
                continue

            has_normals = any(
                isinstance(e, (list, tuple)) and len(e) >= 3 and e[2] is not None
                for e in culled_entries
            )
            _set_draw_state(depth_mode, line_width)
            current_shader = _bind_onion_shader(
                mvp_effective, None, color_vec, has_normals,
                use_wf, use_fn, use_sh, eff_power, eff_fade, eff_shadow, eff_spec, cam_w,
            )
            if not current_shader:
                continue

            for entry in culled_entries:
                gpu_batch = cache.get_entry_gpu_batch(
                    entry,
                    use_wireframe=use_wf,
                    use_fresnel=(not use_wf),
                    use_shading=use_sh,
                )
                if gpu_batch:
                    gpu_batch.draw(current_shader)

    # Ghost markers (persistent pinned frames)
    if ghost_markers and global_settings.show_ghosts:
        shader = cache.get_shader()
        if shader:
            shader.bind()
            shader.uniform_float("ModelViewProjectionMatrix", mvp_matrix)
            gc = tuple(global_settings.ghost_color)
            shader.uniform_float("color", gc)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.line_width_set(1.0)
            for i, marker in enumerate(ghost_markers):
                if not _is_marker_visible_in_viewport(
                    marker, vis_keys, vis_names, all_keys, vis_part_keys=vis_part_keys
                ):
                    continue
                ghost_batch = cache.get_ghost_batch(i)
                if ghost_batch:
                    ghost_batch.draw(shader)

    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    gpu.state.line_width_set(1.0)

    if onion_batches and vis_obj_settings and not screen_animation_playing(context):
        _preupload_cached_vbos_budgeted(vis_obj_settings)


def draw_ghost_labels():
    """Draw frame number labels on ghost overlays using O(1) C++ centroids."""
    context = bpy.context

    if not context.area or context.area.type != 'VIEW_3D':
        return

    scene = context.scene
    if not hasattr(scene, 'onion_skin_settings'):
        return

    settings = scene.onion_skin_settings
    if not settings.enabled or not settings.show_ghost_labels or not settings.show_ghosts or not scene.onion_skin_objects:
        return

    ghost_markers = cache.get_ghost_markers()
    if not ghost_markers:
        return

    vis_part_keys = {}
    _, _, vis_keys, vis_names, all_keys = _collect_viewport_visible_info(
        context, out_part_keys=vis_part_keys
    )

    region = getattr(context, 'region', None)
    space = getattr(context, 'space_data', None)
    rv3d = getattr(space, 'region_3d', None) if space else None
    if region is None or rv3d is None:
        return

    font_id = 0
    blf.size(font_id, 13)
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)

    for i, marker in enumerate(ghost_markers):
        if not _is_marker_visible_in_viewport(
            marker, vis_keys, vis_names, all_keys, vis_part_keys=vis_part_keys
        ):
            continue
        frame_num = marker[0]
        cx, cy, cz = cache.get_ghost_centroid(i)
        pos_2d = location_3d_to_region_2d(region, rv3d, (cx, cy, cz))
        if pos_2d:
            blf.position(font_id, pos_2d.x + 5, pos_2d.y + 5, 0)
            blf.draw(font_id, str(frame_num))


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

_dirty_while_disabled = False
_tracked_ptr_to_identity = {}


def _prune_orphan_ghost_markers(scene):
    """Drop ghost markers whose owner object keys are no longer in any remaining tracked entry."""
    markers = cache.get_ghost_markers()
    if not markers or scene is None:
        return
    surviving = set()
    for item in getattr(scene, 'onion_skin_objects', ()):
        obj = item.object
        if not obj or not _is_valid_object(obj):
            continue
        k = get_object_key(obj)
        if k:
            surviving.add(k)
        surviving.add(obj.name)
        s_cfg = properties.get_object_settings(obj, item)
        for robj in resolve_mesh_objects(obj, s_cfg, scene=scene, include_hidden=True):
            rk = get_object_key(robj)
            if rk:
                surviving.add(rk)
            surviving.add(robj.name)
    dead_keys = set()
    for m in markers:
        m_objs = m[4] if len(m) > 4 else ()
        if m_objs and not any(k in surviving for k in m_objs):
            dead_keys.update(m_objs)
    if dead_keys:
        cache.remove_ghost_markers_for_object(dead_keys)


def _track_object_identity(target_obj):
    """Record target_obj's current key/name and migrate caches + ghost markers on rename."""
    if not _is_valid_object(target_obj):
        return False
    try:
        ptr = target_obj.as_pointer()
    except Exception:
        return False
    tkey = get_object_key(target_obj)
    tname = target_obj.name
    new_id = (tkey, tname)
    old_id = _tracked_ptr_to_identity.get(ptr)
    _tracked_ptr_to_identity[ptr] = new_id
    if old_id is not None and old_id != new_id:
        old_key, old_name = old_id
        cache.clear_frame_cache(old_key)
        cache.clear_rigid_cache(old_key)
        cache.clear_rigid_matrix_cache(old_key)
        _clear_object_action_sigs(old_key)
        _clear_ghost_slots(old_key)
        _cached_signatures.pop(old_key, None)
        _cached_outliner_vis.pop(old_key, None)
        clear_topology_cache(old_key)
        if old_name != old_key:
            clear_topology_cache(old_name)
        cache.rename_ghost_marker_keys(old_key, tkey, old_name, tname)
        return True
    return False


def _sync_scene_identities(scene):
    """Seed/sync tracked object identities across scene.onion_skin_objects."""
    if scene is None:
        return
    for item in getattr(scene, 'onion_skin_objects', ()):
        obj = item.object
        if obj and _is_valid_object(obj):
            _track_object_identity(obj)


def _reset_runtime_state(clear_ghosts=True, keep_identities=False):
    """Reset all runtime caches, signature maps, and transient builder flags."""
    global _is_building, _is_evaluating, _was_playing, _was_transforming
    global _pending_build, _needs_full_eval, _progressive_eval, _dirty_while_disabled
    global _last_nonplay_fc_time, _last_modal_edit_time, _dg_dep_cache, _last_modal_eval_end
    _is_building = False
    _is_evaluating = False
    _was_playing = False
    _was_transforming = False
    _pending_build = False
    _needs_full_eval = False
    _progressive_eval = False
    _dirty_while_disabled = False
    _last_nonplay_fc_time = 0.0
    _last_modal_edit_time = 0.0
    _last_modal_eval_end = 0.0
    _dg_dep_cache = None
    _delta_cap_cache.clear()
    _stale_offscreen_action_roots.clear()
    clear_topology_cache()
    _cached_signatures.clear()
    _cached_outliner_vis.clear()
    _cached_coll_sigs.clear()
    if not keep_identities:
        _tracked_ptr_to_identity.clear()
    _clear_object_action_sigs()
    _passive_extract_costs.clear()
    _last_heavy_passive_capture.clear()
    _clear_ghost_slots()
    cache.clear_frame_cache()
    cache.clear_rigid_cache()
    cache.clear_rigid_matrix_cache()
    cache.clear_onion_batches()
    bump_vis_cache_rev()
    if clear_ghosts:
        cache.clear_ghost_markers()


def _cleanup_handlers():
    our_names = {
        '_on_frame_change', '_on_depsgraph_update', '_on_undo_redo',
        '_on_onion_load_pre', '_on_onion_load_post',
    }
    for handler_list in (bpy.app.handlers.frame_change_post,
                         bpy.app.handlers.depsgraph_update_post,
                         bpy.app.handlers.undo_post,
                         bpy.app.handlers.redo_post,
                         bpy.app.handlers.load_pre,
                         bpy.app.handlers.load_post):
        for fn in [h for h in handler_list if getattr(h, '__name__', None) in our_names]:
            try:
                handler_list.remove(fn)
            except ValueError:
                pass


@persistent
def _on_frame_change(scene, depsgraph):
    global _was_playing, _last_nonplay_fc_time
    if _is_building or _is_evaluating:
        return

    if not hasattr(scene, 'onion_skin_settings'):
        return
    settings = scene.onion_skin_settings
    if not settings.enabled or not scene.onion_skin_objects:
        cache.clear_onion_batches()
        return

    context = bpy.context
    if _stale_offscreen_action_roots and not _is_transforming(context):
        _reconcile_offscreen_action_roots(scene)
    if screen_animation_playing(context):
        _was_playing = True
        _build_onion_batches(context, scene.frame_current, depsgraph)
        return

    now_t = time.perf_counter()
    is_rapid_scrub = (now_t - _last_nonplay_fc_time) < 0.18
    _last_nonplay_fc_time = now_t
    if is_rapid_scrub or _is_scrubbing_modal(context) or _is_transforming(context):
        incomplete = _build_onion_batches(
            context, scene.frame_current, depsgraph,
            allow_deforming_eval=True,
            eval_time_budget=0.028,
        )
        if incomplete:
            _trigger_build(needs_full_eval=True, progressive=True)
    else:
        _build_onion_batches(context, scene.frame_current, depsgraph)


_build_timer_registered = False
_prewarm_timer_registered = False
_pending_build = False
_needs_full_eval = False
_progressive_eval = False
_last_build_time = 0.0
_cached_signatures = {}


def _schedule_prewarm(delay=0.35):
    """Schedule non-blocking idle background cache pre-warming."""
    global _prewarm_timer_registered
    if not _prewarm_timer_registered:
        _prewarm_timer_registered = True
        bpy.app.timers.register(_do_prewarm, first_interval=delay)


def _unregister_prewarm():
    """Unregister any pending idle pre-warmer timer."""
    global _prewarm_timer_registered
    if bpy.app.timers.is_registered(_do_prewarm):
        try:
            bpy.app.timers.unregister(_do_prewarm)
        except Exception:
            pass
    _prewarm_timer_registered = False


def _do_prewarm():
    """Incrementally pre-warm deforming trajectory caches during viewport idle time (zero UI lag)."""
    global _prewarm_timer_registered, _is_evaluating
    context = bpy.context
    if not context or not getattr(context, 'scene', None):
        _prewarm_timer_registered = False
        return None
    scene = context.scene
    settings_glob = getattr(scene, 'onion_skin_settings', None)
    if (
        settings_glob is None
        or not settings_glob.enabled
        or not getattr(scene, 'onion_skin_objects', None)
        or screen_animation_playing(context)
    ):
        _prewarm_timer_registered = False
        return None

    if _is_building or _is_evaluating or _pending_build or _needs_full_eval or _is_transforming(context) or _is_scrubbing_modal(context):
        return 0.10

    now_pc = time.perf_counter()
    if (now_pc - _last_modal_edit_time) < 0.35 or (now_pc - _last_nonplay_fc_time) < 0.35:
        return 0.15

    # Flush any pending external scene/Action updates while _is_evaluating is still False
    try:
        context.evaluated_depsgraph_get().update()
    except Exception:
        pass
    if _is_building or _is_evaluating or _pending_build or _needs_full_eval or _is_transforming(context) or _is_scrubbing_modal(context):
        return 0.10

    was_eval = _is_evaluating
    _is_evaluating = True
    more_remaining = False
    try:
        current = scene.frame_current
        view_layer = getattr(context, 'view_layer', None)
        vis_cols = _get_visible_collections(view_layer) if view_layer is not None else None
        dep_arm_map, vis_arm_map, children_map, meta_map, _ = _build_scene_lookup_maps(scene)

        direct_items = []
        direct_missing = {}
        direct_setters = {}
        direct_deps = {}
        fc_map = {}
        delta_chains = {}
        delta_scale = {}

        for item in scene.onion_skin_objects:
            obj = item.object
            if not obj or not item.visible or not _is_valid_object(obj):
                continue
            if is_object_rigid(obj):
                continue
            s_cfg = properties.get_object_settings(obj, item)
            ok = get_object_key(obj)
            if not _is_object_visible_in_viewport(
                obj, s_cfg, view_layer, v3d=None, scene=scene, vis_cols=vis_cols,
                children_map=children_map, arm_map=vis_arm_map, metaball_map=meta_map,
            ):
                continue

            range_start, range_end = get_frame_range(s_cfg, scene)
            if getattr(s_cfg, 'limit_to_keyframes', False):
                kf_start, kf_end = get_keyframe_range_for_object(obj, scene=scene)
                if kf_start is not None and kf_end is not None:
                    if getattr(s_cfg, 'use_frame_range', False):
                        range_start = max(range_start, int(kf_start))
                        range_end = min(range_end, int(kf_end))
                    else:
                        range_start = int(kf_start)
                        range_end = int(kf_end)
            if range_end < range_start:
                continue
            if (range_end - range_start + 1) > 250:
                lo = max(range_start, current - 125)
                hi = min(range_end, lo + 249)
                lo = max(range_start, hi - 249)
                range_start, range_end = lo, hi

            dep_objs, dep_trees, can_direct, fc_entries, delta_chain, scale_sens = _get_or_build_playback_meta(
                obj, ok, scene, dep_arm_map
            )
            if fc_entries is not None:
                fc_map[ok] = fc_entries
            if delta_chain is not None:
                delta_chains[ok] = delta_chain
                delta_scale[ok] = scale_sens

            missing_set = set()
            has_ref = len(cache.get_cached_frames(ok)) > 0
            for f in range(range_start, range_end + 1):
                if cache.has_frame(ok, f):
                    continue
                if (
                    has_ref
                    and fc_entries is not None
                    and delta_chain is not None
                    and _try_reuse_same_pose_cached_frame(ok, f, fc_entries, delta_chain, scale_sens)
                ):
                    continue
                missing_set.add(f)

            if missing_set and can_direct:
                setters, tag_ids = _build_direct_setters(dep_objs, dep_trees)
                if setters is not None:
                    needed_stub = [(f, 'after' if f >= current else 'before', 1.0, abs(f - current)) for f in missing_set]
                    direct_items.append((item, obj, ok, s_cfg, needed_stub))
                    direct_missing[ok] = missing_set
                    direct_setters[ok] = (setters, tag_ids)
                    direct_deps[ok] = dep_objs

        if direct_items:
            inst_parents = _collect_inst_parent_objs(direct_items, scene, view_layer, vis_cols)
            frame_prios = {}
            for _it, _o, ok, _s, _n in direct_items:
                est_c = _passive_extract_costs.get(ok, 0.001)
                for f in direct_missing[ok]:
                    kp = (est_c, abs(f - current), f)
                    if f not in frame_prios or kp < frame_prios[f]:
                        frame_prios[f] = kp
            cand_frames = sorted(frame_prios.keys(), key=lambda f: frame_prios[f])

            snap_objs = set()
            for ok in direct_deps:
                snap_objs.update(direct_deps[ok])
            snaps = {}
            for dobj in snap_objs:
                try:
                    snaps[dobj] = _snapshot_object_state(dobj, include_pose_bones=False)
                except Exception:
                    pass

            start_t = time.perf_counter()
            eval_count = 0
            touched = []
            seen_touched = set()
            try:
                for idx_f, frame in enumerate(cand_frames):
                    raw_f = [
                        e for e in direct_items
                        if frame in direct_missing[e[2]] and not cache.has_frame(e[2], frame)
                    ]
                    items_at_f = []
                    for e in raw_f:
                        ok_e = e[2]
                        if ok_e in fc_map and ok_e in delta_chains:
                            if _try_reuse_same_pose_cached_frame(
                                ok_e, frame, fc_map[ok_e], delta_chains[ok_e], delta_scale.get(ok_e, False)
                            ):
                                continue
                        items_at_f.append(e)
                    if not items_at_f:
                        continue
                    if eval_count >= 1 and (time.perf_counter() - start_t) >= 0.008:
                        more_remaining = True
                        break
                    t_dg0 = time.perf_counter()
                    for _, _, ok_e, _, _ in items_at_f:
                        if ok_e not in seen_touched:
                            seen_touched.add(ok_e)
                            touched.append(ok_e)
                        setters, tag_ids = direct_setters[ok_e]
                        _apply_direct_setters(setters, tag_ids, frame)
                    dg = context.evaluated_depsgraph_get()
                    dg.update()
                    dg_cost = time.perf_counter() - t_dg0
                    eval_count += 1
                    _eval_and_cache_deforming_at_frame(
                        frame, items_at_f, dg, inst_parents,
                        fc_map, delta_chains, scene, dg_cost,
                        view_layer=view_layer, vis_cols=vis_cols, build_gpu=False,
                    )
                    if idx_f + 1 < len(cand_frames) and (time.perf_counter() - start_t) >= 0.008:
                        more_remaining = True
                        break
            finally:
                if eval_count > 0:
                    for ok_e in reversed(touched):
                        setters, tag_ids = direct_setters[ok_e]
                        _restore_direct_setters(setters, tag_ids)
                    for snap_obj, state in snaps.items():
                        if _is_valid_object(snap_obj):
                            _restore_object_state(snap_obj, state)
                    try:
                        context.evaluated_depsgraph_get().update()
                    except Exception:
                        pass
    except Exception:
        more_remaining = False
    finally:
        _is_evaluating = was_eval

    if more_remaining:
        return 0.03
    _prewarm_timer_registered = False
    return None


def _do_build():
    global _build_timer_registered, _pending_build, _needs_full_eval, _progressive_eval
    global _last_build_time, _was_transforming, _last_modal_eval_end
    context = bpy.context
    if not context or not getattr(context, 'scene', None):
        _build_timer_registered = False
        _was_transforming = False
        return None

    if screen_animation_playing(context):
        _build_timer_registered = False
        _was_transforming = False
        return None

    # During interactive modal drag (Graph Editor F-Curves, Dope Sheet, Pose Mode, Transform):
    # Update rigid & delta-xform ghosts instantly and pace heavy CPU skinning without starving continuous drags.
    if _is_transforming(context) or _is_scrubbing_modal(context):
        _was_transforming = True
        if _pending_build:
            scene = context.scene
            if (time.perf_counter() - _last_modal_eval_end) < 0.012:
                return 0.008

            _pending_build = False
            try:
                depsgraph = context.evaluated_depsgraph_get()
                incomplete = _build_onion_batches(
                    context, scene.frame_current, depsgraph,
                    allow_deforming_eval=True,
                    eval_time_budget=0.016,
                    fast_modal_reuse=True,
                )
                _last_modal_eval_end = time.perf_counter()
                if incomplete:
                    _pending_build = True
                    _progressive_eval = True
                else:
                    _needs_full_eval = False
                    _progressive_eval = False
                _safe_redraw(context)
            except Exception:
                pass
            if _pending_build:
                return 0.008
        return 0.016

    if _was_transforming:
        _was_transforming = False
        if _stale_offscreen_action_roots:
            _reconcile_offscreen_action_roots(context.scene)
        if _needs_full_eval or _progressive_eval:
            _pending_build = True

    now = time.time()
    if now - _last_build_time < 0.016:
        return 0.01

    _last_build_time = now
    _pending_build = False

    try:
        scene = context.scene
        depsgraph = context.evaluated_depsgraph_get()
        idle_budget = 0.045 if (time.perf_counter() - _last_nonplay_fc_time) < 0.25 else 0.35
        incomplete = _build_onion_batches(
            context, scene.frame_current, depsgraph,
            allow_deforming_eval=True,
            eval_time_budget=idle_budget if _progressive_eval else None,
        )
        if incomplete:
            _pending_build = True
        else:
            _needs_full_eval = False
            _progressive_eval = False
        _safe_redraw(context)
    except Exception:
        _needs_full_eval = False
        _progressive_eval = False

    if _pending_build or _needs_full_eval:
        return 0.016

    _build_timer_registered = False
    if getattr(bpy.context, 'scene', None) and hasattr(bpy.context.scene, 'onion_skin_settings'):
        sc_pw = bpy.context.scene
        if sc_pw.onion_skin_settings.enabled and sc_pw.onion_skin_objects:
            _schedule_prewarm()
    return None


def _trigger_build(needs_full_eval=True, progressive=False):
    global _build_timer_registered, _pending_build, _needs_full_eval, _progressive_eval
    _pending_build = True
    if needs_full_eval:
        _needs_full_eval = True
    if progressive:
        _progressive_eval = True
    if not _build_timer_registered:
        _build_timer_registered = True
        first_dt = 0.004 if _is_transforming(bpy.context) else 0.01
        bpy.app.timers.register(_do_build, first_interval=first_dt)


def _register_action_roots(holder, root_key, tracked_actions):
    """Map an ID datablock's Action and NLA actions to root_key in tracked_actions."""
    if holder is None:
        return
    anim = getattr(holder, 'animation_data', None)
    if not anim:
        return
    if anim.action:
        tracked_actions.setdefault(anim.action, set()).add(root_key)
    for track in getattr(anim, 'nla_tracks', ()):
        for strip in getattr(track, 'strips', ()):
            if getattr(strip, 'action', None):
                tracked_actions.setdefault(strip.action, set()).add(root_key)


def _validate_deforming_action_frames(
    rkey, root_obj, frames_to_check, dep_objs, dep_trees, fc_entries, non_xform_fcs, xform_fcs, clear_slots=True
):
    """Validate cached deforming frames against evaluated F-curves, updating GPU delta matrices where safe."""
    changed_frames = []
    delta_updated = False
    delta_checked = False
    can_delta = False
    delta_scale_sensitive = False
    ok_chain = False
    xform_chain = None

    for cf in frames_to_check:
        cur_non_xform, cur_xform = _eval_split_action_sigs_presplit(non_xform_fcs, xform_fcs, cf)
        prev_sig = _action_frame_sigs.get((rkey, cf))
        if (
            prev_sig is None
            or not isinstance(prev_sig, tuple)
            or len(prev_sig) != 2
            or prev_sig[0] != cur_non_xform
        ):
            changed_frames.append(cf)
        elif prev_sig[1] != cur_xform:
            if not delta_checked:
                delta_checked = True
                can_delta, delta_scale_sensitive, ok_chain, xform_chain = _get_cached_delta_capability(
                    rkey, root_obj, dep_objs, dep_trees, fc_entries=fc_entries
                )
            base_mat = _deforming_base_mats.get((rkey, cf))
            if can_delta and ok_chain and base_mat is not None:
                new_mat = _eval_world_matrix_analytic(xform_chain, cf)
                scale_ok = (not delta_scale_sensitive) or (
                    max(abs(a - b) for a, b in zip(new_mat.to_scale(), base_mat.to_scale())) < 1e-4
                )
                if scale_ok:
                    _deforming_delta_mats[(rkey, cf)] = new_mat @ base_mat.inverted()
                    _action_frame_sigs[(rkey, cf)] = (cur_non_xform, cur_xform)
                    delta_updated = True
                else:
                    changed_frames.append(cf)
            else:
                changed_frames.append(cf)

    if changed_frames:
        cache.invalidate_frames(rkey, changed_frames)
        if clear_slots:
            _clear_ghost_slots(rkey)
        for cf in changed_frames:
            _action_frame_sigs.pop((rkey, cf), None)
            _deforming_base_mats.pop((rkey, cf), None)
            _deforming_delta_mats.pop((rkey, cf), None)

    return changed_frames, delta_updated, delta_checked, can_delta, delta_scale_sensitive, ok_chain, xform_chain


def _reconcile_offscreen_action_roots(scene, root_keys=None):
    """Reconcile off-screen cached frames deferred during active modal F-Curve drags."""
    targets = list(_stale_offscreen_action_roots) if root_keys is None else [k for k in root_keys if k in _stale_offscreen_action_roots]
    if not targets or scene is None:
        return
    obj_by_key = {}
    for item in getattr(scene, 'onion_skin_objects', ()):
        obj = item.object
        if obj and _is_valid_object(obj):
            obj_by_key[get_object_key(obj)] = obj
    arm_map = None
    for rkey in targets:
        _stale_offscreen_action_roots.discard(rkey)
        root_obj = obj_by_key.get(rkey)
        if root_obj is None:
            cache.clear_frame_cache(rkey)
            _clear_object_action_sigs(rkey)
            continue
        cached_frames = cache.get_cached_frames(rkey)
        if not cached_frames:
            continue
        if arm_map is None:
            arm_map = _build_armature_children_map(scene)
        dep_objs, _, dep_trees = _gather_object_dependencies(root_obj, scene, armature_children_map=arm_map)
        _cached_kf_ranges[rkey] = get_keyframe_range_for_object(root_obj, scene=scene)
        if (dep_objs or dep_trees) and _can_use_per_frame_action_inval(dep_objs, dep_trees):
            fc_entries = _collect_dep_fcurves_sorted(dep_objs, dep_trees)
            non_xform_fcs, xform_fcs = _split_fc_entries_for_root(fc_entries, rkey)
            _validate_deforming_action_frames(
                rkey, root_obj, cached_frames, dep_objs, dep_trees, fc_entries, non_xform_fcs, xform_fcs, clear_slots=False
            )
        else:
            cache.clear_frame_cache(rkey)
            _clear_object_action_sigs(rkey)


@persistent
def _on_depsgraph_update(scene, depsgraph):
    """Selective cache invalidation with full dependency tracking and zero-flicker double buffering."""
    global _last_modal_edit_time, _dirty_while_disabled, _dg_dep_cache
    if _is_building or _is_evaluating:
        return

    if not hasattr(scene, 'onion_skin_settings'):
        return

    # Stale object cleanup (selectively drop orphaned caches/ghosts without clearing other objects)
    stale = [i for i, item in enumerate(scene.onion_skin_objects) if item.object is None]
    if stale:
        for i in reversed(stale):
            scene.onion_skin_objects.remove(i)
        _prune_orphan_ghost_markers(scene)
        cache.clear_frame_cache()
        cache.clear_rigid_cache()
        cache.clear_rigid_matrix_cache()
        cache.clear_onion_batches()
        bump_vis_cache_rev()
        clear_topology_cache()
        _clear_ghost_slots()
        _clear_object_action_sigs()
        _cached_signatures.clear()
        _cached_outliner_vis.clear()
        _cached_coll_sigs.clear()
        _tracked_ptr_to_identity.clear()
        scene.onion_skin_settings.marker_count = len(cache.get_ghost_markers())
        if scene.onion_skin_active_index >= len(scene.onion_skin_objects):
            scene.onion_skin_active_index = max(0, len(scene.onion_skin_objects) - 1)
        if scene.onion_skin_settings.enabled and scene.onion_skin_objects:
            _trigger_build()
        else:
            _safe_redraw(bpy.context)
        return

    global_settings = scene.onion_skin_settings
    if not global_settings.enabled or not scene.onion_skin_objects:
        if not global_settings.enabled and scene.onion_skin_objects:
            _dirty_while_disabled = True
        return

    if screen_animation_playing(bpy.context):
        return

    view_layer = getattr(bpy.context, 'view_layer', None)
    cur_cache_key = (
        scene.as_pointer(),
        view_layer.as_pointer() if view_layer is not None else 0,
        _vis_cache_rev,
    )
    cur_item_sig = tuple(
        (item.object.as_pointer(), bool(item.visible), item.object.name)
        for item in scene.onion_skin_objects
        if item.object and _is_valid_object(item.object)
    )

    used_fast_dep_cache = False
    dep_c = _dg_dep_cache
    if dep_c is not None and dep_c['key'] == cur_cache_key and dep_c['item_sig'] == cur_item_sig:
        fast_ok = True
        try:
            for h, act_ptr, nla_cnt in dep_c['holder_sigs']:
                anim = getattr(h, 'animation_data', None)
                cur_act = anim.action.as_pointer() if (anim and anim.action) else 0
                cur_nla = len(anim.nla_tracks) if anim else 0
                if cur_act != act_ptr or cur_nla != nla_cnt:
                    fast_ok = False
                    break
            if fast_ok:
                for dobj, dname in dep_c['obj_name_sigs']:
                    if dobj.name != dname:
                        fast_ok = False
                        break
            if fast_ok:
                for o_vis, hv, hg in dep_c['vis_obj_sigs']:
                    cur_hv = bool(getattr(o_vis, 'hide_viewport', False))
                    cur_hg = bool(o_vis.hide_get(view_layer=view_layer)) if view_layer is not None else False
                    if cur_hv != hv or cur_hg != hg:
                        fast_ok = False
                        break
            if fast_ok:
                cached_actions = dep_c['tracked_actions']
                cached_objs = dep_c['tracked_obj_roots']
                has_match_action = False
                for update in depsgraph.updates:
                    raw_id = getattr(update, 'id', None)
                    if raw_id is None or getattr(raw_id, 'is_missing', False):
                        continue
                    up_id = getattr(raw_id, 'original', raw_id)
                    if isinstance(up_id, bpy.types.Action):
                        if up_id in cached_actions:
                            has_match_action = True
                    elif isinstance(up_id, (bpy.types.NodeTree, bpy.types.Mesh, bpy.types.Curve, bpy.types.MetaBall, bpy.types.TextCurve, bpy.types.SurfaceCurve)):
                        fast_ok = False
                        break
                    elif isinstance(up_id, bpy.types.Collection):
                        if _get_collection_membership_signature(up_id) != _cached_coll_sigs.get(_get_data_key(up_id)):
                            fast_ok = False
                            break
                    elif isinstance(up_id, bpy.types.Object) and up_id in cached_objs:
                        if get_object_geometry_signature(up_id) != _cached_signatures.get(get_object_key(up_id)):
                            fast_ok = False
                            break
                if not has_match_action:
                    fast_ok = False
        except (ReferenceError, AttributeError):
            fast_ok = False

        if fast_ok:
            used_fast_dep_cache = True
            tracked_obj_roots = dep_c['tracked_obj_roots']
            tracked_data_roots = dep_c['tracked_data_roots']
            tracked_actions = dep_c['tracked_actions']
            tracked_collections = dep_c['tracked_collections']
            tracked_trees = dep_c['tracked_trees']
            root_is_rigid = dep_c['root_is_rigid']
            root_objs = dep_c['root_objs']
            root_settings = dep_c['root_settings']
            root_dep_objs = dep_c['root_dep_objs']
            root_dep_trees = dep_c['root_dep_trees']
            root_dep_topo_keys = dep_c['root_dep_topo_keys']
            vis_child_changed_roots = set()
            vis_unhidden_roots = set()
            vis_hidden_roots = set()
            needs_rename_rebuild = False

    if not used_fast_dep_cache:
        # Reverse dependency maps from every dependency ID to affected root object keys
        tracked_obj_roots = {}      # orig Object -> set[root_key]
        tracked_data_roots = {}     # orig data/shape_keys -> set[root_key]
        tracked_actions = {}        # orig Action -> set[root_key]
        tracked_collections = {}    # orig Collection -> set[root_key]
        tracked_trees = {}          # orig NodeTree -> set[root_key]
        root_is_rigid = {}          # root_key -> bool
        root_objs = {}              # root_key -> orig Object
        root_settings = {}          # root_key -> resolved OnionSkin settings
        root_dep_objs = {}          # root_key -> set[Object]
        root_dep_trees = {}         # root_key -> set[NodeTree]
        root_dep_topo_keys = {}     # root_key -> set[key strings to purge on geometry change]

        dep_arm_map, vis_arm_map, children_map, meta_map, scene_objs = _build_scene_lookup_maps(scene)
        vis_cols = _get_visible_collections(view_layer) if view_layer is not None else None
        active_batch_keys = {b[0] for b in cache.get_onion_batches() if len(b) > 0}
        vis_child_changed_roots = set()
        vis_unhidden_roots = set()
        vis_hidden_roots = set()
        needs_rename_rebuild = False
        vis_tracked_objs = set()

        for item in scene.onion_skin_objects:
            obj = item.object
            if not obj or not _is_valid_object(obj):
                continue
            if _track_object_identity(obj):
                needs_rename_rebuild = True
            root_key = get_object_key(obj)
            if not item.visible:
                _cached_outliner_vis.pop(root_key, None)
                continue

            vis_tracked_objs.add(obj)
            s_cfg = properties.get_object_settings(obj, item)
            if obj.type == 'ARMATURE' and getattr(s_cfg, 'include_children', True):
                for ch in _get_armature_candidate_meshes(
                    obj, s_cfg, scene=scene, children_map=children_map, arm_map=vis_arm_map,
                ):
                    vis_tracked_objs.add(ch)
                    if _track_object_identity(ch):
                        cache.clear_frame_cache(root_key)
                        cache.clear_rigid_cache(root_key)
                        _clear_object_action_sigs(root_key)
                        _clear_ghost_slots(root_key)
                        needs_rename_rebuild = True
            if view_layer is not None:
                cur_vis_sig = _compute_outliner_vis_sig(
                    obj, s_cfg, view_layer, scene=scene, vis_cols=vis_cols,
                    children_map=children_map, arm_map=vis_arm_map, metaball_map=meta_map,
                )
                obj_vis, child_vis = cur_vis_sig
                prev_vis_sig = _cached_outliner_vis.get(root_key)
                _cached_outliner_vis[root_key] = cur_vis_sig
                if prev_vis_sig is not None and prev_vis_sig != cur_vis_sig:
                    if prev_vis_sig[1] != child_vis:
                        prev_map = dict(prev_vis_sig[1])
                        newly_visible = [ck for ck, cv in child_vis if cv and not prev_map.get(ck, False)]
                        removed_keys = set(prev_map.keys()) - {ck for ck, _ in child_vis}
                        if newly_visible or removed_keys:
                            cached_frs = cache.get_cached_frames(root_key)
                            missing_parts = bool(removed_keys) or not cached_frs
                            if not missing_parts and newly_visible:
                                for cf in cached_frs:
                                    cbatch = cache.get_frame(root_key, cf)
                                    if isinstance(cbatch, list):
                                        present = {e[0] for e in cbatch if isinstance(e, (list, tuple)) and len(e) > 0}
                                        if any(pk not in present for pk in newly_visible):
                                            missing_parts = True
                                            break
                            if missing_parts:
                                vis_child_changed_roots.add(root_key)
                            else:
                                vis_unhidden_roots.add(root_key)
                        else:
                            vis_hidden_roots.add(root_key)
                    if not prev_vis_sig[0] and obj_vis:
                        vis_unhidden_roots.add(root_key)
                    elif prev_vis_sig[0] and not obj_vis:
                        vis_hidden_roots.add(root_key)
                elif prev_vis_sig is None and root_key in active_batch_keys:
                    if not obj_vis or (child_vis and not any(v for _, v in child_vis)):
                        vis_hidden_roots.add(root_key)

            root_is_rigid[root_key] = is_object_rigid(obj)
            root_objs[root_key] = obj
            root_settings[root_key] = s_cfg
            topo_set = root_dep_topo_keys.setdefault(root_key, set())
            topo_set.add(root_key)

            dep_objs, dep_colls, dep_trees = _gather_object_dependencies(
                obj,
                scene,
                armature_children_map=dep_arm_map,
                children_map=children_map,
                metaball_map=meta_map,
                scene_objs=scene_objs,
            )
            root_dep_objs[root_key] = dep_objs
            root_dep_trees[root_key] = dep_trees
            for dobj in dep_objs:
                vis_tracked_objs.add(dobj)
                tracked_obj_roots.setdefault(dobj, set()).add(root_key)
                _register_action_roots(dobj, root_key, tracked_actions)
                topo_set.add(get_object_key(dobj))
                if getattr(dobj, 'data', None):
                    d = dobj.data
                    tracked_data_roots.setdefault(d, set()).add(root_key)
                    _register_action_roots(d, root_key, tracked_actions)
                    topo_set.add(_get_data_key(d))
                    sk = getattr(d, 'shape_keys', None)
                    if sk is not None:
                        tracked_data_roots.setdefault(sk, set()).add(root_key)
                        _register_action_roots(sk, root_key, tracked_actions)

            for coll in dep_colls:
                tracked_collections.setdefault(coll, set()).add(root_key)
            for tree in dep_trees:
                tracked_trees.setdefault(tree, set()).add(root_key)
                _register_action_roots(tree, root_key, tracked_actions)

        holder_list = []
        seen_holders = set()
        for dobj in tracked_obj_roots:
            for h in (dobj, getattr(dobj, 'data', None), getattr(getattr(dobj, 'data', None), 'shape_keys', None)):
                if h is not None and h not in seen_holders:
                    seen_holders.add(h)
                    anim = getattr(h, 'animation_data', None)
                    holder_list.append((
                        h,
                        anim.action.as_pointer() if (anim and anim.action) else 0,
                        len(anim.nla_tracks) if anim else 0,
                    ))
        for tree in tracked_trees:
            if tree not in seen_holders:
                seen_holders.add(tree)
                anim = getattr(tree, 'animation_data', None)
                holder_list.append((
                    tree,
                    anim.action.as_pointer() if (anim and anim.action) else 0,
                    len(anim.nla_tracks) if anim else 0,
                ))
        _dg_dep_cache = {
            'key': cur_cache_key,
            'item_sig': cur_item_sig,
            'holder_sigs': holder_list,
            'obj_name_sigs': [(dobj, dobj.name) for dobj in tracked_obj_roots],
            'vis_obj_sigs': [
                (
                    vo,
                    bool(getattr(vo, 'hide_viewport', False)),
                    bool(vo.hide_get(view_layer=view_layer)) if view_layer is not None else False,
                )
                for vo in vis_tracked_objs
            ],
            'tracked_obj_roots': tracked_obj_roots,
            'tracked_data_roots': tracked_data_roots,
            'tracked_actions': tracked_actions,
            'tracked_collections': tracked_collections,
            'tracked_trees': tracked_trees,
            'root_is_rigid': root_is_rigid,
            'root_objs': root_objs,
            'root_settings': root_settings,
            'root_dep_objs': root_dep_objs,
            'root_dep_trees': root_dep_trees,
            'root_dep_topo_keys': root_dep_topo_keys,
        }
        for dobj in tracked_obj_roots:
            dok = get_object_key(dobj)
            if dok not in _cached_signatures:
                _cached_signatures[dok] = get_object_geometry_signature(dobj)

    if not tracked_obj_roots:
        return

    needs_update = False
    action_updated_roots = set()
    non_action_geom_roots = set()
    obj_geom_nosig_roots = set()
    obj_topo_nosig_roots = set()
    xform_invalidated_roots = set()
    topo_invalidated_roots = set()

    for update in depsgraph.updates:
        try:
            raw_id = getattr(update, 'id', None)
            if raw_id is None or getattr(raw_id, 'is_missing', False):
                continue
            up_id = getattr(raw_id, 'original', raw_id)
            up_geom = bool(getattr(update, 'is_updated_geometry', False))
            up_xform = bool(getattr(update, 'is_updated_transform', False))

            if isinstance(up_id, bpy.types.Action):
                roots = tracked_actions.get(up_id)
                if roots:
                    needs_update = True
                    action_updated_roots.update(roots)
                continue

            if isinstance(up_id, bpy.types.NodeTree):
                roots = tracked_trees.get(up_id)
                if roots:
                    needs_update = True
                    non_action_geom_roots.update(roots)
                    topo_invalidated_roots.update(roots)
                continue

            if isinstance(up_id, bpy.types.Collection):
                roots = tracked_collections.get(up_id)
                if roots and (up_geom or up_xform):
                    needs_update = True
                    ckey = _get_data_key(up_id)
                    new_csig = _get_collection_membership_signature(up_id)
                    old_csig = _cached_coll_sigs.get(ckey)
                    _cached_coll_sigs[ckey] = new_csig
                    if old_csig is not None and old_csig != new_csig:
                        non_action_geom_roots.update(roots)
                        topo_invalidated_roots.update(roots)
                    else:
                        obj_geom_nosig_roots.update(roots)
                continue

            if isinstance(up_id, bpy.types.Key):
                roots = tracked_data_roots.get(up_id)
                if roots:
                    needs_update = True
                    non_action_geom_roots.update(roots)
                continue

            if isinstance(up_id, (bpy.types.Mesh, bpy.types.Curve, bpy.types.MetaBall, bpy.types.TextCurve, bpy.types.SurfaceCurve)):
                roots = tracked_data_roots.get(up_id)
                if roots and up_geom:
                    needs_update = True
                    non_action_geom_roots.update(roots)
                    topo_invalidated_roots.update(roots)
                continue

            if isinstance(up_id, bpy.types.Armature):
                roots = tracked_data_roots.get(up_id)
                if roots and (up_geom or up_xform):
                    needs_update = True
                    obj_geom_nosig_roots.update(roots)
                continue

            if isinstance(up_id, bpy.types.Object):
                roots = tracked_obj_roots.get(up_id)
                if not roots:
                    continue
                if not up_geom and not up_xform:
                    continue

                needs_update = True
                ok = get_object_key(up_id)
                new_sig = get_object_geometry_signature(up_id)
                old_sig = _cached_signatures.get(ok)
                sig_changed = (old_sig != new_sig) if up_geom else (old_sig is not None and old_sig != new_sig)
                _cached_signatures[ok] = new_sig

                if sig_changed:
                    non_action_geom_roots.update(roots)
                    if up_id.type != 'ARMATURE':
                        topo_invalidated_roots.update(roots)
                elif up_geom:
                    # Geometry-level update without structural change (e.g. pose bone move, metaball merge, GN input socket)
                    obj_geom_nosig_roots.update(roots)
                    if up_id.type in ('META', 'CURVE', 'SURFACE', 'FONT') or any(m.type == 'NODES' for m in up_id.modifiers):
                        obj_topo_nosig_roots.update(roots)

                if up_xform:
                    xform_invalidated_roots.update(roots)
        except Exception:
            continue

    if needs_update or needs_rename_rebuild or vis_child_changed_roots or vis_unhidden_roots or vis_hidden_roots:
        if action_updated_roots or non_action_geom_roots or xform_invalidated_roots:
            _last_modal_edit_time = time.perf_counter()

        # When an Action changed without structural/data geometry changes, secondary Object
        # updates come from re-evaluating the Action on the current frame. Use per-frame
        # F-curve invalidation instead of wiping all cached ghost frames.
        non_action_geom_roots.update(vis_child_changed_roots)
        pure_action_roots = {rk for rk in action_updated_roots if rk not in non_action_geom_roots}
        geom_invalidated_roots = non_action_geom_roots | (obj_geom_nosig_roots - pure_action_roots)
        effective_topo_roots = topo_invalidated_roots | (obj_topo_nosig_roots - pure_action_roots)
        effective_xform_roots = xform_invalidated_roots - pure_action_roots

        is_pure_action_only = bool(
            action_updated_roots
            and not geom_invalidated_roots
            and not effective_topo_roots
            and not effective_xform_roots
            and not needs_rename_rebuild
            and not vis_child_changed_roots
            and not vis_unhidden_roots
            and not vis_hidden_roots
        )
        if not is_pure_action_only:
            bump_vis_cache_rev()

        for rkey in effective_topo_roots:
            for tkey in root_dep_topo_keys.get(rkey, ()):
                clear_topology_cache(tkey)
        for rkey in geom_invalidated_roots:
            _rigid_analytic_cache.pop(rkey, None)
            _cached_kf_ranges.pop(rkey, None)
            _delta_cap_cache.pop(rkey, None)
            cache.clear_rigid_cache(rkey)
            cache.clear_rigid_matrix_cache(rkey)
            cache.clear_frame_cache(rkey)
            _clear_object_action_sigs(rkey)
            _clear_ghost_slots(rkey)
        for rkey in effective_xform_roots:
            _rigid_analytic_cache.pop(rkey, None)
            _cached_kf_ranges.pop(rkey, None)
            _delta_cap_cache.pop(rkey, None)
            cache.clear_rigid_matrix_cache(rkey)
            # If a deforming object or its world-space parent/target/metaball moved, clear baked world geometry
            if not root_is_rigid.get(rkey, True):
                cache.clear_frame_cache(rkey)
                _clear_object_action_sigs(rkey)
                _clear_ghost_slots(rkey)

        is_modal_tx = _is_transforming(bpy.context) or _is_scrubbing_modal(bpy.context)
        deforming_action_dirty_roots = set()
        delta_updated_roots = set()
        rigid_needs_frameset = False

        for rkey in action_updated_roots:
            _playback_eval_meta.pop(rkey, None)
            _cached_kf_ranges.pop(rkey, None)
            if rkey in geom_invalidated_roots or (rkey in effective_xform_roots and not root_is_rigid.get(rkey, True)):
                continue
            cache.clear_rigid_matrix_cache(rkey)
            root_obj = root_objs.get(rkey)
            if root_is_rigid.get(rkey, True):
                if root_obj is not None and not _can_analytically_eval_rigid(root_obj):
                    rigid_needs_frameset = True
                continue

            cached_frames = cache.get_cached_frames(rkey)
            dep_objs = root_dep_objs.get(rkey, ())
            dep_trees = root_dep_trees.get(rkey, ())
            if root_obj is not None:
                _cached_kf_ranges[rkey] = get_keyframe_range_for_object(root_obj, scene=scene)
            if cached_frames and (dep_objs or dep_trees) and _can_use_per_frame_action_inval(dep_objs, dep_trees):
                fc_entries = _collect_dep_fcurves_sorted(dep_objs, dep_trees)
                non_xform_fcs, xform_fcs = _split_fc_entries_for_root(fc_entries, rkey)
                s_cfg = root_settings.get(rkey)
                needed_now = (
                    get_needed_frames(scene.frame_current, s_cfg, scene, obj=root_obj, use_kf_cache=True)
                    if (root_obj is not None and s_cfg is not None)
                    else []
                )
                needed_set = {f for f, _, _, _ in needed_now}
                if is_modal_tx and len(cached_frames) > len(needed_set):
                    frames_to_check = [cf for cf in cached_frames if cf in needed_set]
                    _stale_offscreen_action_roots.add(rkey)
                else:
                    frames_to_check = cached_frames
                    _stale_offscreen_action_roots.discard(rkey)

                (
                    changed_frames,
                    did_delta,
                    delta_checked,
                    can_delta,
                    delta_scale_sensitive,
                    ok_chain,
                    xform_chain,
                ) = _validate_deforming_action_frames(
                    rkey,
                    root_obj,
                    frames_to_check,
                    dep_objs,
                    dep_trees,
                    fc_entries,
                    non_xform_fcs,
                    xform_fcs,
                    clear_slots=(not is_modal_tx),
                )
                if did_delta:
                    delta_updated_roots.add(rkey)
                if changed_frames:
                    deforming_action_dirty_roots.add(rkey)
                    if rkey in obj_topo_nosig_roots:
                        for tkey in root_dep_topo_keys.get(rkey, ()):
                            clear_topology_cache(tkey)
                if needed_now:
                    missing_needed = [f for f, _, _, _ in needed_now if not cache.has_frame(rkey, f)]
                    if missing_needed:
                        if cache.get_cached_frames(rkey):
                            if not delta_checked:
                                delta_checked = True
                                can_delta, delta_scale_sensitive, ok_chain, xform_chain = _get_cached_delta_capability(
                                    rkey, root_obj, dep_objs, dep_trees, fc_entries=fc_entries
                                )
                            for nf in missing_needed:
                                if can_delta and ok_chain and _try_reuse_same_pose_cached_frame(
                                    rkey, nf, fc_entries, xform_chain, delta_scale_sensitive
                                ):
                                    delta_updated_roots.add(rkey)
                                else:
                                    deforming_action_dirty_roots.add(rkey)
                        else:
                            deforming_action_dirty_roots.add(rkey)
            else:
                deforming_action_dirty_roots.add(rkey)
                cache.clear_frame_cache(rkey)
                _clear_object_action_sigs(rkey)
                if not is_modal_tx:
                    _clear_ghost_slots(rkey)
                if rkey in obj_topo_nosig_roots:
                    for tkey in root_dep_topo_keys.get(rkey, ()):
                        clear_topology_cache(tkey)

        needs_full = bool(
            geom_invalidated_roots
            or vis_unhidden_roots
            or deforming_action_dirty_roots
            or rigid_needs_frameset
            or any(not root_is_rigid.get(rk, True) for rk in effective_xform_roots)
        )
        has_rigid_fast = (not rigid_needs_frameset) and (
            any(root_is_rigid.get(rk, True) for rk in effective_xform_roots)
            or any(
                root_is_rigid.get(rk, True)
                for rk in action_updated_roots
                if rk not in geom_invalidated_roots
            )
        )
        has_instant_work = bool(
            vis_hidden_roots
            or vis_unhidden_roots
            or delta_updated_roots
            or has_rigid_fast
            or (
                not needs_full
                and not _needs_full_eval
                and not _progressive_eval
                and action_updated_roots
            )
        )
        # Fast inline update (zero timer lag) for instant transform-delta, rigid analytic, and hide toggles
        if has_instant_work:
            try:
                ctx = bpy.context
                if ctx and getattr(ctx, 'scene', None) == scene:
                    _build_onion_batches(
                        ctx, scene.frame_current, depsgraph,
                        allow_deforming_eval=False,
                        fast_modal_reuse=is_pure_action_only,
                    )
                    _safe_redraw(ctx)
                    if not needs_full and not _needs_full_eval and not _progressive_eval:
                        return
            except Exception:
                pass
        _trigger_build(needs_full_eval=(needs_full or _needs_full_eval))


@persistent
def _on_undo_redo(scene):
    _reset_runtime_state(clear_ghosts=False, keep_identities=True)
    if scene and hasattr(scene, 'onion_skin_settings'):
        _sync_scene_identities(scene)
        synced = cache.sync_ghost_markers_to_count(scene.onion_skin_settings.marker_count)
        if scene.onion_skin_settings.marker_count != synced:
            scene.onion_skin_settings.marker_count = synced
        if scene.onion_skin_settings.enabled and scene.onion_skin_objects:
            _trigger_build()
    _safe_redraw(bpy.context)


@persistent
def _on_onion_load_pre(_dummy):
    global _build_timer_registered
    if bpy.app.timers.is_registered(_do_build):
        try:
            bpy.app.timers.unregister(_do_build)
        except Exception:
            pass
    _build_timer_registered = False
    _unregister_prewarm()
    _reset_runtime_state(clear_ghosts=True)


@persistent
def _on_onion_load_post(_dummy):
    global _build_timer_registered
    if bpy.app.timers.is_registered(_do_build):
        try:
            bpy.app.timers.unregister(_do_build)
        except Exception:
            pass
    _build_timer_registered = False
    _unregister_prewarm()
    _reset_runtime_state(clear_ghosts=True)
    for sc in bpy.data.scenes:
        if hasattr(sc, 'onion_skin_settings'):
            sc.onion_skin_settings.marker_count = 0
    ctx = bpy.context
    sc = getattr(ctx, 'scene', None)
    if sc and hasattr(sc, 'onion_skin_settings'):
        _sync_scene_identities(sc)
        if sc.onion_skin_settings.enabled and sc.onion_skin_objects:
            _trigger_build()
    _safe_redraw(ctx)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_draw_handler = None
_label_handler = None


def register():
    global _draw_handler, _label_handler
    _cleanup_handlers()
    _reset_runtime_state(clear_ghosts=True)
    sc = getattr(bpy.context, 'scene', None)
    if sc is not None:
        _sync_scene_identities(sc)

    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_onion_skins, (), 'WINDOW', 'POST_VIEW'
    )
    _label_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_ghost_labels, (), 'WINDOW', 'POST_PIXEL'
    )
    bpy.app.handlers.frame_change_post.append(_on_frame_change)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    bpy.app.handlers.undo_post.append(_on_undo_redo)
    bpy.app.handlers.redo_post.append(_on_undo_redo)
    bpy.app.handlers.load_pre.append(_on_onion_load_pre)
    bpy.app.handlers.load_post.append(_on_onion_load_post)


def unregister():
    global _draw_handler, _label_handler, _build_timer_registered
    if _draw_handler:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None
    if _label_handler:
        bpy.types.SpaceView3D.draw_handler_remove(_label_handler, 'WINDOW')
        _label_handler = None

    if bpy.app.timers.is_registered(_do_build):
        try:
            bpy.app.timers.unregister(_do_build)
        except Exception:
            pass
    _build_timer_registered = False
    _unregister_prewarm()

    _reset_runtime_state(clear_ghosts=True)
    _cleanup_handlers()
