# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators for onion skin addon."""

import bpy
from bpy.types import Operator
from bpy.props import IntProperty

from . import cache


class ONION_OT_add_object(Operator):
    """Add selected objects to onion skin list"""
    bl_idname = "onion_skin.add_object"
    bl_label = "Add Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects or context.active_object)

    def execute(self, context):
        from . import drawing, properties

        existing = {item.object for item in context.scene.onion_skin_objects if item.object}
        added = 0

        if getattr(context, 'mode', None) == 'POSE' and context.active_object:
            candidates = [context.active_object]
        else:
            candidates = list(context.selected_objects) if context.selected_objects else (
                [context.active_object] if context.active_object else []
            )

        properties._is_batch_updating = True
        try:
            for obj in candidates:
                if drawing.can_add_to_onion(obj) and obj not in existing:
                    context.scene.onion_skin_objects.add().object = obj
                    drawing._track_object_identity(obj)
                    existing.add(obj)
                    added += 1
        finally:
            properties._is_batch_updating = False

        if added > 0:
            context.scene.onion_skin_active_index = len(context.scene.onion_skin_objects) - 1
            drawing._trigger_build()
            properties._safe_redraw(context)
        elif candidates:
            target = candidates[0]
            for i, item in enumerate(context.scene.onion_skin_objects):
                if item.object == target:
                    context.scene.onion_skin_active_index = i
                    properties._safe_redraw(context)
                    break

        self.report({'INFO'}, f"Added {added} object(s)")
        return {'FINISHED'}


class ONION_OT_add_from_picker(Operator):
    """Add object from eyedropper selection"""
    bl_idname = "onion_skin.add_from_picker"
    bl_label = "Add Picked Object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import drawing, properties

        settings = context.scene.onion_skin_settings
        obj = settings.pick_object

        if not obj:
            return {'CANCELLED'}

        if not drawing.can_add_to_onion(obj):
            self.report({'WARNING'}, "Object type not supported for onion skinning")
            settings.pick_object = None
            return {'CANCELLED'}

        existing = {item.object for item in context.scene.onion_skin_objects if item.object}

        if obj not in existing:
            context.scene.onion_skin_objects.add().object = obj
            drawing._track_object_identity(obj)
            context.scene.onion_skin_active_index = len(context.scene.onion_skin_objects) - 1
            cache.clear_frame_cache(drawing.get_object_key(obj))
            drawing._trigger_build()
            properties._safe_redraw(context)
            self.report({'INFO'}, f"Added: {obj.name}")
        else:
            self.report({'INFO'}, f"Already added: {obj.name}")

        settings.pick_object = None
        return {'FINISHED'}


def _collect_object_marker_keys(obj, settings, scene):
    """Collect all key/name aliases that may tag ghost markers owned by obj."""
    from . import drawing
    keys = set()
    if not obj:
        return keys
    ok = drawing.get_object_key(obj)
    if ok:
        keys.add(ok)
    if getattr(obj, 'name', None):
        keys.add(obj.name)
    for robj in drawing.resolve_mesh_objects(obj, settings, scene=scene, include_hidden=True):
        rk = drawing.get_object_key(robj)
        if rk:
            keys.add(rk)
        if getattr(robj, 'name', None):
            keys.add(robj.name)
    return keys


def _cleanup_removed_object(rem_obj, context):
    """Selectively purge caches and ghost markers for a removed onion-skin object."""
    from . import drawing, properties
    scene = context.scene
    if cache.get_ghost_markers():
        cache.push_ghost_undo_snapshot()
    if rem_obj:
        s_cfg = properties.get_object_settings(rem_obj)
        ok = drawing.get_object_key(rem_obj)
        rem_keys = _collect_object_marker_keys(rem_obj, s_cfg, scene)
        for k in rem_keys:
            drawing.clear_topology_cache(k)
        cache.clear_frame_cache(ok)
        cache.clear_rigid_cache(ok)
        cache.clear_rigid_matrix_cache(ok)
        drawing._clear_object_action_sigs(ok)
        drawing._clear_ghost_slots(ok)
        drawing._cached_signatures.pop(ok, None)
        drawing._cached_outliner_vis.pop(ok, None)
        cache.remove_ghost_markers_for_object(rem_keys)
    else:
        cache.clear_frame_cache()
        cache.clear_rigid_cache()
        cache.clear_rigid_matrix_cache()
        drawing._prune_orphan_ghost_markers(scene)
    scene.onion_skin_settings.marker_count = len(cache.get_ghost_markers())
    if not scene.onion_skin_objects:
        cache.clear_onion_batches()
        drawing._clear_ghost_slots()
    else:
        drawing._trigger_build()
    properties._safe_redraw(context)


class ONION_OT_remove_object(Operator):
    """Remove object from onion skin list by index"""
    bl_idname = "onion_skin.remove_object"
    bl_label = "Remove"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty()

    def execute(self, context):
        scene = context.scene
        if 0 <= self.index < len(scene.onion_skin_objects):
            rem_obj = scene.onion_skin_objects[self.index].object
            scene.onion_skin_objects.remove(self.index)
            if scene.onion_skin_active_index >= len(scene.onion_skin_objects):
                scene.onion_skin_active_index = max(0, len(scene.onion_skin_objects) - 1)
            _cleanup_removed_object(rem_obj, context)
        return {'FINISHED'}


class ONION_OT_remove_selected(Operator):
    """Remove the active object from the list"""
    bl_idname = "onion_skin.remove_selected"
    bl_label = "Remove Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.onion_skin_objects) > 0

    def execute(self, context):
        scene = context.scene
        index = scene.onion_skin_active_index

        if 0 <= index < len(scene.onion_skin_objects):
            rem_obj = scene.onion_skin_objects[index].object
            scene.onion_skin_objects.remove(index)
            if index >= len(scene.onion_skin_objects):
                scene.onion_skin_active_index = max(0, len(scene.onion_skin_objects) - 1)
            _cleanup_removed_object(rem_obj, context)
        return {'FINISHED'}


class ONION_OT_clear_all(Operator):
    """Remove all objects from the list"""
    bl_idname = "onion_skin.clear_all"
    bl_label = "Clear All"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import properties

        if cache.get_ghost_markers():
            cache.push_ghost_undo_snapshot()
        context.scene.onion_skin_objects.clear()
        cache.clear_frame_cache()
        cache.clear_rigid_cache()
        cache.clear_rigid_matrix_cache()
        cache.clear_onion_batches()
        cache.clear_ghost_markers(keep_undo=True)
        context.scene.onion_skin_settings.marker_count = 0
        context.scene.onion_skin_active_index = 0
        properties._safe_redraw(context)
        return {'FINISHED'}


class ONION_OT_add_marker(Operator):
    """Pin current frame as a persistent ghost overlay"""
    bl_idname = "onion_skin.add_marker"
    bl_label = "Add Ghost"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.onion_skin_objects) > 0

    def execute(self, context):
        from . import drawing, properties

        scene = context.scene
        global_settings = scene.onion_skin_settings
        frame = scene.frame_current

        view_layer = getattr(context, 'view_layer', None)
        vis_cols = drawing._get_visible_collections(view_layer) if view_layer is not None else None
        space = getattr(context, 'space_data', None)
        v3d = space if (space is not None and getattr(space, 'type', None) == 'VIEW_3D') else None

        targets = []
        target_settings = {}
        seen = set()
        selected_objs = set(context.selected_objects)
        matched_selected_tracked = False

        for item in scene.onion_skin_objects:
            obj = item.object
            if not obj:
                continue
            s_cfg = properties.get_object_settings(obj, item)
            if selected_objs:
                is_sel = obj in selected_objs
                if not is_sel and obj.type == 'ARMATURE' and getattr(s_cfg, 'include_children', True):
                    cands = drawing._get_armature_candidate_meshes(obj, s_cfg, scene=scene)
                    is_sel = any(m in selected_objs for m in cands)
                if not is_sel:
                    continue
                matched_selected_tracked = True
            if not item.visible or obj in seen:
                continue
            if not drawing._is_object_visible_in_viewport(
                obj, s_cfg, view_layer, v3d=v3d, scene=scene, vis_cols=vis_cols
            ):
                continue
            targets.append(obj)
            target_settings[obj] = s_cfg
            seen.add(obj)

        if not targets and not matched_selected_tracked:
            for item in scene.onion_skin_objects:
                obj = item.object
                if not obj or not item.visible or obj in seen:
                    continue
                s_cfg = properties.get_object_settings(obj, item)
                if not drawing._is_object_visible_in_viewport(
                    obj, s_cfg, view_layer, v3d=v3d, scene=scene, vis_cols=vis_cols
                ):
                    continue
                targets.append(obj)
                target_settings[obj] = s_cfg
                seen.add(obj)

        if not targets:
            msg = "Selected object(s) are hidden or disabled" if matched_selected_tracked else "No objects to capture"
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()

        idx = scene.onion_skin_active_index
        use_wireframe = target_settings[targets[0]].use_wireframe
        if 0 <= idx < len(scene.onion_skin_objects):
            active_item = scene.onion_skin_objects[idx]
            active_obj = active_item.object
            if active_obj in target_settings:
                use_wireframe = target_settings[active_obj].use_wireframe

        res = drawing.extract_tracked_objects_merged(
            targets, depsgraph, use_wireframe, need_normals=False,
            view_layer=view_layer, vis_cols=vis_cols, v3d=v3d,
        )
        verts, normals, indices, prim_type = res[0], res[1], res[2], res[3]
        aabb_min = res[4] if len(res) > 4 else None
        aabb_max = res[5] if len(res) > 5 else None
        centroid = res[6] if len(res) > 6 else None

        if verts is None or indices is None:
            self.report({'WARNING'}, "No mesh data at this frame")
            return {'CANCELLED'}

        has_lv = v3d is not None and getattr(v3d, 'local_view', None) is not None
        obj_names = []
        for o in targets:
            s_cfg = target_settings[o]
            if has_lv and o.type == 'ARMATURE' and getattr(s_cfg, 'include_children', True):
                lv_parts = drawing.resolve_mesh_objects(
                    o, s_cfg, scene=scene, view_layer=view_layer, vis_cols=vis_cols, v3d=v3d
                )
                if lv_parts:
                    obj_names.extend(drawing.get_object_key(p) for p in lv_parts)
                else:
                    obj_names.append(drawing.get_object_key(o))
            else:
                obj_names.append(drawing.get_object_key(o))
        cache.add_ghost_marker(
            frame, verts, indices, prim_type, obj_names,
            normals=normals, aabb_min=aabb_min, aabb_max=aabb_max, centroid=centroid
        )
        global_settings.marker_count = len(cache.get_ghost_markers())
        properties._safe_redraw(context)

        self.report({'INFO'}, f"Ghost added at frame {frame}")
        return {'FINISHED'}


class ONION_OT_remove_markers(Operator):
    """Remove ghost at current frame for selected objects"""
    bl_idname = "onion_skin.remove_markers"
    bl_label = "Remove Ghost"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.onion_skin_settings.marker_count > 0

    def execute(self, context):
        from . import drawing, properties

        scene = context.scene
        frame = scene.frame_current

        # Collect keys/names for the selected onion-skin objects
        selected_objs = set(context.selected_objects)
        obj_names = set()

        for item in scene.onion_skin_objects:
            obj = item.object
            if not obj:
                continue
            obj_settings = properties.get_object_settings(obj, item)
            matched = obj in selected_objs
            if not matched and obj.type == 'ARMATURE' and getattr(obj_settings, 'include_children', True):
                cands = drawing._get_armature_candidate_meshes(obj, obj_settings, scene=scene)
                matched = any(m in selected_objs for m in cands)
            if not matched:
                continue
            obj_names.update(_collect_object_marker_keys(obj, obj_settings, scene))

        if not obj_names:
            # Fallback: if no tracked object selected, target all tracked objects at current frame
            for item in scene.onion_skin_objects:
                obj = item.object
                if obj:
                    obj_settings = properties.get_object_settings(obj, item)
                    obj_names.update(_collect_object_marker_keys(obj, obj_settings, scene))

        if cache.get_ghost_markers():
            cache.push_ghost_undo_snapshot()
        removed = cache.remove_ghost_markers_for(frame, obj_names)
        scene.onion_skin_settings.marker_count = len(cache.get_ghost_markers())
        properties._safe_redraw(context)

        if removed:
            self.report({'INFO'}, f"Removed {removed} ghost(s) at frame {frame}")
        else:
            self.report({'INFO'}, f"No ghost found at frame {frame} for selected objects")
        return {'FINISHED'}


class ONION_OT_hide_all_ghosts(Operator):
    """Toggle visibility of all ghost overlays"""
    bl_idname = "onion_skin.hide_all_ghosts"
    bl_label = "Hide All Ghosts"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.onion_skin_settings.marker_count > 0

    def execute(self, context):
        from . import properties
        settings = context.scene.onion_skin_settings
        settings.show_ghosts = not settings.show_ghosts
        properties._safe_redraw(context)

        state = "visible" if settings.show_ghosts else "hidden"
        self.report({'INFO'}, f"All ghosts {state}")
        return {'FINISHED'}


class ONION_OT_remove_all_ghosts(Operator):
    """Remove all ghost overlays"""
    bl_idname = "onion_skin.remove_all_ghosts"
    bl_label = "Remove All Ghosts"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.onion_skin_settings.marker_count > 0

    def execute(self, context):
        from . import properties
        if cache.get_ghost_markers():
            cache.push_ghost_undo_snapshot()
        cache.clear_ghost_markers(keep_undo=True)
        context.scene.onion_skin_settings.marker_count = 0
        context.scene.onion_skin_settings.show_ghosts = True
        properties._safe_redraw(context)

        self.report({'INFO'}, "All ghosts removed")
        return {'FINISHED'}


classes = (
    ONION_OT_add_object,
    ONION_OT_add_from_picker,
    ONION_OT_remove_object,
    ONION_OT_remove_selected,
    ONION_OT_clear_all,
    ONION_OT_add_marker,
    ONION_OT_remove_markers,
    ONION_OT_hide_all_ghosts,
    ONION_OT_remove_all_ghosts,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
