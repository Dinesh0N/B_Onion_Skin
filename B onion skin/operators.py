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
        return context.selected_objects
    
    def execute(self, context):
        existing = {item.object for item in context.scene.onion_skin_objects if item.object}
        added = 0
        
        for obj in context.selected_objects:
            if obj.type in {'MESH', 'ARMATURE'} and obj not in existing:
                context.scene.onion_skin_objects.add().object = obj
                added += 1
        
        if added > 0:
            cache.clear_cache()
        
        self.report({'INFO'}, f"Added {added} object(s)")
        return {'FINISHED'}


class ONION_OT_add_from_picker(Operator):
    """Add object from eyedropper selection"""
    bl_idname = "onion_skin.add_from_picker"
    bl_label = "Add Picked Object"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        settings = context.scene.onion_skin_settings
        obj = settings.pick_object
        
        if not obj:
            return {'CANCELLED'}
        
        if obj.type not in {'MESH', 'ARMATURE'}:
            self.report({'WARNING'}, "Only mesh or armature objects supported")
            settings.pick_object = None
            return {'CANCELLED'}
        
        existing = {item.object for item in context.scene.onion_skin_objects if item.object}
        
        if obj not in existing:
            context.scene.onion_skin_objects.add().object = obj
            cache.clear_cache()
            self.report({'INFO'}, f"Added: {obj.name}")
        else:
            self.report({'INFO'}, f"Already added: {obj.name}")
        
        settings.pick_object = None
        return {'FINISHED'}


class ONION_OT_remove_object(Operator):
    """Remove object from onion skin list"""
    bl_idname = "onion_skin.remove_object"
    bl_label = "Remove"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty()
    
    def execute(self, context):
        if 0 <= self.index < len(context.scene.onion_skin_objects):
            context.scene.onion_skin_objects.remove(self.index)
            cache.clear_cache()
        return {'FINISHED'}


class ONION_OT_clear_all(Operator):
    """Remove all objects"""
    bl_idname = "onion_skin.clear_all"
    bl_label = "Clear All"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        context.scene.onion_skin_objects.clear()
        cache.clear_cache()
        return {'FINISHED'}


class ONION_OT_remove_selected(Operator):
    """Remove selected object from list"""
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
            scene.onion_skin_objects.remove(index)
            # Adjust active index if needed
            if index >= len(scene.onion_skin_objects) and len(scene.onion_skin_objects) > 0:
                scene.onion_skin_active_index = len(scene.onion_skin_objects) - 1
            cache.clear_cache()
        return {'FINISHED'}


class ONION_OT_clear_cache(Operator):
    """Clear the mesh cache"""
    bl_idname = "onion_skin.clear_cache"
    bl_label = "Clear Cache"
    
    def execute(self, context):
        cache.clear_cache()
        self.report({'INFO'}, "Cache cleared")
        return {'FINISHED'}


class ONION_OT_bake_cache(Operator):
    """Pre-bake all frames for instant playback"""
    bl_idname = "onion_skin.bake_cache"
    bl_label = "Bake Cache"
    
    @classmethod
    def poll(cls, context):
        return len(context.scene.onion_skin_objects) > 0
    
    def execute(self, context):
        from . import drawing
        
        scene = context.scene
        settings = scene.onion_skin_settings
        
        start, end = drawing.get_frame_range(settings, scene)
        step = settings.frame_step
        
        original_frame = scene.frame_current
        baked = 0
        
        for frame in range(start, end + 1, step):
            if not cache.is_frame_cached(frame):
                if drawing.cache_frame(context, frame):
                    baked += 1
        
        scene.frame_set(original_frame)
        self.report({'INFO'}, f"Baked {baked} frames")
        return {'FINISHED'}


class ONION_OT_rebake_cache(Operator):
    """Clear and rebake all frames"""
    bl_idname = "onion_skin.rebake_cache"
    bl_label = "Rebake Cache"
    
    @classmethod
    def poll(cls, context):
        return len(context.scene.onion_skin_objects) > 0
    
    def execute(self, context):
        from . import drawing
        
        scene = context.scene
        settings = scene.onion_skin_settings
        
        cache.clear_cache()
        
        start, end = drawing.get_frame_range(settings, scene)
        step = settings.frame_step
        
        original_frame = scene.frame_current
        
        for frame in range(start, end + 1, step):
            drawing.cache_frame(context, frame)
        
        scene.frame_set(original_frame)
        self.report({'INFO'}, f"Rebaked {cache.get_cache_size()} frames")
        return {'FINISHED'}


classes = (
    ONION_OT_add_object,
    ONION_OT_add_from_picker,
    ONION_OT_remove_object,
    ONION_OT_remove_selected,
    ONION_OT_clear_all,
    ONION_OT_clear_cache,
    ONION_OT_bake_cache,
    ONION_OT_rebake_cache,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
