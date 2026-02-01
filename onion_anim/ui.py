# SPDX-License-Identifier: GPL-3.0-or-later
"""UI Panels for onion skin addon."""

import bpy
from bpy.types import Panel

from . import cache


class ONION_PT_main(Panel):
    """Main panel for onion skin settings"""
    bl_label = "B Onion Skin"
    bl_idname = "ONION_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion Skin'
    
    def draw_header(self, context):
        settings = context.scene.onion_skin_settings
        self.layout.prop(settings, "enabled", text="")
    
    def draw(self, context):
        pass


class ONION_PT_objects(Panel):
    """Objects panel"""
    bl_label = "Objects"
    bl_idname = "ONION_PT_objects"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion Skin'
    bl_parent_id = "ONION_PT_main"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.onion_skin_settings
        
        # Add selected button
        layout.operator("onion_skin.add_object", text="Add Selected", icon='ADD')
        
        # Object eyedropper
        row = layout.row(align=True)
        row.prop(settings, "pick_object", text="")
        sub = row.row(align=True)
        sub.enabled = settings.pick_object is not None
        sub.operator("onion_skin.add_from_picker", text="", icon='CHECKMARK')
        
        # Object list
        if scene.onion_skin_objects:
            box = layout.box()
            col = box.column(align=True)
            for i, item in enumerate(scene.onion_skin_objects):
                row = col.row(align=True)
                if item.object:
                    icon = 'ARMATURE_DATA' if item.object.type == 'ARMATURE' else 'MESH_DATA'
                    row.label(text=item.object.name, icon=icon)
                else:
                    row.label(text="[Missing]", icon='ERROR')
                op = row.operator("onion_skin.remove_object", text="", icon='X', emboss=False)
                op.index = i
            
            layout.operator("onion_skin.clear_all", text="Clear All", icon='TRASH')
        
        # Include children option
        layout.prop(settings, "include_children")


class ONION_PT_frames(Panel):
    """Frame range panel"""
    bl_label = "Frame Range"
    bl_idname = "ONION_PT_frames"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion Skin'
    bl_parent_id = "ONION_PT_main"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.onion_skin_settings
        
        col = layout.column(align=True)
        
        row = col.row(align=True)
        row.prop(settings, "show_before", text="", icon='TRIA_LEFT')
        row.prop(settings, "frames_before", text="Before")
        
        row = col.row(align=True)
        row.prop(settings, "show_after", text="", icon='TRIA_RIGHT')
        row.prop(settings, "frames_after", text="After")
        
        layout.prop(settings, "frame_step")
        
        layout.separator()
        layout.prop(settings, "use_frame_range")
        if settings.use_frame_range:
            row = layout.row(align=True)
            row.prop(settings, "frame_range_start", text="Start")
            row.prop(settings, "frame_range_end", text="End")


class ONION_PT_appearance(Panel):
    """Appearance panel"""
    bl_label = "Appearance"
    bl_idname = "ONION_PT_appearance"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion Skin'
    bl_parent_id = "ONION_PT_main"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.onion_skin_settings
        
        # Colors
        row = layout.row()
        col = row.column()
        col.label(text="Before")
        col.prop(settings, "color_before", text="")
        col = row.column()
        col.label(text="After")
        col.prop(settings, "color_after", text="")
        
        layout.separator()
        
        layout.prop(settings, "opacity_falloff", text="Fade")
        layout.prop(settings, "falloff_curve", text="")
        
        layout.separator()
        
        row = layout.row(align=True)
        row.prop(settings, "use_xray", toggle=True)
        row.prop(settings, "use_wireframe", toggle=True)
        row.prop(settings, "show_mesh_infront", toggle=True)


class ONION_PT_cache(Panel):
    """Cache panel"""
    bl_label = "Cache"
    bl_idname = "ONION_PT_cache"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion Skin'
    bl_parent_id = "ONION_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw_header(self, context):
        stats = cache.get_cache_stats()
        self.layout.label(text=f"{stats['size']}")
    
    def draw(self, context):
        layout = self.layout
        stats = cache.get_cache_stats()
        
        usage = stats['size'] / stats['max_size'] if stats['max_size'] > 0 else 0
        layout.progress(factor=usage, type='BAR', text=f"{stats['size']} / {stats['max_size']}")
        
        row = layout.row(align=True)
        row.operator("onion_skin.bake_cache", text="Bake", icon='RENDER_ANIMATION')
        row.operator("onion_skin.rebake_cache", text="Rebake", icon='FILE_REFRESH')
        
        layout.operator("onion_skin.clear_cache", text="Clear", icon='TRASH')


classes = (
    ONION_PT_main,
    ONION_PT_objects,
    ONION_PT_frames,
    ONION_PT_appearance,
    ONION_PT_cache,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
