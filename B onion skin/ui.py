# SPDX-License-Identifier: GPL-3.0-or-later
"""UI Panels for onion skin addon."""

import bpy
from bpy.types import Panel, UIList

from . import cache
from . import async_cache


class ONION_UL_objects(UIList):
    """UIList for onion skin objects"""
    bl_idname = "ONION_UL_objects"
    
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if item.object:
                obj_icon = 'ARMATURE_DATA' if item.object.type == 'ARMATURE' else 'MESH_DATA'
                layout.label(text=item.object.name, icon=obj_icon)
            else:
                layout.label(text="[Missing]", icon='ERROR')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            if item.object:
                obj_icon = 'ARMATURE_DATA' if item.object.type == 'ARMATURE' else 'MESH_DATA'
                layout.label(text="", icon=obj_icon)
            else:
                layout.label(text="", icon='ERROR')


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
        
        # UIList with side buttons
        row = layout.row()
        
        # Left side: the list
        row.template_list(
            "ONION_UL_objects", "",
            scene, "onion_skin_objects",
            scene, "onion_skin_active_index",
            rows=4
        )
        
        # Right side: add/remove buttons
        col = row.column(align=True)
        col.operator("onion_skin.add_object", text="", icon='ADD')
        col.operator("onion_skin.remove_selected", text="", icon='REMOVE')
        col.operator("onion_skin.clear_all", text="", icon='TRASH')
        
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
        # Show caching indicator if background caching is active
        if async_cache.is_caching():
            self.layout.label(text=f"⟳ {stats['size']}")
        else:
            self.layout.label(text=f"{stats['size']}")
    
    def draw(self, context):
        layout = self.layout
        stats = cache.get_cache_stats()
        
        # Progress bar
        usage = stats['size'] / stats['max_size'] if stats['max_size'] > 0 else 0
        layout.progress(factor=usage, type='BAR', text=f"{stats['size']} / {stats['max_size']}")
        
        # Hit rate
        row = layout.row()
        row.label(text=f"Hit Rate: {stats['hit_rate']:.0f}%")
        if stats.get('dirty', 0) > 0:
            row.label(text=f"Dirty: {stats['dirty']}")
        
        row = layout.row(align=True)
        row.operator("onion_skin.bake_cache", text="Bake", icon='RENDER_ANIMATION')
        row.operator("onion_skin.rebake_cache", text="Rebake", icon='FILE_REFRESH')
        
        layout.operator("onion_skin.clear_cache", text="Clear", icon='TRASH')


classes = (
    ONION_UL_objects,
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
