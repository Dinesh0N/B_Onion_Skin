# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender event handlers for onion skin addon - with intelligent cache invalidation."""

import bpy
from bpy.app.handlers import persistent

from . import cache
from . import async_cache


# Track objects being monitored for changes
_monitored_objects = set()


@persistent
def on_load(dummy):
    """Handle file load - clear cache."""
    cache.clear_cache()
    _monitored_objects.clear()


@persistent
def on_depsgraph_update(scene, depsgraph):
    """Handle depsgraph updates - invalidate cache for modified objects."""
    if not hasattr(scene, 'onion_skin_settings'):
        return
    
    settings = scene.onion_skin_settings
    if not settings.enabled:
        return
    
    # Get our monitored objects
    monitored = set()
    for item in scene.onion_skin_objects:
        if item.object:
            monitored.add(item.object.name)
            # If including armature children, add them too
            if item.object.type == 'ARMATURE' and settings.include_children:
                for child in item.object.children:
                    if child.type == 'MESH':
                        monitored.add(child.name)
    
    # Check if any monitored objects were updated
    needs_invalidate = False
    
    for update in depsgraph.updates:
        if hasattr(update, 'id'):
            obj_id = update.id
            if hasattr(obj_id, 'name') and obj_id.name in monitored:
                # Check if geometry changed
                if update.is_updated_geometry or update.is_updated_transform:
                    needs_invalidate = True
                    break
    
    if needs_invalidate:
        # Only invalidate frames near current frame (performance optimization)
        # Don't clear entire cache, just mark frames as dirty
        cache.mark_all_dirty()


@persistent 
def on_frame_change(scene, depsgraph=None):
    """Handle frame change - trigger viewport update and background caching."""
    if not hasattr(scene, 'onion_skin_settings'):
        return
    
    settings = scene.onion_skin_settings
    if not settings.enabled:
        return
    
    # Tag redraw for all 3D viewports
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    
    # Start background pre-caching for neighboring frames
    try:
        async_cache.start_background_caching(bpy.context)
    except Exception:
        pass


def register():
    if on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load)
    
    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    
    if on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(on_frame_change)


def unregister():
    if on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load)
    
    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
    
    if on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(on_frame_change)
