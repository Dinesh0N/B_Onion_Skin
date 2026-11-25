# SPDX-License-Identifier: GPL-3.0-or-later
"""Property definitions for onion skin addon."""

import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    BoolProperty, IntProperty, FloatProperty,
    FloatVectorProperty, PointerProperty, EnumProperty
)


def _safe_redraw(context):
    """Safely trigger viewport redraw across all windows."""
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


_is_batch_updating = False


def _update_redraw(self, context):
    """Trigger a viewport redraw and rebuild onion skin batches."""
    if _is_batch_updating:
        return
    from . import drawing
    try:
        drawing.bump_vis_cache_rev()
        scene = context.scene
        if hasattr(scene, 'onion_skin_settings') and scene.onion_skin_settings.enabled:
            depsgraph = context.evaluated_depsgraph_get()
            drawing._build_onion_batches(context, scene.frame_current, depsgraph)
        else:
            from . import cache
            cache.clear_onion_batches()
    except Exception:
        pass
    _safe_redraw(context)


def _update_enabled(self, context):
    """Handle global Onion Skin enable/disable cleanly, flushing stale caches if scene edited while disabled."""
    if _is_batch_updating:
        return
    from . import drawing, cache
    drawing.bump_vis_cache_rev()
    if getattr(self, 'enabled', True):
        if getattr(drawing, '_dirty_while_disabled', False):
            drawing._dirty_while_disabled = False
            cache.clear_frame_cache()
            cache.clear_rigid_cache()
            cache.clear_rigid_matrix_cache()
            drawing._clear_object_action_sigs()
            drawing._clear_ghost_slots()
            drawing.clear_topology_cache()
            drawing._cached_signatures.clear()
        _update_redraw(self, context)
    else:
        cache.clear_onion_batches()
        _safe_redraw(context)


def _just_redraw(self, context):
    """Trigger a viewport redraw without rebuilding batches (for visual properties)."""
    if _is_batch_updating:
        return
    from . import drawing
    drawing.bump_vis_cache_rev()
    _safe_redraw(context)


def _resolve_owner_object_key(self, context):
    """Resolve owning object key for ONION_ObjectItem or ONION_ObjectSettings (including fallback)."""
    from . import drawing
    obj = getattr(self, 'object', None)
    if obj is None:
        id_data = getattr(self, 'id_data', None)
        if isinstance(id_data, bpy.types.Object):
            obj = id_data
        elif isinstance(id_data, bpy.types.Scene):
            for it in getattr(id_data, 'onion_skin_objects', ()):
                if getattr(it, 'settings_fallback', None) == self:
                    obj = it.object
                    break
    return drawing.get_object_key(obj) if obj else None


def get_object_settings(obj, item=None):
    """Return writable per-object onion skin settings, falling back to item storage for read-only linked IDs."""
    if obj is not None:
        onion_settings = getattr(obj, 'onion_skin', None)
        if onion_settings is not None:
            try:
                if not onion_settings.is_property_readonly("show_before"):
                    return onion_settings
            except Exception:
                return onion_settings
    if item is not None:
        fb = getattr(item, 'settings_fallback', None)
        if fb is not None:
            return fb
    if obj is not None:
        try:
            sc = getattr(bpy.context, 'scene', None)
            if sc is not None:
                for it in getattr(sc, 'onion_skin_objects', ()):
                    if it.object == obj and getattr(it, 'settings_fallback', None) is not None:
                        return it.settings_fallback
        except Exception:
            pass
        return getattr(obj, 'onion_skin', None)
    return None


def _update_geometry_settings(self, context):
    """Clear frame cache selectively and rebuild."""
    if _is_batch_updating:
        return
    from . import cache, drawing
    drawing.bump_vis_cache_rev()
    obj_key = _resolve_owner_object_key(self, context)
    if obj_key:
        cache.clear_frame_cache(obj_key)
        cache.clear_rigid_cache(obj_key)
        cache.clear_rigid_matrix_cache(obj_key)
        drawing._clear_object_action_sigs(obj_key)
        drawing._clear_ghost_slots(obj_key)
    else:
        cache.clear_frame_cache()
        cache.clear_rigid_cache()
        cache.clear_rigid_matrix_cache()
        drawing._clear_object_action_sigs()
        drawing._clear_ghost_slots()
    _update_redraw(self, context)


class ONION_ObjectSettings(PropertyGroup):
    """Per-object settings for the onion skin system."""

    show_before: BoolProperty(
        name="Show Before",
        description="Show frames before current",
        default=True,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_redraw
    )

    show_after: BoolProperty(
        name="Show After",
        description="Show frames after current",
        default=True,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_redraw
    )

    frames_before: IntProperty(
        name="Frames Before",
        description="Number of ghost frames before current",
        default=3,
        min=1,
        soft_max=50,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_redraw
    )

    frames_after: IntProperty(
        name="Frames After",
        description="Number of ghost frames after current",
        default=3,
        min=1,
        soft_max=50,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_redraw
    )

    frame_step: IntProperty(
        name="Frame Step",
        description="Interval between ghost frames",
        default=2,
        min=1,
        soft_max=10,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_redraw
    )

    color_before: FloatVectorProperty(
        name="Before Color",
        description="Color for frames before current",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.2, 0.5, 1.0, 0.5),
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    color_after: FloatVectorProperty(
        name="After Color",
        description="Color for frames after current",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.3, 0.2, 0.5),
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    opacity_falloff: FloatProperty(
        name="Opacity Falloff",
        description="How quickly ghosts fade with distance",
        default=0.6,
        min=0.0,
        soft_max=1.0,
        subtype='FACTOR',
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    falloff_curve: EnumProperty(
        name="Falloff Curve",
        description="Shape of the opacity falloff",
        items=[
            ('linear', "Linear", "Straight line falloff"),
            ('smooth', "Smooth", "Natural S-curve falloff"),
            ('exponential', "Exponential", "Fast fade at distance"),
        ],
        default='smooth',
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    use_xray: BoolProperty(
        name="X-Ray",
        description="Display ghosts through other geometry",
        default=False,
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    use_wireframe: BoolProperty(
        name="Wireframe",
        description="Draw ghosts as wireframe",
        default=False,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_geometry_settings
    )

    use_fresnel: BoolProperty(
        name="Fresnel",
        description="Use a Fresnel-based shader for a glowing edge effect",
        default=False,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_geometry_settings
    )

    fresnel_power: FloatProperty(
        name="Fresnel Exponent",
        description="Controls how quickly the Fresnel effect fades towards the center",
        default=2.0,
        min=0.1,
        max=10.0,
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    fresnel_fade: FloatProperty(
        name="Fresnel Fade",
        description="Controls the strength of the Fresnel fade-out at the center (0.0 = solid color, 1.0 = pure outline)",
        default=1.0,
        min=0.0,
        max=1.0,
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    use_shading: BoolProperty(
        name="Shading (Specular & Shadow)",
        description="Render onion skin ghosts with 3D shadow, contour darkening, and specular highlights",
        default=True,
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    shadow_strength: FloatProperty(
        name="Shadow",
        description="Strength of 3D directional shadow and contour darkening on ghosts",
        default=0.65,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    specular_strength: FloatProperty(
        name="Specular",
        description="Intensity of 3D specular highlights on ghost surfaces",
        default=0.35,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    specular_shininess: FloatProperty(
        name="Shininess",
        description="Sharpness of the 3D specular highlight on ghost surfaces",
        default=24.0,
        min=2.0,
        max=128.0,
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    use_screen_space_spacing: BoolProperty(
        name="Screen Space Spacing",
        description="Space onion skin ghosts based on 2D viewport screen-space distance and offset",
        default=False,
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    screen_space_mode: EnumProperty(
        name="Spacing Mode",
        description="How screen-space spacing is applied to onion skin ghosts",
        items=[
            ('SPREAD', "Spread", "Spread ghosts evenly across screen horizontal axis by scene unit distance (meters)"),
            ('FILTER', "Adaptive Filter", "Filter frames so ghosts appear only when separated by the minimum screen pixel distance"),
        ],
        default='SPREAD',
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    screen_space_spread_distance: FloatProperty(
        name="Spacing",
        description="Distance between consecutive ghosts along the screen horizontal axis in scene units (meters)",
        default=1.5,
        min=0.0,
        soft_max=20.0,
        max=1000.0,
        subtype='DISTANCE',
        unit='LENGTH',
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    screen_space_distance: FloatProperty(
        name="Min Distance",
        description="Minimum screen pixel distance between displayed ghosts in Adaptive Filter mode",
        default=85.0,
        min=0.0,
        soft_max=400.0,
        max=2000.0,
        subtype='PIXEL',
        override={'LIBRARY_OVERRIDABLE'},
        update=_just_redraw
    )

    include_children: BoolProperty(
        name="Include Children",
        description="Include mesh children of armatures and objects inside collection instances",
        default=True,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_geometry_settings
    )

    limit_to_keyframes: BoolProperty(
        name="Limit to Keyframes",
        description="Hide onion skins before the first and after the last keyframe",
        default=True,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_redraw
    )

    use_frame_range: BoolProperty(
        name="Limit to Range",
        description="Only show onion skin within specified frame range",
        default=False,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_redraw
    )

    frame_range_start: IntProperty(
        name="Start",
        description="Start frame for onion skin display",
        default=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_redraw
    )

    frame_range_end: IntProperty(
        name="End",
        description="End frame for onion skin display",
        default=250,
        override={'LIBRARY_OVERRIDABLE'},
        update=_update_redraw
    )


class ONION_ObjectItem(PropertyGroup):
    """Object reference in the onion skin list."""

    object: PointerProperty(
        name="Object",
        type=bpy.types.Object,
        description="Object to display onion skin for",
        update=_update_geometry_settings
    )

    visible: BoolProperty(
        name="Visible",
        description="Show onion skin for this object",
        default=True,
        update=_update_redraw
    )

    settings_fallback: PointerProperty(
        name="Fallback Settings",
        type=ONION_ObjectSettings,
        description="Per-item settings used when the target object is read-only linked"
    )


class ONION_SceneSettings(PropertyGroup):
    """Global settings for the onion skin system."""

    enabled: BoolProperty(
        name="Enable",
        description="Toggle onion skin display",
        default=True,
        update=_update_enabled
    )

    pick_object: PointerProperty(
        name="Pick Object",
        type=bpy.types.Object,
        description="Click eyedropper to pick an object"
    )

    # Ghost frames
    marker_count: IntProperty(
        name="Marker Count",
        description="Number of active ghost frames",
        default=0,
        min=0,
    )

    ghost_color: FloatVectorProperty(
        name="Ghost Color",
        description="Color for pinned ghost overlays",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.2, 0.8, 0.3, 0.5),
    )

    show_ghosts: BoolProperty(
        name="Show Ghosts",
        description="Toggle visibility of all ghost overlays without removing them",
        default=True,
        update=_just_redraw
    )

    show_ghost_labels: BoolProperty(
        name="Frame Labels",
        description="Show frame numbers on ghost overlays",
        default=True,
        update=_update_redraw
    )


classes = (
    ONION_ObjectSettings,
    ONION_ObjectItem,
    ONION_SceneSettings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.onion_skin_settings = PointerProperty(type=ONION_SceneSettings)
    bpy.types.Object.onion_skin = PointerProperty(
        type=ONION_ObjectSettings,
        override={'LIBRARY_OVERRIDABLE'},
    )
    bpy.types.Scene.onion_skin_objects = bpy.props.CollectionProperty(type=ONION_ObjectItem)
    bpy.types.Scene.onion_skin_active_index = bpy.props.IntProperty(name="Active Object Index")


def unregister():
    try:
        del bpy.types.Scene.onion_skin_active_index
        del bpy.types.Scene.onion_skin_objects
        del bpy.types.Object.onion_skin
        del bpy.types.Scene.onion_skin_settings
    except Exception:
        pass

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
