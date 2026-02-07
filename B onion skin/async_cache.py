# SPDX-License-Identifier: GPL-3.0-or-later
"""Async background caching for onion skin - pre-cache frames during idle time."""

import bpy
from . import cache

# Background caching state
_timer = None
_pending_frames = []
_is_caching = False
_last_playhead = -999


def get_precache_frames(context, direction=0):
    """Get frames to pre-cache based on playback direction.
    
    Args:
        context: Blender context
        direction: Playback direction (negative = backwards, positive = forwards, 0 = both)
    
    Returns:
        List of frame numbers to cache
    """
    from . import drawing
    
    scene = context.scene
    settings = scene.onion_skin_settings
    current = scene.frame_current
    
    range_start, range_end = drawing.get_frame_range(settings, scene)
    step = settings.frame_step
    
    frames = []
    
    # Pre-cache in both directions or based on playback direction
    look_ahead = max(settings.frames_before, settings.frames_after) + 5
    
    if direction >= 0:
        # Cache frames ahead (forward playback)
        for i in range(step, look_ahead + 1, step):
            f = current + i
            if f <= range_end and not cache.is_frame_cached(f):
                frames.append(f)
    
    if direction <= 0:
        # Cache frames behind (backward playback)
        for i in range(step, look_ahead + 1, step):
            f = current - i
            if f >= range_start and not cache.is_frame_cached(f):
                frames.append(f)
    
    # Sort by distance from current (closest first)
    frames.sort(key=lambda f: abs(f - current))
    
    return frames[:10]  # Limit batch size


def _background_cache_step():
    """Timer callback to cache one frame per step."""
    global _pending_frames, _is_caching, _timer
    
    context = bpy.context
    
    # Check if we should continue
    if not context or not hasattr(context, 'scene'):
        _is_caching = False
        return None  # Stop timer
    
    scene = context.scene
    if not hasattr(scene, 'onion_skin_settings'):
        _is_caching = False
        return None
    
    settings = scene.onion_skin_settings
    if not settings.enabled or not scene.onion_skin_objects:
        _is_caching = False
        return None
    
    # If pending frames are exhausted, get more
    if not _pending_frames:
        _pending_frames = get_precache_frames(context)
        if not _pending_frames:
            _is_caching = False
            return None  # No more frames to cache
    
    # Cache one frame
    from . import drawing
    
    frame = _pending_frames.pop(0)
    original_frame = scene.frame_current
    
    try:
        drawing.cache_frame(context, frame)
    except Exception:
        pass
    
    # Restore frame if changed
    if scene.frame_current != original_frame:
        scene.frame_set(original_frame)
    
    # Continue timer if more frames pending
    if _pending_frames:
        return 0.05  # 50ms between frames
    else:
        _is_caching = False
        return None


def start_background_caching(context):
    """Start background caching timer."""
    global _timer, _pending_frames, _is_caching
    
    if _is_caching:
        return  # Already running
    
    if not hasattr(context.scene, 'onion_skin_settings'):
        return
    
    settings = context.scene.onion_skin_settings
    if not settings.enabled:
        return
    
    _pending_frames = get_precache_frames(context)
    if not _pending_frames:
        return
    
    _is_caching = True
    bpy.app.timers.register(_background_cache_step, first_interval=0.1)


def stop_background_caching():
    """Stop background caching."""
    global _is_caching, _pending_frames
    _is_caching = False
    _pending_frames.clear()


def is_caching():
    """Check if background caching is active."""
    return _is_caching


def cleanup():
    """Cleanup on unregister."""
    stop_background_caching()
