# SPDX-License-Identifier: GPL-3.0-or-later
"""Cache management for onion skin - Ultra Performance v2."""

from collections import OrderedDict
import gpu
from gpu_extras.batch import batch_for_shader

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# -----------------------------------------------------------------------------
# Global Cache
# -----------------------------------------------------------------------------

_shader = None
# {frame: (verts_np, indices_np, prim_type)} - Store numpy arrays directly
_frame_cache = OrderedDict()
_batch_cache = {}  # {frame: gpu_batch} - lazy built
_max_cache_size = 500
_last_frame = -999

# Statistics
_cache_hits = 0
_cache_misses = 0

# Dirty tracking - frames that need to be recached
_dirty_frames = set()

# Object hash tracking - detect when objects change
_object_hashes = {}


def get_shader():
    """Get or create the GPU shader."""
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    return _shader


def clear_cache():
    """Clear all cached data."""
    global _frame_cache, _batch_cache, _last_frame, _cache_hits, _cache_misses
    global _dirty_frames, _object_hashes
    _frame_cache.clear()
    _batch_cache.clear()
    _last_frame = -999
    _cache_hits = 0
    _cache_misses = 0
    _dirty_frames.clear()
    _object_hashes.clear()


def get_last_frame():
    return _last_frame


def set_last_frame(frame):
    global _last_frame
    _last_frame = frame


def is_frame_cached(frame):
    """Check if frame is cached and not dirty."""
    if frame in _dirty_frames:
        return False
    return frame in _frame_cache


def mark_frame_dirty(frame):
    """Mark a specific frame as needing recache."""
    global _dirty_frames
    _dirty_frames.add(frame)
    # Remove from batch cache if exists
    if frame in _batch_cache:
        del _batch_cache[frame]


def mark_all_dirty():
    """Mark all cached frames as dirty."""
    global _dirty_frames
    _dirty_frames.update(_frame_cache.keys())
    _batch_cache.clear()


def add_to_cache(frame, verts, indices, prim_type):
    """Add mesh data to cache. verts/indices can be numpy arrays or lists."""
    global _frame_cache, _batch_cache, _cache_misses, _dirty_frames
    
    if verts is None or indices is None:
        return
    
    # Handle empty data
    if HAS_NUMPY:
        if isinstance(verts, np.ndarray):
            if verts.size == 0:
                return
        elif len(verts) == 0:
            return
    else:
        if len(verts) == 0 or len(indices) == 0:
            return
    
    _cache_misses += 1
    
    # Store data directly (numpy arrays or lists)
    _frame_cache[frame] = (verts, indices, prim_type)
    _frame_cache.move_to_end(frame)
    
    # Clear dirty flag
    _dirty_frames.discard(frame)
    
    # Clear batch cache for this frame (will rebuild on draw)
    if frame in _batch_cache:
        del _batch_cache[frame]
    
    # LRU eviction
    while len(_frame_cache) > _max_cache_size:
        old_frame = next(iter(_frame_cache))
        del _frame_cache[old_frame]
        if old_frame in _batch_cache:
            del _batch_cache[old_frame]


def get_batch(frame):
    """Get GPU batch for frame, building lazily if needed."""
    global _batch_cache, _cache_hits
    
    # If frame is dirty, don't return cached batch
    if frame in _dirty_frames:
        return None
    
    # Check batch cache first
    if frame in _batch_cache:
        _cache_hits += 1
        return _batch_cache[frame]
    
    # Build from raw data
    data = _frame_cache.get(frame)
    if data is None:
        return None
    
    verts, indices, prim_type = data
    shader = get_shader()
    
    # Convert to format expected by batch_for_shader
    if HAS_NUMPY and isinstance(verts, np.ndarray):
        # Keep as numpy - gpu_extras handles numpy arrays efficiently
        verts_list = verts.tolist()
        indices_list = indices.tolist()
    elif HAS_NUMPY and hasattr(verts, '__len__'):
        # Already in list format
        verts_list = verts
        indices_list = indices
    else:
        verts_list = list(verts)
        indices_list = list(indices)
    
    try:
        batch = batch_for_shader(shader, prim_type, {"pos": verts_list}, indices=indices_list)
        _batch_cache[frame] = batch
        _cache_hits += 1
        return batch
    except Exception:
        return None


def get_cache_size():
    return len(_frame_cache)


def get_cache_stats():
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0
    return {
        'size': len(_frame_cache),
        'max_size': _max_cache_size,
        'hits': _cache_hits,
        'misses': _cache_misses,
        'hit_rate': hit_rate,
        'dirty': len(_dirty_frames),
    }


def invalidate_frames_near(center_frame, radius=30):
    """Invalidate frames within radius of center frame."""
    for frame in list(_frame_cache.keys()):
        if abs(frame - center_frame) <= radius:
            mark_frame_dirty(frame)


def evict_distant_frames(current_frame, keep_radius=50):
    """Evict frames far from current frame to free memory."""
    global _frame_cache, _batch_cache
    
    to_remove = []
    for frame in _frame_cache:
        if abs(frame - current_frame) > keep_radius:
            to_remove.append(frame)
    
    for frame in to_remove:
        del _frame_cache[frame]
        if frame in _batch_cache:
            del _batch_cache[frame]


def cleanup():
    global _shader
    _shader = None
    clear_cache()
