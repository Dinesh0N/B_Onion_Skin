# SPDX-License-Identifier: GPL-3.0-or-later
"""Cache management for onion skin - Ultra Performance."""

from collections import OrderedDict
import gpu
from gpu_extras.batch import batch_for_shader

# -----------------------------------------------------------------------------
# Global Cache
# -----------------------------------------------------------------------------

_shader = None
_frame_cache = OrderedDict()  # {frame: (verts, indices, prim_type)}
_batch_cache = {}  # {frame: gpu_batch} - lazy built
_max_cache_size = 500  # Increased for large scenes
_last_frame = -999

# Statistics
_cache_hits = 0
_cache_misses = 0


def get_shader():
    """Get or create the GPU shader."""
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    return _shader


def clear_cache():
    """Clear all cached data."""
    global _frame_cache, _batch_cache, _last_frame, _cache_hits, _cache_misses
    _frame_cache.clear()
    _batch_cache.clear()
    _last_frame = -999
    _cache_hits = 0
    _cache_misses = 0


def get_last_frame():
    return _last_frame


def set_last_frame(frame):
    global _last_frame
    _last_frame = frame


def is_frame_cached(frame):
    """Check if frame is cached (no stats update for speed)."""
    return frame in _frame_cache


def add_to_cache(frame, verts, indices, prim_type):
    """Add mesh data to cache. verts/indices are lists."""
    global _frame_cache, _batch_cache, _cache_misses
    
    if not verts or not indices:
        return
    
    _cache_misses += 1
    
    # Store raw data
    _frame_cache[frame] = (verts, indices, prim_type)
    _frame_cache.move_to_end(frame)
    
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
    
    batch = batch_for_shader(shader, prim_type, {"pos": verts}, indices=indices)
    _batch_cache[frame] = batch
    _cache_hits += 1
    
    return batch


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
    }


def cleanup():
    global _shader
    _shader = None
    clear_cache()
