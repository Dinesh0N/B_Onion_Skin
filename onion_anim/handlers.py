# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender event handlers for onion skin addon."""

import bpy
from bpy.app.handlers import persistent

from . import cache


@persistent
def on_load(dummy):
    """Handle file load - clear cache."""
    cache.clear_cache()


def register():
    if on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load)


def unregister():
    if on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load)
