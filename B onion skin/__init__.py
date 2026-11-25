# SPDX-License-Identifier: GPL-3.0-or-later
"""B Onion Skin — GPU onion skinning for 3D animation."""

bl_info = {
    "name": "B Onion Skin",
    "author": "Dinesh007",
    "version": (1, 2, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Onion Skin",
    "description": "GPU onion skinning for 3D animation",
    "category": "Animation",
}

from . import cache
from . import properties
from . import operators
from . import ui
from . import drawing
from .c_core import core_bridge as cb


def register():
    properties.register()
    operators.register()
    ui.register()
    drawing.register()

    cb.ensure_initialized()
    if cb.is_available():
        print("[B Onion Skin] High-performance C++ backend loaded with hardware SIMD and GPU acceleration!")
    else:
        print("[B Onion Skin] Running in fallback Python mode (C++ core library not loaded).")


def unregister():
    drawing.unregister()
    ui.unregister()
    operators.unregister()
    properties.unregister()
    cache.cleanup()


if __name__ == "__main__":
    register()
