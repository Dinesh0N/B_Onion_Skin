# SPDX-License-Identifier: GPL-3.0-or-later
# Blender 5.0 Onion Skin Addon
# Realtime 3D animation onion skinning with smart caching

bl_info = {
    "name": "B onion skin",
    "author": "Dinesh007",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Onion Skin",
    "description": "Realtime onion skinning for 3D animation",
    "doc_url": "",
    "category": "Animation",
}

# -----------------------------------------------------------------------------
# Module Imports
# -----------------------------------------------------------------------------

from . import cache
from . import properties
from . import operators
from . import ui
from . import drawing
from . import handlers


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

def register():
    properties.register()
    operators.register()
    ui.register()
    drawing.register()
    handlers.register()


def unregister():
    handlers.unregister()
    drawing.unregister()
    ui.unregister()
    operators.unregister()
    properties.unregister()
    cache.cleanup()


if __name__ == "__main__":
    register()
