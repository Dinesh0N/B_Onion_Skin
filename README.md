# B Onion Skin

![Blender](https://img.shields.io/badge/Blender-5.0+-orange)
![License](https://img.shields.io/badge/License-GPL--3.0-blue)
![Version](https://img.shields.io/badge/Version-1.0.0-green)

Onion skinning for 3D animation.

Display ghost frames of your animated meshes directly in the viewport to visualize motion, timing, and spacing while you animate.

## Features

- **Display** — Ghosts update instantly as you scrub the timeline
- **Before & After Frames** — See past and future positions with customizable colors
- **Armature Support** — Works with rigged characters and their mesh children
- **Adjustable Opacity** — Smooth falloff so distant frames fade naturally
- **Wireframe Mode** — Lightweight display for complex meshes
- **X-Ray** — See ghosts through geometry
- **Smart Caching** — Pre-bake frames for smooth playback

## Usage

1. Open **View3D → Sidebar → Onion Skin**
2. Select your mesh or armature and click **Add Selected**
3. Enable onion skinning with the checkbox in the panel header
4. Adjust frame count, colors, and opacity to your preference

## Settings

**Objects**
- Add mesh or armature objects to display onion skins
- Armature children are automatically included

**Frame Range**
- Set how many frames before and after to display
- Adjust frame step for faster animations
- Limit display to a specific frame range

**Appearance**
- Custom colors for before/after ghosts
- Opacity falloff with linear, smooth, or exponential curves
- X-Ray, wireframe, and mesh-in-front options

**Cache**
- Bake all frames for instant playback
- Clear or rebake as needed

## Requirements

- Blender 5.0

## License

GPL-3.0-or-later

