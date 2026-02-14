# 3D Texture Map Converter (Professional Edition)

A desktop (offline) **PBR texture-map generator** that converts a source image (typically **albedo/base-color**) into common **material/utility maps** with a modern **dark UI**, **side-by-side preview**, **strength control**, **undo history**, **export-all**, and **batch processing**.

This is designed as a fast, practical “generator + tweak” tool: it gives you *engine-ready starter maps* that you can refine in Substance/Designer, Blender, Krita, Photoshop, etc.

---

## Features

- **10 map types**
  - Normal Map
  - Height Map
  - Ambient Occlusion
  - Roughness Map
  - Metallic Map
  - Curvature Map
  - Displacement Map
  - Specular Map
  - Emissive Map
  - Edge / Cavity Map
- **Strength slider** (0.1 → 5.0) applied per-map (implementation differs per generator)
- **Side-by-side previews** (Original vs Result)
- **Undo stack** (up to 10 states)
- **Export all maps** in one click (PNG output)
- **Batch process** a folder of images into a chosen map type
- **Keyboard shortcuts**
  - `Ctrl+O` Open image
  - `Ctrl+S` Save result
  - `Ctrl+Z` Undo
  - `Ctrl+E` Export all maps
- **Supported inputs**: `.png .jpg .jpeg .bmp .tiff .tga .webp`
- **Supported save formats** (single result): PNG / JPG / BMP / TIFF

---

## Requirements

- Python **3.x**
- Tkinter (usually included with Python on Windows/macOS; may require an extra package on some Linux distros)
- Python deps:
  - Pillow
  - NumPy
  - SciPy

`requirements.txt` is included.

---

## Install

### 1) Clone
```bash
git clone https://github.com/kai9987kai/3d-texture-map-converter.git
cd 3d-texture-map-converter
