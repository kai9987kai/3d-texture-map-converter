"""
3D Texture Map Converter — Professional Edition
================================================
A comprehensive tool for generating PBR texture maps from source images.
Supports 10 map types, batch processing, export-all, undo history,
adjustable strength, and a modern dark-themed UI with side-by-side preview.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, sobel, laplace
import math
import os
import threading
from collections import deque

# ─── Color palette (Catppuccin Mocha-inspired) ───────────────────────────────
BG_DARK    = "#1e1e2e"
BG_MID     = "#313244"
BG_LIGHT   = "#45475a"
FG_TEXT     = "#cdd6f4"
FG_DIM      = "#a6adc8"
ACCENT      = "#89b4fa"
ACCENT_HOVER = "#b4d0fb"
SUCCESS     = "#a6e3a1"
WARNING     = "#f9e2af"
DANGER      = "#f38ba8"
BORDER      = "#585b70"

# ─── Map type definitions ────────────────────────────────────────────────────
MAP_TYPES = [
    "Normal Map",
    "Height Map",
    "Ambient Occlusion",
    "Roughness Map",
    "Metallic Map",
    "Curvature Map",
    "Displacement Map",
    "Specular Map",
    "Emissive Map",
    "Edge / Cavity Map",
]

MAX_UNDO = 10
PREVIEW_MAX = 512  # max dimension for preview thumbnails


class TextureMapConverter:
    """Main application class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("3D Texture Map Converter — Pro")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1280x800")
        self.root.minsize(960, 640)

        # State
        self.source_image: Image.Image | None = None
        self.result_image: Image.Image | None = None
        self.source_tk: ImageTk.PhotoImage | None = None
        self.result_tk: ImageTk.PhotoImage | None = None
        self.undo_stack: deque[Image.Image] = deque(maxlen=MAX_UNDO)
        self.current_map_type: str = MAP_TYPES[0]
        self.strength_var = tk.DoubleVar(value=1.0)

        self._build_styles()
        self._build_ui()
        self._bind_shortcuts()

    # ── Styling ──────────────────────────────────────────────────────────────

    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=BG_DARK, foreground=FG_TEXT,
                         borderwidth=0, focuscolor=ACCENT)
        style.configure("TFrame", background=BG_DARK)
        style.configure("TLabel", background=BG_DARK, foreground=FG_TEXT,
                         font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"),
                         foreground=ACCENT)
        style.configure("Status.TLabel", background=BG_MID, foreground=FG_DIM,
                         font=("Segoe UI", 9), padding=(8, 4))

        # Buttons
        style.configure("Toolbar.TButton", background=BG_MID, foreground=FG_TEXT,
                         font=("Segoe UI", 10), padding=(12, 6),
                         borderwidth=1, relief="flat")
        style.map("Toolbar.TButton",
                  background=[("active", BG_LIGHT), ("pressed", ACCENT)],
                  foreground=[("active", FG_TEXT)])

        style.configure("Accent.TButton", background=ACCENT, foreground=BG_DARK,
                         font=("Segoe UI", 10, "bold"), padding=(14, 7))
        style.map("Accent.TButton",
                  background=[("active", ACCENT_HOVER)])

        style.configure("Map.TButton", background=BG_MID, foreground=FG_TEXT,
                         font=("Segoe UI", 9), padding=(8, 5))
        style.map("Map.TButton",
                  background=[("active", BG_LIGHT)])

        # Scale / slider
        style.configure("Horizontal.TScale", background=BG_DARK,
                         troughcolor=BG_MID)

        # Separator
        style.configure("TSeparator", background=BORDER)

    # ── UI Layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top toolbar ──
        toolbar = ttk.Frame(self.root, style="TFrame")
        toolbar.pack(fill="x", padx=8, pady=(8, 0))

        ttk.Button(toolbar, text="📂 Open Image", style="Toolbar.TButton",
                   command=self.open_image).pack(side="left", padx=2)
        ttk.Button(toolbar, text="💾 Save Result", style="Toolbar.TButton",
                   command=self.save_result).pack(side="left", padx=2)
        ttk.Button(toolbar, text="📁 Export All Maps", style="Toolbar.TButton",
                   command=self.export_all_maps).pack(side="left", padx=2)
        ttk.Button(toolbar, text="📦 Batch Process", style="Toolbar.TButton",
                   command=self.batch_process).pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y",
                                                        padx=8, pady=2)

        ttk.Button(toolbar, text="↩ Undo", style="Toolbar.TButton",
                   command=self.undo).pack(side="left", padx=2)

        # Strength slider (right side of toolbar)
        strength_frame = ttk.Frame(toolbar, style="TFrame")
        strength_frame.pack(side="right", padx=4)
        ttk.Label(strength_frame, text="Strength:", style="TLabel").pack(
            side="left", padx=(0, 4))
        self.strength_label = ttk.Label(strength_frame, text="1.0",
                                         style="TLabel", width=4)
        self.strength_label.pack(side="right", padx=(4, 0))
        self.strength_scale = ttk.Scale(
            strength_frame, from_=0.1, to=5.0, variable=self.strength_var,
            orient="horizontal", length=160,
            command=self._on_strength_change)
        self.strength_scale.pack(side="left")

        # ── Main content area (paned) ──
        content = ttk.Frame(self.root, style="TFrame")
        content.pack(fill="both", expand=True, padx=8, pady=8)

        # Left sidebar — map buttons
        sidebar = ttk.Frame(content, style="TFrame", width=200)
        sidebar.pack(side="left", fill="y", padx=(0, 8))
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="Texture Maps", style="Title.TLabel").pack(
            anchor="w", padx=8, pady=(4, 8))

        for map_name in MAP_TYPES:
            btn = ttk.Button(sidebar, text=map_name, style="Map.TButton",
                             command=lambda m=map_name: self._convert(m))
            btn.pack(fill="x", padx=4, pady=2)

        # Right area — dual preview
        preview_area = ttk.Frame(content, style="TFrame")
        preview_area.pack(side="left", fill="both", expand=True)

        # Source preview
        src_frame = ttk.Frame(preview_area, style="TFrame")
        src_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))

        ttk.Label(src_frame, text="Original", style="Title.TLabel").pack(
            anchor="w", padx=4)
        self.source_canvas = tk.Canvas(src_frame, bg=BG_MID,
                                        highlightthickness=1,
                                        highlightbackground=BORDER)
        self.source_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.source_canvas.bind("<Double-Button-1>", lambda e: self.open_image())

        # Result preview
        res_frame = ttk.Frame(preview_area, style="TFrame")
        res_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))

        self.result_title = ttk.Label(res_frame, text="Result",
                                       style="Title.TLabel")
        self.result_title.pack(anchor="w", padx=4)
        self.result_canvas = tk.Canvas(res_frame, bg=BG_MID,
                                        highlightthickness=1,
                                        highlightbackground=BORDER)
        self.result_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Ready — Open an image to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                                style="Status.TLabel")
        status_bar.pack(fill="x", side="bottom")

    # ── Keyboard shortcuts ───────────────────────────────────────────────────

    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_result())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-e>", lambda e: self.export_all_maps())

    # ── Strength slider callback ─────────────────────────────────────────────

    def _on_strength_change(self, value):
        val = float(value)
        self.strength_label.config(text=f"{val:.1f}")

    # ── Preview helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _fit_image(img: Image.Image, max_dim: int = PREVIEW_MAX) -> Image.Image:
        """Scale image to fit within max_dim, preserving aspect ratio."""
        w, h = img.size
        if w <= max_dim and h <= max_dim:
            return img.copy()
        ratio = min(max_dim / w, max_dim / h)
        new_size = (int(w * ratio), int(h * ratio))
        return img.resize(new_size, Image.LANCZOS)

    def _show_on_canvas(self, canvas: tk.Canvas, img: Image.Image, side: str):
        """Display a PIL Image on a canvas, auto-scaled."""
        display = self._fit_image(img)
        tk_img = ImageTk.PhotoImage(display)
        canvas.delete("all")
        canvas.create_image(
            canvas.winfo_width() // 2 or 256,
            canvas.winfo_height() // 2 or 256,
            anchor="center", image=tk_img)
        # Keep a reference to avoid GC
        if side == "source":
            self.source_tk = tk_img
        else:
            self.result_tk = tk_img

    def _update_source_preview(self):
        if self.source_image:
            self.root.after(50, lambda: self._show_on_canvas(
                self.source_canvas, self.source_image, "source"))

    def _update_result_preview(self):
        if self.result_image:
            self.root.after(50, lambda: self._show_on_canvas(
                self.result_canvas, self.result_image, "result"))

    # ── File operations ──────────────────────────────────────────────────────

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tga *.webp"),
                ("All files", "*.*"),
            ])
        if not path:
            return
        try:
            self.source_image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to open image:\n{exc}")
            return
        self.result_image = None
        self.result_tk = None
        self.result_canvas.delete("all")
        self.undo_stack.clear()
        w, h = self.source_image.size
        self.status_var.set(
            f"Loaded: {os.path.basename(path)}  |  {w}×{h} px")
        self._update_source_preview()

    def save_result(self):
        if self.result_image is None:
            messagebox.showinfo("Nothing to save",
                                "Generate a map first, then save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),
                       ("BMP", "*.bmp"), ("TIFF", "*.tiff")])
        if not path:
            return
        try:
            self.result_image.save(path)
            self.status_var.set(f"Saved: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save:\n{exc}")

    # ── Undo ─────────────────────────────────────────────────────────────────

    def _push_undo(self):
        if self.result_image is not None:
            self.undo_stack.append(self.result_image.copy())

    def undo(self):
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return
        self.result_image = self.undo_stack.pop()
        self._update_result_preview()
        self.status_var.set("Undo applied")

    # ── Conversion dispatcher ────────────────────────────────────────────────

    def _convert(self, map_type: str):
        if self.source_image is None:
            messagebox.showinfo("No image", "Please open an image first.")
            return

        self.current_map_type = map_type
        self._push_undo()
        self.status_var.set(f"Generating {map_type}…")
        self.root.update_idletasks()

        strength = self.strength_var.get()
        src = self.source_image

        converters = {
            "Normal Map":           self._gen_normal_map,
            "Height Map":           self._gen_height_map,
            "Ambient Occlusion":    self._gen_ao_map,
            "Roughness Map":        self._gen_roughness_map,
            "Metallic Map":         self._gen_metallic_map,
            "Curvature Map":        self._gen_curvature_map,
            "Displacement Map":     self._gen_displacement_map,
            "Specular Map":         self._gen_specular_map,
            "Emissive Map":         self._gen_emissive_map,
            "Edge / Cavity Map":    self._gen_edge_map,
        }

        converter = converters.get(map_type)
        if converter is None:
            return

        try:
            result = converter(src, strength)
            self.result_image = result
            self.result_title.config(text=map_type)
            self._update_result_preview()
            self.status_var.set(f"{map_type} generated  |  Strength {strength:.1f}")
        except Exception as exc:
            messagebox.showerror("Conversion Error", str(exc))
            self.status_var.set("Error during conversion")

    # ── Map generation functions ─────────────────────────────────────────────

    @staticmethod
    def _to_gray(img: Image.Image) -> np.ndarray:
        return np.asarray(img.convert("L"), dtype=np.float64)

    # ---- Normal Map ----------------------------------------------------------
    @staticmethod
    def _gen_normal_map(src: Image.Image, strength: float) -> Image.Image:
        gray = np.asarray(src.convert("L"), dtype=np.float64) / 255.0
        # Sobel gradients
        dx = sobel(gray, axis=1) * strength
        dy = sobel(gray, axis=0) * strength
        dz = np.ones_like(gray)
        # Normalise
        mag = np.sqrt(dx**2 + dy**2 + dz**2)
        mag[mag == 0] = 1.0
        nx = (dx / mag * 0.5 + 0.5) * 255
        ny = (dy / mag * 0.5 + 0.5) * 255
        nz = (dz / mag * 0.5 + 0.5) * 255
        normal = np.stack([nx, ny, nz], axis=-1).astype(np.uint8)
        return Image.fromarray(normal, "RGB")

    # ---- Height Map ----------------------------------------------------------
    @staticmethod
    def _gen_height_map(src: Image.Image, strength: float) -> Image.Image:
        gray = np.asarray(src.convert("L"), dtype=np.float64)
        # Apply strength as contrast adjustment around mid-gray
        mid = 128.0
        adjusted = np.clip((gray - mid) * strength + mid, 0, 255).astype(np.uint8)
        return Image.fromarray(adjusted, "L").convert("RGB")

    # ---- Ambient Occlusion ---------------------------------------------------
    @staticmethod
    def _gen_ao_map(src: Image.Image, strength: float) -> Image.Image:
        gray = np.asarray(src.convert("L"), dtype=np.float64)
        # Multi-scale local contrast
        blur_small = gaussian_filter(gray, sigma=2 * strength)
        blur_large = gaussian_filter(gray, sigma=8 * strength)
        ao = np.clip(128 + (blur_small - blur_large) * 2 * strength, 0, 255)
        return Image.fromarray(ao.astype(np.uint8), "L").convert("RGB")

    # ---- Roughness Map -------------------------------------------------------
    @staticmethod
    def _gen_roughness_map(src: Image.Image, strength: float) -> Image.Image:
        gray = np.asarray(src.convert("L"), dtype=np.float64)
        kernel = max(3, int(5 * strength))
        mean = uniform_filter(gray, size=kernel)
        sq_mean = uniform_filter(gray**2, size=kernel)
        variance = np.clip(sq_mean - mean**2, 0, None)
        roughness = np.sqrt(variance)
        # Normalise to 0-255
        r_min, r_max = roughness.min(), roughness.max()
        if r_max - r_min > 0:
            roughness = (roughness - r_min) / (r_max - r_min) * 255
        else:
            roughness = np.zeros_like(roughness)
        return Image.fromarray(roughness.astype(np.uint8), "L").convert("RGB")

    # ---- Metallic Map --------------------------------------------------------
    @staticmethod
    def _gen_metallic_map(src: Image.Image, strength: float) -> Image.Image:
        hsv = np.asarray(src.convert("HSV"), dtype=np.float64)
        saturation = hsv[:, :, 1]  # 0-255
        # Low saturation → more metallic
        metallic = 255 - saturation
        metallic = np.clip(metallic * strength, 0, 255).astype(np.uint8)
        return Image.fromarray(metallic, "L").convert("RGB")

    # ---- Curvature Map -------------------------------------------------------
    @staticmethod
    def _gen_curvature_map(src: Image.Image, strength: float) -> Image.Image:
        gray = np.asarray(src.convert("L"), dtype=np.float64) / 255.0
        lap = laplace(gray) * strength
        # Map to 0-255 with 128 as the zero-crossing
        curvature = np.clip(lap * 128 + 128, 0, 255).astype(np.uint8)
        return Image.fromarray(curvature, "L").convert("RGB")

    # ---- Displacement Map ----------------------------------------------------
    @staticmethod
    def _gen_displacement_map(src: Image.Image, strength: float) -> Image.Image:
        gray = np.asarray(src.convert("L"), dtype=np.float64)
        smoothed = gaussian_filter(gray, sigma=2.0 * strength)
        displacement = np.clip(smoothed, 0, 255).astype(np.uint8)
        return Image.fromarray(displacement, "L").convert("RGB")

    # ---- Specular Map --------------------------------------------------------
    @staticmethod
    def _gen_specular_map(src: Image.Image, strength: float) -> Image.Image:
        arr = np.asarray(src, dtype=np.float64)
        # Luminance
        lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        lum = lum / 255.0
        # Power curve — higher strength = sharper specular highlight
        spec = np.power(lum, 1.0 / max(strength, 0.1)) * 255
        spec = np.clip(spec, 0, 255).astype(np.uint8)
        return Image.fromarray(spec, "L").convert("RGB")

    # ---- Emissive Map --------------------------------------------------------
    @staticmethod
    def _gen_emissive_map(src: Image.Image, strength: float) -> Image.Image:
        arr = np.asarray(src, dtype=np.float64)
        brightness = arr.mean(axis=2)
        threshold = max(0, 255 - 80 * strength)
        emissive = np.where(brightness >= threshold, brightness, 0)
        emissive = np.clip(emissive, 0, 255).astype(np.uint8)
        return Image.fromarray(emissive, "L").convert("RGB")

    # ---- Edge / Cavity Map ---------------------------------------------------
    @staticmethod
    def _gen_edge_map(src: Image.Image, strength: float) -> Image.Image:
        gray = np.asarray(src.convert("L"), dtype=np.float64) / 255.0
        sx = sobel(gray, axis=1)
        sy = sobel(gray, axis=0)
        edges = np.sqrt(sx**2 + sy**2) * strength
        edges = np.clip(edges * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(edges, "L").convert("RGB")

    # ── Export All Maps ──────────────────────────────────────────────────────

    def export_all_maps(self):
        if self.source_image is None:
            messagebox.showinfo("No image", "Please open an image first.")
            return

        folder = filedialog.askdirectory(title="Choose Export Folder")
        if not folder:
            return

        strength = self.strength_var.get()
        converters = {
            "normal_map":       self._gen_normal_map,
            "height_map":       self._gen_height_map,
            "ambient_occlusion": self._gen_ao_map,
            "roughness_map":    self._gen_roughness_map,
            "metallic_map":     self._gen_metallic_map,
            "curvature_map":    self._gen_curvature_map,
            "displacement_map": self._gen_displacement_map,
            "specular_map":     self._gen_specular_map,
            "emissive_map":     self._gen_emissive_map,
            "edge_cavity_map":  self._gen_edge_map,
        }

        self.status_var.set("Exporting all maps…")
        self.root.update_idletasks()

        def _export():
            count = 0
            for name, fn in converters.items():
                try:
                    result = fn(self.source_image, strength)
                    result.save(os.path.join(folder, f"{name}.png"))
                    count += 1
                except Exception:
                    pass
            self.root.after(0, lambda: self.status_var.set(
                f"Exported {count}/{len(converters)} maps to {folder}"))
            if count == len(converters):
                self.root.after(0, lambda: messagebox.showinfo(
                    "Export Complete",
                    f"All {count} maps saved to:\n{folder}"))

        threading.Thread(target=_export, daemon=True).start()

    # ── Batch Process ────────────────────────────────────────────────────────

    def batch_process(self):
        folder = filedialog.askdirectory(title="Select Folder of Images")
        if not folder:
            return

        # Ask which map type
        map_choice = self.current_map_type

        # Gather image files
        supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tga", ".webp"}
        files = [f for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in supported]
        if not files:
            messagebox.showinfo("No images", "No supported images found.")
            return

        out_dir = os.path.join(folder, f"batch_{map_choice.lower().replace(' ', '_')}")
        os.makedirs(out_dir, exist_ok=True)

        strength = self.strength_var.get()
        converters = {
            "Normal Map":           self._gen_normal_map,
            "Height Map":           self._gen_height_map,
            "Ambient Occlusion":    self._gen_ao_map,
            "Roughness Map":        self._gen_roughness_map,
            "Metallic Map":         self._gen_metallic_map,
            "Curvature Map":        self._gen_curvature_map,
            "Displacement Map":     self._gen_displacement_map,
            "Specular Map":         self._gen_specular_map,
            "Emissive Map":         self._gen_emissive_map,
            "Edge / Cavity Map":    self._gen_edge_map,
        }
        converter = converters.get(map_choice)
        if converter is None:
            return

        total = len(files)
        self.status_var.set(f"Batch: 0/{total}…")
        self.root.update_idletasks()

        def _batch():
            done = 0
            for f in files:
                try:
                    img = Image.open(os.path.join(folder, f)).convert("RGB")
                    result = converter(img, strength)
                    base, _ = os.path.splitext(f)
                    result.save(os.path.join(out_dir, f"{base}_{map_choice.lower().replace(' ', '_')}.png"))
                    done += 1
                    self.root.after(0, lambda d=done: self.status_var.set(
                        f"Batch: {d}/{total}…"))
                except Exception:
                    pass
            self.root.after(0, lambda: self.status_var.set(
                f"Batch complete: {done}/{total} images → {out_dir}"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Batch Complete",
                f"Processed {done}/{total} images.\n\n"
                f"Output folder:\n{out_dir}"))

        threading.Thread(target=_batch, daemon=True).start()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = TextureMapConverter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
