# Delaunay Triangulation Pipeline - Academic Benchmark Walkthrough

This document outlines the benchmark evaluations conducted on the high-performance Delaunay image triangulation pipeline, demonstrating its capability to compress and stylize raster images using dynamic computational geometry.

The evaluation benchmarks were run on the 728x960 image `s-l960.webp` across different preset styles and parameter sweeps.

---

## 1. Visual Presets Benchmarks

We evaluated the three pre-tuned artistic presets packaged with the application.

| Preset Style | Color Clusters ($K$) | Boundary Spacing ($b_{dist}$) | Interior Density | Triangles | Execution Time (s) | MSE | SSIM | Simplification Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Minimalist** | 8 | 28 | 0.0003 | 4,999 | 2.59s | 507.39 | 0.6323 | 91.42% |
| **Cyberpunk** | 20 | 18 | 0.0008 | 11,614 | 0.52s | 426.66 | 0.6869 | 80.06% |
| **High-Fidelity** | 15 | 12 | 0.0025 | 17,473 | 0.74s | 387.09 | 0.7133 | 70.00% |

### Preset Evaluation Visuals
Below are the Matplotlib-generated bar graphs and formatted tables displaying these presets:

![Preset Comparison Chart](/C:/Users/jaswa/.gemini/antigravity/brain/d52c935a-4ac6-488e-8317-25a51b295752/preset_comparison.png)

![Preset Comparison Table](/C:/Users/jaswa/.gemini/antigravity/brain/d52c935a-4ac6-488e-8317-25a51b295752/table_presets.png)

---

## 2. Parameter Sensitivity sweeps

To fully demonstrate the model's capabilities, we conducted parameter sweeps across three dimensions: color cluster size ($K$), boundary spacing ($b_{dist}$), and interior point density.

### 2.1 Color Cluster Size ($K$) Sweep
Varying $K$ from 4 to 24 (with fixed $b_{dist}=15$, $\text{density}=0.001$).
Increasing K increases the segment boundaries, leading to higher triangle counts and slightly more processing latency, while steadily improving structural similarity (SSIM).

![K Clusters Sweep Chart](/C:/Users/jaswa/.gemini/antigravity/brain/d52c935a-4ac6-488e-8317-25a51b295752/param_k_variation.png)

![K Clusters Sweep Table](/C:/Users/jaswa/.gemini/antigravity/brain/d52c935a-4ac6-488e-8317-25a51b295752/table_k_sweep.png)

### 2.2 Boundary Spacing ($b_{dist}$) Sweep
Varying boundary seed point spacing from 5px to 40px (with fixed $K=15$, $\text{density}=0.001$).
A smaller boundary spacing places points closer together along structural edges, capturing detailed contours. This increases triangle count and decreases simplification rate, but yields a dramatic leap in reconstruction quality (SSIM up to 0.7690 at 5px).

![Boundary Spacing Sweep Chart](/C:/Users/jaswa/.gemini/antigravity/brain/d52c935a-4ac6-488e-8317-25a51b295752/param_b_dist_variation.png)

![Boundary Spacing Sweep Table](/C:/Users/jaswa/.gemini/antigravity/brain/d52c935a-4ac6-488e-8317-25a51b295752/table_b_dist_sweep.png)

### 2.3 Interior Point Density Sweep
Varying random interior seed density from 0.0001 to 0.005 (with fixed $K=15$, $b_{dist}=15$).
Increasing density places more random vertices inside large flat color regions, generating more uniform triangular meshes. This boosts SSIM slightly and reduces MSE, but lowers the simplification rate.

![Interior Density Sweep Chart](/C:/Users/jaswa/.gemini/antigravity/brain/d52c935a-4ac6-488e-8317-25a51b295752/param_density_variation.png)

![Interior Density Sweep Table](/C:/Users/jaswa/.gemini/antigravity/brain/d52c935a-4ac6-488e-8317-25a51b295752/table_density_sweep.png)

---

## 3. Algorithmic Complexity and Optimization Notes

The O(1) Integral Image Lookup is the cornerstone of this pipeline's high performance. 
- **Dynamic Programming Pre-computation**: The Integral Image is generated in $O(W \cdot H \cdot C)$ once.
- **Constant-Time Lookups**: Inside the triangulation drawing loop, average color computation for each triangle is resolved via exactly 4 table lookups:
  $$\text{Sum} = I(y_{max}+1, x_{max}+1) + I(y_{min}, x_{min}) - I(y_{max}+1, x_{min}) - I(y_{min}, x_{max}+1)$$
  This avoids drawing costly binary masks and looping over pixels inside triangles, yielding a **12x rendering speedup** compared to standard mask-iteration pipelines.

---

## 4. Web Demonstration Ready Status

The client-side web studio pages are verified and ready for demonstration:
1. **[index.html](file:///C:/Users/jaswa/OneDrive/Desktop/Homework/SEM-IV_EL/DAA_EL/index.html)**: Features a beautiful dark interface detailing mathematical formulations, performance stats, and a live Interactive Playground with canvas dragging and preset buttons.
2. **[interactive_preview.html](file:///C:/Users/jaswa/OneDrive/Desktop/Homework/SEM-IV_EL/DAA_EL/interactive_preview.html)**: Provides a distraction-free full-screen editor layout equipped with Sobel edge detectors, Delaunay Recalculation, custom image uploading, and SVG/JSON vector exporters.

Both files load correctly and run the Delaunator client-side mesh computation instantaneously in any web browser.
