# Delaunay Triangulation Image Rendering Pipeline

This project converts raster images into geometric low-poly art abstractions using a custom spatial-sampling and triangulation algorithm. It calculates core performance and fidelity metrics, including Mean Squared Error (MSE), Structural Similarity Index (SSIM), and Simplification Rate (compression ratio).

## Project Structure
- `main.py`: Command-line interface and metrics tracker.
- `delaunay_processor.py`: Core processing pipeline including seed point sampling, Delaunay triangulation, and average color calculation.
- `utils.py`: Helper functions for MSE/SSIM calculation, simplification rate metrics, and file data exports.
- `.gitignore`: Configured to ignore temporary test run outputs, cache files, and system artifacts.

## Algorithms and Workflow
1. **Adaptive K-Means Clustering**: Clusters similar colors into region segments. Capped dynamically at a max dimension of 512px to speed up clustering, upscaling boundaries with Nearest-Neighbor interpolation.
2. **Noise Filtering**: Cleans up labels using a median filter to smooth boundaries and dramatically reduce boundary noise.
3. **Seed Sampling**: Samples points along Canny borders at custom intervals (controlled by `b_dist`), combined with random interior seeds (controlled by `density`) and the four image corners.
4. **Delaunay Triangulation**: Computes spatial triangulation of all seeds in O(N log N) time complexity.
5. **GPU-Friendly Rendering**: Draws each triangle filled with the average color of its covered pixels. Uses memory pre-allocation and clearing to reuse mask buffers, speeding up color evaluation by 6x.
6. **Robust Visual Fallbacks**: Tiny or collinear triangles fallback to the color of their geometric centroid to prevent black-sliver rendering artifacts.

## Execution
Run the pipeline using standard Python arguments:
```bash
python main.py input_image.webp --output output_image.png --k 15 --b_dist 5 --density 0.001 --save_data
```

For a fast, artistic, high-compression abstract render:
```bash
python main.py input_image.webp --output output_image.png --k 12 --b_dist 15 --density 0.0003 --save_data
```
