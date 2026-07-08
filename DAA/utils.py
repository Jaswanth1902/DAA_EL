import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index between two images."""
    # Ensure images are in grayscale for SSIM if they are multi-channel
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g1, g2 = img1, img2
    return ssim(g1, g2)

def save_triangles(triangles, filename):
    """
    Save triangle data to a file.
    Format: x1,y1,x2,y2,x3,y3,r,g,b per line.
    """
    with open(filename, 'w') as f:
        for pts, color in triangles:
            # pts is (3, 2), color is (3,) BGR
            coords = ",".join([f"{p[0]},{p[1]}" for p in pts])
            # OpenCV color is BGR, so we swap index 0 (Blue) and 2 (Red) to get RGB
            color_str = f"{int(color[2])},{int(color[1])},{int(color[0])}"
            f.write(f"{coords},{color_str}\n")

def save_svg(triangles, width, height, filename):
    """
    Save triangle data to a standard SVG file.
    To prevent hairlines between adjacent triangles, we add a stroke matching the fill.
    """
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
    ]
    for pts, color in triangles:
        r, g, b = int(color[2]), int(color[1]), int(color[0])
        pts_str = " ".join([f"{p[0]},{p[1]}" for p in pts])
        svg_lines.append(f'  <polygon points="{pts_str}" fill="rgb({r},{g},{b})" stroke="rgb({r},{g},{b})" stroke-width="0.5" />')
    svg_lines.append('</svg>')
    with open(filename, 'w') as f:
        f.write("\n".join(svg_lines))

def save_json(triangles, width, height, filename):
    """
    Save Delaunay triangulation data to a highly compressed WebGL-friendly JSON file.
    Format:
    {
      "width": width,
      "height": height,
      "vertices": [x0, y0, x1, y1, ...],
      "triangles": [v0_idx, v1_idx, v2_idx, ...],
      "colors": [r0, g0, b0, ...]
    }
    """
    import json
    unique_points = []
    point_to_idx = {}
    indices = []
    colors = []
    
    for pts, color in triangles:
        tri_indices = []
        for p in pts:
            pt_tuple = (float(p[0]), float(p[1]))
            if pt_tuple not in point_to_idx:
                point_to_idx[pt_tuple] = len(unique_points) // 2
                unique_points.append(pt_tuple[0])
                unique_points.append(pt_tuple[1])
            tri_indices.append(point_to_idx[pt_tuple])
        
        indices.extend(tri_indices)
        colors.extend([int(color[2]), int(color[1]), int(color[0])]) # Convert BGR to RGB
        
    data = {
        "width": int(width),
        "height": int(height),
        "vertices": unique_points,
        "triangles": indices,
        "colors": colors
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

def get_simplification_rate(original_shape, triangle_count):
    """Calculate the simplification rate/compression ratio."""
    # Original size in pixels * 3 (channels)
    original_size = original_shape[0] * original_shape[1] * 3
    # Triangle size: 3 points * 2 coords + 3 colors = 9 values per triangle
    # Assuming each value is stored in a compact way (e.g. 4 bytes per float/int)
    compressed_size = triangle_count * 9 * 4 
    rate = (1 - (compressed_size / original_size)) * 100
    return max(0, rate)
