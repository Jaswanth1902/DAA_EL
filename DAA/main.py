import argparse
import os
import time
import cv2
import numpy as np
from delaunay_processor import DelaunayProcessor
from utils import calculate_mse, calculate_ssim, save_triangles, save_svg, save_json, get_simplification_rate

def main():
    parser = argparse.ArgumentParser(description="Advanced Delaunay Triangulation for Efficient Low-Poly Abstraction")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("--style", choices=["cyberpunk", "minimalist", "highfidelity"], help="Apply a pre-tuned visual style")
    parser.add_argument("--k", type=int, help="Number of k-means color regions")
    parser.add_argument("--b_dist", type=int, help="Boundary seed point spacing Bpointdist")
    parser.add_argument("--density", type=float, help="Interior random point density")
    parser.add_argument("--output", default="technical_result.png", help="Path to output image")
    parser.add_argument("--save_data", action="store_true", help="Save triangle data to .txt")
    parser.add_argument("--save_svg", action="store_true", help="Save vector representation to .svg")
    parser.add_argument("--save_json", action="store_true", help="Save WebGL-compressed representation to .json")

    args = parser.parse_args()

    # Pre-tuned styles mapping
    styles = {
        "cyberpunk": {"k": 20, "b_dist": 18, "density": 0.0008},
        "minimalist": {"k": 8, "b_dist": 28, "density": 0.0003},
        "highfidelity": {"k": 15, "b_dist": 12, "density": 0.0025}
    }

    # Set default values if not provided
    k = args.k
    b_dist = args.b_dist
    density = args.density

    if args.style:
        style_params = styles[args.style]
        if k is None:
            k = style_params["k"]
        if b_dist is None:
            b_dist = style_params["b_dist"]
        if density is None:
            density = style_params["density"]

    # Fallback to standard defaults if still None
    if k is None:
        k = 15
    if b_dist is None:
        b_dist = 5
    if density is None:
        density = 0.001

    print(f"[*] Initializing Technical Abstraction Pipeline...")
    if args.style:
        print(f"[*] Applied Style Preset: {args.style.upper()}")

    processor = DelaunayProcessor(args.input)
    
    start_time = time.time()
    print(f"[*] Processing image ({processor.w}x{processor.h})...")
    print(f"[*] Parameters: k={k}, Bpointdist={b_dist}, Density={density}")
    
    processor.process(k=k, b_dist=b_dist, interior_density=density)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"[+] Processing completed in {duration:.2f} seconds.")

    print(f"[*] Rendering output...")
    output_img = processor.render()
    cv2.imwrite(args.output, output_img)
    print(f"[+] Output saved to {args.output}")

    # Calculate Metrics
    mse = calculate_mse(processor.image, output_img)
    ssim_val = calculate_ssim(processor.image, output_img)
    sim_rate = get_simplification_rate(processor.image.shape, len(processor.triangles))

    print("-" * 40)
    print("DAA PROJECT REPORT METRICS:")
    print(f"Triangle Count: {len(processor.triangles)}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Structural Similarity (SSIM): {ssim_val:.4f}")
    print(f"Simplification Rate: {sim_rate:.2f}%")
    print("-" * 40)

    if args.save_data:
        data_path = os.path.splitext(args.output)[0] + ".txt"
        save_triangles(processor.triangles, data_path)
        print(f"[+] Triangle data tuples saved to {data_path}")

    if args.save_svg:
        svg_path = os.path.splitext(args.output)[0] + ".svg"
        save_svg(processor.triangles, processor.w, processor.h, svg_path)
        print(f"[+] SVG vector file saved to {svg_path}")

    if args.save_json:
        json_path = os.path.splitext(args.output)[0] + ".json"
        save_json(processor.triangles, processor.w, processor.h, json_path)
        print(f"[+] WebGL JSON file saved to {json_path}")

if __name__ == "__main__":
    main()
