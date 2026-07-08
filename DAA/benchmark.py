import os
import time
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from delaunay_processor import DelaunayProcessor
from utils import calculate_mse, calculate_ssim, get_simplification_rate

def run_benchmark_run(processor, k, b_dist, density):
    """Runs a single processor pass and computes performance/quality metrics."""
    # Reset points and processor state
    start_time = time.time()
    processor.process(k=k, b_dist=b_dist, interior_density=density)
    process_duration = time.time() - start_time
    
    # Render and measure fidelity
    output_img = processor.render()
    mse = calculate_mse(processor.image, output_img)
    ssim_val = calculate_ssim(processor.image, output_img)
    sim_rate = get_simplification_rate(processor.image.shape, len(processor.triangles))
    
    return {
        "k": k,
        "b_dist": b_dist,
        "density": density,
        "triangles": len(processor.triangles),
        "duration": process_duration,
        "mse": mse,
        "ssim": ssim_val,
        "simplification_rate": sim_rate
    }

def save_table_as_image(headers, rows, filename, title=""):
    """Saves tabular data as a clean, professionally formatted PNG image using Matplotlib."""
    # Compute size dynamically based on row count
    fig, ax = plt.subplots(figsize=(9, max(2.5, len(rows) * 0.4 + 1.2)))
    ax.axis('off')
    
    # Create the table
    tab = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.2, 1.4)
    
    # Style the header cells
    for col_idx in range(len(headers)):
        cell = tab[0, col_idx]
        cell.set_facecolor('#1e3a8a') # Dark blue academic theme
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
        
    # Style data rows (zebra striping)
    for row_idx in range(1, len(rows) + 1):
        for col_idx in range(len(headers)):
            cell = tab[row_idx, col_idx]
            if row_idx % 2 == 0:
                cell.set_facecolor('#f3f4f6') # Light gray zebra stripes
            else:
                cell.set_facecolor('#ffffff')
                
    if title:
        plt.title(title, fontsize=12, weight='bold', pad=10, color='#1e293b')
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    input_image_path = "s-l960.webp"
    if not os.path.exists(input_image_path):
        print(f"Error: {input_image_path} not found.")
        return
        
    print(f"[*] Initializing benchmarks on {input_image_path}...")
    processor = DelaunayProcessor(input_image_path)
    
    # 1. Benchmarking Styles/Presets
    styles = {
        "Minimalist": {"k": 8, "b_dist": 28, "density": 0.0003},
        "Cyberpunk": {"k": 20, "b_dist": 18, "density": 0.0008},
        "High-Fidelity": {"k": 15, "b_dist": 12, "density": 0.0025}
    }
    
    preset_results = []
    for name, params in styles.items():
        print(f"[*] Evaluating Preset: {name} (k={params['k']}, b_dist={params['b_dist']}, density={params['density']})...")
        res = run_benchmark_run(processor, params['k'], params['b_dist'], params['density'])
        res["name"] = name
        preset_results.append(res)
        
    # 2. Benchmarking K-Means Cluster Sweeps
    print("[*] Running parameter sweep: K-Means clusters (K)...")
    k_results = []
    for k in [4, 8, 12, 16, 20, 24]:
        res = run_benchmark_run(processor, k=k, b_dist=15, density=0.001)
        k_results.append(res)
        
    # 3. Benchmarking Boundary Spacing Sweeps
    print("[*] Running parameter sweep: Boundary point spacing (b_dist)...")
    b_dist_results = []
    for b_dist in [5, 10, 15, 20, 25, 30, 40]:
        res = run_benchmark_run(processor, k=15, b_dist=b_dist, density=0.001)
        b_dist_results.append(res)
        
    # 4. Benchmarking Density Sweeps
    print("[*] Running parameter sweep: Interior point density...")
    density_results = []
    for density in [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005]:
        res = run_benchmark_run(processor, k=15, b_dist=15, density=density)
        density_results.append(res)
        
    # Save All to CSV
    csv_file = "benchmark_results.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Preset_Name", "k", "b_dist", "density", "Triangles", "Duration_Sec", "MSE", "SSIM", "Simplification_Rate_Pct"])
        
        # Write presets
        for r in preset_results:
            writer.writerow(["Preset", r["name"], r["k"], r["b_dist"], r["density"], r["triangles"], f"{r['duration']:.4f}", f"{r['mse']:.2f}", f"{r['ssim']:.4f}", f"{r['simplification_rate']:.2f}"])
        # Write k sweep
        for r in k_results:
            writer.writerow(["K_Sweep", "N/A", r["k"], r["b_dist"], r["density"], r["triangles"], f"{r['duration']:.4f}", f"{r['mse']:.2f}", f"{r['ssim']:.4f}", f"{r['simplification_rate']:.2f}"])
        # Write b_dist sweep
        for r in b_dist_results:
            writer.writerow(["BDist_Sweep", "N/A", r["k"], r["b_dist"], r["density"], r["triangles"], f"{r['duration']:.4f}", f"{r['mse']:.2f}", f"{r['ssim']:.4f}", f"{r['simplification_rate']:.2f}"])
        # Write density sweep
        for r in density_results:
            writer.writerow(["Density_Sweep", "N/A", r["k"], r["b_dist"], r["density"], r["triangles"], f"{r['duration']:.4f}", f"{r['mse']:.2f}", f"{r['ssim']:.4f}", f"{r['simplification_rate']:.2f}"])
            
    print(f"[+] All raw benchmark data saved to {csv_file}")
    
    # ---------------- PLOTTING AND VISUALS GENERATION ----------------
    
    # Matplotlib styling for professional reports
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # --- PLOT 1: Style Presets Comparison Bar Graphs ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    names = [r["name"] for r in preset_results]
    colors = ['#f59e0b', '#ec4899', '#3b82f6'] # Amber, Pink, Blue
    
    # Triangles Count
    bars = axs[0, 0].bar(names, [r["triangles"] for r in preset_results], color=colors, edgecolor='#475569', width=0.5)
    axs[0, 0].set_title("Triangle Density by Preset", fontsize=11, weight='bold', color='#1e293b')
    axs[0, 0].set_ylabel("Number of Triangles", fontsize=9, color='#475569')
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.02), f"{yval:,}", ha='center', va='bottom', fontsize=9, weight='bold')
        
    # Processing Latency
    bars = axs[0, 1].bar(names, [r["duration"] for r in preset_results], color=colors, edgecolor='#475569', width=0.5)
    axs[0, 1].set_title("Compute Time by Preset", fontsize=11, weight='bold', color='#1e293b')
    axs[0, 1].set_ylabel("Execution Time (seconds)", fontsize=9, color='#475569')
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.02), f"{yval:.2f}s", ha='center', va='bottom', fontsize=9, weight='bold')
        
    # Reconstruction Quality (SSIM)
    bars = axs[1, 0].bar(names, [r["ssim"] for r in preset_results], color=colors, edgecolor='#475569', width=0.5)
    axs[1, 0].set_title("Structural Similarity Index (SSIM)", fontsize=11, weight='bold', color='#1e293b')
    axs[1, 0].set_ylabel("SSIM Value (0 to 1)", fontsize=9, color='#475569')
    axs[1, 0].set_ylim(0, 1.05)
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        axs[1, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, weight='bold')
        
    # Simplification/Compression Rate
    bars = axs[1, 1].bar(names, [r["simplification_rate"] for r in preset_results], color=colors, edgecolor='#475569', width=0.5)
    axs[1, 1].set_title("Data Compression / Simplification Rate", fontsize=11, weight='bold', color='#1e293b')
    axs[1, 1].set_ylabel("Simplification Rate (%)", fontsize=9, color='#475569')
    axs[1, 1].set_ylim(0, 110)
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, weight='bold')
        
    plt.suptitle("Comparative Evaluation of Triangulation Presets", fontsize=14, weight='bold', color='#0f172a', y=0.98)
    plt.tight_layout()
    plt.savefig("preset_comparison.png", dpi=300)
    plt.close()
    print("[+] Saved preset_comparison.png")
    
    # --- PLOT 2: Parameter K-Means Sweep Charts ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    k_vals = [r["k"] for r in k_results]
    
    # Left subplot: SSIM and Time vs K
    color = '#1e3a8a'
    axs[0].set_xlabel("Number of Color Clusters (K)", weight='bold', color='#334155')
    axs[0].set_ylabel("Structural Similarity Index (SSIM)", color=color, weight='bold')
    line1 = axs[0].plot(k_vals, [r["ssim"] for r in k_results], color=color, marker='o', linewidth=2, label="SSIM")
    axs[0].tick_params(axis='y', labelcolor=color)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    ax0_twin = axs[0].twinx()
    color_t = '#ea580c'
    ax0_twin.set_ylabel("Processing Latency (seconds)", color=color_t, weight='bold')
    line2 = ax0_twin.plot(k_vals, [r["duration"] for r in k_results], color=color_t, marker='s', linewidth=2, linestyle='--', label="Duration")
    ax0_twin.tick_params(axis='y', labelcolor=color_t)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axs[0].legend(lines, labels, loc='lower right')
    axs[0].set_title("Effect of Color Clusters K on SSIM & Duration", fontsize=11, weight='bold')
    
    # Right subplot: Simplification and Triangles vs K
    color = '#10b981'
    axs[1].set_xlabel("Number of Color Clusters (K)", weight='bold', color='#334155')
    axs[1].set_ylabel("Simplification Rate (%)", color=color, weight='bold')
    line3 = axs[1].plot(k_vals, [r["simplification_rate"] for r in k_results], color=color, marker='^', linewidth=2, label="Simplification Rate")
    axs[1].tick_params(axis='y', labelcolor=color)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    ax1_twin = axs[1].twinx()
    color_t = '#6366f1'
    ax1_twin.set_ylabel("Triangle Count", color=color_t, weight='bold')
    line4 = ax1_twin.plot(k_vals, [r["triangles"] for r in k_results], color=color_t, marker='d', linewidth=2, linestyle=':', label="Triangles")
    ax1_twin.tick_params(axis='y', labelcolor=color_t)
    
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    axs[1].legend(lines, labels, loc='upper right')
    axs[1].set_title("Effect of Color Clusters K on Mesh Density", fontsize=11, weight='bold')
    
    plt.suptitle("Algorithmic Sensitivity: K-Means Segmentation Scale", fontsize=13, weight='bold', color='#0f172a')
    plt.tight_layout()
    plt.savefig("param_k_variation.png", dpi=300)
    plt.close()
    print("[+] Saved param_k_variation.png")
    
    # --- PLOT 3: Parameter b_dist (Boundary Spacing) Sweep Charts ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    b_vals = [r["b_dist"] for r in b_dist_results]
    
    # Left subplot: SSIM and Time vs b_dist
    color = '#1e3a8a'
    axs[0].set_xlabel("Boundary Point Spacing (pixels)", weight='bold', color='#334155')
    axs[0].set_ylabel("Structural Similarity Index (SSIM)", color=color, weight='bold')
    line1 = axs[0].plot(b_vals, [r["ssim"] for r in b_dist_results], color=color, marker='o', linewidth=2, label="SSIM")
    axs[0].tick_params(axis='y', labelcolor=color)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    ax0_twin = axs[0].twinx()
    color_t = '#ea580c'
    ax0_twin.set_ylabel("Processing Latency (seconds)", color=color_t, weight='bold')
    line2 = ax0_twin.plot(b_vals, [r["duration"] for r in b_dist_results], color=color_t, marker='s', linewidth=2, linestyle='--', label="Duration")
    ax0_twin.tick_params(axis='y', labelcolor=color_t)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axs[0].legend(lines, labels, loc='upper right')
    axs[0].set_title("Effect of Boundary Spacing on SSIM & Duration", fontsize=11, weight='bold')
    
    # Right subplot: Simplification and Triangles vs b_dist
    color = '#10b981'
    axs[1].set_xlabel("Boundary Point Spacing (pixels)", weight='bold', color='#334155')
    axs[1].set_ylabel("Simplification Rate (%)", color=color, weight='bold')
    line3 = axs[1].plot(b_vals, [r["simplification_rate"] for r in b_dist_results], color=color, marker='^', linewidth=2, label="Simplification Rate")
    axs[1].tick_params(axis='y', labelcolor=color)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    ax1_twin = axs[1].twinx()
    color_t = '#6366f1'
    ax1_twin.set_ylabel("Triangle Count", color=color_t, weight='bold')
    line4 = ax1_twin.plot(b_vals, [r["triangles"] for r in b_dist_results], color=color_t, marker='d', linewidth=2, linestyle=':', label="Triangles")
    ax1_twin.tick_params(axis='y', labelcolor=color_t)
    
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    axs[1].legend(lines, labels, loc='lower right')
    axs[1].set_title("Effect of Boundary Spacing on Mesh Density", fontsize=11, weight='bold')
    
    plt.suptitle("Algorithmic Sensitivity: Boundary Point Spacing (b_dist)", fontsize=13, weight='bold', color='#0f172a')
    plt.tight_layout()
    plt.savefig("param_b_dist_variation.png", dpi=300)
    plt.close()
    print("[+] Saved param_b_dist_variation.png")
    
    # --- PLOT 4: Parameter Density (Interior Point) Sweep Charts ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    d_vals = [r["density"] for r in density_results]
    
    # Left subplot: SSIM and Time vs density
    color = '#1e3a8a'
    axs[0].set_xlabel("Interior Random Point Density", weight='bold', color='#334155')
    axs[0].set_ylabel("Structural Similarity Index (SSIM)", color=color, weight='bold')
    line1 = axs[0].plot(d_vals, [r["ssim"] for r in density_results], color=color, marker='o', linewidth=2, label="SSIM")
    axs[0].tick_params(axis='y', labelcolor=color)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    ax0_twin = axs[0].twinx()
    color_t = '#ea580c'
    ax0_twin.set_ylabel("Processing Latency (seconds)", color=color_t, weight='bold')
    line2 = ax0_twin.plot(d_vals, [r["duration"] for r in density_results], color=color_t, marker='s', linewidth=2, linestyle='--', label="Duration")
    ax0_twin.tick_params(axis='y', labelcolor=color_t)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axs[0].legend(lines, labels, loc='lower right')
    axs[0].set_title("Effect of Interior Density on SSIM & Duration", fontsize=11, weight='bold')
    
    # Right subplot: Simplification and Triangles vs density
    color = '#10b981'
    axs[1].set_xlabel("Interior Random Point Density", weight='bold', color='#334155')
    axs[1].set_ylabel("Simplification Rate (%)", color=color, weight='bold')
    line3 = axs[1].plot(d_vals, [r["simplification_rate"] for r in density_results], color=color, marker='^', linewidth=2, label="Simplification Rate")
    axs[1].tick_params(axis='y', labelcolor=color)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    ax1_twin = axs[1].twinx()
    color_t = '#6366f1'
    ax1_twin.set_ylabel("Triangle Count", color=color_t, weight='bold')
    line4 = ax1_twin.plot(d_vals, [r["triangles"] for r in density_results], color=color_t, marker='d', linewidth=2, linestyle=':', label="Triangles")
    ax1_twin.tick_params(axis='y', labelcolor=color_t)
    
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    axs[1].legend(lines, labels, loc='upper left')
    axs[1].set_title("Effect of Interior Density on Mesh Density", fontsize=11, weight='bold')
    
    plt.suptitle("Algorithmic Sensitivity: Interior Point Density (Density)", fontsize=13, weight='bold', color='#0f172a')
    plt.tight_layout()
    plt.savefig("param_density_variation.png", dpi=300)
    plt.close()
    print("[+] Saved param_density_variation.png")
    
    # ---------------- GENERATE IMAGES OF THE TABLES ----------------
    
    # Style Presets Table
    preset_headers = ["Preset Style", "K Clusters", "Boundary Spacing (px)", "Interior Density", "Triangles", "MSE", "SSIM", "Simplification"]
    preset_rows = []
    for r in preset_results:
        preset_rows.append([
            r["name"],
            str(r["k"]),
            str(r["b_dist"]),
            f"{r['density']:.4f}",
            f"{r['triangles']:,}",
            f"{r['mse']:.2f}",
            f"{r['ssim']:.4f}",
            f"{r['simplification_rate']:.2f}%"
        ])
    save_table_as_image(preset_headers, preset_rows, "table_presets.png", "Comparative Benchmarks for Visual Presets")
    print("[+] Saved table_presets.png")
    
    # K clusters Table
    k_headers = ["K Clusters", "Boundary Spacing", "Interior Density", "Triangles", "Latency (s)", "MSE", "SSIM", "Simplification"]
    k_rows = []
    for r in k_results:
        k_rows.append([
            str(r["k"]),
            str(r["b_dist"]),
            f"{r['density']:.4f}",
            f"{r['triangles']:,}",
            f"{r['duration']:.4f}s",
            f"{r['mse']:.2f}",
            f"{r['ssim']:.4f}",
            f"{r['simplification_rate']:.2f}%"
        ])
    save_table_as_image(k_headers, k_rows, "table_k_sweep.png", "Parameter Sweep: Color Clusters (K)")
    print("[+] Saved table_k_sweep.png")
    
    # Boundary Spacing Table
    b_headers = ["Boundary Spacing", "K Clusters", "Interior Density", "Triangles", "Latency (s)", "MSE", "SSIM", "Simplification"]
    b_rows = []
    for r in b_dist_results:
        b_rows.append([
            str(r["b_dist"]),
            str(r["k"]),
            f"{r['density']:.4f}",
            f"{r['triangles']:,}",
            f"{r['duration']:.4f}s",
            f"{r['mse']:.2f}",
            f"{r['ssim']:.4f}",
            f"{r['simplification_rate']:.2f}%"
        ])
    save_table_as_image(b_headers, b_rows, "table_b_dist_sweep.png", "Parameter Sweep: Boundary Point Spacing (b_dist)")
    print("[+] Saved table_b_dist_sweep.png")

    # Interior Density Table
    d_headers = ["Interior Density", "K Clusters", "Boundary Spacing", "Triangles", "Latency (s)", "MSE", "SSIM", "Simplification"]
    d_rows = []
    for r in density_results:
        d_rows.append([
            f"{r['density']:.4f}",
            str(r["k"]),
            str(r["b_dist"]),
            f"{r['triangles']:,}",
            f"{r['duration']:.4f}s",
            f"{r['mse']:.2f}",
            f"{r['ssim']:.4f}",
            f"{r['simplification_rate']:.2f}%"
        ])
    save_table_as_image(d_headers, d_rows, "table_density_sweep.png", "Parameter Sweep: Interior Seed Point Density")
    print("[+] Saved table_density_sweep.png")
    
    print("[+] Benchmarking completed successfully!")

if __name__ == "__main__":
    main()
