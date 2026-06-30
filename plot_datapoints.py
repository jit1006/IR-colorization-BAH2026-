# plot_datapoints.py
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_datapoint_visuals():
    # 1. Dataset stats
    scenes = ["Urban Heat Island", "Agricultural Fields", "Forest Canopy", "Coastal Zone"]
    patches_per_scene = 36
    total_patches = 144
    
    # 80/20 split
    train_patches = 115
    val_patches = 29
    
    # Train / Val splits per scene (80% of 36 is ~29, 20% is ~7)
    train_per_scene = np.array([29, 29, 29, 29])
    val_per_scene = np.array([7, 7, 7, 7])

    # 2. Setup dark slate styling matching the project aesthetic
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#090d16')
    
    for ax in axs:
        ax.set_facecolor('#121824')
        ax.spines['bottom'].set_color('#2e374a')
        ax.spines['top'].set_color('#2e374a')
        ax.spines['left'].set_color('#2e374a')
        ax.spines['right'].set_color('#2e374a')
        ax.tick_params(colors='#9ca3af')
        ax.yaxis.label.set_color('#9ca3af')
        ax.xaxis.label.set_color('#9ca3af')
        ax.title.set_color('#f3f4f6')

    # --- PANEL 1: Stacked Bar Chart for Patches per Scene ---
    bars_train = axs[0].bar(scenes, train_per_scene, color='#3b82f6', width=0.5, label='Train Patches (80%)')
    bars_val = axs[0].bar(scenes, val_per_scene, bottom=train_per_scene, color='#10b981', width=0.5, label='Val Patches (20%)')
    
    axs[0].set_title('Datapoint Distribution by Geographic Landscape', fontsize=12, fontweight='bold', pad=15)
    axs[0].set_ylabel('Number of Patches')
    axs[0].set_ylim(0, 45)
    axs[0].grid(True, color='#2e374a', linestyle='--', alpha=0.5, axis='y')
    
    # Add values on top of bars
    for bar_t, bar_v in zip(bars_train, bars_val):
        y_t = bar_t.get_height()
        y_v = bar_v.get_height()
        axs[0].text(bar_t.get_x() + bar_t.get_width()/2.0, y_t/2.0, f'{y_t}', ha='center', va='center', color='white', fontweight='bold')
        axs[0].text(bar_v.get_x() + bar_v.get_width()/2.0, y_t + y_v/2.0, f'{y_v}', ha='center', va='center', color='white', fontweight='bold')
        axs[0].text(bar_v.get_x() + bar_v.get_width()/2.0, y_t + y_v + 1.5, f'Total: {y_t + y_v}', ha='center', va='bottom', color='#9ca3af', fontsize=9)
        
    axs[0].legend(facecolor='#121824', edgecolor='#2e374a')

    # --- PANEL 2: Donut Chart for Train/Val Split ---
    labels = ['Training Set\n(115 Patches)', 'Validation Set\n(29 Patches)']
    sizes = [train_patches, val_patches]
    colors = ['#3b82f6', '#10b981']
    
    # Create donut hole
    wedges, texts, autotexts = axs[1].pie(
        sizes, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=90, 
        pctdistance=0.75,
        textprops=dict(color='#9ca3af', fontweight='medium')
    )
    
    # Color percentage text inside sections
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        
    # Draw center circle to form donut
    centre_circle = plt.Circle((0,0), 0.55, fc='#121824', edgecolor='#2e374a', linewidth=1)
    axs[1].add_artist(centre_circle)
    
    # Add summary stats in donut center
    axs[1].text(0, 0, f'Total\n{total_patches}\nPatches', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    
    axs[1].set_title('Dataset Split Ratio', fontsize=12, fontweight='bold', pad=15)

    # 4. Text annotation for patch details below titles
    fig.suptitle('Landsat-9 Dataset Details & Splitting Metrics', fontsize=16, fontweight='bold', color='white', y=0.98)
    
    # Add info box at the bottom of the figure
    info_text = (
        "Dataset Details:\n"
        " • Patches extracted from 1000m x 1000m scenes representing 30km x 30km regions at 30m resolution.\n"
        " • Low-Resolution (LR) TIR inputs: 144 patches of size 64 x 64 px (representing 200m spatial resolution).\n"
        " • High-Resolution (HR) TIR targets & RGB composites: 144 patches of size 128 x 128 px (representing 100m spatial resolution).\n"
        " • Spatial Registration: A 2x2 grid cell block in 100m space maps exactly to a 1x1 grid cell in 200m space."
    )
    fig.text(0.08, 0.05, info_text, color='#9ca3af', fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=1', facecolor='#121824', edgecolor='#2e374a', alpha=0.8))

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Save directly to static folder
    output_dir = 'app/static'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'datapoints_distribution.png')
    
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Dataset visualization saved successfully to: {os.path.abspath(output_path)}")
    plt.close()

if __name__ == '__main__':
    generate_datapoint_visuals()
