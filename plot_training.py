# plot_training.py
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    # 1. Data from actual Super-Resolution (SRNet) training
    sr_epochs = np.arange(1, 21)
    # Combined Loss: L1 + 0.1*SSIM_loss + 1.0*Deg
    sr_train_loss = np.array([
        0.2487, 0.1042, 0.0785, 0.0714, 0.0624, 0.0619, 0.0629, 0.0526, 0.0470, 0.0546,
        0.0429, 0.0446, 0.0396, 0.0388, 0.0399, 0.0382, 0.0319, 0.0392, 0.0339, 0.0289
    ])
    # Combined Loss: L1 + 0.1*SSIM_loss
    sr_val_loss = np.array([
        0.1757, 0.1729, 0.1622, 0.1543, 0.1279, 0.1169, 0.0680, 0.0448, 0.0328, 0.0277,
        0.0325, 0.0349, 0.0324, 0.0288, 0.0223, 0.0242, 0.0248, 0.0250, 0.0206, 0.0234
    ])
    sr_train_ssim = np.array([
        0.5979, 0.7830, 0.8510, 0.8833, 0.8959, 0.9053, 0.9091, 0.9210, 0.9232, 0.9262,
        0.9338, 0.9336, 0.9407, 0.9427, 0.9422, 0.9484, 0.9510, 0.9468, 0.9493, 0.9554
    ])
    sr_val_ssim = np.array([
        0.6242, 0.6285, 0.6466, 0.6799, 0.7376, 0.7659, 0.8390, 0.8902, 0.9009, 0.9256,
        0.9260, 0.9301, 0.9238, 0.9358, 0.9454, 0.9312, 0.9459, 0.9289, 0.9432, 0.9463
    ])

    # 2. Data from actual Colorization GAN training
    gan_epochs = np.arange(1, 9)
    gan_loss_d = np.array([0.3880, 0.0939, 0.0536, 0.0331, 0.0215, 0.0124, 0.0084, 0.0066])
    gan_loss_g_adv = np.array([1.6769, 3.1623, 3.7884, 4.2314, 4.6504, 4.9740, 5.3387, 5.4906])
    
    # Train L1 is logged scaled by 100, convert back to standard L1
    gan_train_l1 = np.array([79.6429, 45.0481, 31.3439, 23.9326, 17.8316, 12.7912, 10.3022, 9.4032]) / 100.0
    gan_val_l1 = np.array([0.6877, 0.4139, 0.1982, 0.1211, 0.1378, 0.1166, 0.0595, 0.0695])

    # 3. Setup styling (Dark Slate aesthetic matching the web dashboard)
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#090d16')
    
    for ax in axs.flat:
        ax.set_facecolor('#121824')
        ax.spines['bottom'].set_color('#2e374a')
        ax.spines['top'].set_color('#2e374a')
        ax.spines['left'].set_color('#2e374a')
        ax.spines['right'].set_color('#2e374a')
        ax.tick_params(colors='#9ca3af')
        ax.yaxis.label.set_color('#9ca3af')
        ax.xaxis.label.set_color('#9ca3af')
        ax.title.set_color('#f3f4f6')
        ax.grid(True, color='#2e374a', linestyle='--', alpha=0.5)

    # --- PANEL 1: SRNet Loss Curves ---
    axs[0, 0].plot(sr_epochs, sr_train_loss, color='#3b82f6', marker='o', linewidth=2, label='Train Loss')
    axs[0, 0].plot(sr_epochs, sr_val_loss, color='#ef4444', marker='x', linewidth=2, label='Val Loss')
    axs[0, 0].set_title('Stage A: SRNet Loss Curves (L1 + SSIM + Deg)', fontsize=12, fontweight='bold', pad=10)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss Value')
    axs[0, 0].set_xticks(np.arange(0, 21, 2))
    axs[0, 0].legend(facecolor='#121824', edgecolor='#2e374a')

    # --- PANEL 2: SRNet SSIM (Structure Accuracy) Curves ---
    axs[0, 1].plot(sr_epochs, sr_train_ssim, color='#10b981', marker='o', linewidth=2, label='Train SSIM')
    axs[0, 1].plot(sr_epochs, sr_val_ssim, color='#f59e0b', marker='x', linewidth=2, label='Val SSIM')
    axs[0, 1].set_title('Stage A: SRNet Structural Accuracy (SSIM)', fontsize=12, fontweight='bold', pad=10)
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('SSIM Score (higher = better)')
    axs[0, 1].set_xticks(np.arange(0, 21, 2))
    axs[0, 1].set_ylim(0.5, 1.0)
    axs[0, 1].legend(facecolor='#121824', edgecolor='#2e374a', loc='lower right')

    # --- PANEL 3: Color GAN Adversarial Losses ---
    axs[1, 0].plot(gan_epochs, gan_loss_g_adv, color='#8b5cf6', marker='o', linewidth=2, label='Gen Adv Loss')
    axs[1, 0].plot(gan_epochs, gan_loss_d, color='#ec4899', marker='s', linewidth=2, label='Discrim Loss')
    axs[1, 0].set_title('Stage B: Color GAN Adversarial Loss Curves', fontsize=12, fontweight='bold', pad=10)
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss Value')
    axs[1, 0].set_xticks(gan_epochs)
    axs[1, 0].legend(facecolor='#121824', edgecolor='#2e374a')

    # --- PANEL 4: Color GAN L1 Reconstruction (Color Accuracy) ---
    axs[1, 1].plot(gan_epochs, gan_train_l1, color='#10b981', marker='o', linewidth=2, label='Train L1')
    axs[1, 1].plot(gan_epochs, gan_val_l1, color='#ef4444', marker='x', linewidth=2, label='Val L1')
    axs[1, 1].set_title('Stage B: Color GAN Reconstruction Accuracy (L1)', fontsize=12, fontweight='bold', pad=10)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('L1 Distance (lower = better)')
    axs[1, 1].set_xticks(gan_epochs)
    axs[1, 1].legend(facecolor='#121824', edgecolor='#2e374a')

    fig.suptitle('ISRO-SAC BAH 2026: Two-Stage Training Visualizations', fontsize=16, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout()
    output_path = 'training_curves.png'
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Beautiful training curve visuals saved to: {os.path.abspath(output_path)}")
    plt.close()

if __name__ == '__main__':
    generate_plots()
