import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# IEEE style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(FIG_DIR, exist_ok=True)

# ==============================================================================
# Visual 1: Grouped Bar Chart (Identification Accuracy & Detection Rate)
# ==============================================================================
def generate_bar_chart():
    methods = [
        'Raw\nDegraded',
        'Low-FaceNet\n(Stage 1)',
        'InstructIR\n(Alone)',
        'Zero-Shot\nCascade',
        'Fine-Tuned\nCascade (Ours)'
    ]

    top1_accuracy = [3.2, 12.6, 27.6, 62.6, 70.2]
    detection_rate = [41.0, 68.0, 56.4, 100.0, 100.0]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    rects1 = ax.bar(x - width/2, top1_accuracy, width, label='Top-1 Identification Acc (%)',
                    color='#2563eb', edgecolor='#1e40af', alpha=0.9, zorder=3)
    rects2 = ax.bar(x + width/2, detection_rate, width, label='Face Detection Rate (%)',
                    color='#f97316', edgecolor='#c2410c', alpha=0.9, zorder=3)

    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 115)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.legend(loc='upper left', framealpha=0.9)

    # Add numeric labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1e3a8a')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#9a3412')

    plt.tight_layout()
    pdf_path = os.path.join(FIG_DIR, 'identification_detection_comparison.pdf')
    png_path = os.path.join(FIG_DIR, 'identification_detection_comparison.png')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {pdf_path} and {png_path}")


# ==============================================================================
# Visual 2: Perception-Distortion Trade-off (PSNR vs Identification Accuracy)
# ==============================================================================
def generate_tradeoff_plot():
    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    # Data points: (PSNR, Accuracy)
    # LFN Baseline: base reference (Delta PSNR = 0)
    # Zero-Shot: +3.48 dB over LFN, 62.6% Acc
    # Fine-Tuned: +2.55 dB over LFN (-0.93 dB vs Zero-Shot), 70.2% Acc
    points = {
        'Low-FaceNet Baseline': (18.20, 12.6, '#dc2626', 's'),
        'Zero-Shot Cascade': (21.68, 62.6, '#2563eb', 'o'),
        'Fine-Tuned Cascade (Ours)': (20.75, 70.2, '#059669', '*')
    }

    for label, (psnr, acc, color, marker) in points.items():
        size = 140 if marker == '*' else 80
        ax.scatter(psnr, acc, color=color, s=size, marker=marker, zorder=4, label=label)

    # Annotations and arrows
    ax.annotate('Highest PSNR (21.68 dB)\nSub-optimal Identity (62.6%)',
                xy=(21.68, 62.6), xytext=(20.2, 53),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#2563eb'),
                fontsize=8.5, color='#1e40af')

    ax.annotate('Optimal Biometric Point\nHighest Top-1 Acc (70.2%)\n[-0.93 dB PSNR trade-off]',
                xy=(20.75, 70.2), xytext=(18.8, 76),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#059669'),
                fontsize=8.5, fontweight='bold', color='#065f46')

    ax.annotate('Low-FaceNet Front-End\n(12.6% Acc, Severe Noise)',
                xy=(18.20, 12.6), xytext=(18.4, 22),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#dc2626'),
                fontsize=8.5, color='#991b1b')

    # Dotted line showing trade-off trajectory
    ax.plot([18.20, 21.68, 20.75], [12.6, 62.6, 70.2],
            linestyle=':', color='#94a3b8', lw=1.5, zorder=2)

    ax.set_xlabel('Reconstruction Quality: PSNR (dB)')
    ax.set_ylabel('Biometric Recognition: Top-1 Accuracy (%)')
    ax.set_xlim(17.5, 22.8)
    ax.set_ylim(5, 88)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    pdf_path = os.path.join(FIG_DIR, 'perception_distortion_tradeoff.pdf')
    png_path = os.path.join(FIG_DIR, 'perception_distortion_tradeoff.png')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {pdf_path} and {png_path}")


if __name__ == '__main__':
    generate_bar_chart()
    generate_tradeoff_plot()
