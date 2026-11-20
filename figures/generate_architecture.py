import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13.5, 5), dpi=300)
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 5)
ax.axis('off')

# Input
ax.text(1.2, 2.7, 'Degraded Input\n(640x480)\nLow-Light + Noise + Blur', 
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='#ef4444', lw=1.5))

ax.annotate('', xy=(2.4, 2.7), xytext=(2.0, 2.7), arrowprops=dict(arrowstyle='->', lw=2, color='#1e293b'))

# Stage 1: Low-FaceNet
ax.text(3.7, 2.7, 'Stage 1: Low-FaceNet\nEnhanceNet Front-End\nRetinex: S / I\n(Luminance Recovery)', 
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#dbeafe', edgecolor='#3b82f6', lw=1.5))

ax.annotate('', xy=(5.5, 2.7), xytext=(5.0, 2.7), arrowprops=dict(arrowstyle='->', lw=2, color='#1e293b'))

# Intermediate output
ax.text(6.4, 2.7, 'Brightened Image\n(High Sensor Noise\n& Motion Blur)', 
        ha='center', va='center', fontsize=8.5, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef3c7', edgecolor='#f59e0b', lw=1.2))

ax.annotate('', xy=(7.7, 2.7), xytext=(7.3, 2.7), arrowprops=dict(arrowstyle='->', lw=2, color='#1e293b'))

# Text prompt
prompt_text = "Natural Language Prompt:\n'Deblur and denoise the image.'\n(MiniLM 256-D Embedding)"
ax.text(8.9, 4.3, prompt_text, 
        ha='center', va='center', fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#d1fae5', edgecolor='#10b981', lw=1.5))

ax.annotate('', xy=(8.9, 3.5), xytext=(8.9, 3.8), arrowprops=dict(arrowstyle='->', lw=2, color='#059669'))

# Stage 2: InstructIR
ax.text(8.9, 2.7, 'Stage 2: InstructIR\n(NAFNet Backbone)\nText-Gated Channel Masks\n(Denoise & Deblur)', 
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ede9fe', edgecolor='#8b5cf6', lw=1.5))

ax.annotate('', xy=(10.7, 2.7), xytext=(10.1, 2.7), arrowprops=dict(arrowstyle='->', lw=2, color='#1e293b'))

# Stage 3: Recognition output
recog_text = "Biometric Output\n(Native 640x480)\n\nRetinaFace (100% Detect)\n+\nFaceNet 128-D Embedding\n(70.2% Top-1 Accuracy)"
ax.text(12.1, 2.7, recog_text, 
        ha='center', va='center', fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fce7f3', edgecolor='#ec4899', lw=1.5))

# Fine-tuning loss
loss_text = "Biometric Fine-Tuning Supervision (Training Phase on LaPa-Face)\nL_Total = L1 + 1.0 * (1 - CosineSimilarity(FaceNet(pred), FaceNet(clean)))"
ax.text(8.9, 1.0, loss_text, 
        ha='center', va='center', fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffedd5', edgecolor='#ea580c', lw=1.2, ls='--'))

ax.annotate('', xy=(8.9, 1.9), xytext=(8.9, 1.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='#ea580c', ls='--'))

fig_dir = r'figures'
os.makedirs(fig_dir, exist_ok=True)
out_path = os.path.join(fig_dir, 'pipeline_architecture.pdf')
plt.tight_layout()
plt.savefig(out_path, format='pdf', bbox_inches='tight')
print('Successfully saved:', out_path)
