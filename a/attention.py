import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Paths
INPUT_DIR = "/Users/admin/VSCODE/JABES/cnn_attention"
OUTPUT_DIR = "/Users/admin/VSCODE/JABES/cnn_attention_output"
MODEL_PATH = "/Users/admin/VSCODE/JABES/crypto_research_minute_fullimage/ADAUSDT/models/model_2024-09_1m_1week_w5.h5"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)

# Function to generate Grad-CAM heatmap
def get_grad_cam_heatmap(model, img_array, last_conv_layer_name='conv2d_1'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img_array]))
        predicted_class = tf.argmax(predictions[0]) if predictions.shape[-1] > 1 else 0
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

# Process and save attention visualizations
for img_name in os.listdir(INPUT_DIR):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, img_name)
        img = Image.open(img_path).convert("RGB").resize((64, 64))
        img_array = img_to_array(img) / 255.0

        # Save original image
        original_output_path = os.path.join(OUTPUT_DIR, f"original_{img_name}")
        img.save(original_output_path)

        # Generate and save heatmap
        heatmap = get_grad_cam_heatmap(model, img_array)
        heatmap_resized = tf.image.resize(np.expand_dims(heatmap, axis=-1), (64, 64))
        heatmap_resized = np.squeeze(heatmap_resized.numpy())

        plt.figure(figsize=(2, 2))
        plt.imshow(heatmap_resized, cmap='hot')
        plt.axis('off')
        heatmap_output_path = os.path.join(OUTPUT_DIR, f"heatmap_{img_name}")
        plt.savefig(heatmap_output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

        print(f"Saved: {original_output_path}")
        print(f"Saved: {heatmap_output_path}")

print("All images processed and saved separately!")
