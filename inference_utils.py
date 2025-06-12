import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json

# Constants
IMG_SIZE = 240
MODEL_PATH = "model/efficientnetb1_plant_final.weights.h5"
CLASS_NAMES_PATH = "model/class_names.json"

# Load CLASS_NAMES
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# Build model EXACTLY like in training
base_model = tf.keras.applications.EfficientNetB1(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
])

# Load weights
model.load_weights(MODEL_PATH)

# Preprocess image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Grad-CAM
def generate_gradcam(img_path, model, class_index, layer_name="efficientnetb1"):
    img_array = preprocess_image(img_path)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-6
    heatmap = heatmap.numpy()

    # Overlay heatmap
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    result_img = Image.fromarray(np.uint8(superimposed_img))

    return result_img

# Inference
def predict_plant_disease(image_path):
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array)[0]

    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index])
    label = CLASS_NAMES[class_index]

    return {label: confidence}


''' gradcam_img = generate_gradcam(image_path, model, class_index)
    we will disable gradcam for now, we need to rebuild the model in kaggle using functional API to for this to work'''
    
''' def build_model(num_classes=15):
    inputs = tf.keras.Input(shape=(240, 240, 3))
    base_model = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet', input_tensor=inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model = build_model()
    model.load_weights("model/efficientnetb1_plant_final.weights.h5")'''
