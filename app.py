import gradio as gr
from inference_utils import predict_plant_disease

title = "🌿 Plant Disease Classifier (EfficientNetB1)"
description = """
Upload a plant leaf image, use your webcam, or paste an image from your clipboard.  
The model will predict the **plant + disease** and display the confidence.
"""

examples = [
    "assets/Pepper__bell___Bacterial_spot.jpg",
    "assets/Pepper__bell___healthy.jpg",
    "assets/Potato___Early_blight.jpg",
    "assets/Potato___healthy.jpg"
]

iface = gr.Interface(
    fn=predict_plant_disease,
    inputs=gr.Image(type="filepath", label="Input Leaf Image"),
    outputs=gr.Label(label="Prediction & Confidence"),
    title=title,
    description=description,
    examples=examples,
    flagging_mode="never",
    theme="default",
    live=False,
    cache_examples=False,
)

iface.launch()
