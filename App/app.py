import gradio as gr
import skops.io as sio
import os
model_path = os.path.join(os.path.dirname(__file__), "Model", "drug_pipeline.skops")


unknown_types = sio.get_untrusted_types(file=model_path)
print("Untrusted types:", unknown_types)

pipe = sio.load(
    model_path,
    trusted=unknown_types
)

print("Model loaded successfully")



def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    if pipe is None:
        return "Model not loaded"

    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]

    return f"Predicted Drug: {predicted_drug}"




inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]

outputs = [gr.Label(num_top_classes=5)]

examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

title = "Drug Classification"
description = "Enter the details to correctly identify Drug type?"
article = "This app is a part of the Beginner's Guide to CI/CD for Machine Learning."


gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
).launch(theme=gr.themes.Soft())
