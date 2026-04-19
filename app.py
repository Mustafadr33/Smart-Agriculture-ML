import gradio as gr
import joblib
import pandas as pd

# Charger le bundle modèle
MODEL_PATH = "models/random_forest_crop_model.joblib"

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_columns = bundle.get("feature_columns", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

# Fonction de prédiction
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=feature_columns,
    )
    prediction = pipeline.predict(input_df)
    return prediction[0]

# Interface Gradio
interface = gr.Interface(
    fn=predict_crop,
    inputs=[
        gr.Number(label="Azote (N)"),
        gr.Number(label="Phosphore (P)"),
        gr.Number(label="Potassium (K)"),
        gr.Number(label="Température (°C)"),
        gr.Number(label="Humidité (%)"),
        gr.Number(label="pH"),
        gr.Number(label="Précipitations (mm)")
    ],
    outputs=gr.Textbox(label="Culture recommandée"),
    title="Système de recommandation de culture",
    description="Entrez les paramètres du sol et du climat pour prédire la meilleure culture."
)

interface.launch()
