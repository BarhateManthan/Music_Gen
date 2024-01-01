from flask import Flask, request, render_template
import torch
import pickle
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from IPython.display import Audio
import scipy

app = Flask(__name__)

# Load the model and processor from the pickle files
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("processor.pkl", "rb") as processor_file:
    processor = pickle.load(processor_file)

# Move the model to GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for generating music based on text input
@app.route('/generate_music', methods=['POST'])
def generate_music():
    text_input = request.form['text_input']

    # Process the text input
    inputs = processor(
        text=[text_input],
        padding=True,
        return_tensors="pt",
    )

    # Generate audio based on the text input
    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=1024)

    # Display the generated audio
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_path = "static/generated_audio.wav"
    scipy.io.wavfile.write(audio_path, rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

    return render_template('result.html', audio_path=audio_path)

if __name__ == '__main__':
    app.run(debug=True)
