from transformers import MusicgenForConditionalGeneration, AutoProcessor
from IPython.display import Audio
import torch
import scipy

# Upgrade pip and install necessary packages

# Load the MusicGen model
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Move the model to GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# Unconditional Generation
unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=1024)

# Display the generated audios
sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)

# Save the generated audio as a .wav file
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

# Text-Conditional Generation
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# Process the text input
inputs = processor(
    text=["A prime ministers royal entry theme song in a gold mine, rock jazz, pop"],
    padding=True,
    return_tensors="pt",
)

# Generate audio based on the text input
audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=1024)

# Display the generated audio
Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)
scipy.io.wavfile.write("musicgen_out_final.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())