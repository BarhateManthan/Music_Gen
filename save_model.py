import pickle
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# Load the MusicGen model
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Save the model as a pickle file
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Load the processor and save it as a pickle file
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
with open("processor.pkl", "wb") as processor_file:
    pickle.dump(processor, processor_file)
