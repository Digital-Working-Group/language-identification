"""
lang_predict.py
Holds scripts for running language identification prediction
"""

import datetime
from pathlib import Path
import json

from datasets import Audio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import torch.nn.functional as F
from mapping_scripts.global_id_utils import global_id_to_lang


def run_prediction(filepath, **kwargs):
  """
  Run LI on a given file
  """
  input_fp_path = Path(filepath)

  model_id = kwargs.get("model_id")
  output_fname = kwargs.get('output_fname', Path(input_fp_path.name).stem)
  output_dir = kwargs.get('output_dir', input_fp_path.parent / "output")

  ## Build output path
  iso_now = datetime.datetime.now().isoformat().replace(':', '-').replace('.', '-')
  output_dir = Path("/app/sample_files/output") / model_id.replace('/', '_') / iso_now
  output_dir.mkdir(parents=True, exist_ok=True)
  mappings_dir = Path("/app/mappings")

  ## STEPS:
  # Load model 
  print("LOADING MODEL")
  model, feature_extractor = load_model(model_id)
  print("LOADING DATA")
  # Load data
  audio_array = load_local_data(filepath)
  ## Get model mappings
  ## I am not entirely sure that we need the mapping in this case, but maybe this is to make it more universal 
  print("LOADING MAPPING")
  model_id_to_global_id, _ = load_mappings(mappings_dir, model_id)
  ## Run predictions
  print("PREDICTING LANGUAGE")
  prediction = predict(model, audio_array, feature_extractor, model_id_to_global_id)
  ## Write output:
  write_lang_id_json(prediction, output_dir /output_fname)

def load_model(model_id):
  """
  Loads the model amd feature extractor according to model_id
  """
  model = AutoModelForAudioClassification.from_pretrained(model_id)
  feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
  return model, feature_extractor

def load_local_data(filepath):
  """
  loads local data into a dataset using HuggingFace Audio to extract audio from the files
  """
  return Audio(sampling_rate=16000).decode_example({"path": filepath, "bytes": None})

def load_mappings(mappings_dir, model_id):
  """
  Loads the mappings for the models 
  """
  model_mappings_dir = mappings_dir / "models" / model_id

  def load_mapping(path: Path):
    with open(path, "r") as in_file:
      mapping = json.load(in_file)
    mapping_integer_keys = {int(k): v for k, v in mapping.items()}
    return mapping_integer_keys

  ## Maps any deprecated language identifiers to their new values
  model_id_to_global_id = load_mapping(model_mappings_dir / "model_id_to_global_id.json")
  global_id_to_model_id = load_mapping(model_mappings_dir / "global_id_to_model_id.json")

  return model_id_to_global_id, global_id_to_model_id

def predict(model, audio_array, feature_extractor, model_id_to_global_id):
  """
  Prediction on an audio_array of a single file using specified model
  """
  inputs = feature_extractor(audio_array["array"], sampling_rate=16000, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=-1)
    # get id with highest predicted score
    predicted_id = probabilities.argmax(dim=-1).item()
    lang_obj = global_id_to_lang(model_id_to_global_id[predicted_id])
    print(f"Predicted id: {predicted_id}")
    # get the cofidence score
    confidence = probabilities.max(dim=-1).values.item()
    ## change to readable language labels
    prediction = {
      "lang": lang_obj.name,
      "confidence": confidence
      }
    print(f'Predicted Lang: {prediction}')
    print(f'lang_obj: {lang_obj}')
    print(f'Confidence: {confidence}')
  return prediction

def write_lang_id_json(prediction, output_fp):
    """
    Writes all output data to a JSON file
    """
    with open(f"{output_fp}.json", "w") as out_file:
        json.dump(prediction, out_file, indent=4)
    print(f"Wrote: {output_fp}")

# """# Make inferences"""
# def make_inferences(output_dir, model, model_dataset, compute_metrics):
#   """
#   Returns prediction output
#   """
#   args = TrainingArguments(
#       output_dir=output_dir,
#       per_device_eval_batch_size=1,
#       logging_steps=25,
#   )

#   trainer = Trainer(
#       args=args,
#       model=model,
#       eval_dataset=model_dataset,
#       compute_metrics=compute_metrics,
#   )

#   return trainer.predict(model_dataset)