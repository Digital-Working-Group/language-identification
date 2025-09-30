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

script_dir = Path(__file__).resolve().parent

def run_prediction(filepath, **kwargs):
    """
    Run LI on a given file
    """
    input_fp_path = Path(filepath)

    model_id = kwargs.get("model_id")
    top_n_predictions = kwargs.get("top_n_predictions", 10)
    output_fname = kwargs.get('output_fname', Path(input_fp_path.name).stem)
    output_dir = kwargs.get('output_dir', input_fp_path.parent / "output")

    iso_now = datetime.datetime.now().isoformat().replace(':', '-').replace('.', '-')
    output_dir = script_dir.parent / "sample_files" / "output" / model_id.replace('/', '_') / iso_now
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir = script_dir.parent / "mappings"

    print(f"LOADING MODEL: {model_id}")
    model, feature_extractor = load_model(model_id)
    print(f"LOADING DATA: {filepath}")
    audio_array = load_local_data(filepath)
    print("LOADING MAPPING")
    model_id_to_global_id, _ = load_mappings(mappings_dir, model_id)
    print("PREDICTING LANGUAGE")
    prediction = predict(model, audio_array, feature_extractor, model_id_to_global_id, top_n_predictions)
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

def predict(model, audio_array, feature_extractor, model_id_to_global_id, top_n_predictions=10):
    """
    Prediction on an audio_array of a single file using specified model
    """
    inputs = feature_extractor(audio_array["array"], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)
        # if num_prediction=None, return all
        if top_n_predictions is None:
            top_n_predictions = probabilities.size(-1)

        # get ids with highest predicted score (number defined by top_n_predictions)
        top_probabilities, top_lang_ids = torch.topk(probabilities, k=top_n_predictions, dim=-1)
        top_probabilities = top_probabilities.tolist()[0]
        top_lang_ids = top_lang_ids.tolist()[0]

        top_predictions = []
        for lang_id, confidence in zip(top_lang_ids, top_probabilities):
            # map to human-readable languages
            global_id = model_id_to_global_id[lang_id]
            lang_obj = global_id_to_lang(global_id)
            prediction = {
              "lang": lang_obj.name,
              "confidence": confidence,
              "model_lang_id": lang_id,
              "global_id": global_id
              }
            top_predictions.append(prediction)
    return top_predictions

def write_lang_id_json(prediction, output_fp):
    """
    Writes all output data to a JSON file
    """
    with open(f"{output_fp}.json", "w") as out_file:
        json.dump(prediction, out_file, indent=4)
    print(f"Wrote: {output_fp}")
