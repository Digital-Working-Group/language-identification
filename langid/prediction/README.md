# Language Identification: Prediction

This repository runs language identification (langid) models on different audio files.

| Table of Contents |
|---|
| [Installation and Setup](#installation-and-setup)|
| [Usage Example](#usage-example) |
| [Models](#models) |
| [Citations](#citations) |

## Installation and Setup

### Python Requirements
```
langid
   |-- prediction
   |   |-- requirements
   |   |   |-- py3-10-11
   |   |   |   |-- Dockerfile
   |   |   |   |-- build_docker.sh
   |   |   |   |-- docker_pip-licenses.md
   |   |   |   |-- pip-licenses.md
   |   |   |   |-- requirements.txt
   |   |   |   |-- run_docker.sh
```

Check your Python version:
```sh
python --version
```
See [Anaconda](https://www.anaconda.com/download/success) as an option to switch between Python versions. This repository has been tested with Python 3.10.11.

Install requirements for Python 3.10.11:
```sh
pip install -r requirements/py3-10-11/requirements.txt ## Python 3.10.11 requirements
```

Note: you may use the pip install command described above even if you are working with a different Python version, but you may need to adjust the requirements.txt file to fit any dependencies specific to that Python version.

### Requirements.txt License Information
License information for each set of requirements.txt can be found in their respective `pip-licenses.md` file within the requirements/python[version] folders.

### Docker Support
[Docker](https://docs.docker.com/engine/install/) support can be found via the `Dockerfile` and `build_docker.sh` and `run_docker.sh` files.

Please see Docker's documentation for more information ([docker build](https://docs.docker.com/build/), [Dockerfile](https://docs.docker.com/build/concepts/dockerfile/), [docker run](https://docs.docker.com/reference/cli/docker/container/run/)).

## Usage Example
See [main.py](main.py) for usage examples.

```python
from langid_predict import run_prediction

def main():
    """
    Runs lang id example
    """
    kwargs_whisper = {'model_id': 'sanchit-gandhi/whisper-medium-fleurs-lang-id'}
    run_prediction('sample_files/first_minute_Sample_HV_Clip.wav', **kwargs_whisper)
    run_prediction('sample_files/100yearsofsolitude_span.wav', **kwargs_whisper)
    run_prediction('sample_files/mandarin_short.wav', **kwargs_whisper)

    kwargs_fb = {'model_id': 'facebook/mms-lid-4017'}
    run_prediction('sample_files/first_minute_Sample_HV_Clip.wav', **kwargs_fb)
    run_prediction('sample_files/100yearsofsolitude_span.wav', **kwargs_fb)
    run_prediction('sample_files/mandarin_short.wav', **kwargs_fb)

if __name__ == '__main__':
    main()
```

### Arguments
The `predict_langid.run_prediction()` function takes in an audio input filepath (`input_fp`) and a set of keyword arguments to define the model and output.

`model_id` is the id of the langid model to use during the predictions task.
See [Models](#models) for a list of suggested and compatible models.

#### run_asr: kwargs
| Keyword Argument | Type | Description | Default Value |
|---|---|---|---|
| output_fname | str | The desired base filename of the output files. | Basename of input_fp. |
| top_n_predictions | int | Return the top N language predictions. If set to None, returns all language predictions. | 10 |
| output_dir | str | The desired root folder to place output files. | "output/" in the base directory of input_fp. |
| model_id | str | The id of the desired model. | None |

Note: If you wish to produce a JSON with all predicted languages and their probabilities, you may set top_n_predictions=None.

### Sample Input and Output Files

```
langid
   |-- prediction
   |   |-- sample_files
   |   |   |-- 100yearsofsolitude_span.wav
   |   |   |-- first_minute_Sample_HV_Clip.wav
   |   |   |-- mandarin_short.wav
   |   |   |-- output
   |   |   |   |-- facebook_mms-lid-4017
   |   |   |   |   |-- 2025-09-30T16-32-32-569644
   |   |   |   |   |   |-- first_minute_Sample_HV_Clip.json
   |   |   |   |   |-- 2025-09-30T16-32-49-504450
   |   |   |   |   |   |-- 100yearsofsolitude_span.json
   |   |   |   |   |-- 2025-09-30T16-32-58-757270
   |   |   |   |   |   |-- mandarin_short.json
   |   |   |   |-- sanchit-gandhi_whisper-medium-fleurs-lang-id
   |   |   |   |   |-- 2025-09-30T16-32-21-003373
   |   |   |   |   |   |-- first_minute_Sample_HV_Clip.json
   |   |   |   |   |-- 2025-09-30T16-32-26-742124
   |   |   |   |   |   |-- 100yearsofsolitude_span.json
   |   |   |   |   |-- 2025-09-30T16-32-29-735893
```

## Output Examples
| Key | Description | Example |
| - | - | - |
| lang | The ISO639 predicted language. If 'is_iso639' is 0, then lang is a string abbreviation of a language that isn't currently in the ISO639 database. | "English" (is_iso639=1), "prp" (is_iso639=0) |
| confidence | The probability/confidence score from the language identification model. | 0.999 |
| model_lang_id | The model's internal ID integer that represents the predicted language. | 2 |
| global_id | A global integer ID that maps to a language. | 3048 |
| is_iso639 | If 0, then "lang" is a string abbreviation of a language that isn't currently in the ISO639 database. If 1, then "lang" is the ISO639 langauge name. | 1 |

```yaml
[
    {
        "lang": "English",
        "confidence": 0.9994128942489624,
        "model_lang_id": 2,
        "global_id": 3048,
        "is_iso639": 1
    },
    {
        "lang": "Welsh",
        "confidence": 1.6988031347864307e-05,
        "model_lang_id": 51,
        "global_id": 1988,
        "is_iso639": 1
    },
    {
        "lang": "Slovenian",
        "confidence": 1.3651256267621648e-05,
        "model_lang_id": 29,
        "global_id": 12475,
        "is_iso639": 1
    },
    {
        "lang": "Tatar",
        "confidence": 1.1638738214969635e-05,
        "model_lang_id": 27,
        "global_id": 12863,
        "is_iso639": 1
    },
    {
        "lang": "Mandarin Chinese",
        "confidence": 8.861955393513199e-06,
        "model_lang_id": 1,
        "global_id": 1677,
        "is_iso639": 1
    },
    ...
]

```
## Models
See HuggingFace for more information about each model.

- [facebook/mms-lid-4017](https://huggingface.co/facebook/mms-lid-4017)
- [sanchit-gandhi/whisper-medium-fleurs](https://huggingface.co/sanchit-gandhi/whisper-medium-fleurs-lang-id)

# Performance
The following table lists performance of select model and [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) dataset combinations on small subsets of the datasets (due to performance limitations).
For more information, a model's card on HuggingFace may have their own benchmark results.

| Model \ Dataset                                   | PolyAI/minds14                                    |
| ------------------------------------------------  | ------------------------------------------------  |
| **sanchit-gandhi/whisper-medium-fleurs-lang-id**  | Accuracy: 62%, F1 Macro: 0.19, F1 Micro: 0.62     |
| **facebook/mms-lid-4017**                         | Accuracy: 88%, F1 Macro: 0.23, F1 Micro: 0.88     |

## Citations

```bibtex
@article{pratap2023mms,
  title={Scaling Speech Technology to 1,000+ Languages},
  author={Vineel Pratap and Andros Tjandra and Bowen Shi and Paden Tomasello and Arun Babu and Sayani Kundu and Ali Elkahky and Zhaoheng Ni and Apoorv Vyas and Maryam Fazel-Zarandi and Alexei Baevski and Yossi Adi and Xiaohui Zhang and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
journal={arXiv},
year={2023}
}

@article{DBLP:journals/corr/abs-2104-08524,
  author    = {Daniela Gerz and
               Pei{-}Hao Su and
               Razvan Kusztos and
               Avishek Mondal and
               Michal Lis and
               Eshan Singhal and
               Nikola Mrksic and
               Tsung{-}Hsien Wen and
               Ivan Vulic},
  title     = {Multilingual and Cross-Lingual Intent Detection from Spoken Data},
  journal   = {CoRR},
  volume    = {abs/2104.08524},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.08524},
  eprinttype = {arXiv},
  eprint    = {2104.08524},
  timestamp = {Mon, 26 Apr 2021 17:25:10 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2104-08524.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
