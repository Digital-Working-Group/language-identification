"""
main.py
Example usage of langid prediction scripts
"""
from langid_predict import run_prediction

def main():
    """
    Runs lang id example
    """
    kwargs_whisper = {'model_id': 'sanchit-gandhi/whisper-medium-fleurs-lang-id'}

    run_prediction('../sample_files/first_minute_Sample_HV_Clip.wav', **kwargs_whisper)
    run_prediction('../sample_files/100yearsofsolitude_span.wav', **kwargs_whisper)

    kwargs_fb = {'model_id': 'facebook/mms-lid-4017'}
    run_prediction('../sample_files/first_minute_Sample_HV_Clip.wav', **kwargs_fb)
    run_prediction('../sample_files/100yearsofsolitude_span.wav', **kwargs_fb)

if __name__ == '__main__':
    main()