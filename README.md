# Repository for speech and gestural alignment --> to be prepared for ICMI24
# Speech and Gestural Alignment Repository
This repository contains resources and scripts necessary for the project to be presented at ICMI24. It is focused on aligning speech with gestures.

## Getting Started
These instructions will guide you to setup and run the project on your local machine.

### Prerequisites
Setup virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt install libsox-dev
```

### Downloading Data
1. **Download Poses:** 
   - Download the pose data from [Google Drive](https://drive.google.com/file/d/15SwxhEXC4JOJ0XYiQ-WcrGSmEVVvdIDB/view).
   - Make sure the downloaded files are extracted and placed in the `data/final_poses/` directory within this repository.

### Data Preparation
2. **Extract Poses:** 
   - Run the following command to pre-process and prepare the poses from the data:
     ```bash
     python data/CABB_gen_audio_video_data_pre_training.py
     ```

### Model Training
3. **Train the Model:** 
   - After preparing the data, you can train the model by executing:
     ```bash
     python main_sup_pl.py
     ```

## Tests
The tests for certain modules can be executed with `pytest` package. Installation:

``` pip install -U pytest```

How to use:
- Run all tests showing printed statements:

    ```python -m pytest tests```

- Run all tests showing printed statements:

    ```python -m pytest -s tests```

- Run certain function in a test:

    ```python -m pytest -s -v tests/test_w2v2.py::test_wav2vec2_lora_setup```
    
    Runs `test_wav2vec2_lora_setup` function from `tests/test_w2v2.py`.
