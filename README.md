

### Installation and Setup
```bash 
pip install -r requirements.txt
``` 

Wandb Login- for the plots of training models by Huggingface Trainer API. It would require obtaining API key for wandb which can be obtained by following link: https://wandb.ai/authorize 
<br >
<br >
Run in terminal and paste the API key
```
wandb login
```

Load and save dataset in local to speed up the runs of files and avoid loading data from API all the time
```
python3 load_and_save_dataset.py
```

### Folder Structure

1. For training llama and saving the checkpoint: `llama_train.py`
2. For saving encodings of dataset for inference of model in cpp: `save_encodings.py`
3. For saving model in torchscript format: `save_model_in_torchscript.py`
4. Saved Torchscript Model: `traced_llama_4064_3.pt`
5. Saved Pytorch Models: in folder `trained-llama`
6. CPP Inference Code: in folder `cpp`

README in cpp folder has instructions for running the inference code






