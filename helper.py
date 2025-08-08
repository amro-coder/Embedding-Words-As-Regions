import string
import re
import numpy as np
import torch
import yaml
from types import SimpleNamespace as NS
from pathlib import Path
import json

from models import *

PARENT_FOLDER_LOCATION = Path(__file__).resolve().parent

remove_punctuation_table = str.maketrans('', '', string.punctuation)
split_numbers_from_letters = re.compile(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])')
replace_tags = {"@card@", "@ord@"}

def clean(tokens):
    cleaned_tokens = []
    append  = cleaned_tokens.append           # local-variable lookup is cheap
    transition = remove_punctuation_table
    split = split_numbers_from_letters.split

    for tok in tokens:
        # 1) ASCII filter
        if not tok.isascii():
            continue

        # 2) tag replacement
        if tok in replace_tags:
            tok = "7" # any number would work. 7 is my lukcy number

        # 3) strip punctuation once
        tok = tok.translate(transition)
        if not tok:
            continue

       # 4) split words from numbers and convert numbers to <num> and words to lowercase words
        if tok.isdigit():
            append("<num>")
            continue
        if tok.isalpha():
            append(tok.lower())
            continue

        # only mixed tokens reach the regex
        for p in split(tok):
            if not p:
                continue
            append("<num>" if p.isdigit() else p.lower())

    return cleaned_tokens

def save_model_weights(model, file_path):
    np_state = {k:v.cpu().numpy() for k,v in model.state_dict().items()}
    np.savez_compressed(file_path, **np_state)
    return

def load_model_weights(model, file_path):
    np_state = np.load(file_path)
    state_dict = {k: torch.from_numpy(np_state[k]) for k in np_state.files}
    model.load_state_dict(state_dict)
    return 

def save_config_file(config, folder_path):
    with open(folder_path / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False) 
    return

def load_config_file(folder_path):
        with open(folder_path / "config.yaml", "r") as config_file:   
            return yaml.safe_load(config_file)
        
def get_pretrained_model(model_folder):
    config = load_config_file(model_folder)
    config = NS(**config)

    # unpack needed parameters
    
    # training paramters
    using_CBOW_style = config.CBOW
    model_name = config.model_name
    num_learnable_parameters = config.num_learnable_parameters

    # Dataset parameters
    num_negative_samples = config.num_negative_samples
    sub_sampling_threshold = config.sub_sampling_threshold
    max_window_size = config.max_window_size
    dataset_path = config.CBOW_training_examples_dir if using_CBOW_style else config.SkipGram_training_examples_dir
    
    dataset_hyperparameters_path = f"{num_negative_samples}_{sub_sampling_threshold}_{max_window_size}"

    train_dataset_folder = PARENT_FOLDER_LOCATION / dataset_path / dataset_hyperparameters_path

    with open(train_dataset_folder / "word_to_index.json", "r", encoding="ascii") as file:
        word_to_index = json.load(file)

    vocab_size = len(word_to_index)
    embedding_size = num_learnable_parameters//2 if model_name != "word2vec" else num_learnable_parameters

    # Get correct model architecture
    if using_CBOW_style:
        if model_name == "word2vec":
            model = Word2VecCBOW(vocab_size, embedding_size, num_negative_samples)
        elif model_name == "word2box":
            model = Word2BoxCBOW(vocab_size, embedding_size, num_negative_samples)
        elif model_name == "word2ellipsoid":
            model = Word2EllipsoidCBOW(vocab_size, embedding_size, num_negative_samples)
        else:
            raise("Enter a vaild model name \nThe valid implemented models are word2vec, word2box, and word2ellipsoid")
    else:
        if model_name == "word2vec":
            model = Word2VecSkipGram(vocab_size, embedding_size, num_negative_samples)
        elif model_name == "word2box":
            model = Word2BoxSkipGram(vocab_size, embedding_size, num_negative_samples)
        elif model_name == "word2ellipsoid":
            model = Word2EllipsoidSkipGram(vocab_size, embedding_size, num_negative_samples)
        else:
            raise("Enter a vaild model name \nThe valid implemented models are word2vec, word2box, and word2ellipsoid")
    
    load_model_weights(model, model_folder / "weights.npz")

    return model, word_to_index
