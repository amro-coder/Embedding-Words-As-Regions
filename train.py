from pathlib import Path
import numpy as np
import json
import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from models import *
from losses import max_margin_loss, negative_sampling_loss
from helper import save_model_weights, save_config_file

import time

PARENT_FOLDER_LOCATION = Path(__file__).resolve().parent

def train_model(config):

    # Extract needed parameters from config

    # training paramters
    report_to_wandb = config.report_to_wandb
    using_CBOW_style = config.CBOW
    model_name = config.model_name
    model_path = config.CBOW_models_dir if using_CBOW_style else config.SkipGram_models_dir
    num_learnable_parameters = config.num_learnable_parameters
    epochs = config.epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    margin = config.margin

    # Dataset parameters
    num_negative_samples = config.num_negative_samples
    sub_sampling_threshold = config.sub_sampling_threshold
    max_window_size = config.max_window_size
    dataset_path = config.CBOW_training_examples_dir if using_CBOW_style else config.SkipGram_training_examples_dir
    
    dataset_hyperparameters_path = f"{num_negative_samples}_{sub_sampling_threshold}_{max_window_size}"
    model_hyperparameters_path = f"{batch_size}_{learning_rate}_{margin}" if model_name != "word2vec" else f"{batch_size}_{learning_rate}"

    train_dataset_folder = PARENT_FOLDER_LOCATION / dataset_path / dataset_hyperparameters_path
    model_parameters_folder = PARENT_FOLDER_LOCATION / model_path / model_name / (dataset_hyperparameters_path + "_" + model_hyperparameters_path)

    print(f"Training a {model_name} model in {"CBOW style" if using_CBOW_style else "SkipGram style"} with hyperparameters",  f"(batch_size={batch_size}, learning_rate={learning_rate}, margin={margin})" if model_name!="word2vec" else f"(batch_size={batch_size}, learning_rate={learning_rate})", "\n")

    if model_parameters_folder.is_dir():
        print(f"A model with above parameters has alreday been trained\n")
        return

    model_parameters_folder.mkdir(parents=True)

    cnt = np.load(train_dataset_folder / "counter.npy")
    with open(train_dataset_folder / "word_to_index.json", "r", encoding="ascii") as file:
        word_to_index = json.load(file)
    index_to_word = list(word_to_index.keys())

    vocab_size = len(word_to_index)
    embedding_size = num_learnable_parameters//2 if model_name != "word2vec" else num_learnable_parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_of_training_examples = 0
    number_of_steps = 0

    for train_file in train_dataset_folder.glob("training_examples*.npy"):
        num_examples = np.load(train_file).shape[0]
        number_of_training_examples += num_examples
        number_of_steps += (num_examples + batch_size - 1) // batch_size
    
    number_of_steps *= epochs
    
    if device == torch.device("cuda"):
        print(f"A GPU is detected and will be used for training. GPU name is {torch.cuda.get_device_name()}\n")
    else:
        print("No GPU was detected! Training on a CPU can be signficantly slow! (up to days)\n")

    print("number of training examples =", number_of_training_examples)
    print("number of steps =", number_of_steps)
    
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
        
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = LinearLR(optimizer,start_factor = 1.0, end_factor = 0, total_iters = number_of_steps)

    if report_to_wandb:
        wandb.login()
        run = wandb.init(
            # project="RandomizedGridSearch",  
            project="MarginExpirements",
            name=f"{model_name}_{"CBOW" if using_CBOW_style else "SkipGram"}_{dataset_hyperparameters_path + "_" + model_hyperparameters_path}",
            config=config,
            reinit=True,
        )

    num_batches = 0
    total_loss = 0
    non_learning_examples = 0
    examples_encountered = 0

    print("Started Training\n")
    model.train()

    for num_epoch in range(epochs):
        epoch_start = time.perf_counter()
        for train_file in train_dataset_folder.glob("training_examples*.npy"):

            dataset = torch.from_numpy(np.load(train_file))
            data_loader = DataLoader(dataset, batch_size=batch_size ,shuffle=True, prefetch_factor=2, pin_memory=True, num_workers=8 , persistent_workers=True)

            for batch in data_loader:

                batch = batch
                examples_encountered += len(batch)
                batch = batch.to(device, non_blocking = True)
                positive_score, negative_score = model(batch)

                loss, batch_non_learning_examples, exact_loss = max_margin_loss(positive_score, negative_score, margin=margin) if model_name != "word2vec" else negative_sampling_loss(positive_score, negative_score)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.detach().item()
                non_learning_examples += batch_non_learning_examples
                num_batches += 1


                # Log values
                if report_to_wandb:
                    wandb.log({

                        "lr": scheduler.get_last_lr()[0],

                        "average_loss": total_loss / num_batches,
                        "total_skipped_examples": non_learning_examples / examples_encountered,

                        "exact_loss_min": torch.min(exact_loss).detach().item(),
                        "exact_loss_max": torch.max(exact_loss).detach().item(),
                        "exact_loss_mean": torch.mean(exact_loss).detach().item(),                        
                            })
            
            print(f"Finished processing the file named {train_file.name}")
        
        print(f"Time taken for epoch number {num_epoch+1} is {time.perf_counter()-epoch_start:.2f} in seconds")

        avg_loss = total_loss / num_batches
        print(f"After {num_epoch+1} epochs, the average loss = {avg_loss}")

    save_model_weights(model, model_parameters_folder / "weights.npz")
    save_config_file({k:v for k,v in vars(config).items()}, model_parameters_folder)

    wandb.finish()  

    return 