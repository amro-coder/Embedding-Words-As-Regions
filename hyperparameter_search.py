if __name__ == "__main__":

    from types import SimpleNamespace as NS
    import random
    from pprint import pprint   # std-lib pretty printer
    from pathlib import Path

    from prepare_dataset import create_training_examples, xml_to_txt
    from train import train_model
    from helper import load_config_file

    PARENT_FOLDER_LOCATION = Path(__file__).resolve().parent

    #############
    NUMBER_OF_MODELS_TO_TRAIN = 60          # This value determines the number of models trained in the randomized grid search. MAKE SURE IT IS <= TO THE NUMBER OF POSSIBLE MODELS TO AVIOD INFINITE LOOPS
    ##############

    # THE BELOW DICT CONTAINS THE SPACE OF EACH HYPERPARAMETER. WE TRAIN NUMBER_OF_MODELS_TO_TRAIN MODELS WITH HYPERPARAMETERS INITALIZED RANDOMLY FROM THE PROVIDED SPACE.
    valid_values_for = {
        "num_negative_samples" : [5, 10], 
        "sub_sampling_threshold" : [0.001, 0.0001, 0.00001],
        "max_window_size" : [2, 5, 10],
        "batch_size" : [8192, 16384, 32768, 65536],
        "learning_rate" : [0.01, 0.025, 0.1],
        "margin" : [5, 20, 50],
    }

    # The hyperparameter values we are doing the search one for each model
    hyperparameters = {
        "word2vec" :        ["num_negative_samples", "sub_sampling_threshold", "max_window_size", "batch_size", "learning_rate"],
        "word2box" :        ["num_negative_samples", "sub_sampling_threshold", "max_window_size", "batch_size", "learning_rate", "margin"],
        "word2ellipsoid" :   ["num_negative_samples", "sub_sampling_threshold", "max_window_size", "batch_size", "learning_rate", "margin"],
    }

    for using_CBOW_style in [True, False][:1]: # Only CBOW style models are currently implemented
        for model_name in ["word2vec", "word2box", "word2ellipsoid"]:
            used_hyperparameters = set()
            while len(used_hyperparameters) < NUMBER_OF_MODELS_TO_TRAIN:
                current_hyperparameters = tuple(random.choice(valid_values_for[parameter]) for parameter in hyperparameters[model_name])
                if current_hyperparameters not in used_hyperparameters:
                    used_hyperparameters.add(current_hyperparameters)

                    config = load_config_file(PARENT_FOLDER_LOCATION) # Load default value for all parameters

                    # Set this run hyperparameter's values
                    config["CBOW"] = using_CBOW_style
                    config["model_name"] = model_name
                    for parmeter, value in zip(hyperparameters[model_name], current_hyperparameters):
                        config[parmeter] = value
                    
                    print("Current Training Configuration : \n")
                    pprint(config)
                    print()

                    config = NS(**config)

                    # xml_to_txt(config) # This line is only required to run once. If you run it multiple times, the code will still automaticlly detect if the dataset has already been cleaned or not.
                    create_training_examples(config)
                    train_model(config)

                    print(f"\nFinished Training {model_name} Model Number {len(used_hyperparameters)} out of {NUMBER_OF_MODELS_TO_TRAIN} ({len(used_hyperparameters)}/{NUMBER_OF_MODELS_TO_TRAIN})\n")

        
