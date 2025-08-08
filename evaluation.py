from scipy.stats import spearmanr
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
import csv
from helper import get_pretrained_model, load_config_file

PARENT_FOLDER_LOCATION = Path(__file__).resolve().parent
test_dataset_path = "Datasets/test"
CBOW_models_path = "Models/CBOW"

similarity_folder = PARENT_FOLDER_LOCATION / test_dataset_path / "similarity"
hyponymy_folder = PARENT_FOLDER_LOCATION / test_dataset_path / "hyponymy"

similarity_datasets = ["EN-RW-STANFORD", "EN-RG-65", "EN-YP-130", "EN-MEN-TR-3k", "EN-MC-30" , "EN-MTurk-287", "EN-SimVerb-3500", "EN-SIMLEX-999", "EN-MTurk-771", "EN-WS-353-SIM", "EN-WS-353-ALL", "EN-WS-353-REL", "EN-VERB-143"]
hyponymy_datasets = ["bless", "leds", "wbless", "eval"]

CBOW_model_folders = {
    "word2vec" : PARENT_FOLDER_LOCATION / CBOW_models_path / "word2vec",
    "word2box" : PARENT_FOLDER_LOCATION / CBOW_models_path / "word2box",
    "word2ellipsoid" : PARENT_FOLDER_LOCATION / CBOW_models_path / "word2ellipsoid"
}

models_names = ["word2vec", "word2ellipsoid", "word2box"]
hyperparameters_names = ["num_negative_samples","sub_sampling_threshold","max_window_size", "batch_size", "learning_rate", "margin"]

model_pairs = [(models_names[i], models_names[j]) for i in range(len(models_names)) for j in range(len(models_names)) if i != j] # The hypothesis we test is: is the first model in the pair better than the second model in the pair?

def get_mean_similarity_score(model_folder, selected_datasets=set(similarity_datasets)):
    model, word_to_index = get_pretrained_model(model_folder)
    model.eval()
    sum_scores = 0
    num_datasets = 0
    for dataset in similarity_folder.glob("*.txt"):
        if dataset.stem not in selected_datasets:
            continue 
        with dataset.open("r") as f:
            test_set = f.readlines()

        num_datasets += 1

        for i in range(len(test_set)):
            test_set[i] = test_set[i].split()
        
        human_similarity = []
        model_similarity = []
        sum_model_scores = 0
        cnt_model_scores = 0
        for word1, word2, human_score in test_set:
            human_similarity.append(float(human_score))
            if word1 in word_to_index and word2 in word_to_index:
                model_score = model.get_similarity(word_to_index[word1],word_to_index[word2])
                model_similarity.append(model_score)
                sum_model_scores += model_score
                cnt_model_scores += 1
            else:
                model_similarity.append("x")

        # filling the score of each pair that contains an unknown word with the average of the rest of the paris scores.
        average_model_scores = sum_model_scores/cnt_model_scores
        for i in range(len(model_similarity)):
            if model_similarity[i] == "x":
                model_similarity[i] = average_model_scores
        
        score = spearmanr(model_similarity, human_similarity)[0] * 100
        # print(f"Score on {dataset.name} dataset = {score:.2f}")
        sum_scores += score

    # print(f"Average Score = {sum_scores/num_datasets:.2f}")
    return sum_scores/num_datasets


def similarity_cross_validation():
    # For each dataset we get the best performing model on the other datasets and declare it as the best model for the current dataset.
    ans = {dataset : {name:{"accuracy":-1,"hyperparameters":""} for name in models_names} for dataset in similarity_datasets}
    for dataset in similarity_datasets:
        for model_name in models_names:
            best_acc = -1
            best_model_folder = ""
            for model_folder in CBOW_model_folders[model_name].glob("*"):
                current_acc = get_mean_similarity_score(model_folder,selected_datasets=set(d for d in similarity_datasets if d!=dataset))
                if current_acc > best_acc:
                    best_acc = current_acc
                    best_model_folder = model_folder

            # Retrive the information of the best performing model
            config = load_config_file(best_model_folder)
            best_model_hyperparameters = {name: config[name] for name in hyperparameters_names}
            best_model_accuracy = get_mean_similarity_score(best_model_folder, selected_datasets=set([dataset]))

            print(dataset, model_name, best_model_accuracy, best_model_hyperparameters)

            # logging the results in ans
            ans[dataset][model_name]["accuracy"]        = best_model_accuracy
            ans[dataset][model_name]["hyperparameters"] = best_model_hyperparameters
    
    # calculate average accuracy per model
    ans["average"] = {}
    for model_name in models_names:
        current_scores = []
        for dataset in similarity_datasets:
            current_scores.append(ans[dataset][model_name]["accuracy"])
        ans["average"][model_name] = sum(current_scores) / len(current_scores)

    return ans

def similarity_bootstrap_test_p_value(ground_truth,number_of_bootstraps = 10000):
    ans = {dataset: {f"Is {model_A_name} better than {model_B_name}?":{"ground_truth":-1,"p_value":-1} for model_A_name, model_B_name in model_pairs} for dataset in similarity_datasets}

    for dataset in similarity_folder.glob("*.txt"):
        with dataset.open("r") as f:
            test_set = f.readlines()

        for i in range(len(test_set)):
            test_set[i] = test_set[i].split()

        for model_A_name, model_B_name in model_pairs:
            hyperparameters_A = [str(ground_truth[dataset.stem][model_A_name]["hyperparameters"][parameter]) for parameter in hyperparameters_names]
            hyperparameters_B = [str(ground_truth[dataset.stem][model_B_name]["hyperparameters"][parameter]) for parameter in hyperparameters_names]

            if model_A_name == "word2vec":
                hyperparameters_A.pop()
            if model_B_name == "word2vec":
                hyperparameters_B.pop()

            model_folder_A = PARENT_FOLDER_LOCATION / CBOW_models_path / model_A_name / "_".join(hyperparameters_A)
            model_folder_B = PARENT_FOLDER_LOCATION / CBOW_models_path / model_B_name / "_".join(hyperparameters_B)

            model_A, word_to_index_A = get_pretrained_model(model_folder_A)
            model_B, word_to_index_B = get_pretrained_model(model_folder_B)

            model_A.eval()
            model_B.eval()

            human_similarity = [float(human_score) for word1, word2, human_score in test_set]
            model_A_similarity = []
            model_B_similarity = []
            
            sum_model_A_scores = 0
            cnt_model_A_scores = 0
            sum_model_B_scores = 0            
            cnt_model_B_scores = 0

            # Model A
            for word1, word2, human_score in test_set:
                if word1 in word_to_index_A and word2 in word_to_index_A:
                    model_score = model_A.get_similarity(word_to_index_A[word1],word_to_index_A[word2])
                    model_A_similarity.append(model_score)
                    sum_model_A_scores += model_score
                    cnt_model_A_scores += 1
                else:
                    model_A_similarity.append("x")

            # filling the score of each pair that contains an unknown word with the average of the rest of the paris scores.
            average_model_scores = sum_model_A_scores/cnt_model_A_scores
            for i in range(len(model_A_similarity)):
                if model_A_similarity[i] == "x":
                    model_A_similarity[i] = average_model_scores

            
            # Model B
            for word1, word2, human_score in test_set:
                if word1 in word_to_index_B and word2 in word_to_index_B:
                    model_score = model_B.get_similarity(word_to_index_B[word1],word_to_index_B[word2])
                    model_B_similarity.append(model_score)
                    sum_model_B_scores += model_score
                    cnt_model_B_scores += 1
                else:
                    model_B_similarity.append("x")

            # filling the score of each pair that contains an unknown word with the average of the rest of the paris scores.
            average_model_scores = sum_model_B_scores/cnt_model_B_scores
            for i in range(len(model_B_similarity)):
                if model_B_similarity[i] == "x":
                    model_B_similarity[i] = average_model_scores
                
            
            score_A = spearmanr(model_A_similarity, human_similarity)[0] * 100
            score_B = spearmanr(model_B_similarity, human_similarity)[0] * 100
            observed_delta = observed_mean = score_A - score_B

            similarity_scores = list(zip(model_A_similarity,model_B_similarity,human_similarity))

            bootstraped_deltas = []
            for num_bootstrap in range(number_of_bootstraps):
                current_bootstrap = random.choices(similarity_scores, k = len(similarity_scores))
                score_A = spearmanr([i[0] for i in current_bootstrap], [i[2] for i in current_bootstrap])[0] * 100
                score_B = spearmanr([i[1] for i in current_bootstrap], [i[2] for i in current_bootstrap])[0] * 100
                bootstraped_deltas.append(score_A - score_B)
            
            bootstraped_mean = sum(bootstraped_deltas) / len(bootstraped_deltas)

            shifted_bootstraped_deltas = [delta - bootstraped_mean for delta in bootstraped_deltas]

            count_extreme = 0
            for delta in shifted_bootstraped_deltas:
                count_extreme += (delta >= observed_delta)
            p_value = count_extreme / len(shifted_bootstraped_deltas)

            # logging the values
            # We are using a 95% confidence interval
            ans[dataset.stem][f"Is {model_A_name} better than {model_B_name}?"]["ground_truth"]       = observed_delta
            ans[dataset.stem][f"Is {model_A_name} better than {model_B_name}?"]["p_value"]            = p_value

            print(dataset.stem, f"Is {model_A_name} better than {model_B_name}?", observed_delta, p_value)

    # bootstrap test for the average (this test does not inculde the uncertainity of the 13 similarity dataset in consideration and just takes the ground_truth value)
    ans["average"] = {f"Is {model_A_name} better than {model_B_name}?":{"ground_truth":-1,"p_value":-1} for model_A_name, model_B_name in model_pairs}
    for model_A_name, model_B_name in model_pairs:

        observed_deltas = [ground_truth[dataset][model_A_name]["accuracy"] - ground_truth[dataset][model_B_name]["accuracy"] for dataset in similarity_datasets]
        observed_mean = sum(observed_deltas) / len(observed_deltas)

        bootstraped_means = []
        for num_bootstrap in range(number_of_bootstraps):
            current_bootstrap = random.choices(observed_deltas, k = len(observed_deltas))
            bootstraped_means.append(sum(current_bootstrap)/len(current_bootstrap))

        mean_of_bootstraped_means = sum(bootstraped_means) / len(bootstraped_means)

        shifted_bootstraped_means = [mean - mean_of_bootstraped_means for mean in bootstraped_means]

        count_extreme = 0
        for mean in shifted_bootstraped_means:
            count_extreme += (mean >= observed_mean)

        p_value = count_extreme / len(shifted_bootstraped_means)

        # logging the values
        # We are using a 95% confidence interval
        ans["average"][f"Is {model_A_name} better than {model_B_name}?"]["ground_truth"]       = observed_mean
        ans["average"][f"Is {model_A_name} better than {model_B_name}?"]["p_value"]            = p_value

        print("average", f"Is {model_A_name} better than {model_B_name}?", observed_mean, p_value)
    return ans



# Hyponymy Functions 



def prepare_hyponymy_dataset(dataset):
    hypernyms = defaultdict(set)
    non_hypernyms = defaultdict(set)
    with dataset.open("r") as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        first_row = True
        if dataset.stem == "eval":
            for word1,word2,label,relation,*_ in tsv_reader:
                if first_row:
                    first_row = False
                    continue
                if label == "True" and relation == "hyper":
                    hypernyms[word1].add(word2)
                elif label == "False":
                    non_hypernyms[word1].add(word2)
        else:
            for word1,word2,label,*_ in tsv_reader:
                if first_row:
                    first_row = False
                    continue
                if label == "True":
                    hypernyms[word1].add(word2)
                else:
                    non_hypernyms[word1].add(word2)

    all_words = set(hypernyms.keys()).union(set(non_hypernyms.keys()))
    for word in all_words:
        if min(len(hypernyms[word]),len(non_hypernyms[word])) == 0:
            del hypernyms[word]
            del non_hypernyms[word]

    return hypernyms,non_hypernyms



def MAP(model_folder, dataset):
    hypernyms,non_hypernyms = prepare_hyponymy_dataset(dataset)
    model, word_to_index = get_pretrained_model(model_folder)
    model.eval()

    for word in hypernyms.keys(): # hypernyms,non_hypernyms have the same set of keys
        if word not in word_to_index:
            del hypernyms[word]
            del non_hypernyms[word]
    
    for word1 in hypernyms.keys():
        temp_set = set()
        for word2 in hypernyms[word1]:
            if word2 not in word_to_index:
                temp_set.add(word2)
        hypernyms[word1] = hypernyms[word1].difference(temp_set)
        
        temp_set = set()
        for word2 in non_hypernyms[word1]:
            if word2 not in word_to_index:
                temp_set.add(word2)
        non_hypernyms[word1] = non_hypernyms[word1].difference(temp_set)
    

    average_precision = []
    for word1 in hypernyms.keys():
        all_words = list(hypernyms[word1].union(non_hypernyms[word1]))
        score = {word2 : model.get_similarity(word_to_index[word1], word_to_index[word2], condition_on_first_word=True) for word2 in all_words}
        all_words.sort(key= lambda word : score[word], reverse=True)
        sum_precision, retrived_true_pairs = 0, 0
        for i in range(len(all_words)):
            if all_words[i] in hypernyms[word1]:
                retrived_true_pairs += 1
                sum_precision += (retrived_true_pairs/(i+1))
        average_precision.append(sum_precision / len(hypernyms[word1])) # len(hypernyms[word1]) reprsent the total number of true pairs

    return sum(average_precision) / len(average_precision), average_precision


def hyponymy_cross_validation():
    # For each dataset we get the best performing model on the other datasets and declare it as the best model for the current dataset.
    ans = {dataset : {name:{"MAP":-1,"hyperparameters":""} for name in models_names} for dataset in hyponymy_datasets}
    for dataset in hyponymy_folder.glob("*"):
        for model_name in models_names:
            best_acc = -1
            best_model_folder = ""
            for model_folder in CBOW_model_folders[model_name].glob("*"):
                other_dataset_scores = []
                for d in hyponymy_folder.glob("*"):
                    if d.stem!=dataset.stem:
                        other_dataset_scores.append(MAP(model_folder,d)[0])
                current_acc = sum(other_dataset_scores)/len(other_dataset_scores)
                if current_acc > best_acc:
                    best_acc = current_acc
                    best_model_folder = model_folder

            # Retrive the information of the best performing model
            config = load_config_file(best_model_folder)
            best_model_hyperparameters = {name: config[name] for name in hyperparameters_names}
            best_model_accuracy = MAP(best_model_folder, dataset)[0]

            print(dataset.stem, model_name, best_model_accuracy, best_model_hyperparameters)

            # logging the results in ans
            ans[dataset.stem][model_name]["MAP"]        = best_model_accuracy
            ans[dataset.stem][model_name]["hyperparameters"] = best_model_hyperparameters

    # calculate average accuracy per model
    ans["average"] = {}
    for model_name in models_names:
        current_scores = []
        for dataset in hyponymy_datasets:
            current_scores.append(ans[dataset][model_name]["MAP"])
        ans["average"][model_name] = sum(current_scores) / len(current_scores)

    return ans



def hyponymy_bootstrap_test_p_value(ground_truth,number_of_bootstraps = 10000):

    ans = {dataset: {f"Is {model_A_name} better than {model_B_name}?":{"ground_truth":-1,"p_value":-1} for model_A_name, model_B_name in model_pairs} for dataset in hyponymy_datasets}

    for dataset in hyponymy_folder.glob("*"):    
        for model_A_name, model_B_name in model_pairs:
            hyperparameters_A = [str(ground_truth[dataset.stem][model_A_name]["hyperparameters"][parameter]) for parameter in hyperparameters_names]
            hyperparameters_B = [str(ground_truth[dataset.stem][model_B_name]["hyperparameters"][parameter]) for parameter in hyperparameters_names]

            if model_A_name == "word2vec":
                hyperparameters_A.pop()
            if model_B_name == "word2vec":
                hyperparameters_B.pop()

            model_folder_A = PARENT_FOLDER_LOCATION / CBOW_models_path / model_A_name / "_".join(hyperparameters_A)
            model_folder_B = PARENT_FOLDER_LOCATION / CBOW_models_path / model_B_name / "_".join(hyperparameters_B)

            MAP_A, average_precision_A = MAP(model_folder_A, dataset)
            MAP_B, average_precision_B = MAP(model_folder_B, dataset)

            observed_delta = observed_mean = MAP_A - MAP_B
            average_precision_delta = [average_precision_A[i] - average_precision_B[i] for i in range(len(average_precision_A))]

            bootstraped_deltas = []
            for num_bootstrap in range(number_of_bootstraps):
                current_bootstrap = random.choices(average_precision_delta, k = len(average_precision_delta))
                bootstraped_deltas.append(sum(current_bootstrap)/len(current_bootstrap))
            
            bootstraped_mean = sum(bootstraped_deltas) / len(bootstraped_deltas)

            shifted_bootstraped_deltas = [delta - bootstraped_mean for delta in bootstraped_deltas] # shifting the mean to zero

            count_extreme = 0
            for delta in shifted_bootstraped_deltas:
                count_extreme += (delta >= observed_delta)
            p_value = count_extreme / len(shifted_bootstraped_deltas)

            # logging the values
            # We are using a 95% confidence interval
            ans[dataset.stem][f"Is {model_A_name} better than {model_B_name}?"]["ground_truth"]       = observed_delta
            ans[dataset.stem][f"Is {model_A_name} better than {model_B_name}?"]["p_value"]            = p_value

            print(dataset.stem, f"Is {model_A_name} better than {model_B_name}?", observed_delta, p_value)

    ans["average"] = {f"Is {model_A_name} better than {model_B_name}?":{"ground_truth":-1,"p_value":-1} for model_A_name, model_B_name in model_pairs}
    for model_A_name, model_B_name in model_pairs:

        observed_deltas = [ground_truth[dataset][model_A_name]["MAP"] - ground_truth[dataset][model_B_name]["MAP"] for dataset in hyponymy_datasets]
        observed_mean = sum(observed_deltas) / len(observed_deltas)

        bootstraped_means = []
        for num_bootstrap in range(number_of_bootstraps):
            current_bootstrap = random.choices(observed_deltas, k = len(observed_deltas))
            bootstraped_means.append(sum(current_bootstrap)/len(current_bootstrap))
        
        mean_of_bootstraped_means = sum(bootstraped_means) / len(bootstraped_means)

        shifted_bootstraped_means = [mean - mean_of_bootstraped_means for mean in bootstraped_means]

        count_extreme = 0
        for mean in shifted_bootstraped_means:
            count_extreme += (mean >= observed_mean)

        p_value = count_extreme / len(bootstraped_means)

        # logging the values
        # We are using a 95% confidence interval
        ans["average"][f"Is {model_A_name} better than {model_B_name}?"]["ground_truth"]       = observed_mean
        ans["average"][f"Is {model_A_name} better than {model_B_name}?"]["p_value"]            = p_value

        print("average", f"Is {model_A_name} better than {model_B_name}?", observed_mean, p_value)
    return ans


