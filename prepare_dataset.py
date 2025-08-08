import os
from pathlib import Path
from collections import Counter
import numpy as np
import json
import random

from helper import clean

PARENT_FOLDER_LOCATION = Path(__file__).resolve().parent

def xml_to_txt(config):
    
    print("Cleaning and preprocessing xml files into a txt file. THIS COULD TAKE A WHILE (IT TAKES 16 MIN ON AN AMD 9800X3D CPU) \n")
    
    # Extract needed parameters from config
    input_dir = PARENT_FOLDER_LOCATION / config.raw_xml_dir
    output_dir = PARENT_FOLDER_LOCATION /config.preprocessed_txt_dir
    min_token_freq = config.min_freq_of_vocab_tokens
    
    input_folder = Path(input_dir)
    output_folder = Path(output_dir)

    if (output_folder / "sentences.txt" ).exists():
        print(f"Dataset has already been cleaned an preprocessed from raw xml files to a txt file\n")
        return

    frequency = Counter()

    # Converty from xml to txt and clean the txt file
    with open(output_folder / "intermediate_sentences.txt"  , "w", encoding="ascii") as intermediate_file:
        for file in input_folder.glob("*.xml"):
            with file.open("r", encoding="ascii", errors="replace") as input_file: # we only keep ascii characters and replace the others with �. This symbol is then used to delete the words that contains a non-ascii character
                part_of_sentence = False
                current_sentence = []
                for line in input_file:
                    if ("<s>" in line ) or ("<text" in line):
                        part_of_sentence = True 
                    elif "</s>" in line:
                        cleaned_sentence = clean(current_sentence)
                        if cleaned_sentence:
                            intermediate_file.write(" ".join(cleaned_sentence) + "\n")
                            frequency.update(cleaned_sentence)
                        current_sentence = []
                        part_of_sentence = False
                    elif part_of_sentence:
                        current_sentence.append(line.split()[1]) # take the lemma of the word only

    # Toknize and drop toknes with freq<100
    with open(output_folder / "intermediate_sentences.txt"  , "r", encoding="ascii") as intermediate_file:
        with open(output_folder / "sentences.txt"  , "w", encoding="ascii") as output_file:
            for line in intermediate_file:
                words = [w for w in line.split() if frequency[w] >= min_token_freq]
                if words:
                    output_file.write(" ".join(words) + "\n")

    # Delete the intermediate file
    os.remove(output_folder / "intermediate_sentences.txt")    


    print("Dataset has been preprocssed and cleaned\n")
    
    return

def create_CBOW_examples(config):

        # Extract parameters needed from config
        input_dir = PARENT_FOLDER_LOCATION / config.preprocessed_txt_dir
        output_dir = PARENT_FOLDER_LOCATION / config.CBOW_training_examples_dir
        max_size_per_training_file_in_gigabytes = config.max_size_per_training_file_in_gigabytes
        num_negative_samples = config.num_negative_samples
        sub_sampling_threshold = config.sub_sampling_threshold
        max_window_size = config.max_window_size

        print(f"Creating CBOW dataset with hyberparameters (num_negative_samples = {num_negative_samples}, sub_sampling_threshold = {sub_sampling_threshold}, max_window_size = {max_window_size})\n")

        input_folder = Path(input_dir)
        output_folder = Path(output_dir)

        output_folder = output_folder / f"{num_negative_samples}_{sub_sampling_threshold}_{max_window_size}"

        if output_folder.is_dir():
                print(f"CBOW dataset with hyberparameters (num_negative_samples = {num_negative_samples}, sub_sampling_threshold = {sub_sampling_threshold}, max_window_size = {max_window_size}) has already been created\n")
                return

        output_folder.mkdir(parents=True)

        # creating the vocabulary and count frequency of each word
        vocab = set()
        cnt = Counter()
        for file in input_folder.glob("*.txt"):
                with file.open("r", encoding="ascii") as input_file:
                        for line in input_file:
                                temp = line.split()
                                vocab.update(temp)
                                cnt.update(temp)

        index_to_word = sorted(vocab)
        word_to_index = {word:index for index, word in enumerate(index_to_word)}

        cnt = {word_to_index[word] : freq for word, freq in cnt.items()}
        cnt = np.array([cnt[i] for i in range(len(vocab))], dtype=np.int32)

        # print some stats
        print("Statistics about the sentences dataset\n")
        print(f"Number of unique words = {len(vocab)}")
        print(f"Total number of tokens = {np.sum(cnt)}")

        frequencies = sorted(cnt)
        print(f"least word's frequency in dataset = {frequencies[0]}")
        print(f"median word's frequency in dataset = {frequencies[len(frequencies)//2]}")
        print(f"highest word's frequency in dataset = {frequencies[-1]}")

        # Saving cnt and word_to_index
        np.save(output_folder / "counter.npy", cnt)
        with open(output_folder / "word_to_index.json", "w", encoding="ascii") as file:
                json.dump(word_to_index, file, indent=4)


        # sub-sampling setup
        word_probability = cnt/np.sum(cnt)
        drop_probability = 1 - np.sqrt(sub_sampling_threshold/word_probability) # could be negative, but it works as intended. (negative drop propabilities => drop_probability = 0)

        # negative-sampling setup
        # we create a table of around 100 million word index where each word appears depending on its probability of being selected as a negative sample
        # This results in a table where sampling a random element from it will result in the wanted probability distribution of negative sampling.

        modified_cnt = cnt**(3/4) # same as word2vec empirical exponent
        negative_sampling_probability = modified_cnt/np.sum(modified_cnt)
        table_target_size = 10**8
        negative_sampling_table = []
        for word in range(len(cnt)):
                negative_sampling_table.extend([word] * int(negative_sampling_probability[word] * table_target_size))
        

        # defining a function that takes a list of words and generate all training examples (negative samples are included)
        # each train example is of the form (input_index, target_index, *negative_samples_indices)
        def get_training_examples(words):
                training_examples = []
                sampled_window_size = random.randint(1,max_window_size) 
                for i in range(len(words)):
                        center_word = words[i]
                        context_words = words[max(i-sampled_window_size,0):i] + words[i+1:min(i+sampled_window_size+1,len(words))]
                        if len(context_words) > 0:
                                # if there is at least one context word, we add enough place holders to get the window size, else we drop the example.
                                context_words.extend([-1] * (max_window_size*2 - len(context_words))) # add a placeholder for edge cases where the window exceeds the sentence boundaries.
                        else:
                                continue

                        negative_samples = set()
                        while len(negative_samples) < num_negative_samples:
                                index = negative_sampling_table[random.randint(0,len(negative_sampling_table)-1)]
                                if index != center_word: # index is not the target
                                        negative_samples.add(index)

                        training_examples.append((center_word, *negative_samples, *context_words))

                return training_examples
        

        # Generating examples
        
        max_bytes = int(max_size_per_training_file_in_gigabytes * (1024 ** 3))
        current_size = 0
        file_num = 1
        training_examples = []
                                

        print("\nGenerating training examples\n")

        for file in input_folder.glob("*.txt"):
                with file.open("r", encoding="ascii") as input_file:
                        for line in input_file:
                                sub_sampled_words = [word_to_index[word] for word in line.split() if random.random() > drop_probability[word_to_index[word]]]
                                new_training_examples = get_training_examples(sub_sampled_words)
                                if new_training_examples:
                                        training_examples.extend(new_training_examples)
                                        current_size += len(new_training_examples) * len(new_training_examples[0]) * 4 # each number takes 4 bytes since it will be saved as np.int32
                                
                                if current_size >= max_bytes:
                                        np.save(output_folder / f"training_examples{file_num}.npy", np.array(training_examples, dtype=np.int32))
                                        print(f"Generated file number {file_num}")
                                        file_num += 1
                                        training_examples = []
                                        current_size = 0
               
        if training_examples:
                np.save(output_folder / f"training_examples{file_num}.npy", np.array(training_examples, dtype=np.int32))
                print(f"Generated file number {file_num}")
                file_num += 1
                training_examples = []
                current_size = 0


        print("\nFinished creating the CBOW dataset\n")

        return



def create_SkipGram_examples(config):
      return





def create_training_examples(config):
      if config.CBOW:
            create_CBOW_examples(config)
      else:
            create_SkipGram_examples(config)
      