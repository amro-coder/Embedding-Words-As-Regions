import json
from pathlib import Path

PARENT_FOLDER_LOCATION = Path(__file__).resolve().parent

from evaluation import similarity_cross_validation, similarity_bootstrap_test_p_value, hyponymy_cross_validation, hyponymy_bootstrap_test_p_value

NUMBER_OF_BOOTSTRAP_TESTS = 1000000

# Similarity

similarity_ground_truth = similarity_cross_validation()

with open("similarity_ground_truth.json", "w") as f:
    json.dump(similarity_ground_truth,f,indent=4)

similiarity_bootstrap_p_value = similarity_bootstrap_test_p_value(similarity_ground_truth,number_of_bootstraps=NUMBER_OF_BOOTSTRAP_TESTS)

with open("similiarity_bootstrap_p_value.json", "w") as f:
    json.dump(similiarity_bootstrap_p_value,f,indent=4)


# Hyponymy


hyponymy_ground_truth = hyponymy_cross_validation()

with open("hyponymy_ground_truth.json", "w") as f:
    json.dump(hyponymy_ground_truth,f,indent=4)

hyponymy_bootstrap_p_value = hyponymy_bootstrap_test_p_value(hyponymy_ground_truth,number_of_bootstraps=NUMBER_OF_BOOTSTRAP_TESTS)

with open("hyponymy_bootstrap_p_value.json", "w") as f:
    json.dump(hyponymy_bootstrap_p_value,f,indent=4)

