# SeekTruth.py : Classify text objects into two categories
#
# Sripal reddy Nomula - srnomula, Harshini Mysore Narasimha Ranga - hmn, sanjana Agrawal - sanagra
#
# Based on skeleton code by D. Crandall, October 2021
#

import sys
import re
from collections import defaultdict

pattern = '\w'
likelihood_p = defaultdict(dict)
prior_p = {}
train_data_list = []
test_data_list = []
counter_classes = {}


def load_file(filename):
    objects = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ', 1)
            labels.append(parsed[0] if len(parsed) > 0 else "")
            objects.append(parsed[1] if len(parsed) > 1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}


# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated class label for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#

def clean_punctuation(str):
    words = str.split()
    words_list = []
    for word in words:
        word_f = re.sub(r'[^a-zA-Z0-9]', r'', word)
        words_list.append(word_f.lower())
    return words_list


def calculate_likelihood(freq):
    for label in freq:
        prior_p[label] = counter_classes[label] / len(train_data_list)
        freq_category_total = sum( each_freq for each_freq in freq[label].values())
        for word in freq[label]:
            likelihood_p[label][word] = freq[label][word] / freq_category_total


def freq_calculate():
    freq = {}
    for obj in train_data_list:

        if obj[0] not in freq:
            # assign the value to 1 and create nested dictionary with word and assign its value to 1
            freq[obj[0]] = {word: 1 for word in obj[1:]}
            counter_classes[obj[0]] = 1
        else:
            counter_classes[obj[0]] += 1
            # get the previous count and add to it
            for ind, word in enumerate(obj[1:]):
                freq[obj[0]][obj[ind+1]] = freq[obj[0]].get(obj[ind+1], 0) + 1
                # freq[obj[0]][obj[ind + 1]] += 1
    calculate_likelihood(freq)


def classify(obj):
    labels = []
    probs = []

    for label in prior_p:
        p = 0
        for word in obj[1:]:  # get count for each _word
            if p == 0:
                # assign for first time
                p = likelihood_p[label].get(word,10 ** -6)
            else:
                p *= likelihood_p[label].get(word, 10 ** -6)
        probs.append(p)
        labels.append(label)
    return labels[probs.index(max(probs))]


def classifier(train_data, test_data):
    # This is just dummy code -- put yours here!
    # print(train_data)
    for ind, obj in enumerate(train_data['objects']):
        # get the words in each sample
        cleaned_obj = clean_punctuation(obj)
        train_data_list.append([train_data['labels'][ind]] + cleaned_obj)
    for ind, obj in enumerate(test_data['objects']):
        # get the words in each sample
        cleaned_obj = clean_punctuation(obj)
        test_data_list.append(cleaned_obj)

    freq_calculate()
    return [ classify(obj) for obj in test_data_list]

    # return [test_data["classes"][0]] * len(test_data["objects"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if (sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results = classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([(results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"]))])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
