import os
import re

import numpy as np


def clean_str(string):
    string = re.findall('..?', string)
    string = ' '.join(string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_data_and_labels(data_list_file):
    """
    Loads data from data_dir, splits the data into phonemes and generates labels.
    Returns split sentences and labels.
    """
    labels_list = list(open(data_list_file).readlines())
    x_text = []
    y = []
    for label_file in labels_list:
        label_file = label_file.strip()
        examples = list(open(os.path.dirname(data_list_file) + '/' + label_file, "r").readlines())
        examples = [clean_str(s) for s in examples]
        x_text += examples
        label = [0] * len(labels_list)
        label[int(label_file) - 1] = 1
        labels = [label for _ in examples]
        y += labels
    return [x_text, np.array(y)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
