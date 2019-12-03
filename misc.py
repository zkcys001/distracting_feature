import os
import pickle

from argparse import ArgumentError

SEED = 0

def split_arr(arr, k):
    for i in range(0, len(arr), k):
        yield arr[i:i + k]

def str_to_bool(arg):
    if arg == 'True':
        return True
    elif arg == 'False':
        return False
    else:
        raise ArgumentError('bool expected')

def save_file(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)