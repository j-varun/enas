import os
import sys

try:
  import cPickle as pickle
except ImportError:
  import _pickle as pickle

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print(file_name)
    full_name = os.path.join(data_path, file_name)
    with open(full_name, 'rb') as finp:
      data = pickle.load(finp, encoding='bytes')
      batch_images = data[b"data"].astype(np.float32) / 255.0
      batch_labels = np.array(data[b"labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels

def _read_fmnist_data(data_path):
  """Reads Fashion-Mnist data. Returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = {},{}
  data = input_data.read_data_sets(data_path)
  images["train"] = data.train.images.reshape(-1, 1, 28, 28) / 255.0
  images["test"] = data.test.images.reshape(-1, 1, 28, 28) / 255.0

  images["train"] = np.transpose(images["train"], [0, 2, 3, 1])
  images["test"] = np.transpose(images["test"], [0, 2, 3, 1])

  labels["train"] = np.array(data.train.labels, dtype = np.int32)
  labels["test"] = np.array(data.test.labels, dtype = np.int32)
  print("Read and processed data..")
  print(labels["test"])

  return images, labels


def valid_split_data(images, labels, num_valids=5000):
  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None
  return images, labels

def read_data(data_path, num_valids=5000, dataset = "cifar"):
  print("-" * 80)
  print("Reading data")
  print(os.getcwd())

  images, labels = {}, {}
  if(dataset == "fmnist"):
    print("Fashion-Mnist")
    images, labels = _read_fmnist_data(data_path)
    images, labels = valid_split_data(images, labels, num_valids)
    return images, labels

  if dataset == "stacking":
    images["path"] = data_path
    return images, labels
  else:
    train_files = [
      "data_batch_1",
      "data_batch_2",
      "data_batch_3",
      "data_batch_4",
      "data_batch_5",
    ]
    test_file = [
      "test_batch",
    ]
    images["train"], labels["train"] = _read_data(data_path, train_files)

    images, labels = valid_split_data(images, labels, num_valids)

    images["test"], labels["test"] = _read_data(data_path, test_file)

    print("Prepropcess: [subtract mean], [divide std]")
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    print("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = (images["train"] - mean) / std
    if num_valids:
      images["valid"] = (images["valid"] - mean) / std
    images["test"] = (images["test"] - mean) / std

    return images, labels

