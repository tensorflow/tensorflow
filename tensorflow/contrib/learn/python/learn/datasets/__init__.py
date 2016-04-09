"""Module inclues reference datasets and utilities to load datasets."""
#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import collections
from os import path

import numpy as np

    
Dataset = collections.namedtuple('Dataset', ['data', 'target'])


def load_csv(filename, target_dtype):
    with open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        header = next(data_file)
        n_samples = int(header[0])
        n_features = int(header[1])
        target_names = np.array(header[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=target_dtype)

    return Dataset(data=data, target=target)


def load_iris():
    module_path = path.dirname(__file__)
    return load_csv(path.join(module_path, 'data', 'iris.csv'),
                    target_dtype=np.int)


def load_boston():
    module_path = path.dirname(__file__)
    return load_csv(path.join(module_path, 'data', 'boston_house_prices.csv'),
                    target_dtype=np.float)


# List of all available datasets.
DATASETS = {
    'iris': load_iris,
    'boston': load_boston,
}


def load_dataset(name):
    """Loads dataset by name.
    
    Args:
        name: Name of the dataset to load.
    
    Returns:
        Features and targets for given dataset. Can be numpy or iterator.
    """
    if name not in DATASETS:
        raise ValueError("Name of dataset is not found: %s" % name)
    return DATASETS[name]()
        
