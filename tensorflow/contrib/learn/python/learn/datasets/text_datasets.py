"""Text datasets."""
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

import os
import tarfile

import numpy as np

from tensorflow.python.platform import gfile
from tensorflow.contrib.learn.python.learn.datasets import base

DBPEDIA_URL = 'https://googledrive.com/host/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M/dbpedia_csv.tar.gz'


def get_dbpedia(data_dir):
    train_path = os.path.join(data_dir, 'dbpedia_csv/train.csv')
    test_path = os.path.join(data_dir, 'dbpedia_csv/test.csv')
    if not (gfile.Exists(train_path) and gfile.Exists(test_path)):
        archive_path = base.maybe_download('dbpedia_csv.tar.gz', data_dir, DBPEDIA_URL)
        tfile = tarfile.open(archive_path, 'r:*')
        tfile.extractall(data_dir)
    train = base.load_csv(train_path, np.int32, 0, has_header=False)
    test = base.load_csv(test_path, np.int32, 0, has_header=False)
    datasets = base.Datasets(train=train, validation=None, test=test)
    return datasets


def load_dbpedia():
    return get_dbpedia('dbpedia_data')

