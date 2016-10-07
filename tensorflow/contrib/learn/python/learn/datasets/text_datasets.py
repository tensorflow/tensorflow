# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Text datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile

DBPEDIA_URL = 'https://googledrive.com/host/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M/dbpedia_csv.tar.gz'


def maybe_download_dbpedia(data_dir):
  """Download if DBpedia data is not present."""
  train_path = os.path.join(data_dir, 'dbpedia_csv/train.csv')
  test_path = os.path.join(data_dir, 'dbpedia_csv/test.csv')
  if not (gfile.Exists(train_path) and gfile.Exists(test_path)):
    archive_path = base.maybe_download(
        'dbpedia_csv.tar.gz', data_dir, DBPEDIA_URL)
    tfile = tarfile.open(archive_path, 'r:*')
    tfile.extractall(data_dir)


def load_dbpedia(size='small', test_with_fake_data=False):
  """Get DBpedia datasets from CSV files."""
  if not test_with_fake_data:
    data_dir = os.path.join(os.getenv('TF_EXP_BASE_DIR', ''), 'dbpedia_data')
    maybe_download_dbpedia(data_dir)

    train_path = os.path.join(data_dir, 'dbpedia_csv', 'train.csv')
    test_path = os.path.join(data_dir, 'dbpedia_csv', 'test.csv')

    if size == 'small':
      # Reduce the size of original data by a factor of 1000.
      base.shrink_csv(train_path, 1000)
      base.shrink_csv(test_path, 1000)
      train_path = train_path.replace('train.csv', 'train_small.csv')
      test_path = test_path.replace('test.csv', 'test_small.csv')
  else:
    module_path = os.path.dirname(__file__)
    train_path = os.path.join(module_path, 'data', 'text_train.csv')
    test_path = os.path.join(module_path, 'data', 'text_test.csv')

  train = base.load_csv_without_header(
      train_path, target_dtype=np.int32, features_dtype=np.str, target_column=0)
  test = base.load_csv_without_header(
      test_path, target_dtype=np.int32, features_dtype=np.str, target_column=0)

  return base.Datasets(train=train, validation=None, test=test)
