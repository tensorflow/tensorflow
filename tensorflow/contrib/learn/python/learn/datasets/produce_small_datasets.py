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
"""Produce DBpedia datasets of a smaller size (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets import text_datasets
from tensorflow.python.platform import app


def main(unused_argv):
  text_datasets.maybe_download_dbpedia('dbpedia_data')
  # Reduce the size of original data by a factor of 1000.
  base.shrink_csv('dbpedia_data/dbpedia_csv/train.csv', 1000)
  base.shrink_csv('dbpedia_data/dbpedia_csv/test.csv', 1000)


if __name__ == '__main__':
  app.run()
