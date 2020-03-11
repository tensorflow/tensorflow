# Lint as: python3
# coding=utf-8
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Split data into train, validation and test dataset according to person.

That is, use some people's data as train, some other people's data as
validation, and the rest ones' data as test. These data would be saved
separately under "/person_split".

It will generate new files with the following structure:
├──person_split
│   ├── test
│   ├── train
│   └──valid
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from data_split import read_data
from data_split import write_data


def person_split(whole_data, train_names, valid_names, test_names):  # pylint: disable=redefined-outer-name
  """Split data by person."""
  random.seed(30)
  random.shuffle(whole_data)
  train_data = []  # pylint: disable=redefined-outer-name
  valid_data = []  # pylint: disable=redefined-outer-name
  test_data = []  # pylint: disable=redefined-outer-name
  for idx, data in enumerate(whole_data):  # pylint: disable=redefined-outer-name,unused-variable
    if data["name"] in train_names:
      train_data.append(data)
    elif data["name"] in valid_names:
      valid_data.append(data)
    elif data["name"] in test_names:
      test_data.append(data)
  print("train_length:" + str(len(train_data)))
  print("valid_length:" + str(len(valid_data)))
  print("test_length:" + str(len(test_data)))
  return train_data, valid_data, test_data


if __name__ == "__main__":
  data = read_data("./data/complete_data")
  train_names = [
      "hyw", "shiyun", "tangsy", "dengyl", "jiangyh", "xunkai", "negative3",
      "negative4", "negative5", "negative6"
  ]
  valid_names = ["lsj", "pengxl", "negative2", "negative7"]
  test_names = ["liucx", "zhangxy", "negative1", "negative8"]
  train_data, valid_data, test_data = person_split(data, train_names,
                                                   valid_names, test_names)
  if not os.path.exists("./person_split"):
    os.makedirs("./person_split")
  write_data(train_data, "./person_split/train")
  write_data(valid_data, "./person_split/valid")
  write_data(test_data, "./person_split/test")
