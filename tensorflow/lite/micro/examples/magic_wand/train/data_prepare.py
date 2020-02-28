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

"""Prepare data for further process.

Read data from "/slope", "/ring", "/wing", "/negative" and save them
in "/data/complete_data" in python dict format.

It will generate a new file with the following structure:
├── data
│   └── complete_data
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import os
import random

LABEL_NAME = "gesture"
DATA_NAME = "accel_ms2_xyz"
folders = ["wing", "ring", "slope"]
names = [
    "hyw", "shiyun", "tangsy", "dengyl", "zhangxy", "pengxl", "liucx",
    "jiangyh", "xunkai"
]


def prepare_original_data(folder, name, data, file_to_read):  # pylint: disable=redefined-outer-name
  """Read collected data from files."""
  if folder != "negative":
    with open(file_to_read, "r") as f:
      lines = csv.reader(f)
      data_new = {}
      data_new[LABEL_NAME] = folder
      data_new[DATA_NAME] = []
      data_new["name"] = name
      for idx, line in enumerate(lines):  # pylint: disable=unused-variable,redefined-outer-name
        if len(line) == 3:
          if line[2] == "-" and data_new[DATA_NAME]:
            data.append(data_new)
            data_new = {}
            data_new[LABEL_NAME] = folder
            data_new[DATA_NAME] = []
            data_new["name"] = name
          elif line[2] != "-":
            data_new[DATA_NAME].append([float(i) for i in line[0:3]])
      data.append(data_new)
  else:
    with open(file_to_read, "r") as f:
      lines = csv.reader(f)
      data_new = {}
      data_new[LABEL_NAME] = folder
      data_new[DATA_NAME] = []
      data_new["name"] = name
      for idx, line in enumerate(lines):
        if len(line) == 3 and line[2] != "-":
          if len(data_new[DATA_NAME]) == 120:
            data.append(data_new)
            data_new = {}
            data_new[LABEL_NAME] = folder
            data_new[DATA_NAME] = []
            data_new["name"] = name
          else:
            data_new[DATA_NAME].append([float(i) for i in line[0:3]])
      data.append(data_new)


def generate_negative_data(data):  # pylint: disable=redefined-outer-name
  """Generate negative data labeled as 'negative6~8'."""
  # Big movement -> around straight line
  for i in range(100):
    if i > 80:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative8"}
    elif i > 60:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative7"}
    else:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative6"}
    start_x = (random.random() - 0.5) * 2000
    start_y = (random.random() - 0.5) * 2000
    start_z = (random.random() - 0.5) * 2000
    x_increase = (random.random() - 0.5) * 10
    y_increase = (random.random() - 0.5) * 10
    z_increase = (random.random() - 0.5) * 10
    for j in range(128):
      dic[DATA_NAME].append([
          start_x + j * x_increase + (random.random() - 0.5) * 6,
          start_y + j * y_increase + (random.random() - 0.5) * 6,
          start_z + j * z_increase + (random.random() - 0.5) * 6
      ])
    data.append(dic)
  # Random
  for i in range(100):
    if i > 80:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative8"}
    elif i > 60:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative7"}
    else:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative6"}
    for j in range(128):
      dic[DATA_NAME].append([(random.random() - 0.5) * 1000,
                             (random.random() - 0.5) * 1000,
                             (random.random() - 0.5) * 1000])
    data.append(dic)
  # Stay still
  for i in range(100):
    if i > 80:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative8"}
    elif i > 60:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative7"}
    else:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative6"}
    start_x = (random.random() - 0.5) * 2000
    start_y = (random.random() - 0.5) * 2000
    start_z = (random.random() - 0.5) * 2000
    for j in range(128):
      dic[DATA_NAME].append([
          start_x + (random.random() - 0.5) * 40,
          start_y + (random.random() - 0.5) * 40,
          start_z + (random.random() - 0.5) * 40
      ])
    data.append(dic)


# Write data to file
def write_data(data_to_write, path):
  with open(path, "w") as f:
    for idx, item in enumerate(data_to_write):  # pylint: disable=unused-variable,redefined-outer-name
      dic = json.dumps(item, ensure_ascii=False)
      f.write(dic)
      f.write("\n")


if __name__ == "__main__":
  data = []  # pylint: disable=redefined-outer-name
  for idx1, folder in enumerate(folders):
    for idx2, name in enumerate(names):
      prepare_original_data(folder, name, data,
                            "./%s/output_%s_%s.txt" % (folder, folder, name))
  for idx in range(5):
    prepare_original_data("negative", "negative%d" % (idx + 1), data,
                          "./negative/output_negative_%d.txt" % (idx + 1))
  generate_negative_data(data)
  print("data_length: " + str(len(data)))
  if not os.path.exists("./data"):
    os.makedirs("./data")
  write_data(data, "./data/complete_data")
