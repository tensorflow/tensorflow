# Lint as: python3
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

"""Test for data_split.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import unittest
from data_split import read_data
from data_split import split_data


class TestSplit(unittest.TestCase):

  def setUp(self):  # pylint: disable=g-missing-super-call
    self.data = read_data("./data/complete_data")
    self.num_dic = {"wing": 0, "ring": 0, "slope": 0, "negative": 0}
    with open("./data/complete_data", "r") as f:
      lines = f.readlines()
      self.num = len(lines)

  def test_read_data(self):
    self.assertEqual(len(self.data), self.num)
    self.assertIsInstance(self.data, list)
    self.assertIsInstance(self.data[0], dict)
    self.assertEqual(
        set(list(self.data[-1])), set(["gesture", "accel_ms2_xyz", "name"]))

  def test_split_data(self):
    with open("./data/complete_data", "r") as f:
      lines = f.readlines()
      for idx, line in enumerate(lines):  # pylint: disable=unused-variable
        dic = json.loads(line)
        for ges in self.num_dic:
          if dic["gesture"] == ges:
            self.num_dic[ges] += 1
    train_data_0, valid_data_0, test_data_100 = split_data(self.data, 0, 0)
    train_data_50, valid_data_50, test_data_0 = split_data(self.data, 0.5, 0.5)
    train_data_60, valid_data_20, test_data_20 = split_data(self.data, 0.6, 0.2)
    len_60 = int(self.num_dic["wing"] * 0.6) + int(
        self.num_dic["ring"] * 0.6) + int(self.num_dic["slope"] * 0.6) + int(
            self.num_dic["negative"] * 0.6)
    len_50 = int(self.num_dic["wing"] * 0.5) + int(
        self.num_dic["ring"] * 0.5) + int(self.num_dic["slope"] * 0.5) + int(
            self.num_dic["negative"] * 0.5)
    len_20 = int(self.num_dic["wing"] * 0.2) + int(
        self.num_dic["ring"] * 0.2) + int(self.num_dic["slope"] * 0.2) + int(
            self.num_dic["negative"] * 0.2)
    self.assertEqual(len(train_data_0), 0)
    self.assertEqual(len(train_data_50), len_50)
    self.assertEqual(len(train_data_60), len_60)
    self.assertEqual(len(valid_data_0), 0)
    self.assertEqual(len(valid_data_50), len_50)
    self.assertEqual(len(valid_data_20), len_20)
    self.assertEqual(len(test_data_100), self.num)
    self.assertEqual(len(test_data_0), (self.num - 2 * len_50))
    self.assertEqual(len(test_data_20), (self.num - len_60 - len_20))


if __name__ == "__main__":
  unittest.main()
