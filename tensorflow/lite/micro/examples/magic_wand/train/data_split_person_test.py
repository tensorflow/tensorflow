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

"""Test for data_split_person.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from data_split_person import person_split
from data_split_person import read_data


class TestSplitPerson(unittest.TestCase):

  def setUp(self):  # pylint: disable=g-missing-super-call
    self.data = read_data("./data/complete_data")

  def test_person_split(self):
    train_names = ["dengyl"]
    valid_names = ["liucx"]
    test_names = ["tangsy"]
    dengyl_num = 63
    liucx_num = 63
    tangsy_num = 30
    train_data, valid_data, test_data = person_split(self.data, train_names,
                                                     valid_names, test_names)
    self.assertEqual(len(train_data), dengyl_num)
    self.assertEqual(len(valid_data), liucx_num)
    self.assertEqual(len(test_data), tangsy_num)
    self.assertIsInstance(train_data, list)
    self.assertIsInstance(valid_data, list)
    self.assertIsInstance(test_data, list)
    self.assertIsInstance(train_data[0], dict)
    self.assertIsInstance(valid_data[0], dict)
    self.assertIsInstance(test_data[0], dict)


if __name__ == "__main__":
  unittest.main()
