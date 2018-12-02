# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Base class for testing `tf.data.experimental.SqlDataset`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sqlite3

from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SqlDatasetTestBase(test_base.DatasetTestBase):
  """Base class for setting up and testing SqlDataset."""

  def _createSqlDataset(self, output_types, num_repeats=1):
    dataset = readers.SqlDataset(self.driver_name, self.data_source_name,
                                 self.query, output_types).repeat(num_repeats)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()
    return init_op, get_next

  def setUp(self):
    self.data_source_name = os.path.join(test.get_temp_dir(), "tftest.sqlite")
    self.driver_name = array_ops.placeholder_with_default(
        array_ops.constant("sqlite", dtypes.string), shape=[])
    self.query = array_ops.placeholder(dtypes.string, shape=[])

    conn = sqlite3.connect(self.data_source_name)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS students")
    c.execute("DROP TABLE IF EXISTS people")
    c.execute("DROP TABLE IF EXISTS townspeople")
    c.execute(
        "CREATE TABLE IF NOT EXISTS students (id INTEGER NOT NULL PRIMARY KEY, "
        "first_name VARCHAR(100), last_name VARCHAR(100), motto VARCHAR(100), "
        "school_id VARCHAR(100), favorite_nonsense_word VARCHAR(100), "
        "desk_number INTEGER, income INTEGER, favorite_number INTEGER, "
        "favorite_big_number INTEGER, favorite_negative_number INTEGER, "
        "favorite_medium_sized_number INTEGER, brownie_points INTEGER, "
        "account_balance INTEGER, registration_complete INTEGER)")
    c.executemany(
        "INSERT INTO students (first_name, last_name, motto, school_id, "
        "favorite_nonsense_word, desk_number, income, favorite_number, "
        "favorite_big_number, favorite_negative_number, "
        "favorite_medium_sized_number, brownie_points, account_balance, "
        "registration_complete) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [("John", "Doe", "Hi!", "123", "n\0nsense", 9, 0, 2147483647,
          9223372036854775807, -2, 32767, 0, 0, 1),
         ("Jane", "Moe", "Hi again!", "1000", "nonsense\0", 127, -20000,
          -2147483648, -9223372036854775808, -128, -32768, 255, 65535, 0)])
    c.execute(
        "CREATE TABLE IF NOT EXISTS people (id INTEGER NOT NULL PRIMARY KEY, "
        "first_name VARCHAR(100), last_name VARCHAR(100), state VARCHAR(100))")
    c.executemany(
        "INSERT INTO PEOPLE (first_name, last_name, state) VALUES (?, ?, ?)",
        [("Benjamin", "Franklin", "Pennsylvania"), ("John", "Doe",
                                                    "California")])
    c.execute(
        "CREATE TABLE IF NOT EXISTS townspeople (id INTEGER NOT NULL PRIMARY "
        "KEY, first_name VARCHAR(100), last_name VARCHAR(100), victories "
        "FLOAT, accolades FLOAT, triumphs FLOAT)")
    c.executemany(
        "INSERT INTO townspeople (first_name, last_name, victories, "
        "accolades, triumphs) VALUES (?, ?, ?, ?, ?)",
        [("George", "Washington", 20.00,
          1331241.321342132321324589798264627463827647382647382643874,
          9007199254740991.0),
         ("John", "Adams", -19.95,
          1331241321342132321324589798264627463827647382647382643874.0,
          9007199254740992.0)])
    conn.commit()
    conn.close()
