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
"""Tests for experimental sql input op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sqlite3

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SqlDatasetTest(test.TestCase):

  def setUp(self):
    self.data_source_name = os.path.join(test.get_temp_dir(), "tftest.sqlite")
    self.driver_name = array_ops.placeholder(dtypes.string, shape=[])
    self.query = array_ops.placeholder(dtypes.string, shape=[])
    self.output_types = (dtypes.string, dtypes.string, dtypes.string)

    conn = sqlite3.connect(self.data_source_name)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS students")
    c.execute("DROP TABLE IF EXISTS people")
    c.execute(
        "CREATE TABLE IF NOT EXISTS students (id INTEGER NOT NULL PRIMARY KEY,"
        " first_name VARCHAR(100), last_name VARCHAR(100), motto VARCHAR(100))")
    c.execute(
        "INSERT INTO students (first_name, last_name, motto) VALUES ('John', "
        "'Doe', 'Hi!'), ('Apple', 'Orange', 'Hi again!')")
    c.execute(
        "CREATE TABLE IF NOT EXISTS people (id INTEGER NOT NULL PRIMARY KEY, "
        "first_name VARCHAR(100), last_name VARCHAR(100), state VARCHAR(100))")
    c.execute(
        "INSERT INTO people (first_name, last_name, state) VALUES ('Benjamin',"
        " 'Franklin', 'Pennsylvania'), ('John', 'Doe', 'California')")
    conn.commit()
    conn.close()

    dataset = dataset_ops.SqlDataset(self.driver_name, self.data_source_name,
                                     self.query, self.output_types).repeat(2)
    iterator = dataset.make_initializable_iterator()
    self.init_op = iterator.initializer
    self.get_next = iterator.get_next()

  # Test that SqlDataset can read from a database table.
  def testReadResultSet(self):
    with self.test_session() as sess:
      for _ in range(2):  # Run twice to verify statelessness of db operations.
        sess.run(
            self.init_op,
            feed_dict={
                self.driver_name: "sqlite",
                self.query: "SELECT first_name, last_name, motto FROM students "
                            "ORDER BY first_name DESC"
            })
        for _ in range(2):  # Dataset is repeated. See setUp.
          self.assertEqual((b"John", b"Doe", b"Hi!"), sess.run(self.get_next))
          self.assertEqual((b"Apple", b"Orange", b"Hi again!"),
                           sess.run(self.get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(self.get_next)

  # Test that SqlDataset works on a join query.
  def testReadResultSetJoinQuery(self):
    with self.test_session() as sess:
      sess.run(
          self.init_op,
          feed_dict={
              self.driver_name: "sqlite",
              self.query:
                  "SELECT students.first_name, state, motto FROM students "
                  "INNER JOIN people "
                  "ON students.first_name = people.first_name "
                  "AND students.last_name = people.last_name"
          })
      for _ in range(2):
        self.assertEqual((b"John", b"California", b"Hi!"),
                         sess.run(self.get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.get_next)

  # Test that an `OutOfRangeError` is raised on the first call to `get_next`
  # if result set is empty.
  def testReadEmptyResultSet(self):
    with self.test_session() as sess:
      sess.run(
          self.init_op,
          feed_dict={
              self.driver_name: "sqlite",
              self.query: "SELECT first_name, last_name, motto FROM students "
                          "WHERE first_name = 'Nonexistent'"
          })
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.get_next)

  # Test that an error is raised when `driver_name` is invalid.
  def testReadResultSetWithInvalidDriverName(self):
    with self.test_session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(
            self.init_op,
            feed_dict={
                self.driver_name: "sqlfake",
                self.query: "SELECT first_name, last_name, motto FROM students "
                            "ORDER BY first_name DESC"
            })

  # Test that an error is raised when a column name in `query` is nonexistent
  def testReadResultSetWithInvalidColumnName(self):
    with self.test_session() as sess:
      sess.run(
          self.init_op,
          feed_dict={
              self.driver_name: "sqlite",
              self.query:
                  "SELECT first_name, last_name, fake_column FROM students "
                  "ORDER BY first_name DESC"
          })
      with self.assertRaises(errors.UnknownError):
        sess.run(self.get_next)

  # Test that an error is raised when there is a syntax error in `query`.
  def testReadResultSetOfQueryWithSyntaxError(self):
    with self.test_session() as sess:
      sess.run(
          self.init_op,
          feed_dict={
              self.driver_name: "sqlite",
              self.query:
                  "SELEmispellECT first_name, last_name, motto FROM students "
                  "ORDER BY first_name DESC"
          })
      with self.assertRaises(errors.UnknownError):
        sess.run(self.get_next)

  # Test that an error is raised when the number of columns in `query`
  # does not match the length of `output_types`.
  def testReadResultSetWithMismatchBetweenColumnsAndOutputTypes(self):
    with self.test_session() as sess:
      sess.run(
          self.init_op,
          feed_dict={
              self.driver_name: "sqlite",
              self.query: "SELECT first_name, last_name FROM students "
                          "ORDER BY first_name DESC"
          })
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(self.get_next)

  # Test that no results are returned when `query` is an insert query rather
  # than a select query. In particular, the error refers to the number of
  # output types passed to the op not matching the number of columns in the
  # result set of the query (namely, 0 for an insert statement.)
  def testReadResultSetOfInsertQuery(self):
    with self.test_session() as sess:
      sess.run(
          self.init_op,
          feed_dict={
              self.driver_name: "sqlite",
              self.query:
                  "INSERT INTO students (first_name, last_name, motto) "
                  "VALUES ('Foo', 'Bar', 'Baz'), ('Fizz', 'Buzz', 'Fizzbuzz')"
          })
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(self.get_next)


if __name__ == "__main__":
  test.main()
