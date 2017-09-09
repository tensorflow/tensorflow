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

  def _createSqlDataset(self, output_types, num_repeats=1):
    dataset = dataset_ops.SqlDataset(self.driver_name, self.data_source_name,
                                     self.query,
                                     output_types).repeat(num_repeats)
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
    c.execute(
        "CREATE TABLE IF NOT EXISTS students (id INTEGER NOT NULL PRIMARY KEY,"
        " first_name VARCHAR(100), last_name VARCHAR(100), motto VARCHAR(100),"
        " school_id VARCHAR(100), favorite_nonsense_word VARCHAR(100), "
        "grade_level INTEGER, income INTEGER, favorite_number INTEGER)")
    c.executemany(
        "INSERT INTO students (first_name, last_name, motto, school_id, "
        "favorite_nonsense_word, grade_level, income, favorite_number) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [("John", "Doe", "Hi!", "123", "n\0nsense", 9, 0, 2147483647),
         ("Jane", "Moe", "Hi again!", "1000", "nonsense\0", 11, -20000,
          -2147483648)])
    c.execute(
        "CREATE TABLE IF NOT EXISTS people (id INTEGER NOT NULL PRIMARY KEY, "
        "first_name VARCHAR(100), last_name VARCHAR(100), state VARCHAR(100))")
    c.executemany(
        "INSERT INTO people (first_name, last_name, state) VALUES (?, ?, ?)",
        [("Benjamin", "Franklin", "Pennsylvania"), ("John", "Doe",
                                                    "California")])
    conn.commit()
    conn.close()

  # Test that SqlDataset can read from a database table.
  def testReadResultSet(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string), 2)
    with self.test_session() as sess:
      for _ in range(2):  # Run twice to verify statelessness of db operations.
        sess.run(
            init_op,
            feed_dict={
                self.driver_name: "sqlite",
                self.query: "SELECT first_name, last_name, motto FROM students "
                            "ORDER BY first_name DESC"
            })
        for _ in range(2):  # Dataset is repeated. See setUp.
          self.assertEqual((b"John", b"Doe", b"Hi!"), sess.run(get_next))
          self.assertEqual((b"Jane", b"Moe", b"Hi again!"), sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  # Test that SqlDataset works on a join query.
  def testReadResultSetJoinQuery(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.driver_name: "sqlite",
              self.query:
                  "SELECT students.first_name, state, motto FROM students "
                  "INNER JOIN people "
                  "ON students.first_name = people.first_name "
                  "AND students.last_name = people.last_name"
          })
      self.assertEqual((b"John", b"California", b"Hi!"), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that SqlDataset can read a database entry with a null-terminator
  # in the middle of the text and place the entry in a `string` tensor.
  def testReadResultSetNullTerminator(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.driver_name: "sqlite",
              self.query:
                  "SELECT first_name, last_name, favorite_nonsense_word "
                  "FROM students ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", b"Doe", b"n\0nsense"), sess.run(get_next))
      self.assertEqual((b"Jane", b"Moe", b"nonsense\0"), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  # Test that SqlDataset works when used on two different queries.
  # Because the output types of the dataset must be determined at graph-creation
  # time, the two queries must have the same number and types of columns.
  def testReadResultSetReuseSqlDataset(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, last_name, motto FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", b"Doe", b"Hi!"), sess.run(get_next))
      self.assertEqual((b"Jane", b"Moe", b"Hi again!"), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, last_name, state FROM people "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", b"Doe", b"California"), sess.run(get_next))
      self.assertEqual((b"Benjamin", b"Franklin", b"Pennsylvania"),
                       sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that an `OutOfRangeError` is raised on the first call to
  # `get_next_str_only` if result set is empty.
  def testReadEmptyResultSet(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, last_name, motto FROM students "
                          "WHERE first_name = 'Nonexistent'"
          })
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that an error is raised when `driver_name` is invalid.
  def testReadResultSetWithInvalidDriverName(self):
    init_op = self._createSqlDataset((dtypes.string, dtypes.string,
                                      dtypes.string))[0]
    with self.test_session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(
            init_op,
            feed_dict={
                self.driver_name: "sqlfake",
                self.query: "SELECT first_name, last_name, motto FROM students "
                            "ORDER BY first_name DESC"
            })

  # Test that an error is raised when a column name in `query` is nonexistent
  def testReadResultSetWithInvalidColumnName(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query:
                  "SELECT first_name, last_name, fake_column FROM students "
                  "ORDER BY first_name DESC"
          })
      with self.assertRaises(errors.UnknownError):
        sess.run(get_next)

  # Test that an error is raised when there is a syntax error in `query`.
  def testReadResultSetOfQueryWithSyntaxError(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query:
                  "SELEmispellECT first_name, last_name, motto FROM students "
                  "ORDER BY first_name DESC"
          })
      with self.assertRaises(errors.UnknownError):
        sess.run(get_next)

  # Test that an error is raised when the number of columns in `query`
  # does not match the length of `output_types`.
  def testReadResultSetWithMismatchBetweenColumnsAndOutputTypes(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, last_name FROM students "
                          "ORDER BY first_name DESC"
          })
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next)

  # Test that no results are returned when `query` is an insert query rather
  # than a select query. In particular, the error refers to the number of
  # output types passed to the op not matching the number of columns in the
  # result set of the query (namely, 0 for an insert statement.)
  def testReadResultSetOfInsertQuery(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query:
                  "INSERT INTO students (first_name, last_name, motto) "
                  "VALUES ('Foo', 'Bar', 'Baz'), ('Fizz', 'Buzz', 'Fizzbuzz')"
          })
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next)

  def testReadResultSetInt32(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int32))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, grade_level FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 9), sess.run(get_next))
      self.assertEqual((b"Jane", 11), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testReadResultSetInt32NegativeAndZero(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int32))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, income FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 0), sess.run(get_next))
      self.assertEqual((b"Jane", -20000), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  def testReadResultSetInt32MaxValues(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int32))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, favorite_number FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 2147483647), sess.run(get_next))
      self.assertEqual((b"Jane", -2147483648), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  # Test that `SqlDataset` can read a numeric `varchar` from a SQLite database
  # table and place it in an `int32` tensor.
  def testReadResultSetInt32VarCharColumnAsInt(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int32))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, school_id FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 123), sess.run(get_next))
      self.assertEqual((b"Jane", 1000), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)


if __name__ == "__main__":
  test.main()
