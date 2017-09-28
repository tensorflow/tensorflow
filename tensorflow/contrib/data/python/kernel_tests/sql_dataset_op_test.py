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

from tensorflow.contrib.data.python.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SqlDatasetTest(test.TestCase):

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

  # Test that SqlDataset can read from a database table.
  def testReadResultSet(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.string), 2)
    with self.test_session() as sess:
      for _ in range(2):  # Run twice to verify statelessness of db operations.
        sess.run(
            init_op,
            feed_dict={
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

  # Test that `SqlDataset` can read an integer from a SQLite database table and
  # place it in an `int8` tensor.
  def testReadResultSetInt8(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int8))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, desk_number FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 9), sess.run(get_next))
      self.assertEqual((b"Jane", 127), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that `SqlDataset` can read a negative or 0-valued integer from a
  # SQLite database table and place it in an `int8` tensor.
  def testReadResultSetInt8NegativeAndZero(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int8,
                                                dtypes.int8))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, income, favorite_negative_number "
                          "FROM students "
                          "WHERE first_name = 'John' ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 0, -2), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  # Test that `SqlDataset` can read a large (positive or negative) integer from
  # a SQLite database table and place it in an `int8` tensor.
  def testReadResultSetInt8MaxValues(self):
    init_op, get_next = self._createSqlDataset((dtypes.int8, dtypes.int8))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query:
                  "SELECT desk_number, favorite_negative_number FROM students "
                  "ORDER BY first_name DESC"
          })
      self.assertEqual((9, -2), sess.run(get_next))
      # Max and min values of int8
      self.assertEqual((127, -128), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  # Test that `SqlDataset` can read an integer from a SQLite database table and
  # place it in an `int16` tensor.
  def testReadResultSetInt16(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int16))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, desk_number FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 9), sess.run(get_next))
      self.assertEqual((b"Jane", 127), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that `SqlDataset` can read a negative or 0-valued integer from a
  # SQLite database table and place it in an `int16` tensor.
  def testReadResultSetInt16NegativeAndZero(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int16,
                                                dtypes.int16))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, income, favorite_negative_number "
                          "FROM students "
                          "WHERE first_name = 'John' ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 0, -2), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  # Test that `SqlDataset` can read a large (positive or negative) integer from
  # a SQLite database table and place it in an `int16` tensor.
  def testReadResultSetInt16MaxValues(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int16))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, favorite_medium_sized_number "
                          "FROM students ORDER BY first_name DESC"
          })
      # Max value of int16
      self.assertEqual((b"John", 32767), sess.run(get_next))
      # Min value of int16
      self.assertEqual((b"Jane", -32768), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  # Test that `SqlDataset` can read an integer from a SQLite database table and
  # place it in an `int32` tensor.
  def testReadResultSetInt32(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int32))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, desk_number FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 9), sess.run(get_next))
      self.assertEqual((b"Jane", 127), sess.run(get_next))

  # Test that `SqlDataset` can read a negative or 0-valued integer from a
  # SQLite database table and place it in an `int32` tensor.
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

  # Test that `SqlDataset` can read a large (positive or negative) integer from
  # a SQLite database table and place it in an `int32` tensor.
  def testReadResultSetInt32MaxValues(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int32))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, favorite_number FROM students "
                          "ORDER BY first_name DESC"
          })
      # Max value of int32
      self.assertEqual((b"John", 2147483647), sess.run(get_next))
      # Min value of int32
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

  # Test that `SqlDataset` can read an integer from a SQLite database table
  # and place it in an `int64` tensor.
  def testReadResultSetInt64(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int64))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, desk_number FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 9), sess.run(get_next))
      self.assertEqual((b"Jane", 127), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that `SqlDataset` can read a negative or 0-valued integer from a
  # SQLite database table and place it in an `int64` tensor.
  def testReadResultSetInt64NegativeAndZero(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int64))
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

  # Test that `SqlDataset` can read a large (positive or negative) integer from
  # a SQLite database table and place it in an `int64` tensor.
  def testReadResultSetInt64MaxValues(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.int64))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query:
                  "SELECT first_name, favorite_big_number FROM students "
                  "ORDER BY first_name DESC"
          })
      # Max value of int64
      self.assertEqual((b"John", 9223372036854775807), sess.run(get_next))
      # Min value of int64
      self.assertEqual((b"Jane", -9223372036854775808), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  # Test that `SqlDataset` can read an integer from a SQLite database table and
  # place it in a `uint8` tensor.
  def testReadResultSetUInt8(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.uint8))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, desk_number FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 9), sess.run(get_next))
      self.assertEqual((b"Jane", 127), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that `SqlDataset` can read the minimum and maximum uint8 values from a
  # SQLite database table and place them in `uint8` tensors.
  def testReadResultSetUInt8MinAndMaxValues(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.uint8))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, brownie_points FROM students "
                          "ORDER BY first_name DESC"
          })
      # Min value of uint8
      self.assertEqual((b"John", 0), sess.run(get_next))
      # Max value of uint8
      self.assertEqual((b"Jane", 255), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  # Test that `SqlDataset` can read an integer from a SQLite database table
  # and place it in a `uint16` tensor.
  def testReadResultSetUInt16(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.uint16))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, desk_number FROM students "
                          "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", 9), sess.run(get_next))
      self.assertEqual((b"Jane", 127), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that `SqlDataset` can read the minimum and maximum uint16 values from a
  # SQLite database table and place them in `uint16` tensors.
  def testReadResultSetUInt16MinAndMaxValues(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.uint16))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, account_balance FROM students "
                          "ORDER BY first_name DESC"
          })
      # Min value of uint16
      self.assertEqual((b"John", 0), sess.run(get_next))
      # Max value of uint16
      self.assertEqual((b"Jane", 65535), sess.run(get_next))
    with self.assertRaises(errors.OutOfRangeError):
      sess.run(get_next)

  # Test that `SqlDataset` can read a 0-valued and 1-valued integer from a
  # SQLite database table and place them as `True` and `False` respectively
  # in `bool` tensors.
  def testReadResultSetBool(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.bool))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query:
                  "SELECT first_name, registration_complete FROM students "
                  "ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", True), sess.run(get_next))
      self.assertEqual((b"Jane", False), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that `SqlDataset` can read an integer that is not 0-valued or 1-valued
  # from a SQLite database table and place it as `True` in a `bool` tensor.
  def testReadResultSetBoolNotZeroOrOne(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.bool))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query: "SELECT first_name, favorite_medium_sized_number "
                          "FROM students ORDER BY first_name DESC"
          })
      self.assertEqual((b"John", True), sess.run(get_next))
      self.assertEqual((b"Jane", True), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that `SqlDataset` can read a float from a SQLite database table
  # and place it in a `float64` tensor.
  def testReadResultSetFloat64(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.float64))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query:
                  "SELECT first_name, last_name, victories FROM townspeople "
                  "ORDER BY first_name"
          })
      self.assertEqual((b"George", b"Washington", 20.0), sess.run(get_next))
      self.assertEqual((b"John", b"Adams", -19.95), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that `SqlDataset` can read a float from a SQLite database table beyond
  # the precision of 64-bit IEEE, without throwing an error. Test that
  # `SqlDataset` identifies such a value as equal to itself.
  def testReadResultSetFloat64OverlyPrecise(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.float64))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query:
                  "SELECT first_name, last_name, accolades FROM townspeople "
                  "ORDER BY first_name"
          })
      self.assertEqual(
          (b"George", b"Washington",
           1331241.321342132321324589798264627463827647382647382643874),
          sess.run(get_next))
      self.assertEqual(
          (b"John", b"Adams",
           1331241321342132321324589798264627463827647382647382643874.0),
          sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Test that `SqlDataset` can read a float from a SQLite database table,
  # representing the largest integer representable as a 64-bit IEEE float
  # such that the previous integer is also representable as a 64-bit IEEE float.
  # Test that `SqlDataset` can distinguish these two numbers.
  def testReadResultSetFloat64LargestConsecutiveWholeNumbersNotEqual(self):
    init_op, get_next = self._createSqlDataset((dtypes.string, dtypes.string,
                                                dtypes.float64))
    with self.test_session() as sess:
      sess.run(
          init_op,
          feed_dict={
              self.query:
                  "SELECT first_name, last_name, triumphs FROM townspeople "
                  "ORDER BY first_name"
          })
      self.assertNotEqual((b"George", b"Washington", 9007199254740992.0),
                          sess.run(get_next))
      self.assertNotEqual((b"John", b"Adams", 9007199254740991.0),
                          sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
