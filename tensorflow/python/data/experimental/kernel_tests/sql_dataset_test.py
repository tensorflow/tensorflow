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
"""Tests for `tf.data.experimental.SqlDataset`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized

import sqlite3

from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SqlDatasetTestBase(test_base.DatasetTestBase):
  """Base class for setting up and testing SqlDataset."""

  def _createSqlDataset(self,
                        query,
                        output_types,
                        driver_name="sqlite",
                        num_repeats=1):
    dataset = readers.SqlDataset(driver_name, self.data_source_name, query,
                                 output_types).repeat(num_repeats)
    return dataset

  def setUp(self):
    super(SqlDatasetTestBase, self).setUp()
    self.data_source_name = os.path.join(test.get_temp_dir(), "tftest.sqlite")

    conn = sqlite3.connect(self.data_source_name)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS students")
    c.execute("DROP TABLE IF EXISTS people")
    c.execute("DROP TABLE IF EXISTS townspeople")
    c.execute("DROP TABLE IF EXISTS data")
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
    c.execute("CREATE TABLE IF NOT EXISTS data (col1 INTEGER)")
    c.executemany("INSERT INTO DATA VALUES (?)", [(0,), (1,), (2,)])
    conn.commit()
    conn.close()


class SqlDatasetTest(SqlDatasetTestBase, parameterized.TestCase):

  # Test that SqlDataset can read from a database table.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSet(self):
    for _ in range(2):  # Run twice to verify statelessness of db operations.
      dataset = self._createSqlDataset(
          query="SELECT first_name, last_name, motto FROM students "
          "ORDER BY first_name DESC",
          output_types=(dtypes.string, dtypes.string, dtypes.string),
          num_repeats=2)
      self.assertDatasetProduces(
          dataset,
          expected_output=[(b"John", b"Doe", b"Hi!"),
                           (b"Jane", b"Moe", b"Hi again!")] * 2,
          num_test_iterations=2)

  # Test that SqlDataset works on a join query.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetJoinQuery(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT students.first_name, state, motto FROM students "
            "INNER JOIN people "
            "ON students.first_name = people.first_name "
            "AND students.last_name = people.last_name",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))

    self.assertEqual((b"John", b"California", b"Hi!"),
                     self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that SqlDataset can read a database entry with a null-terminator
  # in the middle of the text and place the entry in a `string` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetNullTerminator(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, favorite_nonsense_word "
            "FROM students ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))

    self.assertEqual((b"John", b"Doe", b"n\0nsense"), self.evaluate(get_next()))
    self.assertEqual((b"Jane", b"Moe", b"nonsense\0"),
                     self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that SqlDataset works when used on two different queries.
  # Because the output types of the dataset must be determined at graph-creation
  # time, the two queries must have the same number and types of columns.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetReuseSqlDataset(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, motto FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))
    self.assertEqual((b"John", b"Doe", b"Hi!"), self.evaluate(get_next()))
    self.assertEqual((b"Jane", b"Moe", b"Hi again!"), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, state FROM people "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))
    self.assertEqual((b"John", b"Doe", b"California"),
                     self.evaluate(get_next()))
    self.assertEqual((b"Benjamin", b"Franklin", b"Pennsylvania"),
                     self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that an `OutOfRangeError` is raised on the first call to
  # `get_next_str_only` if result set is empty.
  @combinations.generate(test_base.default_test_combinations())
  def testReadEmptyResultSet(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, motto FROM students "
            "WHERE first_name = 'Nonexistent'",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that an error is raised when `driver_name` is invalid.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetWithInvalidDriverName(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = self._createSqlDataset(
          driver_name="sqlfake",
          query="SELECT first_name, last_name, motto FROM students "
          "ORDER BY first_name DESC",
          output_types=(dtypes.string, dtypes.string, dtypes.string))
      self.assertDatasetProduces(dataset, expected_output=[])

  # Test that an error is raised when a column name in `query` is nonexistent
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetWithInvalidColumnName(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, fake_column FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))
    with self.assertRaises(errors.UnknownError):
      self.evaluate(get_next())

  # Test that an error is raised when there is a syntax error in `query`.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetOfQueryWithSyntaxError(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELEmispellECT first_name, last_name, motto FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))
    with self.assertRaises(errors.UnknownError):
      self.evaluate(get_next())

  # Test that an error is raised when the number of columns in `query`
  # does not match the length of `, output_types`.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetWithMismatchBetweenColumnsAndOutputTypes(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())

  # Test that no results are returned when `query` is an insert query rather
  # than a select query. In particular, the error refers to the number of
  # output types passed to the op not matching the number of columns in the
  # result set of the query (namely, 0 for an insert statement.)
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetOfInsertQuery(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="INSERT INTO students (first_name, last_name, motto) "
            "VALUES ('Foo', 'Bar', 'Baz'), ('Fizz', 'Buzz', 'Fizzbuzz')",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read an integer from a SQLite database table and
  # place it in an `int8` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt8(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, desk_number FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int8)))
    self.assertEqual((b"John", 9), self.evaluate(get_next()))
    self.assertEqual((b"Jane", 127), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a negative or 0-valued integer from a
  # SQLite database table and place it in an `int8` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt8NegativeAndZero(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, income, favorite_negative_number "
            "FROM students "
            "WHERE first_name = 'John' ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int8, dtypes.int8)))
    self.assertEqual((b"John", 0, -2), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a large (positive or negative) integer from
  # a SQLite database table and place it in an `int8` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt8MaxValues(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT desk_number, favorite_negative_number FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.int8, dtypes.int8)))
    self.assertEqual((9, -2), self.evaluate(get_next()))
    # Max and min values of int8
    self.assertEqual((127, -128), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read an integer from a SQLite database table and
  # place it in an `int16` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt16(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, desk_number FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int16)))
    self.assertEqual((b"John", 9), self.evaluate(get_next()))
    self.assertEqual((b"Jane", 127), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a negative or 0-valued integer from a
  # SQLite database table and place it in an `int16` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt16NegativeAndZero(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, income, favorite_negative_number "
            "FROM students "
            "WHERE first_name = 'John' ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int16, dtypes.int16)))
    self.assertEqual((b"John", 0, -2), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a large (positive or negative) integer from
  # a SQLite database table and place it in an `int16` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt16MaxValues(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, favorite_medium_sized_number "
            "FROM students ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int16)))
    # Max value of int16
    self.assertEqual((b"John", 32767), self.evaluate(get_next()))
    # Min value of int16
    self.assertEqual((b"Jane", -32768), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read an integer from a SQLite database table and
  # place it in an `int32` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt32(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, desk_number FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int32)))
    self.assertEqual((b"John", 9), self.evaluate(get_next()))
    self.assertEqual((b"Jane", 127), self.evaluate(get_next()))

  # Test that `SqlDataset` can read a negative or 0-valued integer from a
  # SQLite database table and place it in an `int32` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt32NegativeAndZero(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, income FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int32)))
    self.assertEqual((b"John", 0), self.evaluate(get_next()))
    self.assertEqual((b"Jane", -20000), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a large (positive or negative) integer from
  # a SQLite database table and place it in an `int32` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt32MaxValues(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, favorite_number FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int32)))
    # Max value of int32
    self.assertEqual((b"John", 2147483647), self.evaluate(get_next()))
    # Min value of int32
    self.assertEqual((b"Jane", -2147483648), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a numeric `varchar` from a SQLite database
  # table and place it in an `int32` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt32VarCharColumnAsInt(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, school_id FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int32)))
    self.assertEqual((b"John", 123), self.evaluate(get_next()))
    self.assertEqual((b"Jane", 1000), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read an integer from a SQLite database table
  # and place it in an `int64` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt64(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, desk_number FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int64)))
    self.assertEqual((b"John", 9), self.evaluate(get_next()))
    self.assertEqual((b"Jane", 127), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a negative or 0-valued integer from a
  # SQLite database table and place it in an `int64` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt64NegativeAndZero(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, income FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int64)))
    self.assertEqual((b"John", 0), self.evaluate(get_next()))
    self.assertEqual((b"Jane", -20000), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a large (positive or negative) integer from
  # a SQLite database table and place it in an `int64` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetInt64MaxValues(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, favorite_big_number FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.int64)))
    # Max value of int64
    self.assertEqual((b"John", 9223372036854775807), self.evaluate(get_next()))
    # Min value of int64
    self.assertEqual((b"Jane", -9223372036854775808), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read an integer from a SQLite database table and
  # place it in a `uint8` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetUInt8(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, desk_number FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.uint8)))
    self.assertEqual((b"John", 9), self.evaluate(get_next()))
    self.assertEqual((b"Jane", 127), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read the minimum and maximum uint8 values from a
  # SQLite database table and place them in `uint8` tensors.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetUInt8MinAndMaxValues(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, brownie_points FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.uint8)))
    # Min value of uint8
    self.assertEqual((b"John", 0), self.evaluate(get_next()))
    # Max value of uint8
    self.assertEqual((b"Jane", 255), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read an integer from a SQLite database table
  # and place it in a `uint16` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetUInt16(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, desk_number FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.uint16)))
    self.assertEqual((b"John", 9), self.evaluate(get_next()))
    self.assertEqual((b"Jane", 127), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read the minimum and maximum uint16 values from a
  # SQLite database table and place them in `uint16` tensors.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetUInt16MinAndMaxValues(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, account_balance FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.uint16)))
    # Min value of uint16
    self.assertEqual((b"John", 0), self.evaluate(get_next()))
    # Max value of uint16
    self.assertEqual((b"Jane", 65535), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a 0-valued and 1-valued integer from a
  # SQLite database table and place them as `True` and `False` respectively
  # in `bool` tensors.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetBool(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, registration_complete FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.bool)))
    self.assertEqual((b"John", True), self.evaluate(get_next()))
    self.assertEqual((b"Jane", False), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read an integer that is not 0-valued or 1-valued
  # from a SQLite database table and place it as `True` in a `bool` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetBoolNotZeroOrOne(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, favorite_medium_sized_number "
            "FROM students ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.bool)))
    self.assertEqual((b"John", True), self.evaluate(get_next()))
    self.assertEqual((b"Jane", True), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a float from a SQLite database table
  # and place it in a `float64` tensor.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetFloat64(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, victories FROM townspeople "
            "ORDER BY first_name",
            output_types=(dtypes.string, dtypes.string, dtypes.float64)))
    self.assertEqual((b"George", b"Washington", 20.0),
                     self.evaluate(get_next()))
    self.assertEqual((b"John", b"Adams", -19.95), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a float from a SQLite database table beyond
  # the precision of 64-bit IEEE, without throwing an error. Test that
  # `SqlDataset` identifies such a value as equal to itself.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetFloat64OverlyPrecise(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, accolades FROM townspeople "
            "ORDER BY first_name",
            output_types=(dtypes.string, dtypes.string, dtypes.float64)))
    self.assertEqual(
        (b"George", b"Washington",
         1331241.321342132321324589798264627463827647382647382643874),
        self.evaluate(get_next()))
    self.assertEqual(
        (b"John", b"Adams",
         1331241321342132321324589798264627463827647382647382643874.0),
        self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that `SqlDataset` can read a float from a SQLite database table,
  # representing the largest integer representable as a 64-bit IEEE float
  # such that the previous integer is also representable as a 64-bit IEEE float.
  # Test that `SqlDataset` can distinguish these two numbers.
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetFloat64LargestConsecutiveWholeNumbersNotEqual(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, triumphs FROM townspeople "
            "ORDER BY first_name",
            output_types=(dtypes.string, dtypes.string, dtypes.float64)))
    self.assertNotEqual((b"George", b"Washington", 9007199254740992.0),
                        self.evaluate(get_next()))
    self.assertNotEqual((b"John", b"Adams", 9007199254740991.0),
                        self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that SqlDataset can stop correctly when combined with batch
  @combinations.generate(test_base.default_test_combinations())
  def testReadResultSetWithBatchStop(self):
    dataset = self._createSqlDataset(
        query="SELECT * FROM data", output_types=(dtypes.int32))
    dataset = dataset.map(lambda x: array_ops.identity(x))
    get_next = self.getNext(dataset.batch(2))
    self.assertAllEqual(self.evaluate(get_next()), [0, 1])
    self.assertAllEqual(self.evaluate(get_next()), [2])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())


class SqlDatasetCheckpointTest(SqlDatasetTestBase,
                               checkpoint_test_base.CheckpointTestBase,
                               parameterized.TestCase):

  def _build_dataset(self, num_repeats):
    data_source_name = os.path.join(test.get_temp_dir(), "tftest.sqlite")
    driver_name = array_ops.placeholder_with_default(
        array_ops.constant("sqlite", dtypes.string), shape=[])
    query = ("SELECT first_name, last_name, motto FROM students ORDER BY "
             "first_name DESC")
    output_types = (dtypes.string, dtypes.string, dtypes.string)
    return readers.SqlDataset(driver_name, data_source_name, query,
                              output_types).repeat(num_repeats)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    num_repeats = 4
    num_outputs = num_repeats * 2
    verify_fn(self, lambda: self._build_dataset(num_repeats), num_outputs)


if __name__ == "__main__":
  test.main()
