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

from tensorflow.python.data.experimental.kernel_tests import sql_dataset_test_base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class SqlDatasetTest(sql_dataset_test_base.SqlDatasetTestBase):

  # Test that SqlDataset can read from a database table.
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
  def testReadEmptyResultSet(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, motto FROM students "
            "WHERE first_name = 'Nonexistent'",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # Test that an error is raised when `driver_name` is invalid.
  def testReadResultSetWithInvalidDriverName(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = self._createSqlDataset(
          driver_name="sqlfake",
          query="SELECT first_name, last_name, motto FROM students "
          "ORDER BY first_name DESC",
          output_types=(dtypes.string, dtypes.string, dtypes.string))
      self.assertDatasetProduces(dataset, expected_output=[])

  # Test that an error is raised when a column name in `query` is nonexistent
  def testReadResultSetWithInvalidColumnName(self):
    get_next = self.getNext(
        self._createSqlDataset(
            query="SELECT first_name, last_name, fake_column FROM students "
            "ORDER BY first_name DESC",
            output_types=(dtypes.string, dtypes.string, dtypes.string)))
    with self.assertRaises(errors.UnknownError):
      self.evaluate(get_next())

  # Test that an error is raised when there is a syntax error in `query`.
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


if __name__ == "__main__":
  test.main()
