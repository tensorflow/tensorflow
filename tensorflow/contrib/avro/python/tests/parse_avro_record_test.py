# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import numpy as np

from tensorflow.python.framework.errors import OpError
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.platform import test
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util

from tensorflow.contrib.avro.python.utils.avro_serialization import AvroSerializer
from tensorflow.contrib.avro.python.parse_avro_record import parse_avro_record


class ParseAvroRecordTest(test_util.TensorFlowTestCase):
    """
    Tests the parsing of avro records into tensorflow tensors
    """

    def __init__(self, *args, **kwargs):
        super(ParseAvroRecordTest, self).__init__(*args, **kwargs)

    def setUp(self):
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)

    def assert_same_tensor(self, tensor_expected, tensor_actual):

        def assert_same_array(array_expected, array_actual):
            if np.issubdtype(array_actual.dtype, np.number):
                self.assertAllClose(array_expected, array_actual)
            else:
                self.assertAllEqual(array_expected, array_actual)

        # If we have a sparse tensor we need to check indices and values
        if isinstance(tensor_actual, sparse_tensor.SparseTensorValue):
            self.assertAllEqual(tensor_expected.indices, tensor_actual.indices)
            assert_same_array(tensor_expected.values, tensor_actual.values)
        # If we have a dense tensor we can directly use the values (these are numpy arrays)
        else:
            assert_same_array(tensor_expected, tensor_actual)

    @staticmethod
    def serialize_all_records(schema, records_to_serialize):
        serializer = AvroSerializer(schema)
        return [serializer.serialize(record) for record in records_to_serialize]

    def _compare_all_tensors(self, schema, serialized_records, features, tensors_expected):
        with self.test_session() as sess:
            str_input = array_ops.placeholder(tf_types.string)
            parsed = parse_avro_record(str_input, schema, features)
            tensors = sess.run(parsed, feed_dict={str_input: serialized_records})
            for key, value in features.iteritems():
                tensor_expected = tensors_expected[key]
                tensor_actual = tensors[key]
                self.assert_same_tensor(tensor_expected, tensor_actual)

    def _parse_must_fail(self, schema, serialized_records, features):
        with self.test_session() as sess:
            str_input = array_ops.placeholder(tf_types.string)
            parsed = parse_avro_record(str_input, schema, features)
            with self.assertRaises(OpError) as error:
                print(sess.run(parsed, feed_dict={str_input: serialized_records}))
            logging.info(error)

    def test_primitive_types(self):
        logging.info("Running {}".format(self.test_primitive_types.__func__.__name__))

        schema = '''{"doc": "Primitive types.",
                       "namespace": "com.test.primitive",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                         {"name": "float_type", "type": "float"},
                         {"name": "double_type", "type": "double"},
                         {"name": "long_type", "type": "long"},
                         {"name": "int_type", "type": "int"},
                         {"name": "boolean_type", "type": "boolean"},
                         {"name": "string_type", "type": "string"},
                         {"name": "bytes_type", "type": "bytes"}
                       ]}'''

        records_to_serialize = [
            {'float_type': 0.0,
             'double_type': 0.0,
             'long_type': 0L,
             'int_type': 0,
             'boolean_type': False,
             'string_type': "",
             'bytes_type': ""
             },
            {'float_type': 3.40282306074e+38,
             'double_type': 1.7976931348623157e+308,
             'long_type': 9223372036854775807L,
             'int_type': 2147483648-1,
             'boolean_type': True,
             'string_type': "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
             'bytes_type': "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"
             },
            {'float_type': -3.40282306074e+38,
             'double_type': -1.7976931348623157e+308,
             'long_type': -9223372036854775807L-1L,
             'int_type': -2147483648,
             'boolean_type': True,
             'string_type': "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
             'bytes_type': "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789"
             },
            {'float_type': 2342.322,
             'double_type': 2.2250738585072014e-308,
             'long_type': -234829L,
             'int_type': 213648,
             'boolean_type': False,
             'string_type': "alkdfjiwij2oi2jp",
             'bytes_type': "aljk2ijlqn,w"}]

        features = {'float_type': parsing_ops.FixedLenFeature([], tf_types.float32),
                    'double_type': parsing_ops.FixedLenFeature([], tf_types.float64),
                    'long_type': parsing_ops.FixedLenFeature([], tf_types.int64),
                    'int_type': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'boolean_type': parsing_ops.FixedLenFeature([], tf_types.bool),
                    'string_type': parsing_ops.FixedLenFeature([], tf_types.string),
                    'bytes_type': parsing_ops.FixedLenFeature([], tf_types.string)}

        tensors_expected = {
            'float_type': np.asarray([0.0, 3.40282306074e+38, -3.40282306074e+38, 2342.322]),
            'double_type': np.asarray([0.0, 1.7976931348623157e+308, -1.7976931348623157e+308, 2.2250738585072014e-308]),
            'long_type': np.asarray([0L, 9223372036854775807L, -9223372036854775807L-1L, -234829L]),
            'int_type': np.asarray([0, 2147483648-1, -2147483648, 213648]),
            'boolean_type': np.asarray([False, True, True, False]),
            'string_type': np.asarray(["",
                                       "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
                                       "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
                                       "alkdfjiwij2oi2jp"]),
            'bytes_type': np.asarray(["",
                                      "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
                                      "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
                                      "aljk2ijlqn,w"])
        }

        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_fixed_length_lists(self):
        schema = '''{"doc": "Fixed length lists.",
                   "namespace": "com.test.lists.fixed",
                   "type": "record",
                   "name": "data_row",
                   "fields": [
                     {"name": "float_list_type", "type": {"type": "array", "items": "float"}},
                     {"name": "double_list_type", "type": {"type": "array", "items": "double"}},
                     {"name": "long_list_type", "type": {"type": "array", "items": "long"}},
                     {"name": "int_list_type", "type": {"type": "array", "items": "int"}},
                     {"name": "boolean_list_type", "type": {"type": "array", "items": "boolean"}},
                     {"name": "string_list_type", "type": {"type": "array", "items": "string"}},
                     {"name": "bytes_list_type", "type": {"type": "array", "items": "bytes"}}
                   ]}'''
        records_to_serialize = [
            {'float_list_type': [-1.0001, 0.1, 23.2],
             'double_list_type': [-20.0, 22.33],
             'long_list_type': [-15L, 0L, 3022123019L],
             'int_list_type': [-20, -1, 2934],
             'boolean_list_type': [True],
             'string_list_type': ["abc", "defg", "hijkl"],
             'bytes_list_type': ["abc", "defg", "hijkl"]},
            {'float_list_type': [-3.22, 3298.233, 3939.1213],
             'double_list_type': [-2332.324, 2.665439],
             'long_list_type': [-1543L, 233L, 322L],
             'int_list_type': [-5, 342, -3222],
             'boolean_list_type': [False],
             'string_list_type': ["mnop", "qrs", "tuvwz"],
             'bytes_list_type': ["mnop", "qrs", "tuvwz"]
             }]

        features = {'float_list_type': parsing_ops.FixedLenFeature([3], tf_types.float32),
                    'double_list_type': parsing_ops.FixedLenFeature([2], tf_types.float64),
                    'long_list_type': parsing_ops.FixedLenFeature([3], tf_types.int64),
                    'int_list_type': parsing_ops.FixedLenFeature([3], tf_types.int32),
                    'boolean_list_type': parsing_ops.FixedLenFeature([1], tf_types.bool),
                    'string_list_type': parsing_ops.FixedLenFeature([3], tf_types.string),
                    'bytes_list_type': parsing_ops.FixedLenFeature([3], tf_types.string)}

        tensors_expected = {
            'float_list_type': np.asarray([[-1.0001, 0.1, 23.2], [-3.22, 3298.233, 3939.1213]]),
            'double_list_type': np.asarray([[-20.0, 22.33], [-2332.324, 2.665439]]),
            'long_list_type': np.asarray([[-15L, 0L, 3022123019L], [-1543L, 233L, 322L]]),
            'int_list_type': np.asarray([[-20, -1, 2934], [-5, 342, -3222]]),
            'boolean_list_type': np.asarray([[True], [False]]),
            'string_list_type': np.asarray([["abc", "defg", "hijkl"], ["mnop", "qrs", "tuvwz"]]),
            'bytes_list_type': np.asarray([["abc", "defg", "hijkl"], ["mnop", "qrs", "tuvwz"]])
        }

        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_variable_length_lists(self):
        schema = '''{"doc": "Variable length lists.",
                     "namespace": "com.test.lists.var",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "float_list_type", "type": {"type": "array", "items": "float"}},
                       {"name": "double_list_type", "type": {"type": "array", "items": "double"}},
                       {"name": "long_list_type", "type": {"type": "array", "items": "long"}},
                       {"name": "int_list_type", "type": {"type": "array", "items": "int"}},
                       {"name": "boolean_list_type", "type": {"type": "array", "items": "boolean"}},
                       {"name": "string_list_type", "type": {"type": "array", "items": "string"}},
                       {"name": "bytes_list_type", "type": {"type": "array", "items": "bytes"}}
                    ]}'''

        records_to_serialize = [
            {'float_list_type': [-1.0001, 0.1],
             'double_list_type': [-20.0, 22.33, 234.32334],
             'long_list_type': [-15L, 0L, 3022123019L],
             'int_list_type': [-20, -1],
             'boolean_list_type': [True, False],
             'string_list_type': ["abc", "defg"],
             'bytes_list_type': ["abc", "defgsd"]},
            {'float_list_type': [-3.22, 3298.233, 3939.1213],
             'double_list_type': [-2332.324, 2.665439],
             'long_list_type': [-1543L, 233L, 322L],
             'int_list_type': [-5, 342, -3222],
             'boolean_list_type': [False],
             'string_list_type': ["mnop", "qrs", "tuvwz"],
             'bytes_list_type': ["mnop", "qrs", "tuvwz"]
             }]

        features = {'float_list_type': parsing_ops.VarLenFeature(tf_types.float32),
                    'double_list_type': parsing_ops.VarLenFeature(tf_types.float64),
                    'long_list_type': parsing_ops.VarLenFeature(tf_types.int64),
                    'int_list_type': parsing_ops.VarLenFeature(tf_types.int32),
                    'boolean_list_type': parsing_ops.VarLenFeature(tf_types.bool),
                    'string_list_type': parsing_ops.VarLenFeature(tf_types.string),
                    'bytes_list_type': parsing_ops.VarLenFeature(tf_types.string)}

        tensors_expected = {
            'float_list_type': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]),
                np.asarray([-1.0001, 0.1, -3.22, 3298.233, 3939.1213]),
                np.asarray([2, 3])),
            'double_list_type': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
                np.asarray([-20.0, 22.33, 234.32334, -2332.324, 2.665439]),
                np.asarray([2, 3])),
            'long_list_type': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]),
                np.asarray([-15L, 0L, 3022123019L, -1543L, 233L, 322L]),
                np.asarray([2, 3])),
            'int_list_type': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]),
                np.asarray([-20, -1, -5, 342, -3222]),
                np.asarray([2, 3])),
            'boolean_list_type': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [1, 0]]),
                np.asarray([True, False, False]),
                np.asarray([2, 2])),
            'string_list_type': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]),
                np.asarray(["abc", "defg", "mnop", "qrs", "tuvwz"]),
                np.asarray([2, 3])),
            'bytes_list_type': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]),
                np.asarray(["abc", "defgsd", "mnop", "qrs", "tuvwz"]),
                np.asarray([2, 3]))
        }

        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_sparse_features(self):
        schema = '''{"doc": "Sparse features.",
                     "namespace": "com.test.sparse",
                     "type": "record",
                     "name": "sparse_feature",
                     "fields": [
                       {"name": "sparse_type",
                         "type":
                         {
                            "type": "array",
                            "items": {
                                "type": "record",
                                "name": "sparse_triplet",
                                "fields": [
                                    {"name": "index", "type": "long"},
                                    {"name": "max_index", "type": "int"},
                                    {"name": "value", "type": "float"}
                                ]
                            }
                         }
                     }]}'''

        records_to_serialize = [
            {'sparse_type': [{'index': 0, 'max_index': 10, 'value': 5.0},
                             {'index': 5, 'max_index': 10, 'value': 7.0},
                             {'index': 3, 'max_index': 10, 'value': 1.0}]},
            {'sparse_type': [{'index': 0, 'max_index': 10, 'value': 2.0},
                             {'index': 9, 'max_index': 10, 'value': 1.0}]}]

        features = {'sparse_type': parsing_ops.SparseFeature(
                        index_key='index', value_key='value', dtype=tf_types.float32, size=10)}

        tensors_expected = {
            'sparse_type': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 3], [0, 5], [1, 0], [1, 9]]),
                np.asarray([5.0, 1.0, 7.0, 2.0, 1.0]),
                np.asarray([2, 10]))
        }

        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_nesting(self):
        schema = '''{"doc": "Nested records, arrays, lists, and link resolution.",
                     "namespace": "com.test.nested",
                     "type": "record",
                     "name": "test_nested_record",
                     "fields": [
                         {"name": "nested_record",
                          "type": {
                               "type": "record",
                               "name": "nested_values",
                               "fields": [
                                   {"name": "nested_int", "type": "int"},
                                   {"name": "nested_float_list", "type": {"type": "array", "items": "float"}}
                                   ]}
                         },
                         {"name": "list_of_records",
                          "type": {
                               "type": "array",
                               "items": {
                                   "type": "record",
                                   "name": "person",
                                   "fields": [
                                       {"name": "first_name", "type": "string"},
                                       {"name": "age", "type": "int"}
                                   ]
                               }
                           }
                         },
                         {"name": "map_of_records",
                          "type": {
                               "type": "map",
                               "values": {
                                   "type": "record",
                                   "name": "secondPerson",
                                   "fields": [
                                       {"name": "first_name", "type": "string"},
                                       {"name": "age", "type": "int"}
                                   ]
                               }
                           }
                         }]}'''

        records_to_serialize = [
            {'nested_record': {'nested_int': 0, 'nested_float_list': [0.0, 10.0]},
             'list_of_records': [{'first_name': "Herbert", 'age': 70}],
             'map_of_records': {'first': {'first_name': "Herbert", 'age': 70},
                                'second': {'first_name': "Julia", 'age': 30}}},
            {'nested_record': {'nested_int': 5, 'nested_float_list': [-2.0, 7.0]},
             'list_of_records': [{'first_name': "Doug", 'age': 55},
                                 {'first_name': "Jess", 'age': 66},
                                 {'first_name': "Julia", 'age': 30}],
             'map_of_records': {'first': {'first_name': "Doug", 'age': 55},
                                'second': {'first_name': "Jess", 'age': 66}}},
            {'nested_record': {'nested_int': 7, 'nested_float_list': [3.0, 4.0]},
             'list_of_records': [{'first_name': "Karl", 'age': 32}],
             'map_of_records': {'first': {'first_name': "Karl", 'age': 32},
                                'second': {'first_name': "Joan", 'age': 21}}}]

        features = {'nested_record/nested_int': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'nested_record/nested_float_list': parsing_ops.FixedLenFeature([2], tf_types.float32),
                    'list_of_records/[0]/first_name': parsing_ops.FixedLenFeature([1], tf_types.string),
                    "map_of_records/['second']/age": parsing_ops.FixedLenFeature([1], tf_types.int32)}

        tensors_expected = {
            'nested_record/nested_int': np.asarray([0, 5, 7]),
            'nested_record/nested_float_list': np.asarray([[0.0, 10.0], [-2.0, 7.0], [3.0, 4.0]]),
            'list_of_records/[0]/first_name': np.asarray([["Herbert"], ["Doug"], ["Karl"]]),
            "map_of_records/['second']/age": np.asarray([[30], [66], [21]])
        }

        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_nested_with_asterisk(self):
        schema = '''{"doc": "Nested records in array to use asterisk.",
                     "namespace": "com.test.nested.records",
                     "type": "record",
                     "name": "test_nested_record",
                     "fields": [
                        {"name": "sparse_type",
                            "type":
                            {
                                "type": "array",
                                "items": {
                                    "type": "record",
                                    "name": "sparse_triplet",
                                    "fields": [
                                        {"name": "index", "type": "long"},
                                        {"name": "max_index", "type": "int"},
                                        {"name": "value", "type": "float"}
                                    ]
                                }
                            }
                    }]}'''

        records_to_serialize = [
            {'sparse_type': [{'index': 0, 'max_index': 10, 'value': 5.0},
                             {'index': 5, 'max_index': 10, 'value': 7.0},
                             {'index': 3, 'max_index': 10, 'value': 1.0}]},
            {'sparse_type': [{'index': 0, 'max_index': 10, 'value': 2.0},
                             {'index': 9, 'max_index': 10, 'value': 1.0}]}]

        features = {'sparse_type/[*]/index': parsing_ops.VarLenFeature(tf_types.int64),
                    'sparse_type/[*]/value': parsing_ops.VarLenFeature(tf_types.float32)}

        tensors_expected = {
            'sparse_type/[*]/index': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
                np.asarray([0, 5, 3, 0, 9]),
                np.asarray([2, 3])),
            'sparse_type/[*]/value': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
                np.asarray([5.0, 7.0, 1.0, 2.0, 1.0]),
                np.asarray([2, 3]))
        }

        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_parse_int_as_long_fail(self):
        schema = '''{"doc": "Parse int as long (int64) fails.",
                     "namespace": "com.test.int.type.failure",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "index", "type": "int"}
                    ]}'''
        records_to_serialize = [{'index': 0}]
        features = {'index': parsing_ops.FixedLenFeature([], tf_types.int64)}
        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)

    def test_parse_int_as_sparse_type_fail(self):
        schema = '''{"doc": "Parse int as SparseType fails.",
                     "namespace": "com.test.sparse.type.failure",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "index", "type": "int"}
                     ]}'''
        records_to_serialize = [{'index': 0}]
        features = {'index': parsing_ops.SparseFeature(
            index_key='index', value_key='value', dtype=tf_types.float32, size=10)}
        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)

    def test_parse_float_as_double_fail(self):
        schema = '''{"doc": "Parse float as double fails.",
                     "namespace": "com.test.float.type.failure",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "weight", "type": "float"}
                     ]}'''

        records_to_serialize = [{'weight': 0.5}]
        features = {'weight': parsing_ops.FixedLenFeature([], tf_types.float64)}
        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)

    def test_fixed_length_without_proper_default_fail(self):
        schema = '''{"doc": "Used fixed length without proper default.",
                     "namespace": "com.test.wrong.list.type",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "int_list_type", "type": {"type": "array", "items": "int"}}
                     ]}'''

        records_to_serialize = [{'int_list_type': [0, 1, 2]}, {'int_list_type': [0, 1]}]

        features = {'int_list_type': parsing_ops.FixedLenFeature([], tf_types.int32)}

        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)

    def test_fixed_length_with_default(self):
        schema = '''{"doc": "Fixed length lists with defaults.",
                     "namespace": "com.test.lists.fixed",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "float_list_type", "type": {"type": "array", "items": "float"}},
                       {"name": "double_list_type", "type": {"type": "array", "items": "double"}},
                       {"name": "long_list_type", "type": {"type": "array", "items": "long"}},
                       {"name": "int_list_type", "type": {"type": "array", "items": "int"}},
                       {"name": "boolean_list_type", "type": {"type": "array", "items": "boolean"}},
                       {"name": "string_list_type", "type": {"type": "array", "items": "string"}},
                       {"name": "bytes_list_type", "type": {"type": "array", "items": "bytes"}}
                     ]}'''
        records_to_serialize = [{
            'float_list_type': [-1.0001, 0.1, 23.2],
            'double_list_type': [-20.0, 22.33],
            'long_list_type': [-15L, 0L, 3022123019L],
            'int_list_type': [-20, -1, 2934],
            'boolean_list_type': [True],
            'string_list_type': ["abc", "defg", "hijkl"],
            'bytes_list_type': ["abc", "defg", "hijkl"]},
           {'float_list_type': [-3.22, 3298.233],
            'double_list_type': [-2332.324],
            'long_list_type': [-1543L, 233L],
            'int_list_type': [-5, 342],
            'boolean_list_type': [],
            'string_list_type': ["mnop", "qrs"],
            'bytes_list_type': ["mnop"]
            }]

        features = {'float_list_type': parsing_ops.FixedLenFeature(
                        [3], tf_types.float32, default_value=[0.0, 0.0, 1.0]),
                    'double_list_type': parsing_ops.FixedLenFeature(
                        [2], tf_types.float64, default_value=[0.0, 0.0]),
                    'long_list_type': parsing_ops.FixedLenFeature(
                        [3], tf_types.int64, default_value=[1L, 1L, 1L]),
                    'int_list_type': parsing_ops.FixedLenFeature(
                        [3], tf_types.int32, default_value=[0, 1, 2]),
                    'boolean_list_type': parsing_ops.FixedLenFeature(
                        [], tf_types.bool, default_value=[False]),
                    'string_list_type': parsing_ops.FixedLenFeature(
                        [3], tf_types.string, default_value=['a', 'b', 'c']),
                    'bytes_list_type': parsing_ops.FixedLenFeature(
                        [3], tf_types.string, default_value=['a', 'b', 'c'])}

        tensors_expected = {
            'float_list_type': np.asarray([[-1.0001, 0.1, 23.2], [-3.22, 3298.233, 1.0]]),  # fill in 1.0 from defaults
            'double_list_type': np.asarray([[-20.0, 22.33], [-2332.324, 0.0]]),  # fill in 0.0 from defaults
            'long_list_type': np.asarray([[-15L, 0L, 3022123019L], [-1543L, 233L, 1L]]),  # fill in 1L from defaults
            'int_list_type': np.asarray([[-20, -1, 2934], [-5, 342, 2]]),  # fill in 2 from defaults
            'boolean_list_type': np.asarray([True, False]),  # fill in False from defaults
            'string_list_type': np.asarray([["abc", "defg", "hijkl"], ["mnop", "qrs", "c"]]),  # fill in 'c'
            'bytes_list_type': np.asarray([["abc", "defg", "hijkl"], ["mnop", "b", "c"]])
        }

        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_fixed_length_sequence_with_default(self):
        schema = '''{"doc": "Fixed length lists with defaults.",
                     "namespace": "com.test.lists.fixed",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "float_list_type", "type": {"type": "array", "items": "float"}},
                       {"name": "double_list_type", "type": {"type": "array", "items": "double"}},
                       {"name": "long_list_type", "type": {"type": "array", "items": "long"}},
                       {"name": "int_list_type", "type": {"type": "array", "items": "int"}},
                       {"name": "boolean_list_type", "type": {"type": "array", "items": "boolean"}},
                       {"name": "string_list_type", "type": {"type": "array", "items": "string"}},
                       {"name": "bytes_list_type", "type": {"type": "array", "items": "bytes"}},
                       {"name": "first_short", "type": {"type": "array", "items": "int"}}
                     ]}'''
        records_to_serialize = [
            {'float_list_type': [-1.0001, 0.1, 23.2],
             'double_list_type': [-20.0, 22.33],
             'long_list_type': [-15L, 0L, 3022123019L],
             'int_list_type': [-20, -1, 2934],
             'boolean_list_type': [True],
             'string_list_type': ["abc", "defg", "hijkl"],
             'bytes_list_type': ["abc", "defg", "hijkl"],
             'first_short': [1, 2]},
            {'float_list_type': [-3.22, 3298.233],
             'double_list_type': [-2332.324],
             'long_list_type': [-1543L, 233L],
             'int_list_type': [-5, 342],
             'boolean_list_type': [],
             'string_list_type': ["mnop", "qrs"],
             'bytes_list_type': ["mnop"],
             'first_short': [1, 2, 3]
             }]
        features = {
            'float_list_type':
                parsing_ops.FixedLenSequenceFeature([], tf_types.float32, default_value=0.5, allow_missing=True),
            'double_list_type':
                parsing_ops.FixedLenSequenceFeature([], tf_types.float64, default_value=1.0, allow_missing=True),
            'long_list_type':
                parsing_ops.FixedLenSequenceFeature([], tf_types.int64, default_value=5L, allow_missing=True),
            'int_list_type':
                parsing_ops.FixedLenSequenceFeature([], tf_types.int32, default_value=2, allow_missing=True),
            'boolean_list_type':
                parsing_ops.FixedLenSequenceFeature([], tf_types.bool, default_value=False, allow_missing=True),
            'string_list_type':
                parsing_ops.FixedLenSequenceFeature([], tf_types.string, default_value="default", allow_missing=True),
            'bytes_list_type':
                parsing_ops.FixedLenSequenceFeature([], tf_types.string, default_value="default", allow_missing=True),
            'first_short':
                parsing_ops.FixedLenSequenceFeature([], tf_types.int32, default_value=3, allow_missing=True)}

        tensors_expected = {
            'float_list_type': np.asarray([[-1.0001, 0.1, 23.2], [-3.22, 3298.233, 0.5]]),  # fill 0.5 as default
            'double_list_type': np.asarray([[-20.0, 22.33], [-2332.324, 1.0]]),  # fill 1.0 as default
            'long_list_type': np.asarray([[-15L, 0L, 3022123019L], [-1543L, 233L, 5L]]),  # fill 5L as default
            'int_list_type': np.asarray([[-20, -1, 2934], [-5, 342, 2]]),  # fill in 2 as default
            'boolean_list_type': np.asarray([[True], [False]]),  # fill in False as default
            'string_list_type': np.asarray([["abc", "defg", "hijkl"], ["mnop", "qrs", "default"]]),  # add default
            'bytes_list_type': np.asarray([["abc", "defg", "hijkl"], ["mnop", "default", "default"]]),  # add default
            'first_short': np.asarray([[1, 2, 3], [1, 2, 3]])  # add 3 as default
        }

        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_fixed_length_for_array(self):
        schema = '''{"doc": "Use fixed length for array features.",
                     "namespace": "com.test.fixed.length.for.array",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "names", "type": {"type": "array", "items": "string"}}
                     ]}'''
        records_to_serialize = [
            {'names': ["Hans", "Herbert", "Heinz"]},
            {'names': ["Gilbert", "Gerald", "Genie"]}
        ]
        features = {'names': parsing_ops.FixedLenFeature([3], tf_types.string)}
        tensors_expected = {
            'names': np.asarray([["Hans", "Herbert", "Heinz"], ["Gilbert", "Gerald", "Genie"]])
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_wrong_spelling_of_feature_name_fail(self):
        schema = '''{"doc": "Wrong spelling of feature name.",
                     "namespace": "com.test.wrong.feature.name",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "int_type", "type": "int"}
                     ]}'''
        records_to_serialize = [{'int_type': 0}]
        features = {'wrong_spelling': parsing_ops.FixedLenFeature([], tf_types.int32)}
        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)

    def test_wrong_index(self):
        schema = '''{"doc": "Wrong spelling of feature name and wrong index.",
                     "namespace": "com.test.wrong.feature",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "list_of_records",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "first_name", "type": "string"}
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [{'list_of_records': [{'first_name': "My name"}]}]
        features = {'list_of_records/[2]/name': parsing_ops.FixedLenFeature([], tf_types.string)}
        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)

    def test_filter_with_variable_length(self):
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.filtering",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "guests",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "name", "type": "string"},
                                      {"name": "gender", "type": "string"}
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [
            {'guests': [{'name': "Hans", 'gender': "male"},
                        {'name': "Mary", 'gender': "female"},
                        {'name': "July", 'gender': "female"}]},
            {'guests': [{'name': "Joel", 'gender': "male"},
                        {'name': "JoAn", 'gender': "female"},
                        {'name': "Marc", 'gender': "male"}]}]
        features = {'guests/[gender=male]/name': parsing_ops.VarLenFeature(tf_types.string),
                    'guests/[gender=female]/name': parsing_ops.VarLenFeature(tf_types.string)}
        tensors_expected = {
            'guests/[gender=male]/name': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [1, 0], [1, 1]]),
                np.asarray(["Hans", "Joel", "Marc"]),
                np.asarray([2, 1])),
            'guests/[gender=female]/name': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [1, 0]]),
                np.asarray(["Mary", "July", "JoAn"]),
                np.asarray([2, 1]))
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_filter_with_fixed_length(self):
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.filtering",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "guests",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "name", "type": "string"},
                                      {"name": "gender", "type": "string"}
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [
            {'guests': [{'name': "Hans", 'gender': "male"},
                        {'name': "Mary", 'gender': "female"},
                        {'name': "July", 'gender': "female"}]},
            {'guests': [{'name': "Joel", 'gender': "male"},
                        {'name': "JoAn", 'gender': "female"},
                        {'name': "Kloy", 'gender': "female"}]}]
        features = {'guests/[gender=male]/name': parsing_ops.FixedLenFeature([1], tf_types.string),
                    'guests/[gender=female]/name': parsing_ops.FixedLenFeature([2], tf_types.string)}
        tensors_expected = {
            'guests/[gender=male]/name': np.asarray([["Hans"], ["Joel"]]),
            'guests/[gender=female]/name': np.asarray([["Mary", "July"], ["JoAn", "Kloy"]])
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_filter_with_empty_result(self):
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.filtering",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "guests",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "name", "type": "string"},
                                      {"name": "gender", "type": "string"}
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [
            {'guests': [{'name': "Hans", 'gender': "male"}]},
            {'guests': [{'name': "Joel", 'gender': "male"}]}]
        features = {'guests/[gender=wrong_value]/name': parsing_ops.VarLenFeature(tf_types.string)}
        tensors_expected = {
            'guests/[gender=wrong_value]/name': sparse_tensor.SparseTensorValue(
                np.empty(shape=[0, 2], dtype=np.int64),
                np.empty(shape=[0], dtype=np.str),
                np.asarray([2, 0]))
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_filter_with_wrong_key_fail(self):
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.filtering",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "guests",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "name", "type": "string"}
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [
            {'guests': [{'name': "Hans"},
                        {'name': "Mary"},
                        {'name': "July"}]}]
        features = {'guests/[wrong_key=female]/name': parsing_ops.VarLenFeature(tf_types.string)}
        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)

    def test_filter_with_wrong_pair_fail(self):
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.filtering",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "guests",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "name", "type": "string"}
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [
            {'guests': [{'name': "Hans"},
                        {'name': "Mary"},
                        {'name': "July"}]}]
        features = {'guests/[forgot_the_separator]/name': parsing_ops.VarLenFeature(tf_types.string)}
        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)

    def test_filter_with_too_many_separators_fail(self):
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.filtering",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "guests",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "name", "type": "string"}
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [
            {'guests': [{'name': "Hans"}, {'name': "Mary"}, {'name': "July"}]}]
        features = {'guests/[used=too=many=separators]/name': parsing_ops.VarLenFeature(tf_types.string)}
        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)

    def test_filter_for_nested_record(self):
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.filtering",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "guests",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "name", "type": "string"},
                                      {"name": "gender", "type": "string"},
                                      {"name": "address",
                                          "type": {
                                              "type": "record",
                                              "name": "postal",
                                              "fields": [
                                                  {"name": "street", "type": "string"},
                                                  {"name": "zip", "type": "int"},
                                                  {"name": "state", "type": "string"}
                                              ]
                                          }
                                      }
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [
            {'guests': [{'name': "Hans", 'gender': "male", 'address': {'street': "California St",
                                                                       'zip': 94040,
                                                                       'state': "CA"}},
                        {'name': "Mary", 'gender': "female", 'address': {'street': "Ellis St",
                                                                         'zip': 29040,
                                                                         'state': "MA"}}]}]
        features = {'guests/[gender=female]/address/street': parsing_ops.VarLenFeature(tf_types.string)}
        tensors_expected = {
            'guests/[gender=female]/address/street': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0]]),
                np.asarray(["Ellis St"]),
                np.asarray([2, 1]))
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_filter_with_bytes_as_type(self):
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.filtering",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "guests",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "name", "type": "bytes"},
                                      {"name": "gender", "type": "bytes"}
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [
            {'guests': [{'name': "Hans", 'gender': "male"},
                        {'name': "Mary", 'gender': "female"},
                        {'name': "July", 'gender': "female"}]},
            {'guests': [{'name': "Joel", 'gender': "male"},
                        {'name': "JoAn", 'gender': "female"},
                        {'name': "Marc", 'gender': "male"}]}]
        features = {'guests/[gender=male]/name': parsing_ops.VarLenFeature(tf_types.string),
                    'guests/[gender=female]/name': parsing_ops.VarLenFeature(tf_types.string)}
        tensors_expected = {
            'guests/[gender=male]/name': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [1, 0], [1, 1]]),
                np.asarray(["Hans", "Joel", "Marc"]),
                np.asarray([2, 1])),
            'guests/[gender=female]/name': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [1, 0]]),
                np.asarray(["Mary", "July", "JoAn"]),
                np.asarray([2, 1]))
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_filter_of_sparse_feature(self):
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.filtering",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "guests",
                         "type": {
                              "type": "array",
                              "items": {
                                  "type": "record",
                                  "name": "person",
                                  "fields": [
                                      {"name": "name", "type": "string"},
                                      {"name": "gender", "type": "string"},
                                      {"name": "address",
                                          "type": {
                                              "type": "array",
                                              "items": {
                                                  "type": "record",
                                                  "name": "postal",
                                                  "fields": [
                                                      {"name": "street", "type": "string"},
                                                      {"name": "zip", "type": "long"},
                                                      {"name": "street_no", "type": "int"}
                                                  ]
                                              }
                                          }
                                      }
                                  ]
                              }
                          }
                        }
                     ]}'''
        records_to_serialize = [
            {'guests': [{'name': "Hans", 'gender': "male",
                         'address': [{'street': "California St", 'zip': 94040, 'state': "CA", 'street_no': 1},
                                     {'street': "New York St", 'zip': 32012, 'state': "NY", 'street_no': 2}]},
                        {'name': "Mary", 'gender': "female",
                         'address': [{'street': "Ellis St", 'zip': 29040, 'state': "MA", 'street_no': 3}]}]}]
        features = {'guests/[gender=female]/address': parsing_ops.SparseFeature(index_key="zip",
                                                                                value_key="street_no",
                                                                                dtype=tf_types.int32,
                                                                                size=94040)}
        tensors_expected = {
            'guests/[gender=female]/address': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 29040]]),
                np.asarray([3]),
                np.asarray([1, 94040]))
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_null_union_primitive_type(self):
        """
        The union of null and bytes is missing because of an ambiguity between the 0 termination symbol and the
        representation of null in the c implementation for avro.
        """
        schema = '''{"doc": "Primitive types union with null.",
                     "namespace": "com.test.null.union.primitive",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "float_type", "type": [ "null", "float"]},
                       {"name": "double_type", "type": [ "null", "double"]},
                       {"name": "long_type", "type": [ "null", "long"]},
                       {"name": "int_type", "type": [ "null", "int"]},
                       {"name": "boolean_type", "type": [ "null", "boolean"]},
                       {"name": "string_type", "type": [ "null", "string"]}
                     ]}'''
        records_to_serialize = [
            {'float_type': 3.40282306074e+38,
             'double_type': 1.7976931348623157e+308,
             'long_type': 9223372036854775807L,
             'int_type': 2147483648-1,
             'boolean_type': True,
             'string_type': "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"}]
        features = {'float_type': parsing_ops.FixedLenFeature([], tf_types.float32),
                    'double_type': parsing_ops.FixedLenFeature([], tf_types.float64),
                    'long_type': parsing_ops.FixedLenFeature([], tf_types.int64),
                    'int_type': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'boolean_type': parsing_ops.FixedLenFeature([], tf_types.bool),
                    'string_type': parsing_ops.FixedLenFeature([], tf_types.string)}
        tensors_expected = {
            'float_type': np.asarray([3.40282306074e+38]),
            'double_type': np.asarray([1.7976931348623157e+308]),
            'long_type': np.asarray([9223372036854775807L]),
            'int_type': np.asarray([2147483648-1]),
            'boolean_type': np.asarray([True]),
            'string_type': np.asarray(["SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"])
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_null_union_non_primitive_types(self):
        schema = '''{"doc": "Unions between null and a non-primitive type.",
                     "namespace": "com.test.null.union.non.primitive",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                         {"name": "array_type", "type": [ "null", {"type": "array", "items": {"type": "float"}}]},
                         {"name": "map_type", "type": [ "null", {"type": "map", "values": {"type": "double"}}]}
                     ]}'''
        records_to_serialize = [
            {'array_type': [1.0, 2.0, 3.0],
             'map_type': {'one': 1.0, 'two': 2.0}}]
        features = {
            "array_type/[0]": parsing_ops.FixedLenFeature([], tf_types.float32),
            "map_type/['one']": parsing_ops.FixedLenFeature([], tf_types.float64)}
        tensors_expected = {
            "array_type/[0]": np.asarray([1.0]),
            "map_type/['one']": np.asarray([1.0])
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_nested_unions_with_nulls(self):
        """
        Covers these unions
        array, null
        record, null
        string, null
        float, null
        """
        schema = '''{"doc": "Test filtering",
                     "namespace": "com.test.nested.union.with.null",
                     "type": "record",
                     "name": "row",
                     "fields" : [ {
                        "name" : "features",
                        "type" : [ "null", {
                          "type" : "array",
                          "items" : [ "null", {
                            "type" : "record",
                            "name" : "nameTermValue",
                            "fields" : [ {
                              "name" : "name",
                              "type" : [ "null", "string" ]
                            }, {
                              "name" : "value",
                              "type" : [ "null", "float" ]
                            } ]
                          } ]
                        } ]
                      } ]
                     }'''
        records_to_serialize = [
            {'features': [{'name': "First", 'value': 1.0},
                          {'name': "Second", 'value': 2.0},
                          {'name': "Third", 'value': 3.0}]},
            {'features': [{'name': "First", 'value': 1.0},
                          {'name': "Second", 'value': 2.0},
                          {'name': "Third", 'value': 3.0}]}]
        features = {'features/[name=First]/value': parsing_ops.FixedLenFeature([], tf_types.float32, default_value=0),
                    'features/[name=Second]/value': parsing_ops.FixedLenFeature([], tf_types.float32, default_value=0),
                    'features/[name=Third]/value': parsing_ops.FixedLenFeature([], tf_types.float32, default_value=0)}
        tensors_expected = {
            'features/[name=First]/value': np.asarray([1.0, 1.0]),
            'features/[name=Second]/value': np.asarray([2.0, 2.0]),
            'features/[name=Third]/value': np.asarray([3.0, 3.0])
        }
        self._compare_all_tensors(schema,
                                  ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                  features,
                                  tensors_expected)

    def test_primitive_types_and_union_with_null_fail(self):
        """
        This test case will fail because we do not auto-convert to null's
        """
        schema = '''{"doc": "Primitive types union with null.",
                     "namespace": "com.test.primitive.and.null",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "float_type", "type": [ "null", "float"]},
                       {"name": "double_type", "type": [ "null", "double"]},
                       {"name": "long_type", "type": [ "null", "long"]},
                       {"name": "int_type", "type": [ "null", "int"]},
                       {"name": "boolean_type", "type": [ "null", "boolean"]},
                       {"name": "string_type", "type": [ "null", "string"]}
                     ]}'''
        records_to_serialize = [
            {'float_type': None,
             'double_type': None,
             'long_type': None,
             'int_type': None,
             'boolean_type': None,
             'string_type': None}]
        features = {'float_type': parsing_ops.FixedLenFeature([], tf_types.float32),
                    'double_type': parsing_ops.FixedLenFeature([], tf_types.float64),
                    'long_type': parsing_ops.FixedLenFeature([], tf_types.int64),
                    'int_type': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'boolean_type': parsing_ops.FixedLenFeature([], tf_types.bool),
                    'string_type': parsing_ops.FixedLenFeature([], tf_types.string)}
        self._parse_must_fail(schema,
                              ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                              features)


if __name__ == "__main__":
    test.main()
