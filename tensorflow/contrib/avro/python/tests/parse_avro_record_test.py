from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import numpy as np

from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.platform import test
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.contrib.avro.python.utils.avro_serialization import AvroSerializer

from tensorflow.contrib.avro.python.parse_avro_record import parse_avro_record


class ParseAvroRecordTest(test_util.TensorFlowTestCase):

    def __init__(self, *args, **kwargs):
        super(ParseAvroRecordTest, self).__init__(*args, **kwargs)

    def setUp(self):
        """
        Setup fixture for test cases.
        """
        log_root = logging.getLogger()  # set logging level
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

    def compare_all_tensors(self, schema, serialized_records, features, tensors_expected):
        with self.test_session() as sess:
            str_input = array_ops.placeholder(tf_types.string)
            parsed = parse_avro_record(str_input, schema, features)
            tensors = sess.run(parsed, feed_dict={str_input: serialized_records})
            for key, value in features.iteritems():
                tensor_expected = tensors_expected[key]
                tensor_actual = tensors[key]
                self.assert_same_tensor(tensor_expected, tensor_actual)

    def test_primitive_types(self):
        logging.info("Running {}".format(self.test_primitive_types.__func__.__name__))

        schema = '''{"doc": "Primitive types.",
                       "namespace": "com.test.primitive",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                         {"name": "index", "type": "int"},
                         {"name": "float_type", "type": "float"},
                         {"name": "double_type", "type": "double"},
                         {"name": "long_type", "type": "long"},
                         {"name": "int_type", "type": "int"},
                         {"name": "boolean_type", "type": "boolean"},
                         {"name": "string_type", "type": "string"},
                         {"name": "bytes_type", "type": "bytes"}
                       ]}'''

        records_to_serialize = [
            {'index': 0,
             'float_type': 0.0,
             'double_type': 0.0,
             'long_type': 0L,
             'int_type': 0,
             'boolean_type': False,
             'string_type': "",
             'bytes_type': ""
             },
            {'index': 1,
             'float_type': 3.40282306074e+38,
             'double_type': 1.7976931348623157e+308,
             'long_type': 9223372036854775807L,
             'int_type': 2147483648-1,
             'boolean_type': True,
             'string_type': "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
             'bytes_type': "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"
             },
            {'index': 2,
             'float_type': -3.40282306074e+38,
             'double_type': -1.7976931348623157e+308,
             'long_type': -9223372036854775807L-1L,
             'int_type': -2147483648,
             'boolean_type': True,
             'string_type': "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
             'bytes_type': "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789"
             },
            {'index': 3,
             'float_type': 2342.322,
             'double_type': 2.2250738585072014e-308,
             'long_type': -234829L,
             'int_type': 213648,
             'boolean_type': False,
             'string_type': "alkdfjiwij2oi2jp",
             'bytes_type': "aljk2ijlqn,w"}]

        features = {'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'float_type': parsing_ops.FixedLenFeature([], tf_types.float32),
                    'double_type': parsing_ops.FixedLenFeature([], tf_types.float64),
                    'long_type': parsing_ops.FixedLenFeature([], tf_types.int64),
                    'int_type': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'boolean_type': parsing_ops.FixedLenFeature([], tf_types.bool),
                    'string_type': parsing_ops.FixedLenFeature([], tf_types.string),
                    'bytes_type': parsing_ops.FixedLenFeature([], tf_types.string)}

        tensors_expected = {
            'index': np.asarray([0, 1, 2, 3]),
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

        self.compare_all_tensors(schema,
                                 ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                 features,
                                 tensors_expected)

    def test_fixed_length_lists(self):
        logging.info("Running {}".format(self.test_primitive_types.__func__.__name__))
        schema = '''{"doc": "Fixed length lists.",
                   "namespace": "com.test.lists.fixed",
                   "type": "record",
                   "name": "data_row",
                   "fields": [
                     {"name": "index", "type": "int"},
                     {"name": "float_list_type", "type": {"type": "array", "items": "float"}},
                     {"name": "double_list_type", "type": {"type": "array", "items": "double"}},
                     {"name": "long_list_type", "type": {"type": "array", "items": "long"}},
                     {"name": "int_list_type", "type": {"type": "array", "items": "int"}},
                     {"name": "boolean_list_type", "type": {"type": "array", "items": "boolean"}},
                     {"name": "string_list_type", "type": {"type": "array", "items": "string"}},
                     {"name": "bytes_list_type", "type": {"type": "array", "items": "bytes"}}
                   ]}'''
        records_to_serialize = [
            {'index': 0,
             'float_list_type': [-1.0001, 0.1, 23.2],
             'double_list_type': [-20.0, 22.33],
             'long_list_type': [-15L, 0L, 3022123019L],
             'int_list_type': [-20, -1, 2934],
             'boolean_list_type': [True],
             'string_list_type': ["abc", "defg", "hijkl"],
             'bytes_list_type': ["abc", "defg", "hijkl"]},
            {'index': 1,
             'float_list_type': [-3.22, 3298.233, 3939.1213],
             'double_list_type': [-2332.324, 2.665439],
             'long_list_type': [-1543L, 233L, 322L],
             'int_list_type': [-5, 342, -3222],
             'boolean_list_type': [False],
             'string_list_type': ["mnop", "qrs", "tuvwz"],
             'bytes_list_type': ["mnop", "qrs", "tuvwz"]
             }]

        features = {'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'float_list_type': parsing_ops.FixedLenFeature([3], tf_types.float32),
                    'double_list_type': parsing_ops.FixedLenFeature([2], tf_types.float64),
                    'long_list_type': parsing_ops.FixedLenFeature([3], tf_types.int64),
                    'int_list_type': parsing_ops.FixedLenFeature([3], tf_types.int32),
                    'boolean_list_type': parsing_ops.FixedLenFeature([1], tf_types.bool),
                    'string_list_type': parsing_ops.FixedLenFeature([3], tf_types.string),
                    'bytes_list_type': parsing_ops.FixedLenFeature([3], tf_types.string)}

        tensors_expected = {
            'index': np.asarray([0, 1]),
            'float_list_type': np.asarray([[-1.0001, 0.1, 23.2], [-3.22, 3298.233, 3939.1213]]),
            'double_list_type': np.asarray([[-20.0, 22.33], [-2332.324, 2.665439]]),
            'long_list_type': np.asarray([[-15L, 0L, 3022123019L], [-1543L, 233L, 322L]]),
            'int_list_type': np.asarray([[-20, -1, 2934], [-5, 342, -3222]]),
            'boolean_list_type': np.asarray([[True], [False]]),
            'string_list_type': np.asarray([["abc", "defg", "hijkl"], ["mnop", "qrs", "tuvwz"]]),
            'bytes_list_type': np.asarray([["abc", "defg", "hijkl"], ["mnop", "qrs", "tuvwz"]])
        }

        self.compare_all_tensors(schema,
                                 ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                 features,
                                 tensors_expected)

    def test_variable_length_lists(self):
        schema = '''{"doc": "Variable length lists.",
                     "namespace": "com.test.lists.var",
                     "type": "record",
                     "name": "data_row",
                     "fields": [
                       {"name": "index", "type": "int"},
                       {"name": "float_list_type", "type": {"type": "array", "items": "float"}},
                       {"name": "double_list_type", "type": {"type": "array", "items": "double"}},
                       {"name": "long_list_type", "type": {"type": "array", "items": "long"}},
                       {"name": "int_list_type", "type": {"type": "array", "items": "int"}},
                       {"name": "boolean_list_type", "type": {"type": "array", "items": "boolean"}},
                       {"name": "string_list_type", "type": {"type": "array", "items": "string"}},
                       {"name": "bytes_list_type", "type": {"type": "array", "items": "bytes"}}
                    ]}'''

        records_to_serialize = [
            {'index': 0,
             'float_list_type': [-1.0001, 0.1],
             'double_list_type': [-20.0, 22.33, 234.32334],
             'long_list_type': [-15L, 0L, 3022123019L],
             'int_list_type': [-20, -1],
             'boolean_list_type': [True, False],
             'string_list_type': ["abc", "defg"],
             'bytes_list_type': ["abc", "defgsd"]},
            {'index': 1,
             'float_list_type': [-3.22, 3298.233, 3939.1213],
             'double_list_type': [-2332.324, 2.665439],
             'long_list_type': [-1543L, 233L, 322L],
             'int_list_type': [-5, 342, -3222],
             'boolean_list_type': [False],
             'string_list_type': ["mnop", "qrs", "tuvwz"],
             'bytes_list_type': ["mnop", "qrs", "tuvwz"]
             }]

        features = {'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'float_list_type': parsing_ops.VarLenFeature(tf_types.float32),
                    'double_list_type': parsing_ops.VarLenFeature(tf_types.float64),
                    'long_list_type': parsing_ops.VarLenFeature(tf_types.int64),
                    'int_list_type': parsing_ops.VarLenFeature(tf_types.int32),
                    'boolean_list_type': parsing_ops.VarLenFeature(tf_types.bool),
                    'string_list_type': parsing_ops.VarLenFeature(tf_types.string),
                    'bytes_list_type': parsing_ops.VarLenFeature(tf_types.string)}

        tensors_expected = {
            'index': np.asarray([0, 1]),
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

        self.compare_all_tensors(schema,
                                 ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                 features,
                                 tensors_expected)

    def test_sparse_features(self):
        schema = '''{"doc": "Sparse features.",
                     "namespace": "com.test.sparse",
                     "type": "record",
                     "name": "sparse_feature",
                     "fields": [
                       {"name": "index", "type": "int"},
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
            {'index': 0, 'sparse_type': [{'index': 0, 'max_index': 10, 'value': 5.0},
                                         {'index': 5, 'max_index': 10, 'value': 7.0},
                                         {'index': 3, 'max_index': 10, 'value': 1.0}]},
            {'index': 1, 'sparse_type': [{'index': 0, 'max_index': 10, 'value': 2.0},
                                         {'index': 9, 'max_index': 10, 'value': 1.0}]}]

        features = {'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'sparse_type': parsing_ops.SparseFeature(
                        index_key='index', value_key='value', dtype=tf_types.float32, size=10)}

        tensors_expected = {
            'index': np.asarray([0, 1]),
            'sparse_type': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 3], [0, 5], [1, 0], [1, 9]]),
                np.asarray([5.0, 1.0, 7.0, 2.0, 1.0]),
                np.asarray([2, 10]))
        }

        self.compare_all_tensors(schema,
                                 ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                 features,
                                 tensors_expected)

    def test_nesting(self):
        schema = '''{"doc": "Nested records, arrays, lists, and link resolution.",
                     "namespace": "com.test.nested",
                     "type": "record",
                     "name": "test_nested_record",
                     "fields": [
                         {"name": "index", "type": "int"},
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
            {'index': 0,
             'nested_record': {'nested_int': 0, 'nested_float_list': [0.0, 10.0]},
             'list_of_records': [{'first_name': "Herbert", 'age': 70}],
             'map_of_records': {'first': {'first_name': "Herbert", 'age': 70},
                                'second': {'first_name': "Julia", 'age': 30}}},
            {'index': 1,
             'nested_record': {'nested_int': 5, 'nested_float_list': [-2.0, 7.0]},
             'list_of_records': [{'first_name': "Doug", 'age': 55},
                                 {'first_name': "Jess", 'age': 66},
                                 {'first_name': "Julia", 'age': 30}],
             'map_of_records': {'first': {'first_name': "Doug", 'age': 55},
                                'second': {'first_name': "Jess", 'age': 66}}},
            {'index': 2,
             'nested_record': {'nested_int': 7, 'nested_float_list': [3.0, 4.0]},
             'list_of_records': [{'first_name': "Karl", 'age': 32}],
             'map_of_records': {'first': {'first_name': "Karl", 'age': 32},
                                'second': {'first_name': "Joan", 'age': 21}}}]

        features = {'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'nested_record/nested_int': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'nested_record/nested_float_list': parsing_ops.FixedLenFeature([2], tf_types.float32),
                    'list_of_records/[0]/first_name': parsing_ops.FixedLenFeature([1], tf_types.string),
                    "map_of_records/['second']/age": parsing_ops.FixedLenFeature([1], tf_types.int32)}

        tensors_expected = {
            'index': np.asarray([0, 1, 2]),
            'nested_record/nested_int': np.asarray([0, 5, 7]),
            'nested_record/nested_float_list': np.asarray([[0.0, 10.0], [-2.0, 7.0], [3.0, 4.0]]),
            'list_of_records/[0]/first_name': np.asarray([["Herbert"], ["Doug"], ["Karl"]]),
            "map_of_records/['second']/age": np.asarray([[30], [66], [21]])
        }

        self.compare_all_tensors(schema,
                                 ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                 features,
                                 tensors_expected)

    def test_nested_with_asterisk(self):
        schema = '''{"doc": "Nested records in array to use asterisk.",
                     "namespace": "com.test.nested.records",
                     "type": "record",
                     "name": "test_nested_record",
                     "fields": [
                        {"name": "index", "type": "int"},
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
            {'index': 0, 'sparse_type': [{'index': 0, 'max_index': 10, 'value': 5.0},
                                         {'index': 5, 'max_index': 10, 'value': 7.0},
                                         {'index': 3, 'max_index': 10, 'value': 1.0}]},
            {'index': 1, 'sparse_type': [{'index': 0, 'max_index': 10, 'value': 2.0},
                                         {'index': 9, 'max_index': 10, 'value': 1.0}]}]

        features = {'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                    'sparse_type/[*]/index': parsing_ops.VarLenFeature(tf_types.int64),
                    'sparse_type/[*]/value': parsing_ops.VarLenFeature(tf_types.float32)}

        tensors_expected = {
            'index': np.asarray([0, 1]),
            'sparse_type/[*]/index': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
                np.asarray([0, 5, 3, 0, 9]),
                np.asarray([2, 3])),
            'sparse_type/[*]/value': sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
                np.asarray([5.0, 7.0, 1.0, 2.0, 1.0]),
                np.asarray([2, 3]))
        }

        self.compare_all_tensors(schema,
                                 ParseAvroRecordTest.serialize_all_records(schema, records_to_serialize),
                                 features,
                                 tensors_expected)


if __name__ == "__main__":
    test.main()
