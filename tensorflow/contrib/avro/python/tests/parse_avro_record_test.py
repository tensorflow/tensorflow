from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import collections

from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.framework.errors import OpError
from tensorflow.python.framework import ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from tensorflow.contrib.avro.python.parse_avro_record import parse_avro_record
from tensorflow.contrib.avro.python.utils.avro_record_utilities import get_record_value, serialize, parse_schema
from tensorflow.contrib.avro.python.utils.tensor_utilities import fill_in_fixed_len, fill_in_fixed_len_sequence, \
    get_end_indices, get_n_elements_per_batch
import tensorflow.contrib.avro.python.utils.numerr as nr


class DataForTest(collections.namedtuple('DataForTest', ['schema', 'data', 'features', 'should_pass'])):
    """
    Test data contains a schema, data, features (for parsing), and whether this test is expected to pass or fail.
    """


class ParseAvroRecordTest(test.TestCase):

    def __init__(self, *args, **kwargs):
        super(ParseAvroRecordTest, self).__init__(*args, **kwargs)

    def setUp(self):
        """
        Setup fixture for test cases.
        """
        log_root = logging.getLogger()  # set logging level
        log_root.setLevel(logging.INFO)

    def run_test(self, test_case):
        """
        Runs a test case.
        """
        # Run over all test cases
        schema_object = parse_schema(test_case.schema)
        with ops.Graph().as_default() as g, self.test_session(graph=g) as sess:
            str_input = array_ops.placeholder(tf_types.string)

            parsed = parse_avro_record(str_input, test_case.schema, test_case.features)
            records = test_case.data
            # To test batch processing summarize all test data points into one batch
            serialized = [serialize(record, schema_object) for record in records]
            # If this test case should pass ensure that all thresholds are met
            if test_case.should_pass:
                # Here is where we execute our parser within TensorFlow
                tensors = sess.run(parsed, feed_dict={str_input: serialized})
                # Go over all key, value pairs in the features; keys are strings and values are TensorFlow type info
                for key, value in test_case.features.iteritems():
                    # Get all intended tensor values from the test data
                    tensor_be = len(records)*[None]
                    for i_record, record in enumerate(records):
                        record_value = get_record_value(record, key, value.dtype)
                        tensor_be[i_record] = record_value
                    # Get the actual tensor
                    tensor_is = tensors[key]
                    # Apply different test for the different tensor types: fixed length, var length, sparse
                    if isinstance(value, parsing_ops.FixedLenSequenceFeature):
                        logging.info("Comparing fixed length feature {0}.".format(key))
                        tensor_be = fill_in_fixed_len_sequence(tensor_be,
                                                               end_indices=get_end_indices(tensor_is),
                                                               n_elements_per_batch=get_n_elements_per_batch(tensor_is),
                                                               default_value=value.default_value)
                        assert nr.almost_equal_dense_tensor(tensor_is, tensor_be), \
                            "Value for field {0} has the relative error of {1} but should be < {2}.".format(
                                key, nr.relative_error_for_dense_tensor(tensor_is, tensor_be),
                                nr.ALMOST_EQUALS_THRESHOLD)
                    elif isinstance(value, parsing_ops.FixedLenFeature):
                        logging.info("Comparing fixed length feature {0}.".format(key))
                        tensor_be = fill_in_fixed_len(tensor_be,
                                                      end_indices=get_end_indices(tensor_is),
                                                      n_elements_per_batch=get_n_elements_per_batch(tensor_is),
                                                      default_values=value.default_value)
                        assert nr.almost_equal_dense_tensor(tensor_is, tensor_be), \
                            "Value for field {0} has the relative error of {1} but should be < {2}.".format(
                                key, nr.relative_error_for_dense_tensor(tensor_is, tensor_be),
                                nr.ALMOST_EQUALS_THRESHOLD)
                    elif isinstance(value, parsing_ops.VarLenFeature):
                        logging.info("Comparing variable length features {0}.".format(key))
                        assert nr.almost_equal_var_len_tensor(tensor_is, tensor_be), \
                            "Value for field {0} has the relative error of {1} but should be < {2}.".format(
                                key, nr.relative_error_for_var_len_tensor(tensor_is, tensor_be),
                                nr.ALMOST_EQUALS_THRESHOLD)
                    elif isinstance(value, parsing_ops.SparseFeature):
                        logging.info("Comparing sparse tensors {0}.".format(key))
                        assert nr.almost_equal_sparse_tensor(tensor_is, tensor_be, value), \
                            "Value for field {0} has the relative error of {1} but should be < {2}.".format(
                                key, nr.relative_error_for_sparse_tensor(tensor_is, tensor_be, value),
                                nr.ALMOST_EQUALS_THRESHOLD)
                    else:
                        logging.info("Error in feature type")
            else:
                # We assume this fails, so the print should not happen
                with self.assertRaises(OpError) as error:
                    print(sess.run(parsed, feed_dict={str_input: serialized}))
                # Log the error message so we can inspect it on the command line
                logging.info(error)

    def test_primitive_types(self):
        """
        Tests primitive types and the range for the different types of data.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Primitive types.",
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
                       ]}''',
            data=[{'index': 0,
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
                   'bytes_type': "aljk2ijlqn,w"}],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'float_type': parsing_ops.FixedLenFeature([], tf_types.float32),
                      'double_type': parsing_ops.FixedLenFeature([], tf_types.float64),
                      'long_type': parsing_ops.FixedLenFeature([], tf_types.int64),
                      'int_type': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'boolean_type': parsing_ops.FixedLenFeature([], tf_types.bool),
                      'string_type': parsing_ops.FixedLenFeature([], tf_types.string),
                      'bytes_type': parsing_ops.FixedLenFeature([], tf_types.string)},
            should_pass=True))

    def test_fixed_length_lists(self):
        """
        Tests fixed length lists features; where each row has the same number of items in the array.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Fixed length lists.",
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
                       ]}''',
            data=[{'index': 0,
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
                   }],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'float_list_type': parsing_ops.FixedLenFeature([3], tf_types.float32),
                      'double_list_type': parsing_ops.FixedLenFeature([2], tf_types.float64),
                      'long_list_type': parsing_ops.FixedLenFeature([3], tf_types.int64),
                      'int_list_type': parsing_ops.FixedLenFeature([3], tf_types.int32),
                      'boolean_list_type': parsing_ops.FixedLenFeature([], tf_types.bool),
                      'string_list_type': parsing_ops.FixedLenFeature([3], tf_types.string),
                      'bytes_list_type': parsing_ops.FixedLenFeature([3], tf_types.string)},
            should_pass=True))

    def test_variable_length_lists(self):
        """
        Test variable length features, where each row has a variable number of items in the array.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Variable length lists.",
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
                       ]}''',
            data=[{'index': 0,
                   'float_list_type': [-1.0001, 0.1],
                   'double_list_type': [-20.0, 22.33, 234.32334],
                   'long_list_type': [-15L, 0L, 3022123019L],
                   'int_list_type': [-20, -1],
                   'boolean_list_type': [True, False],
                   'string_list_type': ["abc", "defg"],
                   'bytes_list_type': ["abc", "defg"]},
                  {'index': 1,
                   'float_list_type': [-3.22, 3298.233, 3939.1213],
                   'double_list_type': [-2332.324, 2.665439],
                   'long_list_type': [-1543L, 233L, 322L],
                   'int_list_type': [-5, 342, -3222],
                   'boolean_list_type': [False],
                   'string_list_type': ["mnop", "qrs", "tuvwz"],
                   'bytes_list_type': ["mnop", "qrs", "tuvwz"]
                   }],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'double_list_type': parsing_ops.VarLenFeature(tf_types.float64),
                      'long_list_type': parsing_ops.VarLenFeature(tf_types.int64),
                      'int_list_type': parsing_ops.VarLenFeature(tf_types.int32),
                      'boolean_list_type': parsing_ops.VarLenFeature(tf_types.bool),
                      'string_list_type': parsing_ops.VarLenFeature(tf_types.string),
                      'bytes_list_type': parsing_ops.VarLenFeature(tf_types.string)},
            should_pass=True))

    def test_sparse_features(self):
        """
        Test sparse feature. A sparse feature maps an array of records that have data for keys and values.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Sparse features.",
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
                        }]}''',
            data=[{'index': 0, 'sparse_type': [{'index': 0, 'max_index': 10, 'value': 5.0},
                                               {'index': 5, 'max_index': 10, 'value': 7.0},
                                               {'index': 3, 'max_index': 10, 'value': 1.0}]},
                  {'index': 1, 'sparse_type': [{'index': 0, 'max_index': 10, 'value': 2.0},
                                               {'index': 9, 'max_index': 10, 'value': 1.0}]}],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'sparse_type': parsing_ops.SparseFeature(
                          index_key='index', value_key='value', dtype=tf_types.float32, size=10)},
            should_pass=True))

    def test_nesting(self):
        """
        Test nested records.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Nested records, arrays, lists, and link resolution.",
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
                          }]}''',
            data=[{'index': 0,
                   'nested_record': {'nested_int': 0, 'nested_float_list': [0.0, 10.0]},
                   'list_of_records': [{'first_name': "Herbert", 'age': 70}],
                   'map_of_records': {'first': {'first_name': "Herbert", 'age': 70},
                                      'second': {'first_name': "Julia", 'age': 30}}},
                  {'index': 1,
                   'nested_record': {'nested_int': 0, 'nested_float_list': [0.0, 10.0]},
                   'list_of_records': [{'first_name': "Doug", 'age': 55},
                                       {'first_name': "Jess", 'age': 66},
                                       {'first_name': "Julia", 'age': 30}],
                   'map_of_records': {'first': {'first_name': "Doug", 'age': 55},
                                      'second': {'first_name': "Jess", 'age': 66}}},
                  {'index': 2,
                   'nested_record': {'nested_int': 0, 'nested_float_list': [0.0, 10.0]},
                   'list_of_records': [{'first_name': "Karl", 'age': 32}],
                   'map_of_records': {'first': {'first_name': "Karl", 'age': 32},
                                      'second': {'first_name': "Joan", 'age': 21}}}],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'nested_record/nested_int': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'nested_record/nested_float_list': parsing_ops.FixedLenFeature([2], tf_types.float32),
                      'list_of_records/[0]/first_name': parsing_ops.FixedLenFeature([], tf_types.string),
                      "map_of_records/['second']/age": parsing_ops.FixedLenFeature([], tf_types.int32)},
            should_pass=True
        ))

    def test_nested_with_asterisk(self):
        """
        Test nested records with the asterisk notation. In this case we want all items from an array/map.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Nested records in array to use asterisk.",
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
                        }]}''',
            data=[{'index': 0, 'sparse_type': [{'index': 0, 'max_index': 10, 'value': 5.0},
                                               {'index': 5, 'max_index': 10, 'value': 7.0},
                                               {'index': 3, 'max_index': 10, 'value': 1.0}]},
                  {'index': 1, 'sparse_type': [{'index': 0, 'max_index': 10, 'value': 2.0},
                                               {'index': 9, 'max_index': 10, 'value': 1.0}]}],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'sparse_type/[*]/index': parsing_ops.VarLenFeature(tf_types.int64),
                      'sparse_type/[*]/value': parsing_ops.VarLenFeature(tf_types.float32)},
            should_pass=True))

    def test_parse_int_as_long_fail(self):
        """
        Test a failure case where we want to parse an int as a long value.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Parse int as long (int64) fails.",
                       "namespace": "com.test.int.type.failure",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                         {"name": "index", "type": "int"}
                       ]}''',
            data=[{'index': 0}],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int64)},
            should_pass=False
        ))

    def test_parse_int_as_sparse_type_fail(self):
        """
        Test parsing of an integer as a sparse tensor.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Parse int as SparseType fails.",
                       "namespace": "com.test.sparse.type.failure",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                         {"name": "index", "type": "int"}
                       ]}''',
            data=[{'index': 0}],
            features={'index': parsing_ops.SparseFeature(
                index_key='index', value_key='value', dtype=tf_types.float32, size=10)},
            should_pass=False
        ))

    def test_parse_float_as_double_fail(self):
        """
        Test parsing of a float as double. This fails because we do not provide up-casting in the parser.

        Types must match.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Parse float as double fails.",
                       "namespace": "com.test.float.type.failure",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                         {"name": "weight", "type": "float"}
                       ]}''',
            data=[{'weight': 0.5}],
            features={'weight': parsing_ops.FixedLenFeature([], tf_types.float64)},
            should_pass=False
        ))

    def test_fixed_length_without_proper_default_fail(self):
        """
        Tries to use 'FixedLenFeature' without a proper default.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Used fixed length without proper default.",
                       "namespace": "com.test.wrong.list.type",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                         {"name": "int_list_type", "type": {"type": "array", "items": "int"}}
                       ]}''',
            data=[{'int_list_type': [0, 1, 2]},
                  {'int_list_type': [0, 1]}],
            features={'int_list_type': parsing_ops.FixedLenFeature([], tf_types.int32)},
            should_pass=False
        ))

    def test_fixed_length_with_default(self):
        """
        Use defaults for each position in the tensor.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Fixed length lists with defaults.",
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
                       ]}''',
            data=[{'index': 0,
                   'float_list_type': [-1.0001, 0.1, 23.2],
                   'double_list_type': [-20.0, 22.33],
                   'long_list_type': [-15L, 0L, 3022123019L],
                   'int_list_type': [-20, -1, 2934],
                   'boolean_list_type': [True],
                   'string_list_type': ["abc", "defg", "hijkl"],
                   'bytes_list_type': ["abc", "defg", "hijkl"]},
                  {'index': 1,
                   'float_list_type': [-3.22, 3298.233],
                   'double_list_type': [-2332.324],
                   'long_list_type': [-1543L, 233L],
                   'int_list_type': [-5, 342],
                   'boolean_list_type': [],
                   'string_list_type': ["mnop", "qrs"],
                   'bytes_list_type': ["mnop"]
                   }],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'float_list_type': parsing_ops.FixedLenFeature(
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
                          [3], tf_types.string, default_value=['a', 'b', 'c'])},
            should_pass=True
        ))

    def test_fixed_length_sequence_with_default(self):
        """
        Test fixed length sequence where we provide a single default value.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Fixed length lists with defaults.",
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
                       ]}''',
            data=[{'index': 0,
                   'float_list_type': [-1.0001, 0.1, 23.2],
                   'double_list_type': [-20.0, 22.33],
                   'long_list_type': [-15L, 0L, 3022123019L],
                   'int_list_type': [-20, -1, 2934],
                   'boolean_list_type': [True],
                   'string_list_type': ["abc", "defg", "hijkl"],
                   'bytes_list_type': ["abc", "defg", "hijkl"]},
                  {'index': 1,
                   'float_list_type': [-3.22, 3298.233],
                   'double_list_type': [-2332.324],
                   'long_list_type': [-1543L, 233L],
                   'int_list_type': [-5, 342],
                   'boolean_list_type': [],
                   'string_list_type': ["mnop", "qrs"],
                   'bytes_list_type': ["mnop"]
                   }],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'float_list_type': parsing_ops.FixedLenSequenceFeature(
                          [], tf_types.float32, default_value=0.0, allow_missing=True),
                      'double_list_type': parsing_ops.FixedLenSequenceFeature(
                          [], tf_types.float64, default_value=1.0, allow_missing=True),
                      'long_list_type': parsing_ops.FixedLenSequenceFeature(
                          [], tf_types.int64, default_value=5L, allow_missing=True),
                      'int_list_type': parsing_ops.FixedLenSequenceFeature(
                          [], tf_types.int32, default_value=0, allow_missing=True),
                      'boolean_list_type': parsing_ops.FixedLenSequenceFeature(
                          [], tf_types.bool, default_value=False, allow_missing=True),
                      'string_list_type': parsing_ops.FixedLenSequenceFeature(
                          [], tf_types.string, default_value="my_default", allow_missing=True),
                      'bytes_list_type': parsing_ops.FixedLenSequenceFeature(
                          [], tf_types.string, default_value="my_default", allow_missing=True)},
            should_pass=True
        ))

    def test_fixed_length_for_array(self):
        """
        Tests the use of a fixed length array.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Use fixed length for array features.",
                       "namespace": "com.test.fixed.length.for.array",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                         {"name": "names", "type": {"type": "array", "items": "string"}}
                       ]}''',
            data=[{'names': ["Hans", "Herbert", "Heinz"]},
                  {'names': ["Gilbert", "Gerald", "Genie"]}],
            features={'names': parsing_ops.FixedLenFeature([3], tf_types.string)},
            should_pass=True
        ))

    def test_wrong_spelling_of_feature_name_fail(self):
        """
        Test failure for a wrong spelling of a feature name. Essentially, referencing a non-existing feature.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Wrong spelling of feature name.",
                       "namespace": "com.test.wrong.feature.name",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                         {"name": "int_type", "type": "int"}
                       ]}''',
            data=[{'int_type': 0}],
            features={'wrong_spelling': parsing_ops.FixedLenFeature([], tf_types.int32)},
            should_pass=False
        ))

    def test_wrong_spelling_of_feature_name_and_index(self):
        """
        Test wrong spelling of a feature name and wrong index.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Wrong spelling of feature name and wrong index.",
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
                                        {"name": "first_name", "type": "string"},
                                        {"name": "age", "type": "int"}
                                    ]
                                }
                            }
                          }
                       ]}''',
            data=[{'list_of_records': [{'first_name': "My name", 'age': 33}]}],
            features={'list_of_records/[2]/name': parsing_ops.FixedLenFeature([], tf_types.string)},
            should_pass=False
        ))

    def test_filtering_variable_length(self):
        """
        Test the filtering feature, where we filter array items through values for other record attributes.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans", 'gender': "male"},
                              {'name': "Mary", 'gender': "female"},
                              {'name': "July", 'gender': "female"}]},
                  {'guests': [{'name': "Joel", 'gender': "male"},
                              {'name': "JoAn", 'gender': "female"},
                              {'name': "Kloy", 'gender': "female"}]}],
            features={'guests/[gender=male]/name': parsing_ops.VarLenFeature(tf_types.string),
                      'guests/[gender=female]/name': parsing_ops.VarLenFeature(tf_types.string)},
            should_pass=True))

    def test_filtering_fixed_length(self):
        """
        Tests the filtering feature, where we filter array items through values for other record attributes.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans", 'gender': "male"},
                              {'name': "Mary", 'gender': "female"},
                              {'name': "July", 'gender': "female"}]},
                  {'guests': [{'name': "Joel", 'gender': "male"},
                              {'name': "JoAn", 'gender': "female"},
                              {'name': "Kloy", 'gender': "female"}]}],
            features={'guests/[gender=male]/name': parsing_ops.FixedLenFeature([1], tf_types.string),
                      'guests/[gender=female]/name': parsing_ops.FixedLenFeature([2], tf_types.string)},
            should_pass=True))

    def test_filtering_empty(self):
        """
        Tests a filter that returns an empty result.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans", 'gender': "male"},
                              {'name': "Mary", 'gender': "female"},
                              {'name': "July", 'gender': "female"}]},
                  {'guests': [{'name': "Joel", 'gender': "male"},
                              {'name': "JoAn", 'gender': "female"},
                              {'name': "Kloy", 'gender': "female"}]}],
            features={'guests/[gender=wrong_value]/name': parsing_ops.VarLenFeature(tf_types.string),
                      'guests/[gender=female]/name': parsing_ops.VarLenFeature(tf_types.string)},
            should_pass=True))

    def test_filtering_wrong_key_fail(self):
        """
        Tests a filter with a wrong key. This test expects failure.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans"},
                              {'name': "Mary"},
                              {'name': "July"}]}],
            features={'guests/[wrong_key=female]/name': parsing_ops.VarLenFeature(tf_types.string)},
            should_pass=False))

    def test_filtering_wrong_pair_fail(self):
        """
        In this case the parser thinks its a map but its an array.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans"},
                              {'name': "Mary"},
                              {'name': "July"}]}],
            features={'guests/[forgot_the_separator]/name': parsing_ops.VarLenFeature(tf_types.string)},
            should_pass=False))

    def test_filtering_too_many_separators(self):
        """
        A filter that contains too many '='.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans"},
                              {'name': "Mary"},
                              {'name': "July"}]}],
            features={'guests/[used=too=many=separators]/name': parsing_ops.VarLenFeature(tf_types.string)},
            should_pass=False))

    def test_filtering_nested_record(self):
        """
        Tests the filtering of nested records.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans", 'gender': "male", 'address': {'street': "California St",
                                                                             'zip': 94040,
                                                                             'state': "CA"}},
                              {'name': "Mary", 'gender': "female", 'address': {'street': "Ellis St",
                                                                               'zip': 29040,
                                                                               'state': "MA"}}]}],
            features={'guests/[gender=male]/name': parsing_ops.VarLenFeature(tf_types.string),
                      'guests/[gender=female]/address/street': parsing_ops.VarLenFeature(tf_types.string)},
            should_pass=True))

    def test_filtering_for_bytes(self):
        """
        Tests the filtering for bytes fields.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans", 'gender': "male"},
                              {'name': "Mary", 'gender': "female"},
                              {'name': "July", 'gender': "female"}]},
                  {'guests': [{'name': "Joel", 'gender': "male"},
                              {'name': "JoAn", 'gender': "female"},
                              {'name': "Kloy", 'gender': "female"}]}],
            features={'guests/[gender=male]/name': parsing_ops.VarLenFeature(tf_types.string),
                      'guests/[gender=female]/name': parsing_ops.VarLenFeature(tf_types.string)},
            should_pass=True))

    def test_filtering_fixed_len_feature(self):
        """
        Tests filtering into fixed length features.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans", 'gender': "male"},
                              {'name': "Mary", 'gender': "female"}]},
                  {'guests': [{'name': "Joel", 'gender': "male"},
                              {'name': "JoAn", 'gender': "female"}]}],
            features={'guests/[gender=male]/name': parsing_ops.FixedLenFeature([], tf_types.string),
                      'guests/[gender=female]/name': parsing_ops.FixedLenFeature([], tf_types.string)},
            should_pass=True))

    def test_filtering_sparse_feature(self):
        """
        Apply a filter where the result is a sparse tensor.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                       ]}''',
            data=[{'guests': [{'name': "Hans", 'gender': "male",
                               'address': [{'street': "California St", 'zip': 94040, 'state': "CA", 'street_no': 1},
                                           {'street': "New York St", 'zip': 32012, 'state': "NY", 'street_no': 2}]},
                              {'name': "Mary", 'gender': "female",
                               'address': [{'street': "Ellis St", 'zip': 29040, 'state': "MA", 'street_no': 3}]}]}],
            features={'guests/[gender=female]/address': parsing_ops.SparseFeature(index_key="zip",
                                                                                  value_key="street_no",
                                                                                  dtype=tf_types.int32,
                                                                                  size=94040)},
            should_pass=True))

    def test_null_union_primitive_type(self):
        """
        Tests parsing a union with a primitive type.

        The union of null and bytes is missing because of an ambiguity between the 0 termination symbol and the
        representation of null in the c implementation for avro.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Primitive types union with null.",
                           "namespace": "com.test.null.union.primitive",
                           "type": "record",
                           "name": "data_row",
                           "fields": [
                             {"name": "index", "type": "int"},
                             {"name": "float_type", "type": [ "null", "float"]},
                             {"name": "double_type", "type": [ "null", "double"]},
                             {"name": "long_type", "type": [ "null", "long"]},
                             {"name": "int_type", "type": [ "null", "int"]},
                             {"name": "boolean_type", "type": [ "null", "boolean"]},
                             {"name": "string_type", "type": [ "null", "string"]}
                           ]}''',
            data=[{'index': 0,
                   'float_type': 3.40282306074e+38,
                   'double_type': 1.7976931348623157e+308,
                   'long_type': 9223372036854775807L,
                   'int_type': 2147483648-1,
                   'boolean_type': True,
                   'string_type': "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"
                   }],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'float_type': parsing_ops.FixedLenFeature([], tf_types.float32),
                      'double_type': parsing_ops.FixedLenFeature([], tf_types.float64),
                      'long_type': parsing_ops.FixedLenFeature([], tf_types.int64),
                      'int_type': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'boolean_type': parsing_ops.FixedLenFeature([], tf_types.bool),
                      'string_type': parsing_ops.FixedLenFeature([], tf_types.string)},
            should_pass=True))

    def test_null_union_non_primitive_types(self):
        """
        Tests parsing a union with non-primitive types.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Unions between null and a non-primitive type.",
                       "namespace": "com.test.null.union.non.primitive",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                             {"name": "index", "type": "int"},
                             {"name": "array_type", "type": [ "null", {"type": "array", "items": {"type": "float"}}]},
                             {"name": "map_type", "type": [ "null", {"type": "map", "values": {"type": "double"}}]}
                           ]}''',
            data=[{'index': 0,
                   'array_type': [1.0, 2.0, 3.0],
                   'map_type': {'one': 1.0, 'two': 2.0}}],
            features={"index": parsing_ops.FixedLenFeature([], tf_types.int32),
                      "array_type/[0]": parsing_ops.FixedLenFeature([], tf_types.float32),
                      "map_type/['one']": parsing_ops.FixedLenFeature([], tf_types.float64)},
            should_pass=True))

    def test_nested_unions_with_nulls(self):
        """
        Tests a schema that has union's with nulls.

        In particular this test case covers the following unions

        array, null
        record, null
        string, null
        float, null
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Test filtering",
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
                          "name" : "term",
                          "type" : [ "null", "string" ]
                        }, {
                          "name" : "value",
                          "type" : [ "null", "float" ]
                        } ]
                      } ]
                    } ]
                  } ]
                 }''',
            data=[{'features': [{'name': "First", 'term': "First", 'value': 1.0},
                                {'name': "Second", 'term': "First", 'value': 2.0},
                                {'name': "Third", 'term': "First", 'value': 3.0}]},
                  {'features': [{'name': "First", 'term': "First", 'value': 1.0},
                                {'name': "Second", 'term': "First", 'value': 2.0},
                                {'name': "Third", 'term': "First", 'value': 3.0}]}],
            features={'features/[name=First]/value': parsing_ops.FixedLenFeature([], tf_types.float32, default_value=0),
                      'features/[name=Second]/value': parsing_ops.FixedLenFeature([], tf_types.float32, default_value=0),
                      'features/[name=Third]/value': parsing_ops.FixedLenFeature([], tf_types.float32, default_value=0)},
            should_pass=True))

    def test_primitive_types_and_union_with_null_fail(self):
        """
        Tests union of primitive types with null.

        This test case will fail because we do not auto-convert null's anymore.
        """
        self.run_test(DataForTest(
            schema='''{"doc": "Primitive types union with null.",
                       "namespace": "com.test.primitive.and.null",
                       "type": "record",
                       "name": "data_row",
                       "fields": [
                         {"name": "index", "type": "int"},
                         {"name": "float_type", "type": [ "null", "float"]},
                         {"name": "double_type", "type": [ "null", "double"]},
                         {"name": "long_type", "type": [ "null", "long"]},
                         {"name": "int_type", "type": [ "null", "int"]},
                         {"name": "boolean_type", "type": [ "null", "boolean"]},
                         {"name": "string_type", "type": [ "null", "string"]}
                       ]}''',
            data=[{'index': 0,
                   'float_type': None,
                   'double_type': None,
                   'long_type': None,
                   'int_type': None,
                   'boolean_type': None,
                   'string_type': None
                   }],
            features={'index': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'float_type': parsing_ops.FixedLenFeature([], tf_types.float32),
                      'double_type': parsing_ops.FixedLenFeature([], tf_types.float64),
                      'long_type': parsing_ops.FixedLenFeature([], tf_types.int64),
                      'int_type': parsing_ops.FixedLenFeature([], tf_types.int32),
                      'boolean_type': parsing_ops.FixedLenFeature([], tf_types.bool),
                      'string_type': parsing_ops.FixedLenFeature([], tf_types.string)},
            should_pass=False))
