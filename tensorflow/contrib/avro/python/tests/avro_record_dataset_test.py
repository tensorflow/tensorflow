from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import os
import shutil
import tempfile

from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors import OpError, OutOfRangeError
from tensorflow.core.protobuf import config_pb2
from tensorflow.contrib.avro.python.avro_record_dataset import AvroRecordDataset
from tensorflow.contrib.avro.python.utils.avro_record_utilities import read_schema, parse_schema, \
    read_records_resolved, deserialize, write_records_to_file


class AvroRecordDatasetTest(test_util.TensorFlowTestCase):

    def __init__(self, *args, **kwargs):
        super(AvroRecordDatasetTest, self).__init__(*args, **kwargs)

        # Set by setup method
        self.output_dir = ''
        self.filename = ''

        # Cover all primitive avro types in this schema
        # Note, python uses double precision throughout and int is represented in python through long
        self.full_schema = """{
            "namespace": "test.dataset",
            "doc": "Test schema for avro records.",
            "type": "record",
            "name": "row",
            "fields": [
                {"name": "index", "type": "int"},
                {"name": "boolean_type", "type": "boolean"},
                {"name": "bytes_type", "type": "bytes"},
                {"name": "int_type", "type": "int"},
                {"name": "long_type", "type": "long"},
                {"name": "float_type", "type": "float"},
                {"name": "double_type", "type": "double"},
                {"name": "string_type", "type": "string"},
                {"name": "features",
                 "type": ["null", {
                     "type": "array",
                     "items": ["null", {
                         "type": "record",
                         "name": "triplet",
                         "fields": [
                             {"name": "name", "type": ["null", "string"]},
                             {"name": "term", "type": ["null", "string"]},
                             {"name": "value", "type": ["null", "float"]}
                         ]}]
                 }]},
                {"name": "map_features",
                 "type": {
                     "type": "map",
                     "values": ["null", {
                         "name": "tri",
                         "type": "record",
                         "fields": [
                             {"name": "name", "type": ["null", "string"]},
                             {"name": "term", "type": ["null", "string"]},
                             {"name": "value", "type": ["null", "float"]}
                         ]}]
                 }}]}"""

        # Notes on max/min for different types types that we use in the test case below
        #                       max                 min
        #   int32               2^31-1              -2^31-1
        #   int64               2^63-1              -2^63-1
        #   float32 (single)    (2-2^-23)x2^127     -(2-2^-23)x2^127
        #   float64 (double)    (2-2^-52)x2^1023    -(2-2^-52)x2^1023
        self.test_records = [
            # This row checks empty or default values
            {"index": 0, "boolean_type": True, "bytes_type": "", "int_type": 0, "long_type": 0L,
             "float_type": 0.0, "double_type": 0.0, "string_type": "",
             "features": [],
             "map_features": {"first": {"name": "skill", "term": "coding", "value": 1.0},
                              "second": {"name": "skill", "term": "writing", "value": 1.0}}},
            # This row checks the largest values or special characters
            {"index": 1, "boolean_type": False, "bytes_type": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\"?",
             "int_type": 2147483648-1, "long_type": 9223372036854775807L,
             "float_type": 3.40282306074e+38, "double_type": 1.7976931348623157e+308,
             "string_type": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\"?",
             "features": [{"name": "skill", "term": "coding", "value": 1.0},
                          {"name": "skill", "term": "writing", "value": 1.0}],
             "map_features": {}},
            # This row checks the smallest values or characters
            {"index": 2, "boolean_type": False, "bytes_type": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
             "int_type": -2147483648, "long_type": -9223372036854775807L-1L,
             "float_type": -3.40282306074e+38, "double_type": -1.7976931348623157e+308,
             "string_type": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
             "features": [{"name": "region", "term": "az", "value": 1.0},
                          {"name": "skill", "term": "writing", "value": 1.0}],
             "map_features": {}},
            # This row checks some random data
            {"index": 3, "boolean_type": False, "bytes_type": "alkdfjiwij2oi2jp",
             "int_type": 213648, "long_type": -234829L,
             "float_type": 2342.322, "double_type": 2.2250738585072014e-308, "string_type": "aljk2ijlqn,w",
             "features": [{"name": "region", "term": "ca", "value": 1.0},
                          {"name": "skill", "term": "writing", "value": 1.0},
                          {"name": "region", "term": "az", "value": 1.0}],
             "map_features": {}},
        ]

    def setUp(self):
        # Setup logging
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)

        # Create the output directory
        self.output_dir = tempfile.mkdtemp()

        # Write test records
        self.filename = os.path.join(self.output_dir, "test.avro")
        write_records_to_file(records=self.test_records, writers_schema=self.full_schema, filename=self.filename)

    def tearDown(self):
        # Cleanup
        shutil.rmtree(self.output_dir)

    def _run_dataset_and_drain_data(self, filename, schema):
        """
        Runs the datset in a tensorflow session while draining the data.  Use this method for tests that are expected
        to fail because of filename issues, schema issues, or configuration issues.

        Note, we can't use the compare_records_in_file method in these cases of failure because the fastavro will first
        try to read the files and fail before the tensorflow operator is executed.

        :param filename: The filename.
        :param schema: The schema used for the dataset.
        :return:
        """
        with self.test_session() as sess:
            dataset = AvroRecordDataset(filenames=filename, schema=schema)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(iterator.initializer)
            while True:
                try:
                    sess.run(next_element)
                except OutOfRangeError:
                    logging.info("Done")
                break

    def compare_records_in_file(self, filename_be, filename_is, schema_resolved=None):
        """
        Compares records between files.  Note that the order of records in the two different files needs to match.

        :param filename_be: The filename for the ground-truth records.
        :param filename_is: The filename for the records as they are.
        :param schema_resolved: An optional schema for schema resolution.
        """
        self.compare_records_in_files([filename_be], [filename_is], schema_resolved=schema_resolved)

    def compare_records_in_files(self, filenames_be, filenames_is, schema_resolved=None):
        """
        Reads the string data from an avro encoded file and uses pyavroc to deserialize all of that string.
        The deserialized string name and values are then compared to the ground-truth data.

        :param filenames_be: A list of filenames for the ground-truth records.
        :param filenames_is: A list of filenames for the records as they are.
        :param schema_resolved: An optional schema for schema resolution.
        """

        logging.info("Taking records to be in files '{}' with those that are in '{}'.".format(filenames_be, filenames_is))
        config = config_pb2.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

        with self.test_session() as sess:

            dataset = AvroRecordDataset(filenames=filenames_is, schema=schema_resolved)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()

            writers_schema = read_schema(filenames_be[0])

            if schema_resolved is None:
                schema_resolved = writers_schema

            schema_resolved_object = parse_schema(schema_resolved)

            # Read "be" records from file
            records_be = read_records_resolved(filenames=filenames_be, schema_resolved=schema_resolved)

            sess.run(iterator.initializer)
            while True:
                i_record = 0
                try:
                    # If no schema is supplied use the full schema to deserialize
                    record_is = deserialize(sess.run(next_element), schema_object=schema_resolved_object)
                    record_be = records_be[record_is['index']]
                    for name, value_actual in record_is.iteritems():
                        # The field must be present in the read record
                        assert name in record_be, "Could not find {0} in read record.".format(name)
                        value_expected = record_be[name]
                        # The types of the fields must be the same
                        assert type(value_expected) == type(value_actual), \
                            "The field {} has type {} but should be type {}" \
                            .format(name, type(value_actual), type(value_expected))
                        # For floating points use approximate equality
                        if type(value_expected) is float:
                            self.assertAllClose(value_expected, value_actual)
                        # Anything except floating points need to match exactly
                        else:
                            assert value_expected == value_actual, "The field {} in record {} is {} but should be {}." \
                                .format(name, i_record, value_actual, value_expected)
                    i_record += 1
                except OutOfRangeError:
                    logging.info("Done")
                    break

    def test_broken_schema(self):
        """
        Test a broken schema. This schema is missing a ' in the definition and also a type.
        """
        broken_schema = '''{
            "namespace": "test.dataset",
            "doc": "Test broken schema with a syntax errors in the json string.",
            "type": "record",
            "name": "row",
            "fields": [
                {"name": "index", "type": "int"},
                {"name": "boolean_type"}
            ]}'''

        with self.assertRaises(OpError) as error:
            self._run_dataset_and_drain_data(filename=self.filename, schema=broken_schema)
        logging.info(error)

    def test_incompatible_schema(self):
        """
        Test an incompatible schema by introducing an additional attribute without default value.
        """
        incompatible_schema = '''{
            "namespace": "test.dataset",
            "doc": "Test schema with additional field which is incompatible.",
            "type": "record",
            "name": "row",
            "fields": [
                {"name": "index", "type": "int"},
                {"name": "boolean_type", "type": "boolean"},
                {"name": "crazy_type", "type": "boolean"}
            ]}'''

        with self.assertRaises(OpError) as error:
            self._run_dataset_and_drain_data(filename=self.filename, schema=incompatible_schema)
        logging.info(error)

    def test_wrong_file_name(self):
        """
        Test case with a wrong file name.
        """
        with self.assertRaises(OpError) as error:
            self._run_dataset_and_drain_data(filename=self.filename+".abc", schema=self.full_schema)
        logging.info(error)

    def test_no_schema_resolution_by_supplying_no_schema(self):
        """
        Test no schema resolution by supplying no schema.
        """
        self.compare_records_in_file(self.filename, self.filename, schema_resolved=None)

    def test_no_schema_resolution_by_supplying_full_schema(self):
        """
        Test no schema resolution by supplying the original schema.
        """
        self.compare_records_in_file(self.filename, self.filename, schema_resolved=self.full_schema)

    def test_sub_schema(self):
        """
        Test with sub schema.
        """
        sub_schema = '''{"namespace": "test.dataset",
                         "doc": "Test sub-schema with fields removed.",
                         "type": "record",
                         "name": "row",
                         "fields": [
                             {"name": "index", "type": "int"},
                             {"name": "boolean_type", "type": "boolean"}
                         ]}'''
        self.compare_records_in_file(self.filename, self.filename, schema_resolved=sub_schema)

    def test_expanded_schema_with_default(self):
        """
        Test with expanded schema with a default type.
        """
        expanded_schema = '''{"namespace": "test.dataset",
                              "doc": "Test expanded schema with additional value that have defaults.",
                              "type": "record",
                              "name": "row",
                              "fields": [
                                  {"name": "index", "type": "int"},
                                  {"name": "boolean_type", "type": "boolean", "default": false},
                                  {"name": "string_type_with_default", "type": "string", "default": "unknown"}
                              ]}'''
        with self.assertRaises(OpError) as error:
            self.compare_records_in_file(self.filename, self.filename, schema_resolved=expanded_schema)
        logging.info(error)

    def test_collapsed_record_in_array(self):
        """
        Test removal of a field from a record in an array. This does not work in pyavroc's deserializer and here we do
        not support this case either. One solution could be to select a branch of the union. This is what the current
        error message "Invalid argument: Could not find size of value, Union has no selected branch" indicates.
        """
        # A schema where we removed
        collapse_record_in_array_schema = '''{"namespace": "test.dataset",
                                              "doc": "Test schema with removed field in list of records.",
                                              "type": "record",
                                              "name": "row",
                                              "fields": [
                                                  {"name": "index", "type": "int"},
                                                  {"name": "features",
                                                   "type": [ "null", {
                                                     "type": "array",
                                                     "items": ["null", {
                                                         "type": "record",
                                                         "name": "triplet",
                                                         "fields": [
                                                             {"name": "name", "type": [ "null", "string"]},
                                                             {"name": "value", "type": [ "null", "float" ]}
                                                         ]
                                                      }]
                                                  } ]}
                                              ]}'''
        with self.assertRaises(OpError) as error:
            self.compare_records_in_file(self.filename, self.filename, schema_resolved=collapse_record_in_array_schema)
        logging.info(error)

    def test_collapse_record_in_map(self):
        """
        Test case: Remove a field from a record in a map. The problem and resolution is the same as in
        TestCollapseRecordInArray
        """
        collapse_record_in_map_schema = '''{"namespace": "test.dataset",
                                            "doc": "Test schema with removed field in map of records.",
                                            "type": "record",
                                            "name": "row",
                                            "fields": [
                                                {"name": "index", "type": "int"},
                                                {"name": "map_features",
                                                 "type": {
                                                    "type": "map",
                                                    "values": ["null", {
                                                        "name": "tri",
                                                        "type": "record",
                                                        "fields": [
                                                            {"name": "name", "type": [ "null", "string"]},
                                                            {"name": "value", "type": [ "null", "float" ]}
                                                            ]
                                                       }]
                                                   }
                                                }
                                            ]}'''
        self.compare_records_in_file(self.filename, self.filename, schema_resolved=collapse_record_in_map_schema)

    def test_up_cast(self):
        """
        Test case: Up-casting a single precision floating point to a double precision floating point should be a possible
        schema resolution
        """
        up_cast_schema = '''{"namespace": "test.dataset",
                             "doc": "Test schema with up-cast of float to double.",
                             "type": "record",
                             "name": "row",
                             "fields": [
                                 {"name": "index", "type": "int"},
                                 {"name": "float_type", "type": "double"}
                            ]}'''
        self.compare_records_in_file(self.filename, self.filename, schema_resolved=up_cast_schema)


if __name__ == "__main__":
    test.main()
