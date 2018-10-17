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
import os
import six
import shutil
import tempfile

from avro.io import DatumWriter
from avro.datafile import DataFileWriter

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors import OpError, OutOfRangeError
from tensorflow.contrib.avro.python.avro_record_dataset import AvroRecordDataset
from tensorflow.contrib.avro.python.utils.avro_serialization import \
    AvroDeserializer, AvroParser, AvroSchemaReader, AvroFileToRecords


class AvroRecordDatasetTest(test_util.TensorFlowTestCase):
    """
    Tests the avro record dataset; especially schema resolution that is possible
    within that dataset
    """

    def __init__(self, *args, **kwargs):
        super(AvroRecordDatasetTest, self).__init__(*args, **kwargs)

        # Set by setup
        self.output_dir = ''
        self.filename = ''

        # Cover all primitive avro types in this schema
        # Note, python handles precision of data types dynamically
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
            # Check empty and defaults
            {
                "index": 0,
                "boolean_type": True,
                "bytes_type": b"",
                "int_type": 0,
                "long_type": 0,
                "float_type": 0.0,
                "double_type": 0.0,
                "string_type": "",
                "features": [],
                "map_features": {
                    "first": {
                        "name": "skill",
                        "term": "coding",
                        "value": 1.0
                    },
                    "second": {
                        "name": "skill",
                        "term": "writing",
                        "value": 1.0
                    }
                }
            },
            # Check largest values and special characters
            {
                "index":
                1,
                "boolean_type":
                False,
                "bytes_type":
                b"SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\"?",
                "int_type":
                2147483648 - 1,
                "long_type":
                9223372036854775807,
                "float_type":
                3.40282306074e+38,
                "double_type":
                1.7976931348623157e+308,
                "string_type":
                "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\"?",
                "features": [{
                    "name": "skill",
                    "term": "coding",
                    "value": 1.0
                }, {
                    "name": "skill",
                    "term": "writing",
                    "value": 1.0
                }],
                "map_features": {}
            },
            # Check smallest values and all characters/digits
            {
                "index":
                2,
                "boolean_type":
                False,
                "bytes_type":
                b"ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
                "int_type":
                -2147483648,
                "long_type":
                -9223372036854775807 - 1,
                "float_type":
                -3.40282306074e+38,
                "double_type":
                -1.7976931348623157e+308,
                "string_type":
                "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
                "features": [{
                    "name": "region",
                    "term": "az",
                    "value": 1.0
                }, {
                    "name": "skill",
                    "term": "writing",
                    "value": 1.0
                }],
                "map_features": {}
            },
            # Random data
            {
                "index":
                3,
                "boolean_type":
                False,
                "bytes_type":
                b"alkdfjiwij2oi2jp",
                "int_type":
                213648,
                "long_type":
                -234829,
                "float_type":
                2342.322,
                "double_type":
                2.2250738585072014e-308,
                "string_type":
                "aljk2ijlqn,w",
                "features": [{
                    "name": "region",
                    "term": "ca",
                    "value": 1.0
                }, {
                    "name": "skill",
                    "term": "writing",
                    "value": 1.0
                }, {
                    "name": "region",
                    "term": "az",
                    "value": 1.0
                }],
                "map_features": {}
            },
        ]

    @staticmethod
    def _write_records_to_file(records,
                               filename,
                               writer_schema,
                               codec='deflate'):
        """
        Writes the string data into an avro encoded file

        :param records: Records to write
        :param filename: Filename for the file to be written
        :param writer_schema: The schema used when writing the records
        :param codec: Compression codec used to write the avro file
        """
        schema = AvroParser(writer_schema).get_schema_object()
        with open(filename, 'wb') as out:
            writer = DataFileWriter(
                out, DatumWriter(), schema, codec=codec)
            for record in records:
                writer.append(record)
            writer.close()

    def setUp(self):
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)

        # Write test records into temporary output directory
        self.output_dir = tempfile.mkdtemp()
        self.filename = os.path.join(self.output_dir, "test.avro")
        AvroRecordDatasetTest._write_records_to_file(
            records=self.test_records,
            writer_schema=self.full_schema,
            filename=self.filename)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def _load_records(self, reader_schema):
        """
        Loads records using the avro dataset

        Typically used in tests that fail due to loading

        :param reader_schema: The schema used when reading the dataset
        """
        with self.test_session() as sess:
            dataset = AvroRecordDataset(
                filenames=[self.filename], schema=reader_schema)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(iterator.initializer)
            while True:
                try:
                    sess.run(next_element)
                except OutOfRangeError:
                    logging.info("Done")
                break

    def _load_and_compare_records(self, reader_schema=None):
        """
        Reads records using avro and the avro dataset and compares the contents
        of the records

        :param reader_schema: Optional readers schema
        """

        # Parse cases in sequence
        config = config_pb2.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:

            dataset = AvroRecordDataset(
                filenames=[self.filename], schema=reader_schema)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()

            writer_schema = AvroSchemaReader(self.filename).get_schema_json()

            if reader_schema is None:
                reader_schema = writer_schema

            records_expected = AvroFileToRecords(filename=self.filename,
                                                 reader_schema=reader_schema).get_records()

            sess.run(iterator.initializer)

            deserializer = AvroDeserializer(reader_schema)
            while True:
                i_record = 0
                try:
                    record_actual = deserializer.deserialize(
                        sess.run(next_element))
                    record_expected = records_expected[record_actual['index']]
                    for name, value_actual in six.iteritems(record_actual):
                        # The field must be present in the read record
                        assert name in record_expected, "Could not find {0} in read record.".format(
                            name)
                        value_expected = record_expected[name]
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

    def test_broken_schema_fail(self):
        """
        Test a broken schema. This schema is missing a ' in the definition and a type
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
            self._load_records(reader_schema=broken_schema)
        logging.info(error)

    def test_incompatible_schema_fail(self):
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
            self._load_records(reader_schema=incompatible_schema)
        logging.info(error)

    def test_wrong_file_name_fail(self):
        """
        Test case with a wrong file name.
        """
        with self.assertRaises(OpError) as error:
            self.filename = self.filename + ".abc"
            self._load_records(reader_schema=self.full_schema)
        logging.info(error)

    def test_no_schema_resolution_by_supplying_no_schema(self):
        """
        Test no schema resolution by supplying no schema.
        """
        self._load_and_compare_records(reader_schema=None)

    def test_no_schema_resolution_by_supplying_full_schema(self):
        """
        Test no schema resolution by supplying the original schema.
        """
        self._load_and_compare_records(reader_schema=self.full_schema)

    def test_resolve_to_sub_schema(self):
        """
        Resolve to sub-schema
        """
        sub_schema = '''{"namespace": "test.dataset",
                         "doc": "Test sub-schema with fields removed.",
                         "type": "record",
                         "name": "row",
                         "fields": [
                             {"name": "index", "type": "int"},
                             {"name": "boolean_type", "type": "boolean"}
                         ]}'''
        self._load_and_compare_records(reader_schema=sub_schema)

    def test_extend_with_default_fail(self):
        """
        Expand schema with a default type
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
            self._load_and_compare_records(reader_schema=expanded_schema)
        logging.info(error)

    def test_remove_field_in_record_in_array_fail(self):
        """
        Remove a field from a record in an array.

        This does not work in pyavroc's deserializer and here we do not support this case either.
        One solution could be to select a branch of the union. This is what the current error message

            "Invalid argument: Could not find size of value, Union has no selected branch"

        indicates.
        """
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
            self._load_and_compare_records(
                reader_schema=collapse_record_in_array_schema)
        logging.info(error)

    def test_remove_field_in_record_in_map_fail(self):
        """
        Remove a field from a record in a map. The problem and resolution is the same as in TestCollapseRecordInArray
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
        self._load_and_compare_records(
            reader_schema=collapse_record_in_map_schema)

    def test_up_cast(self):
        """
        Up-cast a single to double precision
        """
        up_cast_schema = '''{"namespace": "test.dataset",
                             "doc": "Test schema with up-cast of float to double.",
                             "type": "record",
                             "name": "row",
                             "fields": [
                                 {"name": "index", "type": "int"},
                                 {"name": "float_type", "type": "double"}
                            ]}'''
        self._load_and_compare_records(reader_schema=up_cast_schema)


if __name__ == "__main__":
    test.main()
