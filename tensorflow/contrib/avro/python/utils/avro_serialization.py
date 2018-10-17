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

import six

from io import BytesIO
from avro.io import DatumReader, DatumWriter, BinaryDecoder, BinaryEncoder
from avro.datafile import DataFileReader

if six.PY2:
    from avro.schema import parse as parse

if six.PY3:
    from avro.schema import Parse as parse


class AvroFileToRecords(object):
    def __init__(self, filename, reader_schema):
        """
        Reads records as strings where each row is serialized separately

        :param filename: The filename from where to load the records
        :param reader_schema: Schema used for reading

        :return: An array of serialized string with one string per record
        """
        self.records = []
        schema_object = AvroParser(reader_schema).get_schema_object()

        with open(filename, 'rb') as file_handle:
            if six.PY2:
                datum_reader = DatumReader(readers_schema=schema_object)
            elif six.PY3:
                datum_reader = DatumReader(reader_schema=schema_object)
            else:
                raise RuntimeError("Only python 2 and python 3 are supported!")

            reader = DataFileReader(file_handle, datum_reader)

            self.records += [record for record in reader]

    def get_records(self):
        return self.records


class AvroSchemaReader(object):
    def __init__(self, filename):
        """
        Reads the schema from a file into json string
        """
        with open(filename, 'rb') as file_handle:
            reader = DataFileReader(file_handle, DatumReader())
            self.schema_json = ""
            if six.PY2:
                self.schema_json = str(reader.datum_reader.writers_schema)

            elif six.PY3:
                self.schema_json = str(reader.datum_reader.writer_schema)

            else:
                raise RuntimeError("Only python 2 and python 3 are supported!")

    def get_schema_json(self):
        return self.schema_json


class AvroParser(object):

    def __init__(self, schema_json):
        """
        Create an avro parser mostly to abstract away the API change between
        avro and avro-python3

        :param schema_json:
        """
        self.schema_object = parse(schema_json)

    def get_schema_object(self):
        return self.schema_object


class AvroDeserializer(object):

    def __init__(self, schema_json):
        """
        Create an avro deserializer.

        :param schema_json: Json string of the schema.
        """
        schema_object = AvroParser(schema_json).get_schema_object()
        # No schema resolution
        self.datum_reader = DatumReader(schema_object, schema_object)

    def deserialize(self, serialized_bytes):
        """
        Deserialize an avro record from bytes.

        :param serialized_bytes: The serialized bytes input.

        :return: The de-serialized record structure in python as map-list object.
        """
        return self.datum_reader.read(BinaryDecoder(BytesIO(serialized_bytes)))


class AvroSerializer(object):

    def __init__(self, schema_json):
        """
        Create an avro serializer.

        :param schema_json: Json string of the schema.
        """
        self.datum_writer = DatumWriter(
            AvroParser(schema_json).get_schema_object())

    def serialize(self, datum):
        """
        Serialize a datum into a avro formatted string.

        :param datum: The avro datum.

        :return: The serialized bytes.
        """
        writer = BytesIO()
        self.datum_writer.write(datum, BinaryEncoder(writer))
        return writer.getvalue()

