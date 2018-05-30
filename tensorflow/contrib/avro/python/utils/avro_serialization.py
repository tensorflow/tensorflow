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


from avro.io import DatumReader, DatumWriter, BinaryDecoder, BinaryEncoder
from avro.schema import parse
from StringIO import StringIO


class AvroDeserializer(object):

    def __init__(self, schema_json):
        """
        Create an avro deserializer.

        :param schema_json: Json string of the schema.
        """
        schema_object = parse(schema_json)
        self.datum_reader = DatumReader(writers_schema=schema_object, readers_schema=schema_object)

    def deserialize(self, serialized_str):
        """
        Deserialize an avro record from a string.

        :param serialized_str: The serialized input string.

        :return: The de-serialized record structure in python as map-list object.
        """
        # Use the same schema for reading as used for writing -- no schema resolution here
        return self.datum_reader.read(BinaryDecoder(StringIO(serialized_str)))


class AvroSerializer(object):

    def __init__(self, schema_json):
        """
        Create an avro serializer.

        :param schema_json:
        """
        schema_object = parse(schema_json)
        self.datum_writer = DatumWriter(writers_schema=schema_object)

    def serialize(self, datum):
        """
        Serialize a datum into a avro formatted string.

        :param datum: The avro datum.

        :return: The serialized string.
        """
        string_writer = StringIO()
        self.datum_writer.write(datum, BinaryEncoder(string_writer))
        return string_writer.getvalue()
