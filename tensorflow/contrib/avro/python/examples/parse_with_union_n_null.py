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
"""Example of parsing Avro data with filtering and filling."""

import tensorflow as tf

from tensorflow.contrib.avro.python.parse_avro_record import parse_avro_record
from tensorflow.contrib.avro.python.utils.avro_serialization import AvroSerializer

schema = '''{"doc": "Test filtering",
             "namespace": "com.linkedin.test.filter_n_fill",
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
             }'''
data = [{'features': [{'name': "First", 'term': "First", 'value': 1.0},
                      {'name': "Second", 'term': "First", 'value': 2.0},
                      {'name': "Third", 'term': "First", 'value': 3.0}]},
        {'features': [{'name': "First", 'term': "First", 'value': 1.0},
                      {'name': "Second", 'term': "First", 'value': 2.0},
                      {'name': "Third", 'term': "First", 'value': 3.0}]}]

features = {'features/[name=First]/value': tf.FixedLenFeature([], tf.float32, default_value=0),
            'features/[name=Second]/value': tf.FixedLenFeature([], tf.float32, default_value=0),
            'features/[name=Third]/value': tf.FixedLenFeature([], tf.float32, default_value=0)}

if __name__ == '__main__':
    serializer = AvroSerializer(schema)  # Create a serializer
    serialized = [serializer.serialize(datum) for datum in data]  # Serialize data into a batch
    with tf.Session() as sess:
        input_str = tf.placeholder(tf.string)  # Variable to feed the serialized string
        tensors = parse_avro_record(input_str, schema, features)  # Use the parse function
        print sess.run(tensors, feed_dict={input_str: serialized})
