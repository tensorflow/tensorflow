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
"""Example of parsing Avro data with filtering."""

import tensorflow as tf
from tensorflow.contrib.avro.python.parse_avro_record import parse_avro_record
from tensorflow.contrib.avro.python.utils.avro_serialization import AvroSerializer

schema = '''{"doc": "Test filtering",
             "namespace": "com.linkedin.test.filtering",
             "type": "record",
             "name": "data_row",
             "fields": [{
                  "name": "guests",
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
             }]}'''
data = [{
    'guests': [{
        'name': "Hans",
        'gender': "male"
    }, {
        'name': "Mary",
        'gender': "female"
    }, {
        'name': "July",
        'gender': "female"
    }]
}, {
    'guests': [{
        'name': "Joel",
        'gender': "male"
    }, {
        'name': "JoAn",
        'gender': "female"
    }, {
        'name': "Kloy",
        'gender': "female"
    }]
}]
features = {
    'guests/[gender=male]/name': tf.VarLenFeature(tf.string),
    'guests/[gender=female]/name': tf.VarLenFeature(tf.string)
}

if __name__ == '__main__':
    serializer = AvroSerializer(schema)  # Create a serializer
    serialized = [serializer.serialize(datum) for datum in data
                 ]  # Serialize data into a batch
    with tf.Session() as sess:
        input_str = tf.placeholder(
            tf.string)  # Variable to feed the serialized string
        tensors = parse_avro_record(input_str, schema,
                                    features)  # Use the parse function
        print sess.run(tensors, feed_dict={input_str: serialized})
