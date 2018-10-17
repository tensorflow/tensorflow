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
"""Example of parsing Avro data with a simple schema"""

import tensorflow as tf

from tensorflow.contrib.avro.python.parse_avro_record import parse_avro_record
from tensorflow.contrib.avro.python.utils.avro_serialization import AvroSerializer

schema = """
{
   "doc":"Simple example",
   "namespace":"com.test.person",
   "type":"record",
   "name":"data_row",
   "fields":[
      {
         "name":"index",
         "type":"long"
      },
      {
         "name":"first_name",
         "type":"string"
      },
      {
         "name":"age",
         "type":"int"
      }
   ]
}
"""

data = [{
    'index': 0,
    'first_name': "Karl",
    'age': 22
}, {
    'index': 1,
    'first_name': "Julia",
    'age': 42
}, {
    'index': 2,
    'first_name': "Liberty",
    'age': 90
}]

features = {
    'index': tf.FixedLenFeature([], tf.int64),
    'first_name': tf.FixedLenFeature([], tf.string),
    'age': tf.FixedLenFeature([], tf.int32)
}

if __name__ == '__main__':
    # Create a serializer
    serializer = AvroSerializer(schema)

    # Serialize data into a batch
    serialized = [serializer.serialize(datum) for datum in data]

    with tf.Session() as sess:
        # Variable to feed the serialized string
        input_str = tf.placeholder(tf.string)
        # Use the parse function
        parsed = parse_avro_record(input_str, schema, features)
        # Pull tensors
        index, first_name, age = parsed['index'], parsed['first_name'], parsed[
            'age']
        # Evaluate
        index_, first_name_, age_ = sess.run(
            [index, first_name, age], feed_dict={input_str: serialized})
        # Print outputs
        print("index: ", index_)
        print("first_name: ", first_name_)
        print("age: ", age_)
