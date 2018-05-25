"""
Example of parse Avro data and fill in with default values.
"""

import tensorflow as tf

from tensorflow.contrib.avro.python.parse_avro_record import parse_avro_record
from tensorflow.contrib.avro.python.utils.avro_serialization import AvroSerializer

schema = '''{"doc": "Fixed length lists.",
                   "namespace": "com.linkedin.test.lists.fixed",
                   "type": "record",
                   "name": "data_row",
                   "fields": [
                     {"name": "float_list_type", "type": {"type": "array", "items": "float"}},
                     {"name": "boolean_list_type", "type": {"type": "array", "items": "boolean"}},
                     {"name": "string_list_type", "type": {"type": "array", "items": "string"}},
                     {"name": "bytes_list_type", "type": {"type": "array", "items": "bytes"}}
                   ]}'''

data = [{'float_list_type': [-1.0001, 0.1, 23.2],
         'boolean_list_type': [True],
         'string_list_type': ["a", "b", "c"],
         'bytes_list_type': ["a", "b", "c"]},
        {'float_list_type': [],
         'boolean_list_type': [],
         'string_list_type': ["d1", "e1"],
         'bytes_list_type': ["d1"]}]

features = {'float_list_type': tf.FixedLenFeature([3], tf.float32, default_value=[0, 0, 0]),
            'boolean_list_type': tf.FixedLenFeature([], tf.bool, default_value=False),
            'string_list_type': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value="f"),
            'bytes_list_type': tf.FixedLenFeature([3], tf.string, default_value=["d", "e", "f"])}


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
        # Evaluate
        evaluated = sess.run(parsed, feed_dict={input_str: serialized})
        print("float list: {}".format(evaluated['float_list_type']))
        print("boolean list: {}".format(evaluated['boolean_list_type']))
        print("string list: {}".format(evaluated['string_list_type']))
        print("bytes list: {}".format(evaluated['bytes_list_type']))
