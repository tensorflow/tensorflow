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
"""Example creating avro data, reading that data into a dataset, and parsing
that data into tensors"""
import logging
import os
import shutil
import tempfile
import tensorflow as tf

from avro.datafile import DataFileWriter
from avro.io import DatumWriter
from avro.schema import parse
from tensorflow.contrib.avro.python.avro_record_dataset import AvroRecordDataset
from tensorflow.contrib.avro.python.parse_avro_record import parse_avro_record

if __name__ == '__main__':
    # Set the logging level
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)

    # Create temporary directory for data
    output_dir = tempfile.mkdtemp()
    filename = os.path.join(output_dir, "row-data.avro")
    filenames = [filename]

    # Define simple schema with an integer index and string data
    schema = '''{"doc": "Row index with data string.",
                 "type": "record",
                 "name": "row",
                 "fields": [
                     {"name": "index", "type": "int"},
                     {"name": "data", "type": "string"}
                 ]}'''

    # Define some sample data
    data = [{
        'index': 1,
        "data": "First row"
    }, {
        'index': 2,
        "data": "Second row"
    }, {
        'index': 3,
        "data": "Third row"
    }, {
        'index': 4,
        "data": "Fourth row"
    }, {
        'index': 5,
        "data": "Fifth row"
    }]

    # Write the avro data into the sample file
    with open(filename, 'wb') as out:
        writer = DataFileWriter(
            out, DatumWriter(), writers_schema=parse(schema))
        for datum in data:
            writer.append(datum)
        writer.close()

    # Define the function for parsing avro records into TensorFlow tensors
    def _parse_function(serialized):
        return parse_avro_record(
            serialized=serialized,
            schema=schema,
            features={
                "index": tf.FixedLenFeature([], tf.int32),
                "data": tf.FixedLenFeature([], tf.string)
            })

    # Create the dataset and apply the parse function
    dataset = AvroRecordDataset(filenames=filenames)
    dataset = dataset.batch(
        1)  # A batch dimension is required for the parse function
    dataset = dataset.map(_parse_function)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        while True:
            try:
                logging.info("Row is '{}'".format(sess.run(next_element)))
            except tf.errors.OutOfRangeError:
                logging.info("Done")
                break

    # Cleanup
    shutil.rmtree(output_dir)
