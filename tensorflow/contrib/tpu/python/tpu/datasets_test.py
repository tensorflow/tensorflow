# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TPU datasets tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.tpu.python.tpu import datasets
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.lib.io import python_io
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

_NUM_FILES = 10
_NUM_ENTRIES = 200


class DatasetsTest(test.TestCase):

  def setUp(self):
    super(DatasetsTest, self).setUp()
    self._coord = server_lib.Server.create_local_server()
    self._worker = server_lib.Server.create_local_server()

    self._cluster_def = cluster_pb2.ClusterDef()
    worker_job = self._cluster_def.job.add()
    worker_job.name = 'tpu_worker'
    worker_job.tasks[0] = self._worker.target[len('grpc://'):]
    coord_job = self._cluster_def.job.add()
    coord_job.name = 'coordinator'
    coord_job.tasks[0] = self._coord.target[len('grpc://'):]

    session_config = config_pb2.ConfigProto(cluster_def=self._cluster_def)

    self._sess = session.Session(self._worker.target, config=session_config)

  def testTextLineDataset(self):
    all_contents = []
    for i in range(_NUM_FILES):
      filename = os.path.join(self.get_temp_dir(), 'text_line.%d.txt' % i)
      contents = []
      for j in range(_NUM_ENTRIES):
        contents.append(compat.as_bytes('%d: %d' % (i, j)))
      with open(filename, 'wb') as f:
        f.write(b'\n'.join(contents))
      all_contents.extend(contents)

    dataset = datasets.StreamingFilesDataset(
        os.path.join(self.get_temp_dir(), 'text_line.*.txt'), filetype='text')

    iterator = dataset.make_initializable_iterator()
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = []
    for _ in range(2 * len(all_contents)):
      retrieved_values.append(compat.as_bytes(self._sess.run(get_next)))

    self.assertEqual(set(all_contents), set(retrieved_values))

  def testTFRecordDataset(self):
    all_contents = []
    for i in range(_NUM_FILES):
      filename = os.path.join(self.get_temp_dir(), 'tf_record.%d' % i)
      writer = python_io.TFRecordWriter(filename)
      for j in range(_NUM_ENTRIES):
        record = compat.as_bytes('Record %d of file %d' % (j, i))
        writer.write(record)
        all_contents.append(record)
      writer.close()

    dataset = datasets.StreamingFilesDataset(
        os.path.join(self.get_temp_dir(), 'tf_record*'), filetype='tfrecord')

    iterator = dataset.make_initializable_iterator()
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = []
    for _ in range(2 * len(all_contents)):
      retrieved_values.append(compat.as_bytes(self._sess.run(get_next)))

    self.assertEqual(set(all_contents), set(retrieved_values))

  def testTFRecordDatasetFromDataset(self):
    filenames = []
    all_contents = []
    for i in range(_NUM_FILES):
      filename = os.path.join(self.get_temp_dir(), 'tf_record.%d' % i)
      filenames.append(filename)
      writer = python_io.TFRecordWriter(filename)
      for j in range(_NUM_ENTRIES):
        record = compat.as_bytes('Record %d of file %d' % (j, i))
        writer.write(record)
        all_contents.append(record)
      writer.close()

    filenames = dataset_ops.Dataset.from_tensor_slices(filenames)

    dataset = datasets.StreamingFilesDataset(filenames, filetype='tfrecord')

    iterator = dataset.make_initializable_iterator()
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = []
    for _ in range(2 * len(all_contents)):
      retrieved_values.append(compat.as_bytes(self._sess.run(get_next)))

    self.assertEqual(set(all_contents), set(retrieved_values))

  def testArbitraryReaderFunc(self):

    def MakeRecord(i, j):
      return compat.as_bytes('%04d-%04d' % (i, j))

    record_bytes = len(MakeRecord(10, 200))

    all_contents = []
    for i in range(_NUM_FILES):
      filename = os.path.join(self.get_temp_dir(), 'fixed_length.%d' % i)
      with open(filename, 'wb') as f:
        for j in range(_NUM_ENTRIES):
          record = MakeRecord(i, j)
          f.write(record)
          all_contents.append(record)

    def FixedLengthFile(filename):
      return readers.FixedLengthRecordDataset(filename, record_bytes)

    dataset = datasets.StreamingFilesDataset(
        os.path.join(self.get_temp_dir(), 'fixed_length*'),
        filetype=FixedLengthFile)

    iterator = dataset.make_initializable_iterator()
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = []
    for _ in range(2 * len(all_contents)):
      retrieved_values.append(compat.as_bytes(self._sess.run(get_next)))

    self.assertEqual(set(all_contents), set(retrieved_values))

  def testUnexpectedFiletypeString(self):
    with self.assertRaises(ValueError):
      datasets.StreamingFilesDataset(
          os.path.join(self.get_temp_dir(), '*'), filetype='foo')

  def testUnexpectedFiletypeType(self):
    with self.assertRaises(ValueError):
      datasets.StreamingFilesDataset(
          os.path.join(self.get_temp_dir(), '*'), filetype=3)

  def testUnexpectedFilesType(self):
    with self.assertRaises(ValueError):
      datasets.StreamingFilesDataset(123, filetype='tfrecord')


if __name__ == '__main__':
  test.main()
