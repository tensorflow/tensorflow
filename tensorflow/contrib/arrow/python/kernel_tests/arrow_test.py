# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for ArrowDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import socket
import tempfile
import threading

import pyarrow as pa
from pyarrow.feather import write_feather

from tensorflow.contrib.arrow.python.ops import arrow_dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class ArrowDatasetTest(test.TestCase):

  def testArrowDataset(self):

    data = [
        [1, 2, 3, 4],
        [1.1, 2.2, 3.3, 4.4],
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        [[1], [2, 2], [3, 3, 3], [4, 4, 4]],
    ]

    arrays = [
        pa.array(data[0], type=pa.int32()),
        pa.array(data[1], type=pa.float32()),
        pa.array(data[2], type=pa.list_(pa.int32())),
        pa.array(data[3], type=pa.list_(pa.int32())),
    ]

    names = ["%s_[%s]" % (i, a.type) for i, a in enumerate(arrays)]
    batch = pa.RecordBatch.from_arrays(arrays, names)

    columns = tuple(range(len(arrays)))
    output_types = (dtypes.int32, dtypes.float32, dtypes.int32, dtypes.int32)

    dataset = arrow_dataset_ops.ArrowDataset(
        batch, columns, output_types)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      for row_num in range(len(data[0])):
        value = sess.run(next_element)
        self.assertEqual(value[0], data[0][row_num])
        self.assertAlmostEqual(value[1], data[1][row_num], 2)
        self.assertListEqual(value[2].tolist(), data[2][row_num])
        self.assertListEqual(value[3].tolist(), data[3][row_num])

    df = batch.to_pandas()

    dataset = arrow_dataset_ops.ArrowDataset.from_pandas(
        df, preserve_index=False)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      for row_num in range(len(data[0])):
        value = sess.run(next_element)
        self.assertEqual(value[0], data[0][row_num])
        self.assertAlmostEqual(value[1], data[1][row_num], 2)
        self.assertListEqual(value[2].tolist(), data[2][row_num])
        self.assertListEqual(value[3].tolist(), data[3][row_num])

  def testArrowFeatherDataset(self):
    f = tempfile.NamedTemporaryFile(delete=False)

    names = ["int32", "float32"]

    data = [
        [1, 2, 3, 4],
        [1.1, 2.2, 3.3, 4.4],
    ]

    arrays = [
        pa.array(data[0], type=pa.int32()),
        pa.array(data[1], type=pa.float32()),
    ]

    batch = pa.RecordBatch.from_arrays(arrays, names)
    df = batch.to_pandas()
    write_feather(df, f)
    f.close()

    columns = (0, 1)
    output_types = (dtypes.int32, dtypes.float32)

    dataset = arrow_dataset_ops.ArrowFeatherDataset(
        f.name, columns, output_types)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      for row_num in range(len(data[0])):
        value = sess.run(next_element)
        self.assertEqual(value[0], data[0][row_num])
        self.assertAlmostEqual(value[1], data[1][row_num], 2)

    os.unlink(f.name)

  def testArrowSocketDataset(self):

    data = [
        [1, 2, 3, 4],
        [1.1, 2.2, 3.3, 4.4],
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        [[1], [2, 2], [3, 3, 3], [4, 4, 4]],
    ]

    arrays = [
        pa.array(data[0], type=pa.int32()),
        pa.array(data[1], type=pa.float32()),
        pa.array(data[2], type=pa.list_(pa.int32())),
        pa.array(data[3], type=pa.list_(pa.int32())),
    ]

    names = ["%s_[%s]" % (i, a.type) for i, a in enumerate(arrays)]
    batch = pa.RecordBatch.from_arrays(arrays, names)

    columns = tuple(range(len(arrays)))
    output_types = (dtypes.int32, dtypes.float32, dtypes.int32, dtypes.int32)

    host = '127.0.0.1'
    port_num = 8080

    def run_server():
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.bind((host, port_num))
      s.listen(1)
      conn, _ = s.accept()
      outfile = conn.makefile(mode='wb')
      writer = pa.RecordBatchStreamWriter(outfile, batch.schema)
      writer.write_batch(batch)
      writer.close()
      outfile.flush()
      outfile.close()
      conn.close()
      s.close()

    server = threading.Thread(target=run_server)
    server.start()

    host_wport = host  # TODO + ':%' % port_num

    dataset = arrow_dataset_ops.ArrowStreamDataset(
        host_wport, columns, output_types)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      for row_num in range(len(data[0])):
        value = sess.run(next_element)
        self.assertEqual(value[0], data[0][row_num])
        self.assertAlmostEqual(value[1], data[1][row_num], 2)

    server.join()

if __name__ == "__main__":
  test.main()
