# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for BigQueryReader Op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re
import threading

from six.moves import SimpleHTTPServer
from six.moves import socketserver

from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops.cloud import cloud
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat

_PROJECT = "test-project"
_DATASET = "test-dataset"
_TABLE = "test-table"
# List representation of the test rows in the 'test-table' in BigQuery.
# The schema for each row is: [int64, string, float].
# The values for rows are generated such that some columns have null values. The
# general formula here is:
#   - The int64 column is present in every row.
#   - The string column is only avaiable in even rows.
#   - The float column is only available in every third row.
_ROWS = [[0, "s_0", 0.1], [1, None, None], [2, "s_2", None], [3, None, 3.1],
         [4, "s_4", None], [5, None, None], [6, "s_6", 6.1], [7, None, None],
         [8, "s_8", None], [9, None, 9.1]]
# Schema for 'test-table'.
# The schema currently has three columns: int64, string, and float
_SCHEMA = {
    "kind": "bigquery#table",
    "id": "test-project:test-dataset.test-table",
    "schema": {
        "fields": [{
            "name": "int64_col",
            "type": "INTEGER",
            "mode": "NULLABLE"
        }, {
            "name": "string_col",
            "type": "STRING",
            "mode": "NULLABLE"
        }, {
            "name": "float_col",
            "type": "FLOAT",
            "mode": "NULLABLE"
        }]
    }
}


def _ConvertRowToExampleProto(row):
  """Converts the input row to an Example proto.

  Args:
    row: Input Row instance.

  Returns:
    An Example proto initialized with row values.
  """

  example = example_pb2.Example()
  example.features.feature["int64_col"].int64_list.value.append(row[0])
  if row[1] is not None:
    example.features.feature["string_col"].bytes_list.value.append(
        compat.as_bytes(row[1]))
  if row[2] is not None:
    example.features.feature["float_col"].float_list.value.append(row[2])
  return example


class FakeBigQueryServer(threading.Thread):
  """Fake http server to return schema and data for sample table."""

  def __init__(self, address, port):
    """Creates a FakeBigQueryServer.

    Args:
      address: Server address
      port: Server port. Pass 0 to automatically pick an empty port.
    """
    threading.Thread.__init__(self)
    self.handler = BigQueryRequestHandler
    self.httpd = socketserver.TCPServer((address, port), self.handler)

  def run(self):
    self.httpd.serve_forever()

  def shutdown(self):
    self.httpd.shutdown()
    self.httpd.socket.close()


class BigQueryRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
  """Responds to BigQuery HTTP requests.

    Attributes:
      num_rows: num_rows in the underlying table served by this class.
  """

  num_rows = 0

  def do_GET(self):
    if "data?maxResults=" not in self.path:
      # This is a schema request.
      _SCHEMA["numRows"] = self.num_rows
      response = json.dumps(_SCHEMA)
    else:
      # This is a data request.
      #
      # Extract max results and start index.
      max_results = int(re.findall(r"maxResults=(\d+)", self.path)[0])
      start_index = int(re.findall(r"startIndex=(\d+)", self.path)[0])

      # Send the rows as JSON.
      rows = []
      for row in _ROWS[start_index:start_index + max_results]:
        row_json = {
            "f": [{
                "v": str(row[0])
            }, {
                "v": str(row[1]) if row[1] is not None else None
            }, {
                "v": str(row[2]) if row[2] is not None else None
            }]
        }
        rows.append(row_json)
      response = json.dumps({
          "kind": "bigquery#table",
          "id": "test-project:test-dataset.test-table",
          "rows": rows
      })
    self.send_response(200)
    self.end_headers()
    self.wfile.write(compat.as_bytes(response))


def _SetUpQueue(reader):
  """Sets up a queue for a reader."""
  queue = data_flow_ops.FIFOQueue(8, [types_pb2.DT_STRING], shapes=())
  key, value = reader.read(queue)
  queue.enqueue_many(reader.partitions()).run()
  queue.close().run()
  return key, value


class BigQueryReaderOpsTest(test.TestCase):

  def setUp(self):
    super(BigQueryReaderOpsTest, self).setUp()
    self.server = FakeBigQueryServer("127.0.0.1", 0)
    self.server.start()
    logging.info("server address is %s:%s", self.server.httpd.server_address[0],
                 self.server.httpd.server_address[1])

  def tearDown(self):
    self.server.shutdown()
    super(BigQueryReaderOpsTest, self).tearDown()

  def _ReadAndCheckRowsUsingFeatures(self, num_rows):
    self.server.handler.num_rows = num_rows

    with self.test_session() as sess:
      feature_configs = {
          "int64_col":
              parsing_ops.FixedLenFeature(
                  [1], dtype=dtypes.int64),
          "string_col":
              parsing_ops.FixedLenFeature(
                  [1], dtype=dtypes.string, default_value="s_default"),
      }
      reader = cloud.BigQueryReader(
          project_id=_PROJECT,
          dataset_id=_DATASET,
          table_id=_TABLE,
          num_partitions=4,
          features=feature_configs,
          timestamp_millis=1,
          test_end_point=("%s:%s" % (self.server.httpd.server_address[0],
                                     self.server.httpd.server_address[1])))

      key, value = _SetUpQueue(reader)

      seen_rows = []
      features = parsing_ops.parse_example(
          array_ops.reshape(value, [1]), feature_configs)
      for _ in range(num_rows):
        int_value, str_value = sess.run(
            [features["int64_col"], features["string_col"]])

        # Parse values returned from the session.
        self.assertEqual(int_value.shape, (1, 1))
        self.assertEqual(str_value.shape, (1, 1))
        int64_col = int_value[0][0]
        string_col = str_value[0][0]
        seen_rows.append(int64_col)

        # Compare.
        expected_row = _ROWS[int64_col]
        self.assertEqual(int64_col, expected_row[0])
        self.assertEqual(
            compat.as_str(string_col), ("s_%d" % int64_col) if expected_row[1]
            else "s_default")

      self.assertItemsEqual(seen_rows, range(num_rows))

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        sess.run([key, value])

  def testReadingSingleRowUsingFeatures(self):
    self._ReadAndCheckRowsUsingFeatures(1)

  def testReadingMultipleRowsUsingFeatures(self):
    self._ReadAndCheckRowsUsingFeatures(10)

  def testReadingMultipleRowsUsingColumns(self):
    num_rows = 10
    self.server.handler.num_rows = num_rows

    with self.test_session() as sess:
      reader = cloud.BigQueryReader(
          project_id=_PROJECT,
          dataset_id=_DATASET,
          table_id=_TABLE,
          num_partitions=4,
          columns=["int64_col", "float_col", "string_col"],
          timestamp_millis=1,
          test_end_point=("%s:%s" % (self.server.httpd.server_address[0],
                                     self.server.httpd.server_address[1])))
      key, value = _SetUpQueue(reader)
      seen_rows = []
      for row_index in range(num_rows):
        returned_row_id, example_proto = sess.run([key, value])
        example = example_pb2.Example()
        example.ParseFromString(example_proto)
        self.assertIn("int64_col", example.features.feature)
        feature = example.features.feature["int64_col"]
        self.assertEqual(len(feature.int64_list.value), 1)
        int64_col = feature.int64_list.value[0]
        seen_rows.append(int64_col)

        # Create our expected Example.
        expected_example = example_pb2.Example()
        expected_example = _ConvertRowToExampleProto(_ROWS[int64_col])

        # Compare.
        self.assertProtoEquals(example, expected_example)
        self.assertEqual(row_index, int(returned_row_id))

      self.assertItemsEqual(seen_rows, range(num_rows))

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        sess.run([key, value])


if __name__ == "__main__":
  test.main()
