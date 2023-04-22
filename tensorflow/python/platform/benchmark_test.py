# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test for the tf.test.benchmark."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from google.protobuf import json_format
from tensorflow.core.util import test_log_pb2
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

class BenchmarkTest(test.TestCase, benchmark.TensorFlowBenchmark):

  def testReportBenchmark(self):
    output_dir = self.get_temp_dir() + os.path.sep
    os.environ['TEST_REPORT_FILE_PREFIX'] = output_dir
    proto_file_path = os.path.join(output_dir,
                                   'BenchmarkTest.testReportBenchmark')
    if os.path.exists(proto_file_path):
      os.remove(proto_file_path)

    self.report_benchmark(
        iters=2000,
        wall_time=1000,
        name='testReportBenchmark',
        metrics=[{'name': 'metric_name_1', 'value': 0, 'min_value': 1},
                 {'name': 'metric_name_2', 'value': 90, 'min_value': 0,
                  'max_value': 95}])

    with open(proto_file_path, 'rb') as f:
      benchmark_entries = test_log_pb2.BenchmarkEntries()
      benchmark_entries.ParseFromString(f.read())

      actual_result = json_format.MessageToDict(
          benchmark_entries, preserving_proto_field_name=True,
          including_default_value_fields=True)['entry'][0]
    os.remove(proto_file_path)

    expected_result = {
        'name': 'BenchmarkTest.testReportBenchmark',
        # google.protobuf.json_format.MessageToDict() will convert
        # int64 field to string.
        'iters': '2000',
        'wall_time': 1000,
        'cpu_time': 0,
        'throughput': 0,
        'extras': {},
        'metrics': [
            {
                'name': 'metric_name_1',
                'value': 0,
                'min_value': 1
            },
            {
                'name': 'metric_name_2',
                'value': 90,
                'min_value': 0,
                'max_value': 95
            }
        ]
    }

    self.assertEqual(2000, benchmark_entries.entry[0].iters)
    self.assertDictEqual(expected_result, actual_result)

if __name__ == '__main__':
  test.main()
