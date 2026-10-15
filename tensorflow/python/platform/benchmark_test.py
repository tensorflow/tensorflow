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

import os
import time
from unittest import mock
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
          always_print_fields_with_no_presence=True)['entry'][0]
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

  def testRunOpBenchmarkThroughputZeroOrNegativeDelta(self):
    mock_sess = mock.MagicMock()
    with mock.patch.object(self, 'report_benchmark') as mock_report:
      # Case 1: min_iters=0 (deltas is empty, median_delta=-1)
      res_zero_iters = self.run_op_benchmark(
          mock_sess, op_or_tensor=None, min_iters=0, mbs=10.0, name='zero_iters'
      )
      self.assertEqual(res_zero_iters['throughput'], 0.0)
      self.assertEqual(mock_report.call_args.kwargs['throughput'], 0.0)

      # Case 2: instantaneous execution (delta=0.0, median_delta=0.0)
      with mock.patch.object(time, 'time', return_value=1.0):
        res_zero_delta = self.run_op_benchmark(
            mock_sess,
            op_or_tensor=None,
            min_iters=5,
            mbs=10.0,
            name='zero_delta',
        )
        self.assertEqual(res_zero_delta['throughput'], 0.0)
        self.assertEqual(mock_report.call_args.kwargs['throughput'], 0.0)

  def testBenchmarksMainFilterRegexWithEqual(self):
    captured_regex = []

    with mock.patch.object(
        benchmark, '_run_benchmarks', side_effect=captured_regex.append
    ):
      with mock.patch.object(
          benchmark.app, 'run', side_effect=lambda fn, argv: fn(argv)
      ):
        test_argv = ['test_prog', '--benchmark_filter=Benchmark.*param=foo=bar']
        benchmark.benchmarks_main(true_main=None, argv=test_argv)

    self.assertEqual(captured_regex, ['Benchmark.*param=foo=bar'])


if __name__ == '__main__':
  test.main()
