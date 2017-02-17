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
"""Tests for tensorflow.tools.dist_test.python.benchmark_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import os

from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.tools.dist_test.python import benchmark_util


class BenchmarkUtilTest(googletest.TestCase):

  def testStoreDataWithNoEntries(self):
    output_file = os.path.join(test.get_temp_dir(), 'test_output1.json')
    timing_entries = []
    benchmark_util.store_data_in_json(
        timing_entries, datetime.date(2017, 1, 1), output_file)
    json_output = json.loads(open(output_file, 'r').read())
    self.assertEquals('TestBenchmark', json_output['name'])
    self.assertEquals(u'1483228800', json_output['startTime'])

  def testStoreDataWithEntries(self):
    output_file = os.path.join(test.get_temp_dir(), 'test_output2.json')
    timing_entries = [
        benchmark_util.TimingEntry('test', 0.1, 1)]
    benchmark_util.store_data_in_json(
        timing_entries, datetime.date(2017, 1, 1), output_file)
    json_output = json.loads(open(output_file, 'r').read())

    self.assertEquals(1, len(json_output['entries']['entry']))
    self.assertEquals('test', json_output['entries']['entry'][0]['name'])
    self.assertEquals(0.1, json_output['entries']['entry'][0]['wallTime'])
    self.assertEquals(u'1', json_output['entries']['entry'][0]['iters'])
    self.assertEquals(u'1483228800', json_output['startTime'])
    self.assertEquals('TestBenchmark', json_output['name'])


if __name__ == '__main__':
  googletest.main()
