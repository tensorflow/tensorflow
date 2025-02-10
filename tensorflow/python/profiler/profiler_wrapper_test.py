# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for profiler_wrapper.cc pybind methods."""

from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.profiler.internal import _pywrap_profiler_plugin as profiler_wrapper_plugin


class ProfilerSessionTest(test_util.TensorFlowTestCase):

  def test_xspace_to_tools_data_default_options(self):
    # filenames only used for `tf_data_bottleneck_analysis` and
    # `hlo_proto` tools.
    profiler_wrapper_plugin.xspace_to_tools_data([], 'trace_viewer')

  def _test_xspace_to_tools_data_options(self, options):
    profiler_wrapper_plugin.xspace_to_tools_data([], 'trace_viewer', options)

  def test_xspace_to_tools_data_empty_options(self):
    self._test_xspace_to_tools_data_options({})

  def test_xspace_to_tools_data_int_options(self):
    self._test_xspace_to_tools_data_options({'example_option': 0})

  def test_xspace_to_tools_data_str_options(self):
    self._test_xspace_to_tools_data_options({'example_option': 'example'})

if __name__ == '__main__':
  test.main()
