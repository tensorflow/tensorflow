# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Alias from TF to profiler code as part of moving out of TF.
"""

from tensorflow.python.platform import test
from tensorflow.python.profiler.internal import _pywrap_profiler_plugin


class PywrapTest(test.TestCase):
  def test_xspace_to_tools_data_default_options(self):
    # filenames only used for `tf_data_bottleneck_analysis` and
    # `hlo_proto` tools.
    _pywrap_profiler_plugin.xspace_to_tools_data([], 'trace_viewer')

if __name__ == '__main__':
  test.main()
