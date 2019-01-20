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
"""Basic tests for TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


class ManualTest(trt_test.TfTrtIntegrationTestBase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    super(ManualTest, self).__init__(methodName)
    self._params_map = None

  def _GetEnv(self):
    """Get an environment variable specifying the manual test parameters.

    The value of the environment variable is the string representation of a dict
    which should contain the following keys:
    - 'graph_path': the file path to the serialized frozen graphdef
    - 'input_names': TfTrtIntegrationTestParams.input_names
    - 'input_dims': TfTrtIntegrationTestParams.input_dims
    - 'expected_output_dims': TfTrtIntegrationTestParams.expected_output_dims
    - 'output_name': the name of op to fetch
    - 'expected_engines_to_run': ExpectedEnginesToRun() will return this
    - 'expected_engines_to_build': ExpectedEnginesToBuild() will return this
    - 'max_batch_size': ConversionParams.max_batch_size

    Returns:
      The value of the environment variable.
    """
    return os.getenv('TRT_MANUAL_TEST_PARAMS', '')

  def _GetParamsMap(self):
    """Parse the environment variable as a dict and return it."""
    if self._params_map is None:
      self._params_map = ast.literal_eval(self._GetEnv())
    return self._params_map

  def GetParams(self):
    """Testing conversion of manually provided frozen graph."""
    params_map = self._GetParamsMap()
    gdef = graph_pb2.GraphDef()
    with gfile.Open(params_map['graph_path'], 'rb') as f:
      gdef.ParseFromString(f.read())
    return trt_test.TfTrtIntegrationTestParams(
        gdef=gdef,
        input_names=params_map['input_names'],
        input_dims=[params_map['input_dims']],
        output_names=params_map['output_names'],
        expected_output_dims=[params_map['expected_output_dims']])

  def GetConversionParams(self, run_params):
    """Return a ConversionParams for test."""
    conversion_params = super(ManualTest, self).GetConversionParams(run_params)
    params_map = self._GetParamsMap()
    if 'max_batch_size' in params_map:
      conversion_params = conversion_params._replace(
          max_batch_size=params_map['max_batch_size'])
    return conversion_params

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return self._GetParamsMap()['expected_engines_to_build']

  def ExpectedEnginesToRun(self, run_params):
    """Return the expected engines to run."""
    params_map = self._GetParamsMap()
    if 'expected_engines_to_run' in params_map:
      return params_map['expected_engines_to_run']
    return self.ExpectedEnginesToBuild(run_params)

  def ExpectedAbsoluteTolerance(self, run_params):
    """The absolute tolerance to compare floating point results."""
    params_map = self._GetParamsMap()
    if 'atol' in params_map:
      return params_map['atol']
    return 1.e-3

  def ExpectedRelativeTolerance(self, run_params):
    """The relative tolerance to compare floating point results."""
    params_map = self._GetParamsMap()
    if 'rtol' in params_map:
      return params_map['rtol']
    return 1.e-3

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    return len(self._GetEnv())


if __name__ == '__main__':
  test.main()
