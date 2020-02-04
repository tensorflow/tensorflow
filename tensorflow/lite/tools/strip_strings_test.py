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
"""Tests for strip_strings.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools import strip_strings
from tensorflow.lite.tools import test_utilities
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class StripTensorNamesTest(test_util.TensorFlowTestCase):

  def testStripTensorNames(self):
    # Define mock model
    model_mock = test_utilities.BuildMockModel()

    # Define temporary files
    tmp_dir = self.get_temp_dir()
    model_filename = os.path.join(tmp_dir, 'model.tflite')
    model_stripped_filename = os.path.join(tmp_dir, 'model_stripped.tflite')

    # Validate the mock model
    model = schema_fb.Model.GetRootAsModel(model_mock, 0)
    model_tensors = model.Subgraphs(0).Tensors
    self.assertEqual(b'input_tensor', model_tensors(0).Name())
    self.assertEqual(b'constant_tensor', model_tensors(1).Name())
    self.assertEqual(b'output_tensor', model_tensors(2).Name())

    # Store the model locally in model_filename
    with open(model_filename, 'wb') as model_file:
      model_file.write(model_mock)
    # Invoke the StripTfliteFile function to remove string names
    strip_strings.StripTfliteFile(model_filename, model_stripped_filename)
    # Read the locally stored model in model_stripped_filename
    with open(model_stripped_filename, 'rb') as model_file:
      model_stripped = model_file.read()

    # Validate the model stripped of tensor names
    model_stripped = schema_fb.Model.GetRootAsModel(model_stripped, 0)
    model_stripped_tensors = model_stripped.Subgraphs(0).Tensors
    self.assertEqual(b'', model_stripped_tensors(0).Name())
    self.assertEqual(b'', model_stripped_tensors(1).Name())
    self.assertEqual(b'', model_stripped_tensors(2).Name())

  def testStripSubGraphNames(self):
    # Define mock model
    model_mock = test_utilities.BuildMockModel()

    # Define temporary files
    tmp_dir = self.get_temp_dir()
    model_filename = os.path.join(tmp_dir, 'model.tflite')
    model_stripped_filename = os.path.join(tmp_dir, 'model_stripped.tflite')

    # Validate the mock model
    model = schema_fb.Model.GetRootAsModel(model_mock, 0)
    self.assertEqual(b'subgraph_name', model.Subgraphs(0).Name())

    # Store the model locally in model_filename
    with open(model_filename, 'wb') as model_file:
      model_file.write(model_mock)
    # Invoke the StripTfliteFile function to remove string names
    strip_strings.StripTfliteFile(model_filename, model_stripped_filename)
    # Read the locally stored model in model_stripped_filename
    with open(model_stripped_filename, 'rb') as model_file:
      model_stripped = model_file.read()

    # Validate the model stripped of subgraph names
    model_stripped = schema_fb.Model.GetRootAsModel(model_stripped, 0)
    self.assertEqual(b'', model_stripped.Subgraphs(0).Name())

  def testStripModelDescription(self):
    # Define mock model
    model_mock = test_utilities.BuildMockModel()

    # Define temporary files
    tmp_dir = self.get_temp_dir()
    model_filename = os.path.join(tmp_dir, 'model.tflite')
    model_stripped_filename = os.path.join(tmp_dir, 'model_stripped.tflite')

    # Validate the mock model
    model = schema_fb.Model.GetRootAsModel(model_mock, 0)
    self.assertEqual(b'model_description', model.Description())

    # Store the model locally in model_filename
    with open(model_filename, 'wb') as model_file:
      model_file.write(model_mock)
    # Invoke the StripTfliteFile function to remove string names
    strip_strings.StripTfliteFile(model_filename, model_stripped_filename)
    # Read the locally stored model in model_stripped_filename
    with open(model_stripped_filename, 'rb') as model_file:
      model_stripped = model_file.read()

    # Validate the model stripped of model description
    model_stripped = schema_fb.Model.GetRootAsModel(model_stripped, 0)
    self.assertEqual(b'', model_stripped.Description())


if __name__ == '__main__':
  test.main()
