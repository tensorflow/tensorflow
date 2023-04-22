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
"""Ensures that pywrap_gradient_exclusions.cc is up-to-date."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.eager import gradient_input_output_exclusions
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class GradientInputOutputExclusionsTest(test.TestCase):

  def testGeneratedFileMatchesHead(self):
    expected_contents = gradient_input_output_exclusions.get_contents()
    filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        resource_loader.get_path_to_datafile("pywrap_gradient_exclusions.cc"))
    actual_contents = file_io.read_file_to_string(filename)

    # On windows, one or both of these strings may have CRLF line endings.
    # To make sure, sanitize both:
    sanitized_actual_contents = actual_contents.replace("\r", "")
    sanitized_expected_contents = expected_contents.replace("\r", "")

    self.assertEqual(
        sanitized_actual_contents, sanitized_expected_contents, """
pywrap_gradient_exclusions.cc needs to be updated.
Please regenerate using:
bazel run tensorflow/python/eager:gradient_input_output_exclusions -- $PWD/tensorflow/python/eager/pywrap_gradient_exclusions.cc"""
    )


if __name__ == "__main__":
  test.main()
