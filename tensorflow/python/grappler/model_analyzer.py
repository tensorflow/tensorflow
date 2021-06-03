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
# =============================================================================
"""Provides a proper python API for the symbols exported through swig."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.grappler import _pywrap_model_analyzer as tf_wrap


def GenerateModelReport(metagraph, assume_valid_feeds=True, debug=False):
  """Report what's known statically about each node in the provided metagraph.

  Args:
    metagraph: A TensorFlow MetaGraphDef.
    assume_valid_feeds: If True, assume that the shape of the fed nodes is valid
    debug: Add some information useful for debugging.

  Returns:
    A string containing the report.
  """
  return tf_wrap.GenerateModelReport(
      metagraph.SerializeToString(), assume_valid_feeds, debug)
