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
"""Python wrapper for convert models from dense to sparse format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.lazy_loader import LazyLoader

# Lazy load since some of the performance benchmark skylark rules
# break dependencies. Must use double quotes to match code internal rewrite
# rule.
_sparsification_wrapper = LazyLoader(
    "_sparsification_wrapper", globals(),
    "tensorflow.lite.python.optimize."
    "tensorflow_lite_wrap_sparsification_wrapper")


class Sparsifier(object):
  """Convert a model from dense to sparse format.

  This is an internal class, not a public interface.
  """

  def __init__(self, model_content):
    """Constructor.

    Args:
      model_content: Content of a TF-Lite Flatbuffer file.

    Raises:
      ValueError: If unable to open the model.
    """
    if not model_content:
      raise ValueError("`model_content` must be specified.")
    try:
      self._sparsifier = (
          _sparsification_wrapper.SparsificationWrapper
          .CreateWrapperCPPFromBuffer(model_content))
    except Exception as e:
      raise ValueError("Failed to parse the model: %s." % e)
    if not self._sparsifier:
      raise ValueError("Failed to parse the model.")

  def sparsify(self):
    """Convert the model to sparse format.

    Returns:
      A sparse model.

    """
    return self._sparsifier.SparsifyModel()
