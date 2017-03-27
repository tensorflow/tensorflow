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
"""Implements text_summary in TensorFlow, with TensorBoard support.

The text_summary is basically a wrapper around the generic tensor_summary,
and it uses a TextSummaryPluginAsset class to record which tensor_summaries
are readable by the TensorBoard text plugin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensorflow.python.framework import dtypes
from tensorflow.python.ops.summary_ops import tensor_summary
from tensorflow.python.summary import plugin_asset


def text_summary(name, tensor, collections=None):
  """Summarizes textual data.

  Text data summarized via this plugin will be visible in the Text Dashboard
  in TensorBoard.

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    tensor: a scalar string-type Tensor to summarize.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]

  Returns:
    A  TensorSummary op that is configured so that TensorBoard will recognize
    that it contains textual data. The TensorSummary is a scalar `Tensor` of
    type `string` which contains `Summary` protobufs.

  Raises:
    ValueError: If tensor has the wrong shape or type.
  """
  if tensor.dtype != dtypes.string:
    raise ValueError("Expected tensor %s to have dtype string, got %s" %
                     (tensor.name, tensor.dtype))

  if tensor.shape.ndims != 0:
    raise ValueError("Expected tensor %s to be scalar, has shape %s" %
                     (tensor.name, tensor.shape))

  t_summary = tensor_summary(name, tensor, collections)
  text_assets = plugin_asset.get_plugin_asset(TextSummaryPluginAsset)
  text_assets.register_tensor(t_summary.op.name)
  return t_summary


class TextSummaryPluginAsset(plugin_asset.PluginAsset):
  """Provides a registry of text summaries for the TensorBoard text plugin."""
  plugin_name = "tensorboard_text"

  def __init__(self):
    self._tensor_names = []

  def register_tensor(self, name):
    """Register a new Tensor Summary name as containing textual data."""
    self._tensor_names.append(name)

  def assets(self):
    """Store the tensors registry in a file called tensors.json."""
    return {"tensors.json": json.dumps(self._tensor_names)}
