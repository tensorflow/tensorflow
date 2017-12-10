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

The text_summary is a wrapper around the generic tensor_summary that takes a
string-type tensor and emits a TensorSummary op with SummaryMetadata that
notes that this summary is textual data for the TensorBoard text plugin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.summary_ops import tensor_summary

PLUGIN_NAME = "text"


def text_summary(name, tensor, collections=None):
  """Summarizes textual data.

  Text data summarized via this plugin will be visible in the Text Dashboard
  in TensorBoard. The standard TensorBoard Text Dashboard will render markdown
  in the strings, and will automatically organize 1d and 2d tensors into tables.
  If a tensor with more than 2 dimensions is provided, a 2d subarray will be
  displayed along with a warning message. (Note that this behavior is not
  intrinsic to the text summary api, but rather to the default TensorBoard text
  plugin.)

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    tensor: a string-type Tensor to summarize.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]

  Returns:
    A TensorSummary op that is configured so that TensorBoard will recognize
    that it contains textual data. The TensorSummary is a scalar `Tensor` of
    type `string` which contains `Summary` protobufs.

  Raises:
    ValueError: If tensor has the wrong type.
  """
  if tensor.dtype != dtypes.string:
    raise ValueError("Expected tensor %s to have dtype string, got %s" %
                     (tensor.name, tensor.dtype))

  summary_metadata = summary_pb2.SummaryMetadata(
      plugin_data=summary_pb2.SummaryMetadata.PluginData(
          plugin_name=PLUGIN_NAME))
  t_summary = tensor_summary(
      name=name,
      tensor=tensor,
      summary_metadata=summary_metadata,
      collections=collections)
  return t_summary
