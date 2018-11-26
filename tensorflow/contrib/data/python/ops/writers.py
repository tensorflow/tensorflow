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
"""Python wrappers for tf.data writers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import writers
from tensorflow.python.util import deprecation


class TFRecordWriter(writers.TFRecordWriter):
  """Writes data to a TFRecord file."""

  @deprecation.deprecated(
      None, "Use `tf.data.experimental.TFRecordWriter(...)`.")
  def __init__(self, filename, compression_type=None):
    super(TFRecordWriter, self).__init__(filename, compression_type)
