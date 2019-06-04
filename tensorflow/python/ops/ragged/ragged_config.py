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
"""Configuration parameters for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def auto_cast_partition_dtype():
  """Whether incopmatible row-partitioning dtypes should be auto-converted.

  If true, then operations that combine RaggedTensors but have different
  row-partitioning tensor dtypes will be automatically cast to a
  compatible dtype (`tf.int64`).  If false, then such operations will result
  in an error.

  Returns:
    `bool`
  """
  return False
