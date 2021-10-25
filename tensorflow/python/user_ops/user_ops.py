# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""All user ops."""

from tensorflow.python.user_ops.ops import gen_user_ops as _gen_user_ops

# go/tf-wildcard-import
from tensorflow.python.user_ops.ops.gen_user_ops import *  # pylint: disable=wildcard-import
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=['user_ops.my_fact'])
def my_fact():
  """Example of overriding the generated code for an Op."""
  return _gen_user_ops.fact()
