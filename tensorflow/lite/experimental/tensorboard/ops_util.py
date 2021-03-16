# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Ops util to handle ops for Lite."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.lite.python import wrap_toco
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


class SupportedOp(collections.namedtuple("SupportedOp", ["op"])):
  """Spec of supported ops.

  Args:
    op: string of op name.
  """


@tf_export(v1=["lite.experimental.get_potentially_supported_ops"])
@deprecation.deprecated(
    None, "Deprecated in TF 2.4 and targeted to remove after TF 2.5. This"
    "experimental function in TF v1 is to get a list of op names without real "
    "conversion. To check whether a model can be convertable or not indeed, "
    "please run `tf.lite.TFLiteConverter`.")
def get_potentially_supported_ops():
  """Returns operations potentially supported by TensorFlow Lite.

  The potentially support list contains a list of ops that are partially or
  fully supported, which is derived by simply scanning op names to check whether
  they can be handled without real conversion and specific parameters.

  Given that some ops may be partially supported, the optimal way to determine
  if a model's operations are supported is by converting using the TensorFlow
  Lite converter.

  Returns:
    A list of SupportedOp.
  """
  ops = wrap_toco.wrapped_get_potentially_supported_ops()
  return [SupportedOp(o["op"]) for o in ops]
