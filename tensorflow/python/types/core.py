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
"""Core TensorFlow types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import textwrap

from typing import Union

import numpy as np

from tensorflow.python.types import doc_typealias
from tensorflow.python.util.tf_export import tf_export

if sys.version_info >= (3, 8):
  from typing import Protocol  # pylint:disable=g-import-not-at-top
else:
  from typing_extensions import Protocol  # pylint:disable=g-import-not-at-top

# TODO(mdan): Consider adding ABC once the dependence on isinstance is reduced.
# TODO(mdan): Add type annotations.


# TODO(b/178822082): Revisit this API when tf.types gets more resource.
@tf_export("__internal__.types.Tensor", v1=[])
class Tensor(object):
  """The base class of all dense Tensor objects.

  A dense tensor has a static data type (dtype), and may have a static rank and
  shape. Tensor objects are immutable. Mutable objects may be backed by a Tensor
  which holds the unique handle that identifies the mutable object.
  """

  @property
  def dtype(self):
    pass

  @property
  def shape(self):
    pass


class Symbol(Tensor):
  """Symbolic "graph" Tensor.

  These objects represent the output of an op definition and do not carry a
  value.
  """
  pass


class Value(Tensor):
  """Tensor that can be associated with a value (aka "eager tensor").

  These objects represent the (usually future) output of executing an op
  immediately.
  """

  def numpy(self):
    pass


class TensorProtocol(Protocol):
  """Protocol type for objects that can be converted to Tensor."""

  def __tf_tensor__(self, dtype=None, name=None):
    """Converts this object to a Tensor.

    Args:
      dtype: data type for the returned Tensor
      name: a name for the operations which create the Tensor
    Returns:
      A Tensor.
    """
    pass


# TODO(rahulkamat): Add missing types that are convertible to Tensor.
TensorLike = Union[Tensor, TensorProtocol, int, float, bool, str, complex,
                   tuple, list, np.ndarray]
doc_typealias.document(
    obj=TensorLike,
    doc=textwrap.dedent("""\
      Union of all types that can be converted to a `tf.Tensor` by `tf.convert_to_tensor`.

      This definition may be used in user code. Additional types may be added
      in the future as more input types are supported.

      Example:

      ```
      def foo(x: TensorLike):
        pass
      ```

      This definition passes static type verification for:

      ```
      foo(tf.constant([1, 2, 3]))
      foo([1, 2, 3])
      foo(np.array([1, 2, 3]))
      ```
      """),
)
tf_export("types.experimental.TensorLike").export_constant(
    __name__, "TensorLike")
