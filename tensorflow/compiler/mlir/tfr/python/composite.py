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
"""Op composition registration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# TODO(fengliuai): add the tf_export decrator
class Composite(object):
  """A decorator to register a function as a composition for an TF operator.

  The argument to the decorator must be the name of a TF raw operator the
  function composites for. Decorated function must take positional arguments
  which corresponds to the input and attributes in OpDef of the TF operation.
  # TODO(fengliuai): more documents here.

  Example:
    @composite.Composite('AddN')
    def _compose_add_n(inputs, N):
      if N == 1:
        ....
  """

  # TODO(fengliuai): support input_binding and output_binding so the arguments
  # are not positional.
  def __init__(self,
               op_name,
               inputs=None,
               attrs=None,
               derived_attrs=None,
               outputs=None):
    self._op_name = op_name
    self._inputs = inputs
    self._attrs = attrs
    self._derived_attrs = derived_attrs
    self._outputs = outputs

  def __call__(self, compose_fn):
    # TODO(fengliuai): more sanity check of the input function and make sure
    # the bounded arguments of the function matches the 'inputs' and 'attrs'.
    setattr(compose_fn, '_tfr_op_name', self._op_name)
    return compose_fn
