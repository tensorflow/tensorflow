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

"""Utility functions for TensorFlow models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc


def _enforce_names_consistency(specs):
  """Enforces that either all specs have names or none do."""

  def _has_name(spec):
    return hasattr(spec, 'name') and spec.name is not None

  def _clear_name(spec):
    spec = copy.deepcopy(spec)
    if hasattr(spec, 'name'):
      spec._name = None  # pylint:disable=protected-access
    return spec

  flat_specs = nest.flatten(specs)
  name_inconsistency = (
      any(_has_name(s) for s in flat_specs) and
      not all(_has_name(s) for s in flat_specs))

  if name_inconsistency:
    specs = nest.map_structure(_clear_name, specs)
  return specs


def model_input_signature(model, keep_original_batch_size=False):
  """Inspect model to get its input signature.

  The model's input signature is a list with a single (possibly-nested) object.
  This is due to the Keras-enforced restriction that tensor inputs must be
  passed in as the first argument.

  For example, a model with input {'feature1': <Tensor>, 'feature2': <Tensor>}
  will have input signature: [{'feature1': TensorSpec, 'feature2': TensorSpec}]

  Args:
    model: Keras Model object.
    keep_original_batch_size: A boolean indicating whether we want to keep using
      the original batch size or set it to None. Default is `False`, which means
      that the batch dim of the returned input signature will always be set to
      `None`.

  Returns:
    A list containing either a single TensorSpec or an object with nested
    TensorSpecs. This list does not contain the `training` argument.
  """
  input_specs = model._get_save_spec(dynamic_batch=not keep_original_batch_size)  # pylint: disable=protected-access
  if input_specs is None:
    return None
  input_specs = _enforce_names_consistency(input_specs)
  # Return a list with a single element as the model's input signature.
  if isinstance(input_specs,
                collections_abc.Sequence) and len(input_specs) == 1:
    # Note that the isinstance check filters out single-element dictionaries,
    # which should also be wrapped as a single-element list.
    return input_specs
  else:
    return [input_specs]

