# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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


"""Utilities for using generic resources."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_should_use


_Resource = collections.namedtuple("_Resource",
                                   ["handle", "create", "is_initialized"])


def register_resource(handle, create_op, is_initialized_op, is_shared=True):
  """Registers a resource into the appropriate collections.

  This makes the resource findable in either the shared or local resources
  collection.

  Args:
   handle: op which returns a handle for the resource.
   create_op: op which initializes the resource.
   is_initialized_op: op which returns a scalar boolean tensor of whether
    the resource has been initialized.
   is_shared: if True, the resource gets added to the shared resource
    collection; otherwise it gets added to the local resource collection.

  """
  resource = _Resource(handle, create_op, is_initialized_op)
  if is_shared:
    ops.add_to_collection(ops.GraphKeys.RESOURCES, resource)
  else:
    ops.add_to_collection(ops.GraphKeys.LOCAL_RESOURCES, resource)


def shared_resources():
  """Returns resources visible to all tasks in the cluster."""
  return ops.get_collection(ops.GraphKeys.RESOURCES)


def local_resources():
  """Returns resources intended to be local to this session."""
  return ops.get_collection(ops.GraphKeys.LOCAL_RESOURCES)


def report_uninitialized_resources(resource_list=None,
                                   name="report_uninitialized_resources"):
  """Returns the names of all uninitialized resources in resource_list.

  If the returned tensor is empty then all resources have been initialized.

  Args:
   resource_list: resources to check. If None, will use shared_resources() +
    local_resources().
   name: name for the resource-checking op.

  Returns:
   Tensor containing names of the handles of all resources which have not
   yet been initialized.

  """
  if resource_list is None:
    resource_list = shared_resources() + local_resources()
  with ops.name_scope(name):
    # Run all operations on CPU
    local_device = os.environ.get(
        "TF_DEVICE_FOR_UNINITIALIZED_VARIABLE_REPORTING", "/cpu:0")
    with ops.device(local_device):
      if not resource_list:
        # Return an empty tensor so we only need to check for returned tensor
        # size being 0 as an indication of model ready.
        return array_ops.constant([], dtype=dtypes.string)
      # Get a 1-D boolean tensor listing whether each resource is initialized.
      variables_mask = math_ops.logical_not(
          array_ops.stack([r.is_initialized for r in resource_list]))
      # Get a 1-D string tensor containing all the resource names.
      variable_names_tensor = array_ops.constant(
          [s.handle.name for s in resource_list])
      # Return a 1-D tensor containing all the names of uninitialized resources.
      return array_ops.boolean_mask(variable_names_tensor, variables_mask)


@tf_should_use.should_use_result
def initialize_resources(resource_list, name="init"):
  """Initializes the resources in the given list.

  Args:
   resource_list: list of resources to initialize.
   name: name of the initialization op.

  Returns:
   op responsible for initializing all resources.
  """
  if resource_list:
    return control_flow_ops.group(*[r.create for r in resource_list], name=name)
  return control_flow_ops.no_op(name=name)
