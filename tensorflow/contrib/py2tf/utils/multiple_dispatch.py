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
"""Utilities for type-dependent behavior used in py2tf-generated code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.utils.type_check import is_tensor
from tensorflow.python.ops import control_flow_ops


def run_cond(condition, true_fn, false_fn):
  if is_tensor(condition):
    return control_flow_ops.cond(condition, true_fn, false_fn)
  else:
    return py_cond(condition, true_fn, false_fn)


def py_cond(condition, true_fn, false_fn):
  if condition:
    return true_fn()
  else:
    return false_fn()


def run_while(cond_fn, body_fn, init_args):
  if not isinstance(init_args, (tuple, list)) or not init_args:
    raise ValueError(
        'init_args must be a non-empty list or tuple, found %s' % init_args)

  if is_tensor(*init_args):
    return control_flow_ops.while_loop(cond_fn, body_fn, init_args)
  else:
    return py_while_loop(cond_fn, body_fn, init_args)


def py_while_loop(cond_fn, body_fn, init_args):
  state = init_args
  while cond_fn(*state):
    state = body_fn(*state)
  return state
