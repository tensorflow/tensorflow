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
"""Experimental impl for gen_math_ops.py using unified APIs, for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework.experimental import _math_ops
from tensorflow.python.framework.experimental import context_stack as context
from tensorflow.python.framework.experimental import gradient_registry
from tensorflow.python.framework.experimental import tape_stack


def add(a, b, name=None):
  ctx = context.get_default()
  tape = tape_stack.get_default()
  grad_registry = gradient_registry.get_global_registry()
  return _math_ops.add(ctx, a, b, name, tape, grad_registry)


def mat_mul(a, b, name=None):
  ctx = context.get_default()
  tape = tape_stack.get_default()
  grad_registry = gradient_registry.get_global_registry()
  return _math_ops.mat_mul(ctx, a, b, name, tape, grad_registry)
