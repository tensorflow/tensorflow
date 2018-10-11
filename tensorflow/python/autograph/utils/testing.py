# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Testing utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


def fake_tf():
  """Creates a fake module that looks like TensorFlow, for testing."""
  mod = imp.new_module('tensorflow')
  mod_contents = dict()
  mod_contents.update(gen_math_ops.__dict__)
  mod_contents.update(math_ops.__dict__)
  mod_contents.update(ops.__dict__)
  mod_contents.update(mod.__dict__)
  mod.__dict__.update(mod_contents)
  return mod
