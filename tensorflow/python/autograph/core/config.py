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
"""Global configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph import utils


PYTHON_LITERALS = {
    'None': None,
    'False': False,
    'True': True,
    'float': float,
}


def internal_module_name(name):
  full_name = utils.__name__
  name_start = full_name.find(name)
  name_end = name_start + len(name) + 1
  return full_name[:name_end]


DEFAULT_UNCOMPILED_MODULES = set(((internal_module_name('tensorflow'),),))

COMPILED_IMPORT_STATEMENTS = (
    'from __future__ import print_function',
)
