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
"""Miscellaneous utilities that don't fit anywhere else."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import six


def deprecated_py2_support(module_name):
  if six.PY2:
    legacy_module = __import__(module_name + '_deprecated_py2')
    current_module = sys.modules[module_name]
    current_module.__dict__.update({
        k: v
        for k, v in legacy_module.__dict__.items()
        if not k.startswith('__')
    })
