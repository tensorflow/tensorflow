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
"""Utility module that contains APIs usable in the generated code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.utils.builtins import dynamic_builtin
from tensorflow.contrib.py2tf.utils.builtins import dynamic_print
from tensorflow.contrib.py2tf.utils.builtins import dynamic_range
from tensorflow.contrib.py2tf.utils.context_managers import control_dependency_on_returns
from tensorflow.contrib.py2tf.utils.misc import alias_tensors
from tensorflow.contrib.py2tf.utils.multiple_dispatch import dynamic_is
from tensorflow.contrib.py2tf.utils.multiple_dispatch import dynamic_is_not
from tensorflow.contrib.py2tf.utils.multiple_dispatch import run_cond
from tensorflow.contrib.py2tf.utils.multiple_dispatch import run_while
from tensorflow.contrib.py2tf.utils.py_func import wrap_py_func
from tensorflow.contrib.py2tf.utils.tensor_list import dynamic_list_append
from tensorflow.contrib.py2tf.utils.testing import fake_tf
from tensorflow.contrib.py2tf.utils.type_check import is_tensor
from tensorflow.contrib.py2tf.utils.type_hints import set_element_type
