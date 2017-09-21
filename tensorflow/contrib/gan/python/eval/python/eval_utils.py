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
"""Utility file for visualizing generated images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan.python.eval.python import eval_utils_impl
# pylint: disable=wildcard-import
from tensorflow.contrib.gan.python.eval.python.eval_utils_impl import *
# pylint: enable=wildcard-import
from tensorflow.python.util.all_util import remove_undocumented

__all__ = eval_utils_impl.__all__
remove_undocumented(__name__, __all__)
