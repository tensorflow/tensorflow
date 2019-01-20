# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""All-reduce implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,wildcard-import
from tensorflow.contrib.all_reduce.python.all_reduce import *

from tensorflow.python.util.all_util import remove_undocumented
# pylint: enable=unused-import,line-too-long,wildcard-import

_allowed_symbols = [
    'build_ring_all_reduce',
    'build_recursive_hd_all_reduce',
    'build_shuffle_all_reduce',
    'build_nccl_all_reduce',
    'build_nccl_then_ring',
    'build_nccl_then_recursive_hd',
    'build_nccl_then_shuffle',
    'build_shuffle_then_ring',
    'build_shuffle_then_shuffle'
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
