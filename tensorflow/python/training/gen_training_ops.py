# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Python wrappers for training ops."""
# NOTE(allenl): The generated op wrappers for training ops were originally in
# training/gen_training_ops.py. They moved to ops/gen_training_ops.py when
# training/ became a module, and this is an alias to avoid breaking existing
# imports.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_training_ops import *
# pylint: enable=wildcard-import
