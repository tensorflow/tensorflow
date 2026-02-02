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
"""A module containing TensorFlow ops whose API may change in the future."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(ptucker): Add these to tf.contrib.variables?
# pylint: disable=wildcard-import
from tensorflow.contrib.framework.python.ops.arg_scope import *
from tensorflow.contrib.framework.python.ops.embedding_ops import *
from tensorflow.contrib.framework.python.ops.ops import *
from tensorflow.contrib.framework.python.ops.prettyprint_ops import *
from tensorflow.contrib.framework.python.ops.sampling_ops import *
from tensorflow.contrib.framework.python.ops.variables import *
# pylint: enable=wildcard-import
