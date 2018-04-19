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
# =============================================================================
"""Import custom op for plugin and register it in plugin factory registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensorrt.custom_plugin_examples.ops import gen_inc_op
from tensorflow.contrib.tensorrt.custom_plugin_examples.plugin_wrap import inc_op_register
from tensorflow.contrib.tensorrt.custom_plugin_examples import inc_op as import_inc_op_so

inc_op = gen_inc_op.inc_plugin_trt
inc_op_register()
