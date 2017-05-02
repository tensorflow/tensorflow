## Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

# Configuration for a plugin
#
# Copy this file to plugin_config.py, and insert plugin specific
# information.  The unit test framework will then include a backend
# which tests the device and types named below.
#
# The loader value should name a python module that is used to load
# the device.
#
# See //tensorflow/compiler/tests/plugin/BUILD.tpl

device = "XLA_MY_DEVICE"
types = "DT_FLOAT,DT_INT32"

