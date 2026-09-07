# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
#
# On macOS, Apple ld64 uses this file with -exported_symbols_list.
# A single '*' exports all global symbols, matching the 'global: *;'
# behavior of tf_framework_version_script.lds on Linux.
#
# This allows Python C++ extensions (_pywrap_*.so) and dynamic op plugins
# to resolve internal TensorFlow runtime symbols at load time.
#
# Symbols listed in tf_private_symbols.lds remain unexported via
# -unexported_symbols_list.
*
