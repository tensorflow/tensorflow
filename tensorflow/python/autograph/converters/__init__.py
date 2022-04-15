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
"""Code converters used by Autograph."""

# Naming conventions:
#  * each converter should specialize on a single idiom; be consistent with
#    the Python reference for naming
#  * all converters inherit core.converter.Base
#  * module names describe the idiom that the converter covers, plural
#  * the converter class is named consistent with the module, singular and
#    includes the word Transformer
#
# Example:
#
#   lists.py
#     class ListTransformer(converter.Base)
