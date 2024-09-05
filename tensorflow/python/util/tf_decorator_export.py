# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Exports functions from tf_decorator.py to avoid cycles."""

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export


make_decorator = tf_export.tf_export(
    '__internal__.decorator.make_decorator', v1=[]
)(tf_decorator.make_decorator)
unwrap = tf_export.tf_export('__internal__.decorator.unwrap', v1=[])(
    tf_decorator.unwrap
)
