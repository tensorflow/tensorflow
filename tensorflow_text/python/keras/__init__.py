# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tensorflow Text Layers for Keras API."""

from tensorflow.python.util.all_util import remove_undocumented

# pylint: disable=wildcard-import
from tensorflow_text.python.keras.layers import *

# Public symbols in the "tensorflow_text.layers" package.
_allowed_symbols = [
    "layers",
]

remove_undocumented(__name__, _allowed_symbols)
