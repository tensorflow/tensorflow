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
"""Keras data preprocessing utils for text data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras._impl.keras.preprocessing.text import hashing_trick
from tensorflow.python.keras._impl.keras.preprocessing.text import one_hot
from tensorflow.python.keras._impl.keras.preprocessing.text import text_to_word_sequence
from tensorflow.python.keras._impl.keras.preprocessing.text import Tokenizer

del absolute_import
del division
del print_function
