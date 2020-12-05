# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Text utilities.

`tfds` includes a set of `TextEncoder`s as well as a `Tokenizer` to enable
expressive, performant, and reproducible natural language research.
"""

from tensorflow.data.experimental.core.deprecated.text.subword_text_encoder import SubwordTextEncoder
from tensorflow.data.experimental.core.deprecated.text.text_encoder import ByteTextEncoder
from tensorflow.data.experimental.core.deprecated.text.text_encoder import TextEncoder
from tensorflow.data.experimental.core.deprecated.text.text_encoder import TextEncoderConfig
from tensorflow.data.experimental.core.deprecated.text.text_encoder import Tokenizer
from tensorflow.data.experimental.core.deprecated.text.text_encoder import TokenTextEncoder

__all__ = [
    "ByteTextEncoder",
    "SubwordTextEncoder",
    "TextEncoder",
    "TextEncoderConfig",
    "Tokenizer",
    "TokenTextEncoder",
]
