# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""tf.llm — One-line LLM loading and fine-tuning for TensorFlow.

Public API:
    tf.llm.from_pretrained(model_name, ...)  →  TFLLMModel
    model.compile(optimizer, loss, lora_config)
    model.fit(dataset, epochs)
    model.generate(prompt, max_new_tokens)
    model.save_pretrained(directory)
    model.summary()
"""

from tensorflow.python.llm.llm_model import from_pretrained
from tensorflow.python.llm.llm_model import TFLLMModel

__all__ = [
    "from_pretrained",
    "TFLLMModel",
]

