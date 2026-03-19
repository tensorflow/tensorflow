# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Helper functions for TPU Embedding Contexts."""

import threading
from absl import logging


class _EmbeddingPipeliningState(threading.local):

  def __init__(self):
    super().__init__()
    self.enabled = True

embedding_pipelining_state = _EmbeddingPipeliningState()


class SequentialEmbeddingContext:
  """Disables embedding pipelining for all ops created in the scope."""

  def __init__(self):
    self._original_embedding_pipelining_state_enabled = (
        embedding_pipelining_state.enabled
    )

  def __enter__(self):
    embedding_pipelining_state.enabled = False
    logging.info("Entering SequentialEmbeddingContext.")

  def __exit__(self, exc_type, exc_val, exc_tb):
    embedding_pipelining_state.enabled = (
        self._original_embedding_pipelining_state_enabled
    )
    logging.info("Exiting SequentialEmbeddingContext.")
