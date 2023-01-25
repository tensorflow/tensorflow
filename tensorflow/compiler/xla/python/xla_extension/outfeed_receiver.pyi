# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import enum
from typing import Any, Sequence

from tensorflow.compiler.xla.python import xla_extension

Client = xla_extension.Client
XlaBuilder = xla_extension.XlaBuilder
XlaOp = xla_extension.XlaOp

_CallbackToPython = Any

def start(
    callback_to_python: _CallbackToPython,
    backends: Sequence[Client],
    max_queue_size_bytes: int = ...) -> OutfeedReceiverForPython: ...

class OutfeedReceiverForPython:
  def add_outfeed(
      builder: XlaBuilder,
      token: XlaOp,
      consumer_id: int,
      arrays: Sequence[XlaOp],
      device_idx: int) -> XlaOp: ...
