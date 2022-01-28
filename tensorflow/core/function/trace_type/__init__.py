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
"""Tracing Protocol for tf.function.

TODO(b/202447704): Briefly describe the tracing, retracing, and how trace types
control it.
"""


from tensorflow.core.function.trace_type.signature_builder import make_function_signature
from tensorflow.core.function.trace_type.signature_builder import SignatureContext
from tensorflow.core.function.trace_type.signature_builder import WeakrefDeletionObserver

