# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Interface that provides access to Keras dependencies.

This library is a common interface that contains Keras functions needed by
TensorFlow and TensorFlow Lite and is required as per the dependency inversion
principle (https://en.wikipedia.org/wiki/Dependency_inversion_principle). As per
this principle, high-level modules (eg: TensorFlow and TensorFlow Lite) should
not depend on low-level modules (eg: Keras) and instead both should depend on a
common interface such as this file.
"""


from tensorflow.python.util.tf_export import tf_export

_KERAS_CALL_CONTEXT_FUNCTION = None
_KERAS_CLEAR_SESSION_FUNCTION = None
_KERAS_GET_SESSION_FUNCTION = None
_KERAS_LOAD_MODEL_FUNCTION = None

# TODO(b/169898786): Use the Keras public API when TFLite moves out of TF


# Register functions
@tf_export('__internal__.register_call_context_function', v1=[])
def register_call_context_function(func):
  global _KERAS_CALL_CONTEXT_FUNCTION
  _KERAS_CALL_CONTEXT_FUNCTION = func


@tf_export('__internal__.register_clear_session_function', v1=[])
def register_clear_session_function(func):
  global _KERAS_CLEAR_SESSION_FUNCTION
  _KERAS_CLEAR_SESSION_FUNCTION = func


@tf_export('__internal__.register_get_session_function', v1=[])
def register_get_session_function(func):
  global _KERAS_GET_SESSION_FUNCTION
  _KERAS_GET_SESSION_FUNCTION = func


@tf_export('__internal__.register_load_model_function', v1=[])
def register_load_model_function(func):
  global _KERAS_LOAD_MODEL_FUNCTION
  _KERAS_LOAD_MODEL_FUNCTION = func


# Get functions
def get_call_context_function():
  global _KERAS_CALL_CONTEXT_FUNCTION
  return _KERAS_CALL_CONTEXT_FUNCTION


def get_clear_session_function():
  global _KERAS_CLEAR_SESSION_FUNCTION
  return _KERAS_CLEAR_SESSION_FUNCTION


def get_get_session_function():
  global _KERAS_GET_SESSION_FUNCTION
  return _KERAS_GET_SESSION_FUNCTION


def get_load_model_function():
  global _KERAS_LOAD_MODEL_FUNCTION
  return _KERAS_LOAD_MODEL_FUNCTION
