/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SWIG test helper for lib::tensorflow::Status

%include "tensorflow/python/platform/base.i"
%import(module="tensorflow.python.pywrap_tensorflow") "tensorflow/python/lib/core/status.i"

%inline %{
#include "tensorflow/core/lib/core/status.h"

tensorflow::Status NotOkay() {
  return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT, "Testing 1 2 3");
}

tensorflow::Status Okay() {
  return tensorflow::Status();
}
%}
