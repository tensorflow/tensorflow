/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/nnapi_delegate.h"

#include <cassert>

namespace tflite {

NNAPIAllocation::NNAPIAllocation(const char* filename,
                                 ErrorReporter* error_reporter)
    : MMAPAllocation(filename, error_reporter) {
  // The disabled variant should never be created.
  assert(false);
}

NNAPIAllocation::~NNAPIAllocation() {}

NNAPIDelegate::~NNAPIDelegate() {
#define UNUSED_MEMBER(x) (void)(x)
  UNUSED_MEMBER(nn_model_);
  UNUSED_MEMBER(nn_compiled_model_);
  UNUSED_MEMBER(model_status_);
#undef UNUSED_MEMBER
}

TfLiteStatus NNAPIDelegate::BuildGraph(Interpreter* interpreter) {
  return kTfLiteError;
}

TfLiteStatus NNAPIDelegate::Invoke(Interpreter* interpreter) {
  return kTfLiteError;
}

bool NNAPIDelegate::IsSupported() { return false; }

}  // namespace tflite
