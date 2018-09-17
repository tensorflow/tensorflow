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
#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_C_C_API_INTERNAL_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_C_C_API_INTERNAL_H_

#include "tensorflow/contrib/lite/experimental/c/c_api.h"

#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"

// Internal structures used by the C API. These are likely to change and should
// not be depended on.

struct TFL_Model {
  // Sharing is safe as FlatBufferModel is const.
  std::shared_ptr<const tflite::FlatBufferModel> impl;
};

struct TFL_InterpreterOptions {
  enum {
    kDefaultNumThreads = -1,
  };
  int num_threads = kDefaultNumThreads;
};

struct TFL_Interpreter {
  // Taking a reference to the (const) model data avoids lifetime-related issues
  // and complexity with the TFL_Model's existence.
  std::shared_ptr<const tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> impl;
};

#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_C_C_API_INTERNAL_H_
