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
#ifndef TENSORFLOW_LITE_C_C_API_INTERNAL_H_
#define TENSORFLOW_LITE_C_C_API_INTERNAL_H_

#include <stdarg.h>

#include <memory>
#include <vector>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/mutable_op_resolver.h"

// Internal structures used by the C API. These are likely to change and should
// not be depended on directly by any C API clients.
//
// NOTE: This header does not follow C conventions and does not define a C API.
// It is effectively an (internal) implementation detail of the C API.

struct TfLiteModel {
  // Sharing is safe as FlatBufferModel is const.
  std::shared_ptr<const tflite::FlatBufferModel> impl;
};

struct TfLiteInterpreterOptions {
  enum {
    kDefaultNumThreads = -1,
  };
  int num_threads = kDefaultNumThreads;

  tflite::MutableOpResolver op_resolver;

  void (*error_reporter)(void* user_data, const char* format,
                         va_list args) = nullptr;
  void* error_reporter_user_data = nullptr;

  std::vector<TfLiteDelegate*> delegates;
};

struct TfLiteInterpreter {
  // Taking a reference to the (const) model data avoids lifetime-related issues
  // and complexity with the TfLiteModel's existence.
  std::shared_ptr<const tflite::FlatBufferModel> model;

  // The interpreter does not take ownership of the provided ErrorReporter
  // instance, so we ensure its validity here. Note that the interpreter may use
  // the reporter in its destructor, so it should be declared first.
  std::unique_ptr<tflite::ErrorReporter> optional_error_reporter;

  std::unique_ptr<tflite::Interpreter> impl;
};

#endif  // TENSORFLOW_LITE_C_C_API_INTERNAL_H_
