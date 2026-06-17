/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
/// \file
///
/// Deserialization infrastructure for tflite. Provides functionality
/// to go from a serialized tflite model in flatbuffer format to an
/// in-memory representation of the model.
///
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include "third_party/tensorflow/lite/model_builder.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.

#ifndef TENSORFLOW_LITE_CORE_MODEL_BUILDER_H_
#define TENSORFLOW_LITE_CORE_MODEL_BUILDER_H_

#include <stddef.h>

#include <memory>

#include "tensorflow/compiler/mlir/lite/core/model_builder_base.h"  // IWYU pragma: export
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {

namespace impl {

class FlatBufferModel : public FlatBufferModelBase<FlatBufferModel> {
 public:
   using Ptr = std::unique_ptr<FlatBufferModel>;

  // Use stderr_reporter as the default error reporter.
  static ErrorReporter* GetDefaultErrorReporter() {
    return DefaultErrorReporter();
  }

  // Inherit all constructors from FlatBufferModelBase since inherited factory
  // methods refer to them.
  using FlatBufferModelBase<FlatBufferModel>::FlatBufferModelBase;
};

}  // namespace impl

using FlatBufferModel = impl::FlatBufferModel;

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_MODEL_BUILDER_H_
