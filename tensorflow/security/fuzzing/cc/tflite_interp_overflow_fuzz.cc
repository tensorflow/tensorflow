/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// FuzzTest coverage for TFLite model load, interpreter build, and invoke.
// Models enter through FlatBufferModel::VerifyAndBuildFromBuffer, so inputs
// that fail the flatbuffers structural check are rejected at ingress, as they
// would be in a deployment that verifies models. That check does not cover the
// uint64 external-offset fields (Buffer.offset/size and
// Operator.large_custom_options_offset/size), which point past the flatbuffer
// into the appended data, so the bounds checks on them in
// InterpreterBuilder::ParseNodes and ParseTensors are exercised directly. The
// OSS-Fuzz tensorflow build has no TFLite target that reaches this path.

#include <cstring>
#include <memory>
#include <string>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace {

void FuzzModelBuildAndInvoke(const std::string& model_bytes) {
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      model_bytes.data(), model_bytes.size());
  if (model == nullptr) return;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    return;
  }
  if (interpreter == nullptr) return;
  if (interpreter->AllocateTensors() != kTfLiteOk) return;

  // Zero-initialize runtime inputs so MSan does not flag uninitialized reads.
  // An input a crafted model backs with a constant buffer is kTfLiteMmapRo and
  // points into the read-only model mapping; it is already initialized and
  // writing to it would fault, so leave those alone.
  for (int input_index : interpreter->inputs()) {
    TfLiteTensor* tensor = interpreter->tensor(input_index);
    if (tensor != nullptr && tensor->data.raw != nullptr &&
        tensor->allocation_type != kTfLiteMmapRo) {
      std::memset(tensor->data.raw, 0, tensor->bytes);
    }
  }

  interpreter->Invoke();
}
FUZZ_TEST(TfliteInterpreterBuilder, FuzzModelBuildAndInvoke);

}  // namespace
