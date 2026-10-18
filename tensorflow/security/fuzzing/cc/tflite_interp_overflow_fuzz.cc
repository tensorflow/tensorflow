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

#include <cstdarg>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace {

// Nearly every fuzz input is malformed, and the default reporter writes each
// rejection to stderr, which would dominate the run. The model and, through
// the model, the interpreter builder report into this instead.
class NullErrorReporter : public tflite::ErrorReporter {
 public:
  int Report(const char* /*format*/, va_list /*args*/) override { return 0; }
};

// A valid model to start from, so the fuzzer does not have to rediscover the
// flatbuffer framing before it reaches InterpreterBuilder and Invoke.
std::vector<std::tuple<std::string>> SeedModels() {
  std::ifstream file(
      tensorflow::GetDataDependencyFilepath("tensorflow/lite/testdata/add.bin"),
      std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
  if (bytes.empty()) return {};
  return {std::make_tuple(std::move(bytes))};
}

// Whether `tensor` is a runtime input that is safe and useful to zero.
bool IsZeroableInput(const TfLiteTensor* tensor) {
  if (tensor == nullptr || tensor->data.raw == nullptr || tensor->bytes == 0) {
    return false;
  }
  // A constant input aliases the read-only model mapping, and a persistent
  // read-only one is owned and filled by the planner; neither is
  // uninitialized, and writing to the mapping would fault.
  if (tensor->allocation_type == kTfLiteMmapRo ||
      tensor->allocation_type == kTfLitePersistentRo) {
    return false;
  }
  // These are not plain byte buffers of `bytes` length.
  if (tensor->type == kTfLiteString || tensor->type == kTfLiteResource ||
      tensor->type == kTfLiteVariant) {
    return false;
  }
  return true;
}

void FuzzModelBuildAndInvoke(const std::string& model_bytes) {
  NullErrorReporter error_reporter;
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      model_bytes.data(), model_bytes.size(), /*extra_verifier=*/nullptr,
      &error_reporter);
  if (model == nullptr) return;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    return;
  }
  if (interpreter == nullptr) return;
  if (interpreter->AllocateTensors() != kTfLiteOk) return;

  // Zero-initialize runtime inputs so MSan does not flag uninitialized reads.
  for (int input_index : interpreter->inputs()) {
    TfLiteTensor* tensor = interpreter->tensor(input_index);
    if (IsZeroableInput(tensor)) {
      std::memset(tensor->data.raw, 0, tensor->bytes);
    }
  }

  interpreter->Invoke();
}
FUZZ_TEST(TfliteInterpreterBuilder, FuzzModelBuildAndInvoke)
    .WithSeeds(SeedModels());

}  // namespace
