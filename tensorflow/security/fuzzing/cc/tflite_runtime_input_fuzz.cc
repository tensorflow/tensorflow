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

// Fuzz target: TFLite interpreter, clean verified model + fuzzed RUNTIME input
// values. Exercises the kernel input-validation path for ops that consume
// runtime input values as sizes/indices (the threat model where inference
// inputs are untrusted but the model is benign).
//
// The model is a fixed, clean, verified flatbuffer (embedded in model_bytes.h).
// The fuzzer controls ONLY the runtime input VALUES.

#include <cstdint>
#include <string>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/security/fuzzing/cc/model_bytes.h"

namespace {

void FuzzTfliteRuntimeInput(const std::vector<int32_t>& seq_lengths,
                            const std::vector<float>& input_vals) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
          reinterpret_cast<const char*>(kCleanRevSeqModel),
          kCleanRevSeqModelLen);
  if (!model) return;  // Model is fixed/clean, but guard anyway.

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder builder(*model, resolver);
  if (builder(&interpreter) != kTfLiteOk || !interpreter) return;
  if (interpreter->AllocateTensors() != kTfLiteOk) return;

  // Fill runtime inputs from fuzz values, resolved by tensor name so the
  // target does not depend on subgraph input ordering.
  for (size_t i = 0; i < interpreter->inputs().size(); ++i) {
    const TfLiteTensor* tensor = interpreter->tensor(interpreter->inputs()[i]);
    if (tensor == nullptr || tensor->name == nullptr) continue;
    const std::string name = tensor->name;
    if (name == "input" && tensor->type == kTfLiteFloat32) {
      // Data input (2x4 f32); default 1.0f if the fuzzer gave fewer values.
      float* data = interpreter->typed_input_tensor<float>(i);
      if (data == nullptr) return;
      const size_t n = tensor->bytes / sizeof(float);
      for (size_t j = 0; j < n; ++j) {
        data[j] = j < input_vals.size() ? input_vals[j] : 1.0f;
      }
    } else if (name == "seq_lengths" && tensor->type == kTfLiteInt32) {
      // seq_lengths (2 x int32); default 4 if the fuzzer gave fewer values.
      int32_t* data = interpreter->typed_input_tensor<int32_t>(i);
      if (data == nullptr) return;
      const size_t n = tensor->bytes / sizeof(int32_t);
      for (size_t j = 0; j < n; ++j) {
        data[j] = j < seq_lengths.size() ? seq_lengths[j] : 4;
      }
    }
  }

  interpreter->Invoke();
}

FUZZ_TEST(TfliteRuntimeInput, FuzzTfliteRuntimeInput)
    .WithDomains(fuzztest::VectorOf(fuzztest::Arbitrary<int32_t>()),
                 fuzztest::VectorOf(fuzztest::Arbitrary<float>()));

}  // namespace
