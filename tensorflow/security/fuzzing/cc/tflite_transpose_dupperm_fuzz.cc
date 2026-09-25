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
// inputs are untrusted but the model is benign). This target covers TRANSPOSE
// with an int4 (sub-byte) tensor and a runtime permutation.
//
// The model is a fixed, clean, verified flatbuffer (embedded in
// model_bytes_transpose_dupperm.h): TRANSPOSE(input [8,1] int4, perm int32[2])
// -> output int4; both `input` and `perm` are graph inputs. The fuzzer
// controls ONLY the runtime input VALUES.

#include <cstdint>
#include <cstring>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/security/fuzzing/cc/model_bytes_transpose_dupperm.h"

namespace {

void FuzzTfliteTransposeDupPerm(const std::vector<int32_t>& perm,
                                const std::vector<uint8_t>& input_bytes) {
  static const tflite::FlatBufferModel* const model = []() {
    return tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
               reinterpret_cast<const char*>(kCleanTransposeInt4DupPermModel),
               kCleanTransposeInt4DupPermModelLen)
        .release();
  }();
  if (!model) return;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder builder(*model, resolver);
  if (builder(&interpreter) != kTfLiteOk || !interpreter) return;
  if (interpreter->AllocateTensors() != kTfLiteOk) return;

  // Resolve graph inputs by name, so the target does not depend on subgraph
  // input ordering.
  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* perm_tensor = nullptr;
  for (int idx : interpreter->inputs()) {
    TfLiteTensor* t = interpreter->tensor(idx);
    if (!t || !t->name) continue;
    if (std::strcmp(t->name, "input") == 0) input_tensor = t;
    if (std::strcmp(t->name, "perm") == 0) perm_tensor = t;
  }
  if (!input_tensor || !perm_tensor) return;
  if (perm_tensor->type != kTfLiteInt32 || perm_tensor->dims == nullptr ||
      perm_tensor->dims->size != 1 || perm_tensor->dims->data[0] != 2) {
    return;
  }

  // Fill the packed int4 `input` ([8,1] -> 4 bytes) from fuzz bytes
  // (default 0). Content is not the trigger; the permutation is.
  if (input_tensor->data.raw) {
    for (size_t i = 0; i < input_tensor->bytes; ++i) {
      input_tensor->data.raw[i] =
          i < input_bytes.size() ? input_bytes[i] : 0;
    }
  }

  // Fill `perm` (2 x int32) from fuzz values; default is the benign [1, 0].
  int32_t* perm_data = perm_tensor->data.i32;
  if (!perm_data) return;
  const int32_t kDefaultPerm[2] = {1, 0};
  for (int i = 0; i < 2; ++i) {
    perm_data[i] =
        i < static_cast<int>(perm.size()) ? perm[i] : kDefaultPerm[i];
  }

  interpreter->Invoke();
}

FUZZ_TEST(TfliteTransposeDupPerm, FuzzTfliteTransposeDupPerm)
    .WithDomains(fuzztest::VectorOf(fuzztest::Arbitrary<int32_t>()),
                 fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()));

}  // namespace
