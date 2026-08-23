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
// inputs are untrusted but the model is benign). This target covers
// READ_VARIABLE with a runtime-shaped resource variable.
//
// The model is a fixed, clean, verified flatbuffer (embedded in
// model_bytes_readvar.h): VAR_HANDLE -> handle; FILL(dims, value) -> dyn;
// ASSIGN_VARIABLE(handle, dyn); READ_VARIABLE(handle) -> out (static [1,4096]
// f32). The fuzzer controls ONLY the runtime input VALUES (`dims` int32[2]
// and `value` f32 scalar, both graph inputs).

#include <cstdint>
#include <cstring>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/security/fuzzing/cc/model_bytes_readvar.h"

namespace {

void FuzzTfliteReadvar(const std::vector<int32_t>& dims, uint32_t value_bits) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
          reinterpret_cast<const char*>(kCleanReadvarShrinkModel),
          kCleanReadvarShrinkModelLen);
  if (!model) return;  // Model is fixed/clean, but guard anyway.

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder builder(*model, resolver);
  if (builder(&interpreter) != kTfLiteOk || !interpreter) return;
  if (interpreter->AllocateTensors() != kTfLiteOk) return;

  // Resolve graph inputs by name, so the target does not depend on subgraph
  // input ordering.
  TfLiteTensor* dims_tensor = nullptr;
  TfLiteTensor* value_tensor = nullptr;
  for (int idx : interpreter->inputs()) {
    TfLiteTensor* t = interpreter->tensor(idx);
    if (!t || !t->name) continue;
    if (std::strcmp(t->name, "dims") == 0) dims_tensor = t;
    if (std::strcmp(t->name, "value") == 0) value_tensor = t;
  }
  if (!dims_tensor || !value_tensor) return;
  if (dims_tensor->type != kTfLiteInt32 || dims_tensor->dims == nullptr ||
      dims_tensor->dims->size != 1 || dims_tensor->dims->data[0] != 2) {
    return;
  }

  // Fill `dims` (2 x int32) from fuzz values; default is the benign
  // model-matching [1, 4096] (variable size == output size).
  int32_t* dims_data = dims_tensor->data.i32;
  if (!dims_data) return;
  const int32_t kDefaultDims[2] = {1, 4096};
  for (int i = 0; i < 2; ++i) {
    dims_data[i] =
        i < static_cast<int>(dims.size()) ? dims[i] : kDefaultDims[i];
  }

  // Fill the scalar `value` (f32) from fuzz bits (content is not the bug
  // trigger; it only rides along into the FILL output).
  if (value_tensor->type == kTfLiteFloat32 && value_tensor->data.f) {
    float v;
    std::memcpy(&v, &value_bits, sizeof(v));
    value_tensor->data.f[0] = v;
  }

  interpreter->Invoke();
}

FUZZ_TEST(TfliteReadvar, FuzzTfliteReadvar)
    .WithDomains(fuzztest::VectorOf(fuzztest::Arbitrary<int32_t>()),
                 fuzztest::Arbitrary<uint32_t>());

}  // namespace
