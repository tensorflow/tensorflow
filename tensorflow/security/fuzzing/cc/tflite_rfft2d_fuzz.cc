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
// inputs are untrusted but the model is benign). This target covers RFFT2D
// with a runtime fft_length.
//
// The model is a fixed, clean, verified flatbuffer (embedded in
// model_bytes_rfft2d.h): RFFT2D(input [1,4,4] f32, fft_length int32[2]) ->
// output complex64; both `input` and `fft_length` are graph inputs. The fuzzer
// controls ONLY the runtime input VALUES.
// The fuzzer controls ONLY the runtime input VALUES (`input` 16 x f32 and
// `fft_length` int32[2], both graph inputs).

#include <cstdint>
#include <cstring>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/security/fuzzing/cc/model_bytes_rfft2d.h"

namespace {

void FuzzTfliteRfft2d(const std::vector<int32_t>& fft_length,
                      const std::vector<uint32_t>& input_bits) {
  static const tflite::FlatBufferModel* const model = []() {
    return tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
               reinterpret_cast<const char*>(kCleanRfft2dRtlenModel),
               kCleanRfft2dRtlenModelLen)
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
  TfLiteTensor* fft_length_tensor = nullptr;
  for (int idx : interpreter->inputs()) {
    TfLiteTensor* t = interpreter->tensor(idx);
    if (!t || !t->name) continue;
    if (std::strcmp(t->name, "input") == 0) input_tensor = t;
    if (std::strcmp(t->name, "fft_length") == 0) fft_length_tensor = t;
  }
  if (!input_tensor || !fft_length_tensor) return;
  if (fft_length_tensor->type != kTfLiteInt32 ||
      fft_length_tensor->dims == nullptr ||
      fft_length_tensor->dims->size != 1 ||
      fft_length_tensor->dims->data[0] != 2) {
    return;
  }

  // Fill the f32 `input` ([1,4,4] -> 16 floats) from fuzz bit patterns
  // (default 0.0f). Content is not the trigger; fft_length is.
  if (input_tensor->type == kTfLiteFloat32 && input_tensor->data.f) {
    for (int i = 0; i < 16; ++i) {
      uint32_t bits =
          i < static_cast<int>(input_bits.size()) ? input_bits[i] : 0;
      float v;
      std::memcpy(&v, &bits, sizeof(v));
      input_tensor->data.f[i] = v;
    }
  }

  // Fill `fft_length` (2 x int32) from fuzz values; default is the benign
  // model-matching [4, 4].
  int32_t* fft_length_data = fft_length_tensor->data.i32;
  if (!fft_length_data) return;
  const int32_t kDefaultFftLength[2] = {4, 4};
  for (int i = 0; i < 2; ++i) {
    fft_length_data[i] = i < static_cast<int>(fft_length.size())
                             ? fft_length[i]
                             : kDefaultFftLength[i];
  }

  interpreter->Invoke();
}

FUZZ_TEST(TfliteRfft2d, FuzzTfliteRfft2d)
    .WithDomains(fuzztest::VectorOf(fuzztest::Arbitrary<int32_t>()),
                 fuzztest::VectorOf(fuzztest::Arbitrary<uint32_t>()));

}  // namespace
