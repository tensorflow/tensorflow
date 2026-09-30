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

// FuzzTest coverage for the TFLite untrusted-model ingress path: model
// verification, model load, interpreter build, and invoke. tflite::Verify() is
// the validation layer for untrusted models, so the target runs it first and
// stops on rejection; everything it accepts goes on to ParseTensors and the
// kernels. This covers the verifier's checks on attacker-controlled model
// metadata (including the uint64 external-offset bounds checks in
// VerifyTensors / VerifyOperators) and the runtime behind them. The OSS-Fuzz
// tensorflow build has no TFLite target that reaches this path.

#include <cstring>
#include <memory>
#include <string>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/verifier.h"

namespace {

void FuzzModelBuildAndInvoke(const std::string& model_bytes) {
  // Supported ingress path: an untrusted model is verified before anything is
  // built from it. Inputs the verifier rejects stop here. The error reporter
  // is optional and would only add stderr noise under the fuzzer.
  if (!tflite::Verify(model_bytes.data(), model_bytes.size(),
                      /*error_reporter=*/nullptr)) {
    return;
  }

  auto model = tflite::FlatBufferModel::BuildFromBuffer(model_bytes.data(),
                                                        model_bytes.size());
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
    if (tensor != nullptr && tensor->data.raw != nullptr) {
      std::memset(tensor->data.raw, 0, tensor->bytes);
    }
  }

  // Runs the kernels on whatever the verifier accepted.
  interpreter->Invoke();
}
FUZZ_TEST(TfliteInterpreterBuilder, FuzzModelBuildAndInvoke);

}  // namespace
