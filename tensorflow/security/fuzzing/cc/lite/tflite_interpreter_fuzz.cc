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

#include <cstring>
#include <memory>
#include <string_view>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

// Fuzzer for the TFLite Interpreter load/prepare/invoke pipeline.
//
// Attack surface: tflite::FlatBufferModel::BuildFromBuffer ->
// InterpreterBuilder::operator() -> Interpreter::AllocateTensors ->
// Interpreter::Invoke. This is the same path exercised by
// tf.lite.Interpreter(model_content=...) from Python, so any crash here
// is reachable by loading an untrusted .tflite model.
//
// Uses BuildFromBuffer (not VerifyAndBuildFromBuffer) on purpose: the
// FlatBuffer verifier filters out a large class of inputs that TFLite
// consumers are not actually protected against -- many historical
// kernel-level bugs require a flatbuffer that is verifier-valid but
// semantically inconsistent.
//
// Include set and deps match tensorflow/lite/examples/minimal, which is
// the public-facing "how to embed the TFLite runtime" example.

namespace {

void FuzzTFLiteInterpreter(std::string_view data) {
  if (data.empty()) return;

  auto model = tflite::FlatBufferModel::BuildFromBuffer(data.data(),
                                                        data.size());
  if (model == nullptr) return;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) !=
      kTfLiteOk) {
    return;
  }
  if (interpreter == nullptr) return;

  if (interpreter->AllocateTensors() != kTfLiteOk) return;

  // Zero-fill inputs so Invoke has defined source bytes. The bug classes
  // we want to find (OOB R/W, null deref, stack overflow) do not depend
  // on input *content*, only on structural properties of the flatbuffer.
  for (int i : interpreter->inputs()) {
    TfLiteTensor* t = interpreter->tensor(i);
    if (t != nullptr && t->data.raw != nullptr && t->bytes > 0) {
      std::memset(t->data.raw, 0, t->bytes);
    }
  }

  interpreter->Invoke();
}
FUZZ_TEST(CC_FUZZING, FuzzTFLiteInterpreter);

}  // namespace
