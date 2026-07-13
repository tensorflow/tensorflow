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

// FuzzTest coverage for TFLite model loading, interpreter build, and invoke.
// This exercises interpreter_builder.cc get_readonly_data / ParseTensors, which
// perform an unchecked uint64 addition (offset + size > allocation_->bytes())
// on attacker-controlled external-buffer offsets. On overflow the check is
// bypassed and a wild pointer (allocation_->base() + offset) is used as tensor
// data, which is dereferenced during Invoke. The OSS-Fuzz tensorflow build has
// no TFLite target that reaches this path.

#include <cstring>
#include <memory>
#include <string>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/kernels/register.h"

namespace {

void FuzzModelBuildAndInvoke(const std::string& model_bytes) {
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

  // The unchecked external-offset arithmetic sets a wild read-only tensor
  // pointer that is dereferenced here.
  interpreter->Invoke();
}
FUZZ_TEST(TfliteInterpreterBuilder, FuzzModelBuildAndInvoke);

}  // namespace
