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

// Fuzzes the TensorFlow Lite interpreter end to end: an arbitrary buffer is
// verified as a flatbuffer model, an interpreter is built for it, tensors are
// allocated, inputs are filled deterministically, and the graph is invoked.
//
// This exercises the builtin kernel implementations, the arena planner and the
// shape-propagation paths, none of which previously had OSS-Fuzz coverage.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace fuzzing {
namespace {

// Keep the fuzzer inside the OSS-Fuzz memory budget. Models and arenas larger
// than this are not interesting: they exercise the allocator, not the kernels.
constexpr size_t kMaxModelBytes = 1 << 20;  // 1 MiB
constexpr size_t kMaxArenaBytes = 1 << 26;  // 64 MiB

void FuzzInterpreter(const std::string& model_bytes) {
  if (model_bytes.size() < 8 || model_bytes.size() > kMaxModelBytes) {
    return;
  }

  // VerifyAndBuildFromBuffer applies tflite::VerifyModelBuffer first, so
  // structurally invalid buffers are rejected cheaply.
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::VerifyAndBuildFromBuffer(model_bytes.data(),
                                                model_bytes.size());
  if (model == nullptr) {
    return;
  }

  // Delegates are excluded so the fuzzer exercises the reference and optimized
  // CPU kernels rather than a delegate's own implementation.
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  std::unique_ptr<Interpreter> interpreter;
  if (InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk ||
      interpreter == nullptr) {
    return;
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return;
  }

  size_t total_bytes = 0;
  for (const int tensor_index : interpreter->inputs()) {
    TfLiteTensor* tensor = interpreter->tensor(tensor_index);
    if (tensor == nullptr || tensor->data.raw == nullptr) {
      continue;
    }
    // String tensors own a dynamic buffer with its own layout; writing raw
    // bytes into it would corrupt the interpreter rather than the kernel under
    // test.
    if (tensor->type == kTfLiteString || tensor->type == kTfLiteResource ||
        tensor->type == kTfLiteVariant) {
      return;
    }
    // Check before accumulating so the sum itself cannot wrap.
    if (tensor->bytes > kMaxArenaBytes ||
        total_bytes > kMaxArenaBytes - tensor->bytes) {
      return;
    }
    total_bytes += tensor->bytes;
    std::memset(tensor->data.raw, 1, tensor->bytes);
  }

  interpreter->Invoke();
}
FUZZ_TEST(TfLiteFuzz, FuzzInterpreter);

}  // namespace
}  // namespace fuzzing
}  // namespace tflite
