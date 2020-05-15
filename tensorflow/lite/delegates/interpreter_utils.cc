/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/interpreter_utils.h"

namespace tflite {
namespace delegates {
TfLiteStatus InterpreterUtils::InvokeWithCPUFallback(Interpreter* interpreter) {
  TfLiteStatus status = interpreter->Invoke();
  if (status == kTfLiteOk || interpreter->IsCancelled() ||
      !interpreter->HasDelegates()) {
    return status;
  }
  // Retry without delegation.
  // TODO(b/138706191): retry only if error is due to delegation.
  TF_LITE_REPORT_ERROR(
      interpreter->error_reporter(),
      "Invoke() failed in the presence of delegation. Retrying without.");

  // Copy input data to a buffer.
  // Input data is safe since Subgraph::PrepareOpsAndTensors() passes
  // preserve_inputs=true to ArenaPlanner.
  std::vector<char> buf;
  size_t input_size = 0;

  for (auto i : interpreter->inputs()) {
    TfLiteTensor* t = interpreter->tensor(i);
    input_size += t->bytes;
  }
  buf.reserve(input_size);
  auto bufp = buf.begin();
  for (auto i : interpreter->inputs()) {
    // TF_LITE_ENSURE_STATUS(interpreter->EnsureTensorDataIsReadable(i));
    TfLiteTensor* t = interpreter->tensor(i);
    std::copy(t->data.raw, t->data.raw + t->bytes, bufp);
    bufp += t->bytes;
  }

  TF_LITE_ENSURE_STATUS(interpreter->RemoveAllDelegates());

  // Copy inputs from buffer.
  bufp = buf.begin();
  for (auto i : interpreter->inputs()) {
    TfLiteTensor* t = interpreter->tensor(i);
    std::copy(bufp, bufp + t->bytes, t->data.raw);
    bufp += t->bytes;
  }

  // Invoke again.
  TF_LITE_ENSURE_STATUS(interpreter->Invoke());
  return kTfLiteDelegateError;
}

}  // namespace delegates
}  // namespace tflite
