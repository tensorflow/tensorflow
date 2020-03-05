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

#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {

// TODO(b/141330728): Move this method elsewhere as part clean up.
void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                     ErrorReporter* error_reporter, TfLiteContext* context) {
  context->tensors_size = tensors_size;
  context->tensors = tensors;
  context->impl_ = static_cast<void*>(error_reporter);
  context->GetExecutionPlan = nullptr;
  context->ResizeTensor = nullptr;
  context->ReportError = ReportOpError;
  context->AddTensors = nullptr;
  context->GetNodeAndRegistration = nullptr;
  context->ReplaceNodeSubsetsWithDelegateKernels = nullptr;
  context->recommended_num_threads = 1;
  context->GetExternalContext = nullptr;
  context->SetExternalContext = nullptr;

  for (int i = 0; i < tensors_size; ++i) {
    if (context->tensors[i].is_variable) {
      ResetVariableTensor(&context->tensors[i]);
    }
  }
}

}  // namespace testing
}  // namespace tflite
