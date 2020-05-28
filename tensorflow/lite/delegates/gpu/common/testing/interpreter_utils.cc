/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/testing/interpreter_utils.h"

#include <cstring>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace gpu {
namespace testing {

absl::Status InterpreterInvokeWithOpResolver(
    const ::tflite::Model* model, TfLiteDelegate* delegate,
    const OpResolver& op_resolver, const std::vector<TensorFloat32>& inputs,
    std::vector<TensorFloat32>* outputs) {
  auto interpreter = absl::make_unique<Interpreter>();
  if (InterpreterBuilder(model, op_resolver)(&interpreter) != kTfLiteOk) {
    return absl::InternalError("Unable to create TfLite InterpreterBuilder");
  }
  if (delegate && interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
    return absl::InternalError(
        "Unable to modify TfLite graph with the delegate");
  }
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("Unable to allocate TfLite tensors");
  }
  for (int i = 0; i < inputs.size(); ++i) {
    DCHECK_EQ(interpreter->tensor(interpreter->inputs()[i])->type,
              kTfLiteFloat32);
    float* tflite_data =
        interpreter->typed_tensor<float>(interpreter->inputs()[i]);
    DCHECK_EQ(inputs[i].data.size() * sizeof(float),
              interpreter->tensor(interpreter->inputs()[i])->bytes);
    std::memcpy(tflite_data, inputs[i].data.data(),
                inputs[i].data.size() * sizeof(float));
  }
  if (interpreter->Invoke() != kTfLiteOk) {
    return absl::InternalError("Unable to invoke TfLite interpreter");
  }
  if (!outputs || !outputs->empty()) {
    return absl::InternalError("Invalid outputs pointer");
  }
  outputs->reserve(interpreter->outputs().size());
  for (auto t : interpreter->outputs()) {
    const TfLiteTensor* out_tensor = interpreter->tensor(t);
    TensorFloat32 bhwc;
    bhwc.id = t;
    // TODO(impjdi) Relax this condition to arbitrary batch size.
    if (out_tensor->dims->data[0] != 1) {
      return absl::InternalError("Batch dimension is expected to be 1");
    }
    bhwc.shape.b = out_tensor->dims->data[0];
    switch (out_tensor->dims->size) {
      case 2:
        bhwc.shape.h = 1;
        bhwc.shape.w = 1;
        bhwc.shape.c = out_tensor->dims->data[1];
        break;
      case 3:
        bhwc.shape.h = 1;
        bhwc.shape.w = out_tensor->dims->data[1];
        bhwc.shape.c = out_tensor->dims->data[2];
        break;
      case 4:
        bhwc.shape.h = out_tensor->dims->data[1];
        bhwc.shape.w = out_tensor->dims->data[2];
        bhwc.shape.c = out_tensor->dims->data[3];
        break;
      default:
        return absl::InternalError("Unsupported dimensions size " +
                                   std::to_string(out_tensor->dims->size));
    }
    bhwc.data = std::vector<float>(
        out_tensor->data.f,
        out_tensor->data.f + out_tensor->bytes / sizeof(float));
    outputs->push_back(bhwc);
  }
  return absl::OkStatus();
}

absl::Status InterpreterInvoke(const ::tflite::Model* model,
                               TfLiteDelegate* delegate,
                               const std::vector<TensorFloat32>& inputs,
                               std::vector<TensorFloat32>* outputs) {
  ops::builtin::BuiltinOpResolver builtin_op_resolver;
  return InterpreterInvokeWithOpResolver(model, delegate, builtin_op_resolver,
                                         inputs, outputs);
}

}  // namespace testing
}  // namespace gpu
}  // namespace tflite
