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

#include <memory>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/model_hints.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/subgraph.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

Status SelectDefault(const CreationContext& creation_context,
                     const OperationDef& op_def, ModelHints hints,
                     const std::vector<Value<TensorRef<BHWC>>*>& inputs,
                     const std::vector<Value<TensorRef<BHWC>>*>& outputs,
                     const Node& node, GPUOperationsSubgraph* gpu_subgraph) {
  return UnimplementedError(
      absl::StrCat("No selector for ", node.operation.type));
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
