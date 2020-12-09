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
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/subgraph.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {

absl::Status SelectDefault(const GpuInfo& gpu_info, const OperationDef& op_def,
                           ModelHints hints, const std::vector<Value*>& inputs,
                           const std::vector<Value*>& outputs, const Node& node,
                           GPUOperationsSubgraph* gpu_subgraph) {
  return absl::UnimplementedError(
      absl::StrCat("No selector for ", node.operation.type));
}

}  // namespace gpu
}  // namespace tflite
