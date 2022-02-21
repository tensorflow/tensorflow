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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_OPERATION_SELECTOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_OPERATION_SELECTOR_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/subgraph.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"

namespace tflite {
namespace gpu {

struct SharedWeightsConvDesc {
  int weights_id;
  WeightsDescription desc;
  std::vector<int> global_const_ids;

  bool operator==(const SharedWeightsConvDesc& t) const {
    return weights_id == t.weights_id && desc == t.desc;
  }

  void RemapIds(const absl::flat_hash_map<int, ValueId>& mapping) {
    for (int i = 0; i < global_const_ids.size(); ++i) {
      int local_id = -(global_const_ids[i] + 1);
      if (local_id >= 0) {
        global_const_ids[i] = mapping.at(local_id);
      }
    }
  }
};

absl::Status GPUOperationFromNode(
    const GpuInfo& gpu_info, const OperationDef& op_def, ModelHints hints,
    const std::vector<Value*>& inputs, const std::vector<Value*>& outputs,
    const Node& node, std::vector<SharedWeightsConvDesc>* shared_conv_weights,
    GPUOperationsSubgraph* gpu_subgraph);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_OPERATION_SELECTOR_H_
