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

#include "tensorflow/lite/delegates/gpu/common/selectors/special_selector.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/subgraph.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/dw7x7_conv2to6_concat_conv8to8.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/fc_fc_add.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/thin_pointwise_fuser.h"

namespace tflite {
namespace gpu {
absl::Status GPUSubgraphFromGraph(
    const ModelHints& hints, const GpuInfo& gpu_info,
    CalculationsPrecision precision, const GraphFloat32& graph,
    NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryDW7x7Conv2To6ConcatConv8to8(gpu_info, precision, graph, first_node_id,
                                     tensor_descriptors, consumed_nodes,
                                     gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryThinPointwiseFuser(gpu_info, precision, graph, first_node_id,
                            tensor_descriptors, consumed_nodes, gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryFCFCAdd(gpu_info, precision, graph, first_node_id, tensor_descriptors,
                 consumed_nodes, gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if (TryFusedPointwiseConv(graph, first_node_id, precision, tensor_descriptors,
                            consumed_nodes, gpu_subgraph)
          .ok()) {
    gpu_subgraph->operations[0].name = "slice_mul_reduce_concat";
    return absl::OkStatus();
  }
  if (TryMeanStdDevNormalization(gpu_info, precision, graph, first_node_id,
                                 tensor_descriptors, consumed_nodes,
                                 gpu_subgraph)
          .ok()) {
    gpu_subgraph->operations[0].name = "mean_stddev_normalization";
    return absl::OkStatus();
  }
  return absl::NotFoundError("No special combination.");
}

}  // namespace gpu
}  // namespace tflite
