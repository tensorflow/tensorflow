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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_DW7X7_CONV2TO6_CONCAT_CONV8TO8_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_DW7X7_CONV2TO6_CONCAT_CONV8TO8_H_

#include <map>
#include <set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/subgraph.h"

namespace tflite {
namespace gpu {

// Input: 1 tensors with 2 channels.
// Output: 2 tensors
//   1 - 8 channels, 2x smaller in XY than input
//   2 - 8 channels, 2x smaller in XY than input
// This operation replace folowing sequence of operations:
//   - Depthwise with kernel 7x7 (input -> interm_1)
//   - MaxPooling 2x2 (interm_0 -> interm_2)
//   - Convolution 1x1 2 to 6 channels (+PReLU) (interm_1 -> interm_3)
//   - Concat (interm_3, interm_2 -> output_0)
//   - Convolution 1x1 8 to 8 channels (+PReLU) (output_0 -> output_1)
//
// Limitations:
//  Requires extension cl_qcom_accelerated_image_ops.

bool IsDW7x7Conv2To6ConcatConv8to8Supported(const GpuInfo& gpu_info);

GPUOperation CreateDW7x7Conv2To6ConcatConv8to8(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& dw_attr,
    const Convolution2DAttributes& conv2to6, const PReLUAttributes& prelu0,
    const Convolution2DAttributes& conv8to8, const PReLUAttributes& prelu1);

absl::Status TryDW7x7Conv2To6ConcatConv8to8(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_DW7X7_CONV2TO6_CONCAT_CONV8TO8_H_
