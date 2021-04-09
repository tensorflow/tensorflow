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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

// Validates which operations are supported and returns array of operations to
// replace with GPU kernels. The caller must free the pointer on TfLiteIntArray.
// 'max_delegated_partitions' limits the maximum number of partitions to
// delegate as a graph could possibly have multiple partitions (each partition
// consists of a subset of ops) to be replaced.
TfLiteIntArray* GetOpsToReplace(TfLiteContext* context,
                                bool allow_quant_ops = false,
                                int max_delegated_partitions = 1);

// Extracts TFLite delegate execution plan from the input TFLite context and
// converts it into generic graph format.
//
// If model is quantized, quant_conversion_map maps the dequantized tensor
// (floating-point) to the original tensor (fixed-point) & vice-versa.
// NOTE: Not all of these new tensors will have any data and need memory
// allocated for them. We need to do that only for the overall GPU graph inputs
// & outputs. This should be done by the delegate, by setting the appropriate
// TfLiteNode->temporaries.
absl::Status BuildModel(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    GraphFloat32* graph,
    absl::flat_hash_map<int, int>* quant_conversion_map = nullptr);

// Same as BuildModel, but enforces user-provided input/output indices instead
// of using delegate_params->inputs and delegate_params->outputs for
// inputs/outputs preallocating.
absl::Status BuildModelEnforceIO(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const std::vector<int>& input_ids, const std::vector<int>& output_ids,
    GraphFloat32* graph,
    absl::flat_hash_map<int, int>* quant_conversion_map = nullptr);

// Same as above but also apply all transformations on the final graph.
// Prefer using this method instead of BuildModel.
//
// If model is quantized, quant_conversion_map maps the dequantized tensor
// (floating-point) to the original TFLite tensor (fixed-point) & vice-versa.
// NOTE: Not all of these new tensors will have any data and need memory
// allocated for them. We need to do that only for the overall GPU graph inputs
// & outputs. This should be done by the delegate, by setting the appropriate
// TfLiteNode->temporaries.
absl::Status BuildFinalModel(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    GraphFloat32* graph,
    absl::flat_hash_map<int, int>* quant_conversion_map = nullptr);

// Module-internal converter, exposed for unit testing purpose only.
absl::Status ConvertTfLiteTensorToTensorRef(const TfLiteTensor& tflite_tensor,
                                            TensorRef<BHWC>* tensor_ref);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_H_
