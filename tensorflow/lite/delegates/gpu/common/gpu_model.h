/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_MODEL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_MODEL_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_model_generated.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {

struct GpuNode {
  std::unique_ptr<GPUOperation> gpu_operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  std::string name;

  GpuNode() = default;
  GpuNode(GpuNode&& node) = default;
  GpuNode& operator=(GpuNode&& node) = default;
  GpuNode(const GpuNode&) = delete;
  GpuNode& operator=(const GpuNode&) = delete;
};

struct CreateGpuModelInfo {
  CalculationsPrecision precision;
  TensorStorageType storage_type;
  ModelHints hints;

  // User can require specific layout for some tensors.
  // This will guarantee that tensors with specific ids have exact specified
  // layout.
  // Some restrictions apply:
  //   1) ValueId must be input or output id of GraphFloat32
  //   2) data_type must be equal to DeduceDataTypeFromPrecision(precision);
  //      for example for precision F16, data_type must be FLOAT16
  //   3) Layout must be without Batch dimension if tensor.shape.b == 1
  //      Layout must be with Batch dimension if tensor.shape.b != 1
  // InitFromGraph will fail if gpu can not allocate tensor with requested
  // tensor descriptor
  // WARNING: This is an experimental API and subject to change.
  // IMPORTANT: tensors ids from predefined / external_immutable_tensors /
  // external_mutable_tensors should not intersect.
  absl::flat_hash_map<ValueId, TensorDescriptor> predefined;

  // User can provide immutable external tensors for inference context.
  // Some restrictions apply:
  //   1) ValueId must be input or output id of GraphFloat32
  //   2) Provided ptrs must be valid during life of InferenceContext.
  //   3) data_type must be equal to DeduceDataTypeFromPrecision(precision);
  //      for example for precision F16, data_type must be FLOAT16
  //   4) Layout must be without Batch dimension if tensor.shape.b == 1
  //      Layout must be with Batch dimension if tensor.shape.b != 1
  // InitFromGraph will fail if gpu can not allocate tensor with requested
  // tensor descriptor
  // WARNING: This is an experimental API and subject to change.
  // IMPORTANT: tensors ids from predefined / external_immutable_tensors /
  // external_mutable_tensors should not intersect.
  absl::flat_hash_map<ValueId, GpuSpatialTensor*> external_immutable_tensors;

  // User can provide mutable external tensors for inference context.
  // HINT: Highly recommended to use other options if possible, this options
  // will be with the worst performance.
  // Some restrictions apply:
  //   1) ValueId must be input or output id of GraphFloat32
  //   2) data_type must be equal to DeduceDataTypeFromPrecision(precision);
  //      for example for precision F16, data_type must be FLOAT16
  //   3) Layout must be without Batch dimension if tensor.shape.b == 1
  //      Layout must be with Batch dimension if tensor.shape.b != 1
  // InitFromGraph will fail if gpu can not allocate tensor with requested
  // tensor descriptor
  // WARNING: This is an experimental API and subject to change.
  // IMPORTANT: tensors ids from predefined / external_immutable_tensors /
  // external_mutable_tensors should not intersect.
  absl::flat_hash_map<ValueId, TensorDescriptor> external_mutable_tensors;
};

struct GpuModel {
  std::vector<std::pair<ValueId, ValueId>> input_ids_and_refs;
  std::vector<std::pair<ValueId, ValueId>> variable_ids_and_refs;
  std::vector<std::pair<ValueId, ValueId>> output_ids_and_refs;
  std::vector<GpuNode> nodes;
  absl::flat_hash_map<ValueId, TensorDescriptor> tensors;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors;
};

absl::Status GraphToGpuModel(const GraphFloat32& graph,
                             const CreateGpuModelInfo& create_info,
                             const GpuInfo& gpu_info, GpuModel* gpu_model);

flatbuffers::Offset<data::GpuModel> Encode(
    const GpuModel& gpu_model, flatbuffers::FlatBufferBuilder* builder);

absl::Status Decode(const data::GpuModel* fb_gpu_model, GpuModel* gpu_model);

// This transformations MUST be applied to graph for correct work of GpuModel
// that will be created from graph
absl::Status RunGraphTransformsForGpuModel(GraphFloat32* graph);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_MODEL_H_
