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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/recordable_queue_builder.h"
#include "tensorflow/lite/delegates/gpu/cl/serialization_generated.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

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

namespace cl {

struct CLNode {
  ClOperation cl_operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  // Mostly for debug purposes.
  std::string name;

  CLNode() = default;

  CLNode(CLNode&& node) = default;
  CLNode& operator=(CLNode&& node) = default;
  CLNode(const CLNode&) = delete;
  CLNode& operator=(const CLNode&) = delete;
};

class InferenceContext {
 public:
  absl::Status InitFromGraph(const CreateGpuModelInfo& create_info,
                             const GraphFloat32& graph, Environment* env,
                             std::vector<uint8_t>* serialized_model = nullptr);

  // Applies OpenCL-specific transformations to the graph before the
  // initialization. These transformations are either impossible or useless in
  // other backends.
  absl::Status InitFromGraphWithTransforms(
      const CreateGpuModelInfo& create_info, GraphFloat32* graph,
      Environment* env, std::vector<uint8_t>* serialized_model = nullptr);

  absl::Status AddToQueue(CLCommandQueue* queue);
  absl::Status Profile(ProfilingCommandQueue* queue, ProfilingInfo* result);
  // for profiling and memory statistics
  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;

  absl::Status SetInputTensor(ValueId id, const TensorFloat32& tensor,
                              CLCommandQueue* queue);

  // It will work only with input/output tensor ids. For all other ids we don't
  // have any guarantees.
  Tensor* GetTensor(ValueId id);

  absl::Status GetOutputTensor(ValueId id, CLCommandQueue* queue,
                               TensorFloat32* result);

  const std::vector<ValueId>& GetInputIds() const { return input_ids_; }
  const std::vector<ValueId>& GetOutputIds() const { return output_ids_; }

  absl::Status RestoreDeserialized(
      const absl::Span<const uint8_t> serialized_model, Environment* env,
      CreateGpuModelInfo* create_info = nullptr);

  // Can be used only with ids from external_mutable_tensors in create_info
  // Must be called after initialization and before execution
  absl::Status SetTensor(const ValueId& tensor_id, Tensor* tensor_ptr);

 private:
  enum class TensorMemoryType {
    kStrongShape,
    kBuffer,
    kVariable,
    kConst,
    kExternal
  };

  friend flatbuffers::Offset<data::InferenceContext> Encode(
      const CLDevice& device, const InferenceContext& inference,
      const ProgramCache& program_cache,
      flatbuffers::Offset<data::GpuModel> gpu_model_fb,
      flatbuffers::FlatBufferBuilder* builder);
  friend absl::Status Decode(const CLContext& context, const CLDevice& device,
                             ProgramCache* program_cache,
                             const data::InferenceContext* fb_inference,
                             InferenceContext* inference);

  void CopyFromGpuModel(GpuModel* gpu_model);

  absl::Status AllocateMemory(const GpuInfo& gpu_info, CLContext* context);

  absl::Status AllocateMemoryForConstTensors(CLContext* context);

  absl::Status AllocateMemoryForVariableTensors(CLContext* context);

  absl::Status AllocateMemoryForBuffers(const GpuInfo& gpu_info,
                                        CLContext* context);

  absl::Status AllocateMemoryForStrongShapes(const GpuInfo& gpu_info,
                                             CLContext* context);

  // utility function
  void GetUsages(const std::function<bool(ValueId)>& functor,
                 std::map<ValueId, int2>* usages);

  TensorMemoryType GetTensorMemoryType(const GpuInfo& gpu_info, ValueId id);

  void BindMemoryToOperations();
  absl::Status Compile(const CreationContext& creation_context);
  absl::Status Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                    ProfilingCommandQueue* profiling_queue);
  absl::Status UpdateParams();
  void PrepareExternal();

  void InitRecordableQueue(Environment* env);

  void ReleaseCPURepresentation();

  absl::Status ProfileTime(ProfilingCommandQueue* queue, ProfilingInfo* result);

  struct ExecutionHints {
    bool need_flush = false;

    bool flush_periodically = false;
    int flush_period = 1;

    // In order to reduce memory leak on Mali a pipeline needs to be
    // synchronized with CPU to prevent growing internal global OpenCL kernel
    // pool. One trick is to enqueue an event from a previous run. Most of the
    // time is should already be executed on GPU and should not stall the
    // pipeline.
    bool need_manual_release = false;
    CLEvent prev_enqueue_start_point;

    void Init(const GpuInfo& gpu_info);
  };
  ExecutionHints execution_hints_;

  // Directly mapped nodes from graph, but some of them "inactive" due
  //  to fusion (inactive = fused).
  // Memory is allocated only once, in ConvertOperations, and is not modified
  //  anywhere.
  std::vector<CLNode> nodes_;

  absl::flat_hash_map<ValueId, Tensor*> external_immutable_tensors_;
  absl::flat_hash_map<ValueId, Tensor*> external_mutable_tensors_;
  absl::flat_hash_map<ValueId, std::vector<int>> external_tensor_to_nodes_;
  absl::flat_hash_map<ValueId, TensorDescriptor> tensors_descs_;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors_descs_;
  std::map<ValueId, Tensor> const_tensors_;

  std::map<ValueId, Tensor> variable_tensors_;
  Buffer shared_buffers_parent_;
  std::vector<Buffer> shared_buffers_;
  std::vector<Tensor>
      shared_buffer_tensors_;  // use references to memory from shared_buffers_
  std::map<ValueId, int> graph_ids_to_shared_buffer_tensors_;

  std::map<ValueId, Tensor> strong_shape_tensors_;
  std::map<ValueId, ValueId> graph_ids_to_strong_shape_tensors_;

  std::vector<ValueId> input_ids_;
  std::map<ValueId, ValueId> variable_ids_and_refs_;
  std::vector<ValueId> output_ids_;

  std::unique_ptr<RecordableQueue> recordable_queue_ = nullptr;

  GpuInfo gpu_info_;
};

// Runs OpenCL specific transforms for the graph.
absl::Status RunGraphTransforms(GraphFloat32* graph);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_
