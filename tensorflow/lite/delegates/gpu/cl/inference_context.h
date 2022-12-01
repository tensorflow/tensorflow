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
#include "tensorflow/lite/delegates/gpu/common/gpu_model.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
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

enum class TensorType { kVariable, kConst, kExternal, kRuntime };

class InferenceContext {
 public:
  absl::Status InitFromGraph(const CreateGpuModelInfo& create_info,
                             const GraphFloat32& graph, Environment* env,
                             std::vector<uint8_t>* serialized_model = nullptr);

  absl::Status InitFromGpuModel(
      const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
      Environment* env, std::vector<uint8_t>* serialized_model = nullptr,
      Buffer* shared_buffer = nullptr);

  absl::Status AddToCommanBuffer(cl_command_buffer_khr cb);

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
  uint64_t GetConstantTensorsSize() const;

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
  flatbuffers::Offset<data::InferenceContext> Encode(
      const CLDevice& device, const ProgramCache& program_cache,
      flatbuffers::Offset<tflite::gpu::data::GpuModel> gpu_model_fb,
      flatbuffers::FlatBufferBuilder* builder);

  void InitFromGpuModel(GpuModel* gpu_model);

  absl::Status AllocateMemory(const GpuModel& gpu_model,
                              const GpuInfo& gpu_info,
                              const CreateGpuModelInfo* create_info,
                              CLContext* context);

  absl::Status AllocateConstTensors(const GpuModel& gpu_model,
                                    CLContext* context);

  absl::Status AllocateVariableTensors(const GpuModel& gpu_model,
                                       CLContext* context);

  absl::Status AllocateBufferBasedTensors(const GpuModel& gpu_model,
                                          const GpuInfo& gpu_info,
                                          const CreateGpuModelInfo* create_info,
                                          CLContext* context);

  absl::Status AllocateStrongShapesTensors(
      const GpuModel& gpu_model, const GpuInfo& gpu_info,
      const CreateGpuModelInfo* create_info, CLContext* context);

  void BindMemoryToOperations();
  absl::Status Compile(const CreationContext& creation_context);
  absl::Status Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                    ProfilingCommandQueue* profiling_queue);
  absl::Status UpdateParams();
  void PrepareExternal();

  void InitRecordableQueue(Environment* env);

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

  std::map<ValueId, Tensor> const_tensors_;

  std::map<ValueId, ValueId> variable_ids_and_refs_;
  std::map<ValueId, Tensor> variable_tensors_;

  std::unique_ptr<Buffer> shared_buffers_parent_;
  Buffer* shared_buffers_parent_ptr_ = nullptr;
  std::vector<Buffer> shared_buffers_;
  std::vector<Tensor>
      shared_buffer_tensors_;  // use references to memory from shared_buffers_
  std::map<ValueId, int> graph_ids_to_shared_buffer_tensors_;

  std::map<ValueId, Tensor> strong_shape_tensors_;
  std::map<ValueId, ValueId> graph_ids_to_strong_shape_tensors_;

  std::vector<ValueId> input_ids_;
  std::vector<ValueId> output_ids_;

  std::unique_ptr<RecordableQueue> recordable_queue_ = nullptr;

  GpuInfo gpu_info_;
};

absl::Status GetInOutRefs(const absl::Span<const uint8_t> serialized_model,
                          std::vector<int64_t>* in_refs,
                          std::vector<int64_t>* out_refs);

absl::Status GetTotalBufferSizeForTensors(const GpuModel& gpu_model,
                                          const CreateGpuModelInfo& create_info,
                                          const GpuInfo& gpu_info,
                                          uint64_t* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_
