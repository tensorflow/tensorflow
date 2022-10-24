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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_INFERENCE_CONTEXT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_INFERENCE_CONTEXT_H_

#import <Metal/Metal.h>

#include <list>
#include <map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_model.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_model_generated.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/profiling_info.h"
#include "tensorflow/lite/delegates/gpu/common/task/tuning_type.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/inference_context_generated.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

namespace tflite {
namespace gpu {
namespace metal {

struct MetalNode {
  ComputeTask task;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  // Mostly for debug purposes.
  std::string name;

  MetalNode() = default;

  MetalNode(MetalNode&& node) = default;
  MetalNode& operator=(MetalNode&& node) = default;
  MetalNode(const MetalNode&) = delete;
  MetalNode& operator=(const MetalNode&) = delete;
};

class InferenceContext {
 public:
  InferenceContext() = default;

  // IMPORTANT: If InitFromGraph used, RunGraphTransforms must be applied for
  // this graph upfront, otherwise not guaranteed correct behavior
  absl::Status InitFromGraph(const CreateGpuModelInfo& create_info,
                             const GraphFloat32& graph, id<MTLDevice> device_id,
                             std::vector<uint8_t>* serialized_model = nullptr);

  // Applies specific transformations to the graph before the
  // initialization. These transformations are either impossible or useless in
  // other backends.
  absl::Status InitFromGraphWithTransforms(
      const CreateGpuModelInfo& create_info, GraphFloat32* graph,
      id<MTLDevice> device_id,
      std::vector<uint8_t>* serialized_model = nullptr);

  absl::Status RestoreDeserialized(
      const absl::Span<const uint8_t> serialized_model, id<MTLDevice> device_id,
      CreateGpuModelInfo* create_info = nullptr);

  /// Inserts all GPU compute tasks into the command encoder.
  /// @param inputOutputBuffers Must be created and passed into the method
  /// with pairs ID:buffer
  /// @discussion No GPU synchronization functions are used inside. All GPU
  /// resources must be created
  ///             with the same device which has been used in
  ///             compileModelWithDevice() method.
  void EncodeWithEncoder(id<MTLComputeCommandEncoder> command_encoder);

  /// Inserts all GPU compute tasks into the command buffer. For every task will
  /// be used separate
  ///   encoder.
  /// @param inputOutputBuffers Must be created and passed into the method with
  /// pairs ID:buffer
  /// @discussion No GPU synchronization functions are used inside. All GPU
  /// resources must be created
  ///             with the same device which has been used in
  ///             compileModelWithDevice() method.
  void EncodeWithCommandBuffer(id<MTLCommandBuffer> command_buffer);

  /// Adds all GPU compute tasks to the command queue. For every task will be
  /// used separate
  ///   encoder. Few encoders(flushPeriod) batched into compute buffer that sent
  ///   for execution.
  /// @param inputOutputBuffers Must be created and passed into the method with
  /// pairs ID:buffer
  /// @discussion No GPU synchronization functions are used inside. All GPU
  /// resources must be created
  ///             with the same device which has been used in
  ///             compileModelWithDevice() method.
  void EncodeWithCommandQueue(id<MTLCommandQueue> command_queue,
                              int flush_period);

  API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
  void AddResources(id<MTLComputeCommandEncoder> command_encoder);
  API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
  void EncodeWithICB(id<MTLComputeCommandEncoder> command_encoder);

  void Profile(id<MTLDevice> device, ProfilingInfo* result);
  // Returns size in bytes for all intermediate(runtime) tensors that owned by
  // this inference context. Do not include constant tensors.
  uint64_t GetIntermediateTensorsSize() const;
  uint64_t GetConstantTensorsSize() const;

  // Can be used only with ids from external_mutable_tensors in create_info
  // Must be called after initialization and before execution
  absl::Status SetTensor(const ValueId& tensor_id,
                         MetalSpatialTensor* tensor_ptr);

  MetalSpatialTensor* GetTensor(ValueId tensor_id);
  absl::Status SetInputTensor(ValueId id, const TensorFloat32& tensor);
  absl::Status GetOutputTensor(ValueId id, TensorFloat32* result);

 private:
  enum class TensorMemoryType {
    kStrongShape,
    kBuffer,
    kVariable,
    kConst,
    kExternal
  };

  flatbuffers::Offset<data::InferenceContext> Encode(
      MetalDevice* device,
      flatbuffers::Offset<tflite::gpu::data::GpuModel> gpu_model_fb,
      flatbuffers::FlatBufferBuilder* builder);

  absl::Status Decode(MetalDevice* device,
                      const data::InferenceContext* fb_inference);

  void CopyFromGpuModel(GpuModel* gpu_model);
  absl::Status CompileOperations(MetalDevice* device);
  void PrepareExternal();

  absl::Status AllocateTensors(MetalDevice* device);
  absl::Status AllocateMemoryForConstTensors(MetalDevice* device);
  absl::Status AllocateMemoryForBuffers(MetalDevice* device);
  absl::Status AllocateMemoryForStrongShapes(MetalDevice* device);
  void BindTensorsToOperations();
  absl::Status UpdateParams(const GpuInfo& gpu_info);
  void GetUsages(const std::function<bool(ValueId)>& functor,
                 std::map<ValueId, int2>* usages);
  TensorMemoryType GetTensorMemoryType(const GpuInfo& gpu_info, ValueId id);
  absl::Status Tune(TuningType tuning_type, MetalDevice* device);

  absl::flat_hash_map<ValueId, TensorDescriptor> tensors_descs_;

  std::vector<MetalNode> nodes_;
  std::vector<ValueId> input_ids_;
  std::vector<ValueId> output_ids_;

  absl::flat_hash_map<ValueId, MetalSpatialTensor*> external_immutable_tensors_;
  absl::flat_hash_map<ValueId, MetalSpatialTensor*> external_mutable_tensors_;
  absl::flat_hash_map<ValueId, std::vector<int>> external_tensor_to_nodes_;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors_descs_;
  std::map<ValueId, MetalSpatialTensor> const_tensors_;

  std::map<ValueId, int> graph_ids_to_shared_buffer_tensors_;
  std::vector<id<MTLBuffer>> shared_buffers_;
  std::vector<MetalSpatialTensor>
      shared_buffer_tensors_;  // use references to memory
                               // from _sharedBuffers

  std::map<ValueId, MetalSpatialTensor> strong_shape_tensors_;
  std::map<ValueId, ValueId> graph_ids_to_strong_shape_tensors_;

  id<MTLIndirectCommandBuffer> icb_ = nullptr;
  id<MTLDevice> device_ = nullptr;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_INFERENCE_CONTEXT_H_
