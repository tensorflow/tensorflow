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
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

namespace tflite {
namespace gpu {
namespace metal {

class InferenceContext {
 public:
  struct CreateInferenceInfo {
    CalculationsPrecision precision;
    TensorStorageType storage_type;
    ModelHints hints;
  };

  InferenceContext() = default;

  absl::Status InitFromGraph(const CreateInferenceInfo& create_info,
                             const GraphFloat32& graph,
                             id<MTLDevice> device_id);

  /// Inserts all GPU compute tasks into the command encoder.
  /// @param inputOutputBuffers Must be created and passed into the method
  /// with pairs ID:buffer
  /// @discussion No GPU synchronization functions are used inside. All GPU
  /// resources must be created
  ///             with the same device which has been used in
  ///             compileModelWithDevice() method.
  void EncodeWithEncoder(
      id<MTLComputeCommandEncoder> command_encoder,
      const std::map<ValueId, id<MTLBuffer>>& in_out_buffers);

  /// Inserts all GPU compute tasks into the command buffer. For every task will
  /// be used separate
  ///   encoder.
  /// @param inputOutputBuffers Must be created and passed into the method with
  /// pairs ID:buffer
  /// @discussion No GPU synchronization functions are used inside. All GPU
  /// resources must be created
  ///             with the same device which has been used in
  ///             compileModelWithDevice() method.
  void EncodeWithCommandBuffer(
      id<MTLCommandBuffer> command_buffer,
      const std::map<ValueId, id<MTLBuffer>>& in_out_buffers);

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
  void EncodeWithCommandQueue(
      id<MTLCommandQueue> command_queue,
      const std::map<ValueId, id<MTLBuffer>>& in_out_buffers, int flush_period);

 private:
  struct CompiledModel {
    std::vector<NodeDescriptor> nodes;
  };

  absl::Status Compile(const GraphFloat32& graph, const GpuInfo& gpu_info,
                       CalculationsPrecision precision,
                       CompiledModel* compiled_model);

  void ReserveGraphTensors(const CreateInferenceInfo& create_info,
                           const GpuInfo& gpu_info, const GraphFloat32& graph);

  absl::Status CompileModelWithDevice(CompiledModel* model,
                                      MetalDevice* device);

  absl::Status Merge(CompiledModel* model);
  absl::Status AllocateTensors(MetalDevice* device);
  absl::Status AllocateMemoryForBuffers(MetalDevice* device);
  void BindTensorsToOperations();
  MetalSpatialTensor* GetTensor(ValueId tensor_id);
  void GetUsages(std::map<ValueId, int2>* usages);
  void UpdatePreallocatedTensors(
      const std::map<ValueId, id<MTLBuffer>>& preallocated);

  struct DummyTensor {
    BHWC shape;
    TensorDescriptor descriptor;

    bool operator==(const DummyTensor& b) const {
      return shape == b.shape && descriptor == b.descriptor;
    }
  };

  class TensorReserver {
   public:
    TensorReserver() : next_(0) {}
    ValueId Add(const DummyTensor& dummy) {
      reservations_[next_] = dummy;
      return next_++;
    }
    void Add(ValueId id, const DummyTensor& dummy) {
      reservations_[id] = dummy;
    }
    void SetNext(ValueId id) { next_ = id; }
    DummyTensor Get(ValueId id) { return reservations_[id]; }

    std::vector<std::pair<ValueId, TensorDescriptor>> GetTensorDescs() const {
      std::vector<std::pair<ValueId, TensorDescriptor>> result;
      for (auto& v : reservations_) {
        TensorDescriptor desc = v.second.descriptor;
        desc.shape.b = v.second.shape.b;
        desc.shape.h = v.second.shape.h;
        desc.shape.w = v.second.shape.w;
        desc.shape.d = 1;
        desc.shape.c = v.second.shape.c;
        result.push_back({v.first, desc});
      }
      return result;
    }

    void Add(const std::vector<std::pair<ValueId, TensorDescriptor>>& tensors) {
      for (auto& v : tensors) {
        DummyTensor dummy;
        dummy.descriptor = v.second;
        dummy.shape.b = v.second.shape.b;
        dummy.shape.h = v.second.shape.h;
        dummy.shape.w = v.second.shape.w;
        dummy.shape.c = v.second.shape.c;
        Add(v.first, dummy);
      }
    }

   private:
    absl::flat_hash_map<ValueId, DummyTensor> reservations_;
    ValueId next_;
  };
  TensorReserver tensor_reserver_;

  std::vector<ComputeTask> compute_tasks_;
  // contains indexes of compute_tasks_
  std::vector<int> task_ids_with_preallocated_tensors_;
  std::vector<ValueId> input_ids_;
  std::vector<ValueId> output_ids_;
  CalculationsPrecision precision_;
  std::map<ValueId, MetalSpatialTensor> preallocated_tensors_;

  std::map<ValueId, int> graph_ids_to_shared_buffer_tensors_;
  std::vector<id<MTLBuffer>> shared_buffers_;
  std::vector<MetalSpatialTensor>
      shared_buffer_tensors_;  // use references to memory
                               // from _sharedBuffers
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_INFERENCE_CONTEXT_H_
