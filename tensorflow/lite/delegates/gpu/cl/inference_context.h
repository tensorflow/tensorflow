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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
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
  struct CreateInferenceInfo {
    CalculationsPrecision precision;
    TensorStorageType storage_type;
    ModelHints hints;
  };

  absl::Status InitFromGraph(const CreateInferenceInfo& create_info,
                             const GraphFloat32& graph, Environment* env,
                             std::vector<uint8_t>* serialized_model = nullptr);

  // Applies OpenCL-specific transformations to the graph before the
  // initialization. These transformations are either impossible or useless in
  // other backends.
  absl::Status InitFromGraphWithTransforms(
      const CreateInferenceInfo& create_info, GraphFloat32* graph,
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

  const std::vector<int64_t>& GetInputRefs() const { return in_refs_; }
  const std::vector<int64_t>& GetOutputRefs() const { return out_refs_; }

  absl::Status RestoreDeserialized(
      const absl::Span<const uint8_t> serialized_model, Environment* env);

 private:
  enum class TensorMemoryType { kStrongShape, kBuffer, kVariable, kConst };

  friend flatbuffers::Offset<data::InferenceContext> Encode(
      const InferenceContext& inference,
      flatbuffers::FlatBufferBuilder* builder);
  friend absl::Status Decode(const data::InferenceContext* fb_inference,
                             InferenceContext* inference);

  void CopyInAndOutIds(const GraphFloat32& graph);
  absl::Status ConvertOperations(const GpuInfo& gpu_info,
                                 const GraphFloat32& graph, ModelHints hints);
  void CreateLinks();
  absl::Status ReserveGraphTensors(const CreateInferenceInfo& create_info,
                                   const GpuInfo& gpu_info,
                                   const GraphFloat32& graph);
  absl::Status Merge();
  absl::Status AllocateMemory(CLContext* context);

  absl::Status AllocateMemoryForConstTensors(CLContext* context);

  absl::Status AllocateMemoryForVariableTensors(CLContext* context);

  absl::Status AllocateMemoryForBuffers(CLContext* context);

  absl::Status AllocateMemoryForStrongShapes(CLContext* context);

  // utility function
  void GetUsages(const std::function<bool(ValueId)>& functor,
                 std::map<ValueId, int2>* usages);

  TensorMemoryType GetTensorMemoryType(ValueId id);

  void BindMemoryToOperations();
  absl::Status Compile(const CreationContext& creation_context);
  absl::Status Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                    ProfilingCommandQueue* profiling_queue);
  absl::Status UpdateParams();

  void ReleaseCPURepresentation();

  // performance hacks
  bool need_flush_ = false;

  bool flush_periodically_ = false;
  int flush_period_ = 1;

  // In order to reduce memory leak on Mali a pipeline needs to be synchronized
  // with CPU to prevent growing internal global OpenCL kernel pool. One trick
  // is to enqueue an event from a previous run. Most of the time is should
  // already be executed on GPU and should not stall the pipeline.
  bool need_manual_release_ = false;
  CLEvent prev_enqueue_start_point_;

  CalculationsPrecision precision_;
  TensorStorageType storage_type_;

  // Directly mapped nodes from graph, but some of them "inactive" due
  //  to fusion (inactive = fused).
  // Memory is allocated only once, in ConvertOperations, and is not modified
  //  anywhere.
  std::vector<CLNode> nodes_;

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

  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors_descs_;
  std::map<ValueId, Tensor> const_tensors_;

  std::map<ValueId, Tensor> variable_tensors_;
  std::vector<Buffer> shared_buffers_;
  std::vector<Tensor>
      shared_buffer_tensors_;  // use references to memory from shared_buffers_
  std::map<ValueId, int> graph_ids_to_shared_buffer_tensors_;

  std::map<ValueId, Tensor> strong_shape_tensors_;
  std::map<ValueId, ValueId> graph_ids_to_strong_shape_tensors_;

  std::vector<ValueId> input_ids_;
  std::map<ValueId, ValueId> variable_ids_and_refs_;
  std::vector<ValueId> output_ids_;

  // for serialization
  std::vector<int64_t> in_refs_;
  std::vector<int64_t> out_refs_;
};

// Runs OpenCL specific transforms for the graph.
absl::Status RunGraphTransforms(GraphFloat32* graph);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_
