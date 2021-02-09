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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_ENGINE_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_ENGINE_UTILS_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
using ::stream_executor::port::StatusOr;

// Input/output data format for OpConverterTest::BuildAndRun().
struct InputOutputData {
  void* Buffer() const {
    return const_cast<char*>(tensor.tensor_data().data());
  }

  size_t TotalBytes() const { return tensor.TotalBytes(); }

  string name;
  Tensor tensor;
};

class TRTBaseAllocator;

// Keeps track of the TensorRT execution context and the device memory owned by
// the context, if any. An execution context owns the device memory that TF-TRT
// allocates for the context. In this case, the allocator is not null and is
// used to free the device memory. An execution context doesn't own a device
// memory (1) if the device memory is allocated through TensorRT, or (2) the
// device memory is allocated by TF-TRT for another execution context but
// shared with this context. If this case, the device memory is null.
//
// Currently, the main reason we want to allocate the device memory for an
// execution context in TF-TRT is because the TensorRT API to create an
// execution context with device memory doesn't handle out of memory properly.
//
// To support dynamic shapes, we create multiple execution contexts for an
// engine and may want to support multiple execution contexts sharing the same
// device memory.
class ExecutionContext {
 private:
  // Records the TensorRT execution context `context`, the device memory
  // `device_memory` TF-TRT allocates for the context and the device memory
  // allocator `allocator` used to allocate the memory. If TF-TRT doesn't
  // allocate any device memory for the context, then `device_memory` is null.
  // otherwise, allocator should not be null.
  ExecutionContext(TRTBaseAllocator* allocator, void* device_memory,
                   nvinfer1::IExecutionContext* context)
      : memory_allocator_(allocator),
        device_memory_(device_memory),
        execution_context_(context) {}

 public:
  // Disables copy constructors as the object owns the device memory and the
  // execution context.
  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;

  ExecutionContext(ExecutionContext&& other)
      : memory_allocator_(other.memory_allocator_),
        device_memory_(other.device_memory_),
        execution_context_(other.execution_context_) {
    other.memory_allocator_ = nullptr;
    other.device_memory_ = nullptr;
    other.execution_context_ = nullptr;
  }

  ~ExecutionContext();

  operator nvinfer1::IExecutionContext*() const { return execution_context_; }
  nvinfer1::IExecutionContext* GetIExecutionContext() const {
    return execution_context_;
  }

  static StatusOr<ExecutionContext> Create(nvinfer1::ICudaEngine* cuda_engine,
                                           TRTBaseAllocator* allocator);

 private:
  // The allocator used to allocate and free the device memory owned by the
  // execution context.
  TRTBaseAllocator* memory_allocator_;
  // The device memory owned by the execution context.
  void* device_memory_;
  // The TensorRT execution context.
  nvinfer1::IExecutionContext* execution_context_;
};

// Creates a TensorRT execution context. If an allocator is not given, then the
// execution context is created with device memory allocated by TensorRT.
// Otherwise, uses the allocator to allocate the needed device memory for the
// execution context.
//
// Returns an ExecutionContext object that wraps the above results. If out of
// device memory happens, returns an error status instead.
StatusOr<ExecutionContext> CreateExecutionContext(
    nvinfer1::ICudaEngine* cuda_engine, TRTBaseAllocator* allocator);

using DataVec = std::vector<InputOutputData>;

// Gets the binding index of a tensor in an engine.
//
// The binding index is looked up using the tensor's name and the profile index.
// Profile index should be set to zero, if we do not have optimization profiles.
Status GetTrtBindingIndex(const char* tensor_name, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index);

// Sets input buffers for TRT from a list of input tensors. The input tensors
// are either defined by ctx or by input_vec.
Status SetTrtEngineInputs(nvinfer1::ICudaEngine* cuda_engine,
                          nvinfer1::IExecutionContext* execution_context,
                          const int trt_profile_idx,
                          std::vector<void*>& buffers, bool use_implicit_batch,
                          int num_batch, OpKernelContext* ctx = nullptr,
                          const DataVec* input_vec = nullptr);

// Returns the shape of a binding from TensorRT.
//
// The binding is identified by its binding_index. The batch_size argument is
// ignored if use_implicit_batch==false. The shape is returned in the last
// argument.
Status GetTrtBindingShape(const nvinfer1::ICudaEngine* cuda_engine,
                          const nvinfer1::IExecutionContext* execution_context,
                          int binding_index, bool use_implicit_batch,
                          int batch_size, TensorShape& shape);

// Defines output buffers for TRT. The buffers are allocated by ctx, if ctx is
// not null. Otherwise it is expected that the outputs DataVec is not null, and
// the Tensors in outputs are already allocated.
Status SetTrtEngineOutputs(nvinfer1::ICudaEngine* cuda_engine,
                           nvinfer1::IExecutionContext* execution_context,
                           int trt_profile_idx, std::vector<void*>& buffers,
                           bool use_implicit_batch, int batch_size = 0,
                           OpKernelContext* ctx = nullptr,
                           DataVec* outputs = nullptr);

// Enqueues TensorRT inference job. The batch_size argument is only relevant in
// implicit batch mode.
Status TrtEnqueue(nvinfer1::IExecutionContext* execution_context,
                  std::vector<void*>& buffers, cudaStream_t stream,
                  bool use_implicit_batch, int batch_size = 1);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_ENGINE_UTILS_H_
