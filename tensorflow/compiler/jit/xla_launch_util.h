/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Contains utilities for launching compiled XLA kernels for a KernelContext.

#ifndef TENSORFLOW_COMPILER_JIT_XLA_LAUNCH_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_XLA_LAUNCH_UTIL_H_

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class XlaAllocator;

// Takes a snapshot of the values of resource variable arguments, which are
// the last `num_variables` arguments. We snapshot tensors that back
// resource variables since concurrent updates may modify the shape, and it is
// important that the shapes used for compilation match the true shapes of the
// buffers.
//
// Returns a map of TensorFlow argument index to resource variable.
std::map<int, OptionalTensor> SnapshotResourceVariables(OpKernelContext* ctx,
                                                        int num_variables);

// Adapter class that wraps a Tensorflow allocator as an XLA allocator.
// Assumes that the Tensorflow allocator permits asynchronous deallocation:
// see comment on `AllowsAsynchronousDeallocation()`.
class XlaAllocator : public xla::DeviceMemoryAllocator {
 public:
  XlaAllocator(const perftools::gputools::Platform* platform,
               Allocator* wrapped);
  ~XlaAllocator() override;
  xla::StatusOr<perftools::gputools::DeviceMemoryBase> Allocate(
      int device_ordinal, uint64 size, bool retry_on_failure) override;
  Status Deallocate(int device_ordinal,
                    perftools::gputools::DeviceMemoryBase* mem) override;

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

 private:
  Allocator* wrapped_;
};

// Helper class to perform the marshalling of TensorFlow inputs and outputs to
// ShapedBuffers suitable for passing to an XLA computation.
class XlaComputationLaunchContext {
 public:
  // Create a new launch context. 'allocate_xla_tensors' is true if allocated
  // output tensors and variables are always XlaTensors. If false they are
  // assumed to be "normal" device pointers.
  XlaComputationLaunchContext(int64 num_resource_args, xla::LocalClient* client,
                              xla::DeviceMemoryAllocator* xla_allocator,
                              bool allocate_xla_tensors);

  // Add all inputs within `ctx` as XLA arguments (returned by arguments()).
  // `variables` is a map from TensorFlow argument number to resource variable.
  void PopulateInputs(OpKernelContext* ctx,
                      const XlaCompiler::CompilationResult* kernel,
                      const std::map<int, OptionalTensor>& variables);

  // Given the XLA output in `output`, populate all outputs of `ctx`.
  void PopulateOutputs(OpKernelContext* ctx,
                       const XlaCompiler::CompilationResult* kernel,
                       std::unique_ptr<xla::ScopedShapedBuffer> output);

  // Return the argument list. Only valid after PopulateInputs() has been
  // called.
  const std::vector<xla::ShapedBuffer*>& arguments() const { return arg_ptrs_; }

 private:
  int64 num_resource_args_;
  xla::LocalClient* client_;
  xla::DeviceMemoryAllocator* xla_allocator_;
  bool allocate_xla_tensors_;
  std::vector<std::unique_ptr<xla::ShapedBuffer>> arg_buffers_;
  std::vector<xla::ShapedBuffer*> arg_ptrs_;
};

// A simple TensorBuffer implementation that allows us to create Tensors that
// take ownership of pre-allocated memory.
class XlaTensorBuffer : public TensorBuffer {
 public:
  XlaTensorBuffer(const void* ptr, size_t expected_size, size_t actual_size,
                  Allocator* allocator)
      : expected_size_(expected_size),
        actual_size_(actual_size),
        allocator_(allocator) {
    data_ = const_cast<void*>(ptr);
  }

  ~XlaTensorBuffer() override { allocator_->DeallocateRaw(data_); }

  void* data() const override { return data_; }
  size_t size() const override { return expected_size_; }

  TensorBuffer* root_buffer() override { return this; }

  void FillAllocationDescription(AllocationDescription* proto) const override {
    proto->set_allocated_bytes(actual_size_);
  }

  static Tensor MakeTensor(DataType dtype, const TensorShape& shape,
                           perftools::gputools::DeviceMemoryBase buffer,
                           Allocator* allocator) {
    size_t expected_size = shape.num_elements() * DataTypeSize(dtype);
    auto* tensor_buffer = new XlaTensorBuffer(buffer.opaque(), expected_size,
                                              buffer.size(), allocator);
    Tensor t(dtype, shape, tensor_buffer);
    tensor_buffer->Unref();
    return t;
  }

 private:
  void* data_;
  size_t expected_size_;
  size_t actual_size_;
  Allocator* allocator_;
};

}  // namespace tensorflow

#endif
