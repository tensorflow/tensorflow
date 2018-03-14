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
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/local_client.h"
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
               OpKernelContext* op_context);
  ~XlaAllocator() override;
  xla::StatusOr<perftools::gputools::DeviceMemoryBase> Allocate(
      int device_ordinal, uint64 size, bool retry_on_failure) override;
  Status Deallocate(int device_ordinal,
                    perftools::gputools::DeviceMemoryBase* mem) override;

  // Register an Tensor (input or resource variable) with the allocator. If
  // the operation returns an alias to one of its inputs, then the allocator
  // needs to be able to handle it.
  Status RegisterArgument(const Tensor* t);

  // Makes 'tensor' a wrapper around the data buffer at 'ptr'. The buffer is
  // interpreted as having data type 'dtype' and shape 'shape'.
  Status MakeTensorFromBuffer(perftools::gputools::DeviceMemoryBase buffer,
                              DataType dtype, const TensorShape& shape,
                              Tensor* out_tensor) const;

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

 private:
  OpKernelContext* const op_context_;

  // Map from pointer address to the owning Tensor; used by
  // MakeTensorFromBuffer. Also used to automatically release Tensors when the
  // allocator is freed.
  std::unordered_map<void*, Tensor> tensors_;
};

// Helper class to perform the marshalling of TensorFlow inputs and outputs to
// ShapedBuffers suitable for passing to an XLA computation.
class XlaComputationLaunchContext {
 public:
  XlaComputationLaunchContext(int64 num_resource_args, xla::LocalClient* client,
                              XlaAllocator* xla_allocator);

  // Add all inputs within `ctx` as XLA arguments (returned by arguments()).
  // `variables` is a map from TensorFlow argument number to resource variable.
  void PopulateInputs(OpKernelContext* ctx,
                      const XlaCompiler::CompilationResult* kernel,
                      const std::map<int, OptionalTensor>& variables);

  // Given the XLA output in `output`, populate all outputs of `ctx`.
  void PopulateOutputs(OpKernelContext* ctx,
                       const XlaCompiler::CompilationResult* kernel,
                       std::unique_ptr<xla::ShapedBuffer> output);

  // Return the argument list. Only valid after PopulateInputs() has been
  // called.
  const std::vector<xla::ShapedBuffer*>& arguments() const { return arg_ptrs_; }

 private:
  int64 num_resource_args_;
  xla::LocalClient* client_;
  XlaAllocator* xla_allocator_;
  std::vector<std::unique_ptr<xla::ShapedBuffer>> arg_buffers_;
  std::vector<xla::ShapedBuffer*> arg_ptrs_;
};

}  // namespace tensorflow

#endif
