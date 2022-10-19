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
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Snapshot of resource variables for a TF kernel invocation, mapping from
// parameter number to values at execution time. If the resource variable is not
// initialized, the value will not be present.
using ResourceVarsSnapshot = absl::flat_hash_map<int, std::optional<Tensor>>;

// Information about the state of a variable passed as input to the _XlaCompile
// and _XlaRun operators.  Unlocks the resource variable and decrements its
// refcount on destruction.
class VariableInfo {
 public:
  explicit VariableInfo(int index, absl::string_view name, Var* var,
                        const std::optional<ManagedStackTrace>&
                            definition_stack_trace = std::nullopt);
  VariableInfo(VariableInfo&& other);

  VariableInfo& operator=(VariableInfo&& other);

  VariableInfo(const VariableInfo&) = delete;
  VariableInfo& operator=(const VariableInfo&) = delete;

  // The index of the DT_RESOURCE input to the _XlaCompile/_XlaRun operator.
  // Note that the indices can be different between _XlaCompile and _XlaRun.
  int index() const { return index_; }

  // A pointer to the resource variable.  May be null if this VariableInfo is
  // "empty", i.e. it does not track a resource variable.
  Var* var() const { return var_; }

  // Returns the variable name.
  absl::string_view name() const { return name_; }

  // Returns true if the resource variable lock was successfully acquired by
  // this thread.
  bool lock_held() const { return lock_held_; }
  void set_lock_held() { lock_held_ = true; }

  const std::optional<ManagedStackTrace>& definition_stack_trace() const {
    return definition_stack_trace_;
  }

  ~VariableInfo();

 private:
  int index_;
  std::string name_;
  Var* var_;
  std::optional<ManagedStackTrace> definition_stack_trace_;

  // We can't use a optional<mutex_lock> here because it confuses the compiler's
  // thread safety analysis. Instead we use a boolean flag and release the lock
  // in the VariableInfo destructor.
  bool lock_held_ = false;
};

// Creates a list of updated resource variables.
StatusOr<std::vector<VariableInfo>> GatherVariableInfo(
    OpKernelContext* ctx,
    const XlaCompiler::CompilationResult& compilation_result,
    int missing_ctx_input_prefix);

// Takes a snapshot of the values of resource variable arguments, whose indices
// are specified in `variable_indices` argument. We snapshot tensors that back
// resource variables since concurrent updates may modify the shape, and it is
// important that the shapes used for compilation match the true shapes of the
// buffers.
//
// We snapshot the entire set of resource variables as one atomic operation.
// This models Read->* dependencies between resource variable operations.  See
// jit/resource_operation_safety_analysis for details.
Status SnapshotResourceVariables(OpKernelContext* ctx,
                                 absl::Span<const int> variable_indices,
                                 absl::Span<VariableInfo const> variable_infos,
                                 ResourceVarsSnapshot* result);

// Acquires the mutexes for all the variables in `variables` using a
// deadlock-safe protocol (acquire the mutexes in increasing-address order).
//
// `variables` is allowed to contain instances that don't track a resource
// variable (i.e. variables[i].var() can be null for some i).
Status LockVariables(absl::Span<VariableInfo*> variables)
    TF_EXCLUSIVE_LOCK_FUNCTION();
Status LockVariables(absl::Span<VariableInfo> variables)
    TF_EXCLUSIVE_LOCK_FUNCTION();

// Returns a vector of VariableInfo instances for the resource variable inputs,
// given that *all* inputs are in `inputs`. The input indices for the resource
// variable inputs are in `variable_indices`.
Status GetVariableInfosFromInputs(ResourceMgr* rm, DeviceBase* dev,
                                  absl::Span<const Tensor* const> inputs,
                                  absl::Span<const int> variable_indices,
                                  std::vector<VariableInfo>* result);

// Returns pointers to inputs stored in `ctx`.
std::vector<const Tensor*> InputsFromContext(OpKernelContext* ctx);

StatusOr<std::vector<int>> GetConstantInputIndicesFromContext(
    OpKernelContext* ctx);

std::vector<int> GetResourceVariableIndicesFromContext(OpKernelContext* ctx);

// Helper class to perform the marshalling of TensorFlow inputs and outputs to
// ShapedBuffers suitable for passing to an XLA computation.
class XlaComputationLaunchContext {
 public:
  // Create a new launch context. 'allocate_xla_tensors' is true if allocated
  // output tensors and variables are always XlaTensors. If false they are
  // assumed to be "normal" device pointers.
  // If 'use_multiple_streams' is true, tensors may be defined and used on
  // multiple streams and so se::Events must be defined and waited for. If
  // 'use_multiple_streams' is true, 'allocate_xla_tensors' must also be true
  // because we track inter-stream dependencies through events inside XlaTensor
  // objects.
  XlaComputationLaunchContext(xla::LocalClient* client,
                              se::DeviceMemoryAllocator* xla_allocator,
                              int device_ordinal, bool allocate_xla_tensors,
                              bool use_multiple_streams);

  // Builds a XlaCompiler::Argument vector from the arguments to an XlaLaunch
  // op.
  // Precondition: variables in `variable_args` are locked.
  static StatusOr<std::vector<XlaCompiler::Argument>> BuildXlaCompilerArguments(
      absl::Span<int const> must_be_constant_idxs,
      absl::Span<const Tensor* const> inputs,
      absl::Span<VariableInfo const> variable_args, Device* device);

  // Add all inputs within `ctx` as XLA arguments (returned by arguments()).
  // `variables` is a map from TensorFlow argument number to resource variable.
  //
  // Assumes that the first `missing_ctx_input_prefix` inputs to the kernel are
  // missing and adjusts input indices accordingly.  All elements in kernel's
  // input_mapping must be greater than or equal to `missing_ctx_input_prefix`
  // (in other words, no inputs actually required by the kernel can be missing).
  StatusOr<std::vector<xla::ExecutionInput>> PopulateInputs(
      OpKernelContext* ctx,
      const XlaCompiler::CompilationResult* compilation_result,
      const std::map<int, const Tensor*>& resource_vars,
      int missing_ctx_input_prefix,
      const xla::HloInputOutputAliasConfig& input_output_alias);

  // Given the XLA output in `output`, populate all outputs of `ctx`.  Also
  // writes out the resource variable updates.
  //
  // Updates to all resource variables are written in a single atomic operation.
  // This models *->Write dependencies between resource variable operations.
  // See jit/resource_operation_safety_analysis for details.
  //
  //
  // Assumes that the first `missing_ctx_input_prefix` inputs to the
  // compilation_result are missing and adjusts input indices accordingly.
  Status PopulateOutputs(
      OpKernelContext* ctx,
      const XlaCompiler::CompilationResult* compilation_result,
      xla::ScopedShapedBuffer output, int missing_ctx_input_prefix,
      absl::Span<VariableInfo> variable_infos,
      const xla::HloInputOutputAliasConfig& input_output_alias,
      const std::map<int, const Tensor*>& resource_vars);

 private:
  xla::LocalClient* client_;
  se::DeviceMemoryAllocator* xla_allocator_;
  bool allocate_xla_tensors_;
  bool use_multiple_streams_;
  int device_ordinal_;
};

// A simple TensorBuffer implementation that allows us to create Tensors that
// take ownership of pre-allocated memory.
class XlaTensorBuffer : public TensorBuffer {
 public:
  XlaTensorBuffer(const void* ptr, size_t expected_size, size_t actual_size,
                  Allocator* allocator)
      : TensorBuffer(const_cast<void*>(ptr)),
        expected_size_(expected_size),
        actual_size_(actual_size),
        allocator_(allocator) {}

  ~XlaTensorBuffer() override {
    if (data()) {
      allocator_->DeallocateRaw(data());
    }
  }

  size_t size() const override { return expected_size_; }

  TensorBuffer* root_buffer() override { return this; }

  void FillAllocationDescription(AllocationDescription* proto) const override {
    proto->set_requested_bytes(static_cast<int64_t>(expected_size_));
    proto->set_allocator_name(allocator_->Name());
    proto->set_ptr(reinterpret_cast<uintptr_t>(data()));
    if (allocator_->TracksAllocationSizes()) {
      auto ab = static_cast<int64_t>(allocator_->AllocatedSize(data()));
      proto->set_allocated_bytes(ab);
      int64_t id = allocator_->AllocationId(data());
      if (id > 0) {
        proto->set_allocation_id(id);
      }
      if (RefCountIsOne()) {
        proto->set_has_single_reference(true);
      }
    }
  }

 private:
  size_t expected_size_;
  size_t actual_size_;
  Allocator* allocator_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_LAUNCH_UTIL_H_
