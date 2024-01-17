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

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/shaped_buffer.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Creates a list of updated resource variables.
StatusOr<std::vector<VariableInfo>> GatherVariableInfo(
    OpKernelContext* ctx,
    const XlaCompiler::CompilationResult& compilation_result,
    int missing_ctx_input_prefix);

// Returns pointers to inputs stored in `ctx`.
std::vector<const Tensor*> InputsFromContext(OpKernelContext* ctx);

StatusOr<std::vector<int>> GetConstantInputIndicesFromContext(
    OpKernelContext* ctx);

Status SetOutputForConstant(
    OpKernelContext* ctx, bool requires_copy_to_device,
    const XlaCompiler::CompilationResult* compilation_result, int output_num);

// Converts input tensors and variables which are parameters of the
// XlaComputation into PjRtBuffers to be fed as input to the
// PjRtLoadedExecutable.
//
// Assumes that the first `num_missing_prefix_ctx_inputs` inputs to the
// compilation_result are missing in `inputs` and adjusts indexing into `inputs`
// accordingly.
// `input_mapping` is a vector that maps from the parameters of the
// XlaComputation to their original argument positions. This can be sourced from
// `XlaCompiler::CompilationResult::input_mapping`.
// `variable_snapshots` is a map of {index of the input to the
// compilation_result -> underlying Tensor the variable is/was pointing to (i.e.
// the value of the variable at the time of lowering/compilation)}.
//
// The obtained PjRtBuffers are populated to `args` vector.
// `non_donatable_input_indices` will also be set, which contains the indices of
// the input that should not be donated to output.
//
// There can be three types of input: 1. Tensor with PjRtTensorBuffer; 2.
// Tensor with AsyncValueTensor; 3. Tensor with raw device mem pointer.
// For case 3, we need to create a PjRtBuffer from the raw device mem pointer,
// and we need to ensure the PjRtBuffer persists till XLA computation is
// complete. Therefore we put the newly created PjRtBuffer into `owned_args`.
// Caller is responsible to ensure `owned_args` lives till the end of XLA
// computation.
Status PreparePjRtExecutableArguments(
    int num_missing_prefix_ctx_inputs, const std::vector<int>& input_mapping,
    const std::vector<const Tensor*>& inputs,
    const absl::flat_hash_map<int, const Tensor*>& variable_snapshots,
    xla::PjRtClient* pjrt_client, xla::PjRtDevice* pjrt_device,
    bool use_pjrt_tensor_buffer, std::vector<xla::PjRtBuffer*>* args,
    std::vector<std::unique_ptr<xla::PjRtBuffer>>* owned_args,
    absl::flat_hash_set<int>* non_donatable_input_indices);

// Populates the OpKernelContext outputs with the outputs of the
// PjRtLoadedExecutable. Requires the `compilation_result` used to build the
// PjRtLoadedExecutable.
// This function only looks at variables that were updated, so `variables` can
// either be all the variables or only the ones that were updated.
// Assumes that the first `num_missing_prefix_ctx_inputs` inputs to the
// compilation_result are missing in `inputs` and adjusts indexing into `inputs`
// accordingly.
Status PopulateCtxOutputsFromPjRtExecutableOutputs(
    int num_missing_prefix_ctx_inputs, const std::vector<const Tensor*>& inputs,
    const std::vector<VariableInfo>& variables,
    const XlaCompiler::CompilationResult& compilation_result,
    bool use_pjrt_tensor_buffer,
    std::vector<std::unique_ptr<xla::PjRtBuffer>>& executable_outputs,
    OpKernelContext* ctx);

// Returns the options used for executing a PjRtLoadedExecutable.
xla::ExecuteOptions GetPjRtExecuteOptions(
    const DeviceType& device_type,
    absl::flat_hash_set<int> non_donatable_input_indices);

// Returns the device ordinal from the parsed name of the device.
int GetDeviceOrdinal(const DeviceBase* device);

// Returns the device type from the OpKernelContext.
DeviceType GetDeviceType(OpKernelContext* ctx);

// Runs `executable` and populates the outputs in `ctx`. `inputs` and
// `variables` are the input arguments to the computation, usually read from the
// OpKernelContext, `ctx`. Requires the device-appropriate `pjrt_client` and the
// `compilation_result` used to build the `executable`.
Status RunPjRtExecutable(
    const std::vector<const Tensor*>& inputs,
    const std::vector<VariableInfo>& variables,
    const XlaCompiler::CompilationResult& compilation_result,
    xla::PjRtClient* pjrt_client, xla::PjRtLoadedExecutable* executable,
    OpKernelContext* ctx);

// Same as the above function but takes in `updated_variables` and
// `variable_snapshots` which is a map of {index of the input to the
// compilation_result -> underlying Tensor the variable is/was pointing to
// (i.e. the value of the variable at the time of lowering/compilation)}.
// Assumes that the first `num_missing_prefix_ctx_inputs` inputs to the
// compilation_result are missing in `inputs` and adjusts indexing into `inputs`
// accordingly.
Status RunPjRtExecutable(
    int num_missing_prefix_ctx_inputs, const std::vector<const Tensor*>& inputs,
    const absl::flat_hash_map<int, const Tensor*>& variable_snapshots,
    const std::vector<VariableInfo>& updated_variables,
    const XlaCompiler::CompilationResult& compilation_result,
    xla::PjRtClient* pjrt_client, xla::PjRtLoadedExecutable* executable,
    OpKernelContext* ctx);

// Similar to the above function but it does not take an OpKernelContext, and
// it returns the output in PjRtBuffers, instead of populating results into
// OpKernelContext.
StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> RunPjRtExecutable(
    int num_missing_prefix_ctx_inputs, const std::vector<const Tensor*>& inputs,
    const absl::flat_hash_map<int, const Tensor*>& variable_snapshots,
    const std::vector<VariableInfo>& updated_variables,
    const DeviceType& device_type, bool use_pjrt_tensor_buffer,
    const XlaCompiler::CompilationResult& compilation_result,
    xla::PjRtDevice* device, xla::PjRtClient* pjrt_client,
    xla::PjRtLoadedExecutable* executable);

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
