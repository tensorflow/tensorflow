/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_CUSTOM_CALL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_CUSTOM_CALL_THUNK_H_

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// Thunk to run an XLA FFI custom call on a GPU.
//
// This thunk handles custom calls registered via the XLA FFI mechanism, which
// provides a type-safe API for registering external functions with XLA runtime.
//
// For legacy (non-FFI) custom calls, see LegacyCustomCallThunk.
//
// Note that not all kCustomCall HLOs in XLA:GPU end up being run by this thunk.
// XLA itself creates kCustomCall instructions when lowering kConvolution HLOs
// into calls to cudnn.  These internally-created custom-calls are run using
// ConvolutionThunk, not CustomCallThunk.  There's no ambiguity because they
// have special call target names (e.g. "__cudnn$convForward") that only the
// compiler is allowed to create.
//
// Also implements TracedCommand so it can be recorded directly into command
// buffers; the default TracedCommand::Record() traces ExecuteOnStream() on the
// command-buffer trace stream, reusing the FFI invocation logic.
class CustomCallThunk : public TracedCommand {
 public:
  // An owning equivalent of XLA_FFI_Handler_Bundle that allows using lambdas
  // with captures.
  //
  // The members can be initialized with xla::ffi::Ffi::Bind().To(...).
  struct OwnedHandlerBundle {
    std::unique_ptr<xla::ffi::Ffi> initialize;
    std::unique_ptr<xla::ffi::Ffi> instantiate;
    std::unique_ptr<xla::ffi::Ffi> prepare;
    std::unique_ptr<xla::ffi::Ffi> execute;
  };

  // A per-execution state that holds state for prepare and initialize stages.
  struct PrepareAndInitState {
    ffi::ExecutionState prepare;
    ffi::ExecutionState init;
  };

  // Creates a serializable custom call thunk. The callback is resolved using
  // XLA FFI.
  static absl::StatusOr<std::unique_ptr<CustomCallThunk>> Create(
      ThunkInfo thunk_info, std::string target_name,
      std::vector<NullableShapedSlice> operands,
      std::vector<NullableShapedSlice> results,
      xla::ffi::AttributesMap attributes,
      const HloComputation* called_computation, absl::string_view platform_name,
      const se::GpuComputeCapability& gpu_compute_capability,
      std::unique_ptr<xla::ffi::ExecutionState> execution_state = nullptr,
      std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options =
          std::nullopt);

  // Creates a serializable custom call thunk from the given XLA FFI handler
  // bundle. Note that `target_name` needs to refer to a registered XLA FFI
  // handler which matches the given bundle.
  static absl::StatusOr<std::unique_ptr<CustomCallThunk>> Create(
      ThunkInfo thunk_info, std::string target_name,
      XLA_FFI_Handler_Bundle bundle, std::vector<NullableShapedSlice> operands,
      std::vector<NullableShapedSlice> results,
      xla::ffi::AttributesMap attributes,
      const HloComputation* called_computation,
      const se::GpuComputeCapability& gpu_compute_capability,
      std::unique_ptr<xla::ffi::ExecutionState> execution_state = nullptr,
      std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options =
          std::nullopt);

  // Creates a custom call thunk from a bundle of handlers created with
  // xla::ffi::Bind(). Any pointer or reference lambda captures must be valid
  // for the lifetime of the thunk.
  static absl::StatusOr<std::unique_ptr<CustomCallThunk>> Create(
      ThunkInfo thunk_info, std::string target_name, OwnedHandlerBundle bundle,
      std::vector<NullableShapedSlice> operands,
      std::vector<NullableShapedSlice> results,
      xla::ffi::AttributesMap attributes,
      const HloComputation* called_computation,
      const se::GpuComputeCapability& gpu_compute_capability,
      std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options =
          std::nullopt);

  absl::Status Prepare(const PrepareParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const std::string& target_name() const { return target_name_; }

  std::optional<XLA_FFI_Handler_Bundle> bundle() const {
    const XLA_FFI_Handler_Bundle* c_bundle =
        std::get_if<XLA_FFI_Handler_Bundle>(&bundle_);
    return c_bundle ? std::make_optional(*c_bundle) : std::nullopt;
  }

  std::optional<ffi::CallFrame> call_frame() const {
    return call_frame_ ? std::make_optional(call_frame_->Copy()) : std::nullopt;
  }

  std::shared_ptr<ffi::ExecutionState> execution_state() const {
    return execution_state_;
  }

  const std::vector<NullableShapedSlice>& operands() const { return operands_; }
  const std::vector<NullableShapedSlice>& results() const { return results_; }

  BufferUses buffer_uses() const override {
    BufferUses res;
    res.reserve(operands_.size() + results_.size());
    for (const NullableShapedSlice& shaped_slice : operands_) {
      if (!shaped_slice.has_value()) {
        continue;
      }
      res.push_back(BufferUse::Read(shaped_slice->slice, shaped_slice->shape));
    }
    for (const NullableShapedSlice& shaped_slice : results_) {
      if (!shaped_slice.has_value()) {
        continue;
      }
      res.push_back(BufferUse::Write(shaped_slice->slice, shaped_slice->shape));
    }
    return res;
  }

  absl::StatusOr<ThunkProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<CustomCallThunk>> FromProto(
      ThunkInfo thunk_info, const CustomCallThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      const HloModule* absl_nullable hlo_module,
      absl::string_view platform_name,
      const se::GpuComputeCapability& gpu_compute_capability,
      std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options);

 private:
  CustomCallThunk(
      ThunkInfo thunk_info, std::string target_name,
      std::variant<XLA_FFI_Handler_Bundle, OwnedHandlerBundle> bundle,
      std::vector<NullableShapedSlice> operands,
      std::vector<NullableShapedSlice> results, ffi::CallFrame call_frame,
      xla::ffi::AttributesMap attributes,
      std::unique_ptr<ffi::ExecutionState> execution_state,
      const HloComputation* called_computation,
      std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options);

  absl::StatusOr<ObjectPool<xla::ffi::CallFrame>::BorrowedObject>
  BuildCallFrame(const BufferAllocations* absl_nullable buffer_allocations);

  xla::ffi::InvokeContext BuildInvokeContext(
      RunId run_id, se::Stream* absl_nullable stream,
      Thunk::ExecutionScopedState* absl_nullable execution_scoped_state,
      const BufferAllocations* absl_nullable buffer_allocations,
      const CollectiveParams* absl_nullable collective_params,
      CollectiveCliqueRequests* absl_nullable collective_clique_requests,
      CollectiveMemoryRequests* absl_nullable collective_memory_requests,
      const CollectiveCliques* absl_nullable collective_cliques,
      const CollectiveMemory* absl_nullable collective_memory,
      const ffi::ExecutionContext* absl_nullable execution_context,
      absl::Span<se::Stream* const> computation_streams);

  absl::Status ExecuteFfiHandler(
      RunId run_id, XLA_FFI_Handler* handler, XLA_FFI_ExecutionStage stage,
      se::Stream* stream, Thunk::ExecutionScopedState* execution_scoped_state,
      const ffi::ExecutionContext* execution_context,
      const BufferAllocations* buffer_allocations,
      const CollectiveParams* absl_nullable collective_params,
      CollectiveCliqueRequests* absl_nullable collective_clique_requests,
      CollectiveMemoryRequests* absl_nullable collective_memory_requests,
      const CollectiveCliques* absl_nullable collective_cliques,
      const CollectiveMemory* absl_nullable collective_memory,
      absl::Span<se::Stream* const> computation_streams);

  absl::Status ExecuteFfiHandler(
      RunId run_id, xla::ffi::Ffi& handler, xla::ffi::ExecutionStage stage,
      se::Stream* stream, Thunk::ExecutionScopedState* execution_scoped_state,
      const ffi::ExecutionContext* execution_context,
      const BufferAllocations* buffer_allocations,
      const CollectiveParams* absl_nullable collective_params,
      CollectiveCliqueRequests* absl_nullable collective_clique_requests,
      CollectiveMemoryRequests* absl_nullable collective_memory_requests,
      const CollectiveCliques* absl_nullable collective_cliques,
      const CollectiveMemory* absl_nullable collective_memory,
      absl::Span<se::Stream* const> computation_streams);

  std::string target_name_;

  // Nulled shape slices represent null pointer arguments to the thunk.
  std::vector<NullableShapedSlice> operands_;
  std::vector<NullableShapedSlice> results_;

  // XLA FFI handler bundle: either a C API bundle (from the global FFI
  // registry) or an owned bundle (from xla::ffi::Bind()).
  std::variant<XLA_FFI_Handler_Bundle, OwnedHandlerBundle> bundle_;
  std::optional<xla::ffi::AttributesMap> attributes_;

  // Reference call frame pre-initialized at construction time.
  std::optional<ffi::CallFrame> call_frame_;

  // A pool of call frames used at run time. Newly created call frames are
  // copied from the reference call frame and updated with buffer addresses.
  std::optional<ObjectPool<ffi::CallFrame>> call_frames_;

  // Execution state bound to the FFI handler instance. Optional.
  std::shared_ptr<ffi::ExecutionState> execution_state_;

  // TODO(ezhulenev): Currently we assume that HloModule that owns this
  // computation is owned by a GpuExecutable and stays alive for as long as
  // thunk is alive, however in general it might not be true and we can destroy
  // underlying HloModule. We have to make a copy of HloComputation for a thunk,
  // and also pass some form of relatively-ABI-stable representation to external
  // custom calls, i.e. we can pass it as HloComputationProto or as MLIR
  // bytecode of the computation serialized to StableHLO. Today we assume that
  // custom calls that access called computation can only be linked statically.
  const HloComputation* called_computation_ = nullptr;

  std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_CUSTOM_CALL_THUNK_H_
