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

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

// Thunk to run a GPU custom call.
//
// This thunk's `ExecuteOnStream` implementation executes a host function
// `call_target` which is expected to enqueue operations onto the GPU.
//
// Note that not all kCustomCall HLOs in XLA:GPU end up being run by this thunk.
// XLA itself creates kCustomCall instructions when lowering kConvolution HLOs
// into calls to cudnn.  These internally-created custom-calls are run using
// ConvolutionThunk, not CustomCallThunk.  There's no ambiguity because they
// have special call target names (e.g. "__cudnn$convForward") that only the
// compiler is allowed to create.
class CustomCallThunk : public Thunk {
 public:
  using CustomCallTarget =
      std::function<void(stream_executor::Stream*, void**, const char*, size_t,
                         XlaCustomCallStatus*)>;

  // We keep buffer allocation slice together with its shape to be able to fill
  // FFI arguments with required details.
  struct Slice {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  using Attribute = ffi::CallFrameBuilder::Attribute;
  using AttributesMap = ffi::CallFrameBuilder::AttributesMap;

  static absl::StatusOr<std::unique_ptr<CustomCallThunk>> Create(
      ThunkInfo thunk_info, std::string target_name,
      CustomCallTarget call_target, std::vector<std::optional<Slice>> operands,
      std::vector<std::optional<Slice>> results, const std::string& opaque);

  static absl::StatusOr<std::unique_ptr<CustomCallThunk>> Create(
      ThunkInfo thunk_info, std::string target_name,
      XLA_FFI_Handler_Bundle bundle, std::vector<std::optional<Slice>> operands,
      std::vector<std::optional<Slice>> results, AttributesMap attributes,
      const HloComputation* called_computation);

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequestsInterface& resource_requests) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const std::string& target_name() const { return target_name_; }
  CustomCallTarget call_target() const { return call_target_; }
  std::optional<XLA_FFI_Handler_Bundle> bundle() const { return bundle_; }
  const AttributesMap& attributes() const { return attributes_; }

  const std::vector<std::optional<Slice>>& operands() const {
    return operands_;
  }
  const std::vector<std::optional<Slice>>& results() const { return results_; }

  absl::string_view opaque() const { return opaque_; }

 private:
  CustomCallThunk(ThunkInfo thunk_info, std::string target_name,
                  CustomCallTarget call_target,
                  std::vector<std::optional<Slice>> operands,
                  std::vector<std::optional<Slice>> results,
                  const std::string& opaque);

  CustomCallThunk(ThunkInfo thunk_info, std::string target_name,
                  XLA_FFI_Handler_Bundle bundle,
                  std::vector<std::optional<Slice>> operands,
                  std::vector<std::optional<Slice>> results,
                  AttributesMap attributes,
                  std::unique_ptr<ffi::ExecutionState> execution_state,
                  const HloComputation* called_computation);

  absl::Status ExecuteCustomCall(const ExecuteParams& params);

  absl::Status ExecuteFfiHandler(RunId run_id, XLA_FFI_Handler* handler,
                                 XLA_FFI_ExecutionStage stage,
                                 se::Stream* stream,
                                 const ffi::ExecutionContext* execution_context,
                                 const BufferAllocations* buffer_allocations);

  std::string target_name_;

  std::vector<std::optional<Slice>> operands_;
  std::vector<std::optional<Slice>> results_;

  // This is a legacy custom call API that is discouraged, and will be
  // deprecated once XLA:FFI mechanism is ready.
  CustomCallTarget call_target_;
  std::string opaque_;

  // XLA FFI provides a right type safe mechanism for registering external
  // functions with XLA runtime. It's under construction, and still misses
  // a lot of features. Long term it will replace legacy custom calls.
  std::optional<XLA_FFI_Handler_Bundle> bundle_;
  AttributesMap attributes_;

  // Execution state bound to the FFI handler. Optional.
  std::unique_ptr<ffi::ExecutionState> execution_state_;

  // TODO(ezhulenev): Currently we assume that HloModule that owns this
  // computation is owned by a GpuExecutable and stays alive for as long as
  // thunk is alive, however in general it might not be true and we can destroy
  // underlying HloModule. We have to make a copy of HloComputation for a thunk,
  // and also pass some form of relatively-ABI-stable representation to external
  // custom calls, i.e. we can pass it as HloComputationProto or as MLIR
  // bytecode of the computation serialized to StableHLO. Today we assume that
  // custom calls that access called computation can only be linked statically.
  const HloComputation* called_computation_ = nullptr;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_CUSTOM_CALL_THUNK_H_
