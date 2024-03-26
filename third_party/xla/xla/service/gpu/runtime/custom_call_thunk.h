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

#ifndef XLA_SERVICE_GPU_RUNTIME_CUSTOM_CALL_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_CUSTOM_CALL_THUNK_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/status.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_types.h"
#endif

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
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  using Stream = stream_executor::gpu::GpuStreamHandle;
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  using Stream = void*;
#endif  //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  using CustomCallTarget = std::function<void(Stream, void**, const char*,
                                              size_t, XlaCustomCallStatus*)>;

  // We keep buffer allocation slice together with its shape to be able to fill
  // FFI arguments with required details.
  struct Slice {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  using Attribute = ffi::CallFrameBuilder::FlatAttribute;
  using AttributesMap = ffi::CallFrameBuilder::FlatAttributesMap;

  CustomCallThunk(ThunkInfo thunk_info, CustomCallTarget call_target,
                  std::vector<std::optional<Slice>> operands,
                  std::vector<std::optional<Slice>> results,
                  const std::string& opaque);

  CustomCallThunk(ThunkInfo thunk_info, XLA_FFI_Handler* handler,
                  std::vector<std::optional<Slice>> operands,
                  std::vector<std::optional<Slice>> results,
                  AttributesMap attributes,
                  const HloComputation* called_computation);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const CustomCallTarget& call_target() const { return call_target_; }
  const std::vector<std::optional<Slice>>& operands() const {
    return operands_;
  }
  const std::vector<std::optional<Slice>>& results() const { return results_; }
  absl::string_view opaque() const { return opaque_; }

 private:
  absl::Status ExecuteCustomCall(const ExecuteParams& params);
  absl::Status ExecuteFfiHandler(const ExecuteParams& params);

  std::vector<std::optional<Slice>> operands_;
  std::vector<std::optional<Slice>> results_;

  // This is a legacy custom call API that is discouraged, and will be
  // deprecated once XLA:FFI mechanism is ready.
  CustomCallTarget call_target_;
  std::string opaque_;

  // XLA FFI provides a right type safe mechanism for registering external
  // functions with XLA runtime. It's under construction, and still misses
  // a lot of features. Long term it will replace legacy custom calls.
  XLA_FFI_Handler* handler_ = nullptr;
  AttributesMap attributes_;

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

// Converts MLIR dictionary attribute attached to a custom call operation to a
// custom call thunk attributes that are forwarded to the FFI handler.
absl::StatusOr<CustomCallThunk::AttributesMap> BuildAttributesMap(
    mlir::DictionaryAttr dict);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_CUSTOM_CALL_THUNK_H_
