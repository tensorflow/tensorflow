/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/gemm_thunk.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

GemmThunk::GemmThunk(ThunkInfo thunk_info, GemmConfig config,
                     const BufferAllocation::Slice& lhs_buffer,
                     const BufferAllocation::Slice& rhs_buffer,
                     const BufferAllocation::Slice& output_buffer,
                     std::optional<const BufferAllocation::Slice> workspace,
                     bool deterministic)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      workspace_(workspace),
      deterministic_(deterministic) {}

absl::Status GemmThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Running GEMM thunk";
  const BufferAllocations& allocs = *params.buffer_allocations;
  se::DeviceMemoryBase workspace(/*opaque=*/nullptr, /*size=*/0);
  if (workspace_.has_value()) {
    workspace = allocs.GetDeviceAddress(workspace_.value());
  }
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));

  return RunGemm(config_, allocs.GetDeviceAddress(lhs_buffer_),
                 allocs.GetDeviceAddress(rhs_buffer_),
                 allocs.GetDeviceAddress(output_buffer_), workspace,
                 deterministic_, stream);
}

absl::Status GemmThunk::Initialize(const InitializeParams& params) {
  if (!params.executor->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support");
  }
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> GemmThunk::ToProto() const {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, Thunk::ToProto());

  auto* gemm_thunk_proto = proto.mutable_gemm_thunk();
  *gemm_thunk_proto->mutable_gemm_config() = config_.ToProto();
  TF_ASSIGN_OR_RETURN(*gemm_thunk_proto->mutable_lhs_buffer(),
                      lhs_buffer_.ToProto());
  TF_ASSIGN_OR_RETURN(*gemm_thunk_proto->mutable_rhs_buffer(),
                      rhs_buffer_.ToProto());
  TF_ASSIGN_OR_RETURN(*gemm_thunk_proto->mutable_output_buffer(),
                      output_buffer_.ToProto());
  if (workspace_.has_value()) {
    TF_ASSIGN_OR_RETURN(*gemm_thunk_proto->mutable_workspace(),
                        workspace_.value().ToProto());
  }
  gemm_thunk_proto->set_deterministic(deterministic_);
  return proto;
}

absl::StatusOr<std::unique_ptr<GemmThunk>> GemmThunk::FromProto(
    ThunkInfo thunk_info, const GemmThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(stream_executor::gpu::GemmConfig config,
                      GemmConfig::FromProto(proto.gemm_config()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice lhs_buffer,
                      BufferAllocation::Slice::FromProto(proto.lhs_buffer(),
                                                         buffer_allocations));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice rhs_buffer,
                      BufferAllocation::Slice::FromProto(proto.rhs_buffer(),
                                                         buffer_allocations));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_buffer,
                      BufferAllocation::Slice::FromProto(proto.output_buffer(),
                                                         buffer_allocations));
  std::optional<BufferAllocation::Slice> workspace;
  if (proto.has_workspace()) {
    TF_ASSIGN_OR_RETURN(workspace, BufferAllocation::Slice::FromProto(
                                       proto.workspace(), buffer_allocations));
  }
  return std::make_unique<GemmThunk>(thunk_info, GemmConfig(config), lhs_buffer,
                                     rhs_buffer, output_buffer, workspace,
                                     proto.deterministic());
}

}  // namespace gpu
}  // namespace xla
