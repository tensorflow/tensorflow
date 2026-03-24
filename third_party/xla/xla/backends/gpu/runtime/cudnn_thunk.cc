/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/backends/gpu/runtime/cudnn_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

CuDnnThunk::CuDnnThunk(std::string fingerprint, ThunkInfo thunk_info,
                       std::vector<ShapedSlice> args,
                       std::vector<bool> output_args,
                       std::optional<int64_t> sdpa_dropout_seed)
    : Thunk(Kind::kCuDnn, std::move(thunk_info)),
      fingerprint_(std::move(fingerprint)),
      graph_(std::make_shared<se::dnn::LazyDnnGraph>(nullptr)),
      args_(std::move(args)),
      output_args_(std::move(output_args)),
      sdpa_dropout_seed_(sdpa_dropout_seed) {}

absl::Status CuDnnThunk::Initialize(const InitializeParams& params) {
  absl::Status ret = absl::OkStatus();
  // Calling AsDnn outside call_once ensures that cuDNN handles get created for
  // all GPUs in programs using cuDNN during the executable initialization
  // phase. It's sufficient to deserialize the graph once using just one of
  // them.
  se::dnn::DnnSupport* dnn = params.stream->parent()->AsDnn();
  absl::call_once(once_flag_, [&] {
    auto result = dnn->DeserializeGraph(
        *params.stream, params.src.dnn_compiled_graphs.at(fingerprint_));
    std::string().swap(fingerprint_);
    if (result.ok()) {
      graph_->swap(*result);
      if (sdpa_dropout_seed_.has_value()) {
        graph_->get()->InitDropoutState(params.local_device_count,
                                        *sdpa_dropout_seed_, 16);
      }
    }
    ret = result.status();
  });
  return ret;
}

absl::Status CuDnnThunk::ExecuteOnStream(const ExecuteParams& params) {
  InitializeParams initialize_params;
  initialize_params.stream = params.stream;
  TF_RETURN_IF_ERROR(Initialize(initialize_params));
  std::vector<se::DeviceAddressBase> buffer_args;
  buffer_args.reserve(args_.size());
  for (const ShapedSlice& arg : args_) {
    auto addr = params.buffer_allocations->GetDeviceAddress(arg.slice);
    if (output_args_[buffer_args.size()]) {
      tsl::profiler::MarkMemoryInitialized(
          addr.opaque(), addr.size(),
          static_cast<tsl::profiler::StreamHandle>(
              params.stream->platform_specific_handle().stream));
    }
    buffer_args.push_back(addr);
  }
  return graph_->get()->Execute(
      *params.stream, absl::Span<se::DeviceAddressBase>(buffer_args),
      params.collective_params->local_device_id.value());
}

absl::StatusOr<ThunkProto> CuDnnThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  proto.mutable_cudnn_thunk()->set_fingerprint(fingerprint_);

  for (const ShapedSlice& arg : args_) {
    ASSIGN_OR_RETURN(*proto.mutable_cudnn_thunk()->add_args(), arg.ToProto());
  }
  for (const bool is_output : output_args_) {
    proto.mutable_cudnn_thunk()->add_output_args(is_output);
  }
  if (sdpa_dropout_seed_.has_value()) {
    proto.mutable_cudnn_thunk()->set_sdpa_dropout_seed(
        static_cast<int64_t>(*sdpa_dropout_seed_));
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<CuDnnThunk>> CuDnnThunk::FromProto(
    ThunkInfo thunk_info, const CudnnThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<ShapedSlice> args;
  args.reserve(proto.args_size());
  for (const ShapedSliceProto& arg : proto.args()) {
    ASSIGN_OR_RETURN(args.emplace_back(),
                     ShapedSlice::FromProto(arg, buffer_allocations));
  }
  std::vector<bool> output_args;
  output_args.reserve(proto.output_args_size());
  for (const bool output_arg : proto.output_args()) {
    output_args.push_back(output_arg);
  }
  std::optional<uint64_t> sdpa_dropout_seed;
  if (proto.has_sdpa_dropout_seed()) {
    sdpa_dropout_seed = static_cast<uint64_t>(proto.sdpa_dropout_seed());
  }
  return std::make_unique<CuDnnThunk>(
      proto.fingerprint(), std::move(thunk_info), std::move(args),
      std::move(output_args), sdpa_dropout_seed);
}

}  // namespace gpu
}  // namespace xla
