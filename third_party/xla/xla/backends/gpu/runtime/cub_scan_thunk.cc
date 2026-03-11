/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/cub_scan_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

absl::StatusOr<ffi::HandlerRegistration> GetPrefixSumInitializer(
    PrimitiveType type) {
  switch (type) {
    case BF16:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_bf16", "CUDA");
    case F16:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_f16", "CUDA");
    case F32:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_f32", "CUDA");
    case F64:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_f64", "CUDA");
    case S8:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_s8", "CUDA");
    case S16:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_s16", "CUDA");
    case S32:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_s32", "CUDA");
    case S64:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_s64", "CUDA");
    case U8:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_u8", "CUDA");
    case U16:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_u16", "CUDA");
    case U32:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_u32", "CUDA");
    case U64:
      return ffi::FindHandler("xla.gpu.ext.cub_prefix_sum_u64", "CUDA");
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported type for cub prefix sum: %s",
                          primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

class CubScanImpl : public CubScanRunnerInterface {
 public:
  explicit CubScanImpl(ffi::HandlerRegistration ffi_handler)
      : ffi_handler_(ffi_handler) {}

  absl::Status Run(const se::DeviceAddressBase& input_buffer,
                   const se::DeviceAddressBase& output_buffer,
                   const se::DeviceAddressBase& scratch_buffer,
                   int64_t num_elements, se::Stream* stream) override {
    ffi::CallFrameBuilder builder(2, 1);
    builder.AddBufferArg(scratch_buffer, PrimitiveType::U8,
                         {static_cast<int64_t>(scratch_buffer.size())});
    builder.AddBufferArg(input_buffer, PrimitiveType::U8,
                         {static_cast<int64_t>(input_buffer.size())});
    builder.AddBufferRet(output_buffer, PrimitiveType::U8,
                         {static_cast<int64_t>(output_buffer.size())});

    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Insert("num_items", static_cast<int64_t>(num_elements));
    builder.AddAttributes(attrs.Build());

    ffi::CallFrame call_frame = builder.Build();
    ffi::InvokeContext context{};
    context.backend_context = ffi::InvokeContext::GpuContext{stream, nullptr};
    return ffi::Invoke(ffi::GetXlaFfiApi(), ffi_handler_.bundle.execute,
                       call_frame, context, XLA_FFI_ExecutionStage_EXECUTE);
  }

  absl::StatusOr<int64_t> GetScratchSize(int64_t num_elements) override {
    size_t scratch_size_bytes = 0;

    ffi::CallFrameBuilder builder(0, 0);
    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Insert("temp_bytes", absl::bit_cast<int64_t>(&scratch_size_bytes));
    attrs.Insert("num_items", static_cast<int64_t>(num_elements));
    builder.AddAttributes(attrs.Build());

    ffi::CallFrame call_frame = builder.Build();

    TF_RETURN_IF_ERROR(ffi::Invoke(
        ffi::GetXlaFfiApi(), ffi_handler_.bundle.initialize, call_frame,
        ffi::InvokeContext{}, XLA_FFI_ExecutionStage_INITIALIZE));

    return scratch_size_bytes;
  }

 private:
  ffi::HandlerRegistration ffi_handler_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<CubScanRunnerInterface>>
CubScanRunnerInterface::Create(PrimitiveType type,
                               const std::string& platform_name) {
  if (platform_name == "CUDA") {
    TF_ASSIGN_OR_RETURN(auto ffi_handler, GetPrefixSumInitializer(type));
    return std::make_unique<CubScanImpl>(ffi_handler);
  }
  return absl::UnimplementedError(absl::StrFormat(
      "CUB prefix sum is not supported on platform %s", platform_name));
}

absl::StatusOr<std::unique_ptr<CubScanThunk>> CubScanThunk::Create(
    ThunkInfo thunk_info, PrimitiveType type,
    const BufferAllocation::Slice& input_slice,
    const BufferAllocation::Slice& output_slice,
    const BufferAllocation::Slice& scratch_slice, int64_t num_elements) {
  TF_ASSIGN_OR_RETURN(auto runner,
                      CubScanRunnerInterface::Create(type, "CUDA"));
  return std::make_unique<CubScanThunk>(
      std::move(thunk_info), std::move(runner), type, "CUDA", input_slice,
      output_slice, scratch_slice, num_elements);
}

CubScanThunk::CubScanThunk(ThunkInfo thunk_info,
                           std::unique_ptr<CubScanRunnerInterface> runner,
                           PrimitiveType type, std::string platform_name,
                           const BufferAllocation::Slice& input_slice,
                           const BufferAllocation::Slice& output_slice,
                           const BufferAllocation::Slice& scratch_slice,
                           int64_t num_elements)
    : Thunk(Thunk::Kind::kCustomCall, std::move(thunk_info)),
      runner_(std::move(runner)),
      type_(type),
      platform_name_(std::move(platform_name)),
      input_slice_(input_slice),
      output_slice_(output_slice),
      scratch_slice_(scratch_slice),
      num_elements_(num_elements) {}

absl::Status CubScanThunk::Initialize(const InitializeParams& params) {
  if (!runner_) {
    TF_ASSIGN_OR_RETURN(runner_,
                        CubScanRunnerInterface::Create(type_, platform_name_));
  }
  return absl::OkStatus();
}

absl::Status CubScanThunk::ExecuteOnStream(const ExecuteParams& params) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  se::DeviceAddressBase input_buffer = allocs.GetDeviceAddress(input_slice_);
  se::DeviceAddressBase output_buffer = allocs.GetDeviceAddress(output_slice_);
  se::DeviceAddressBase scratch_buffer =
      allocs.GetDeviceAddress(scratch_slice_);

  return runner_->Run(input_buffer, output_buffer, scratch_buffer,
                      num_elements_, params.stream);
}

absl::StatusOr<ThunkProto> CubScanThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  auto* cub_scan_thunk_proto = proto.mutable_cub_scan_thunk();
  cub_scan_thunk_proto->set_type(type_);
  cub_scan_thunk_proto->set_platform_name(platform_name_);
  TF_ASSIGN_OR_RETURN(*cub_scan_thunk_proto->mutable_input_slice(),
                      input_slice_.ToProto());
  TF_ASSIGN_OR_RETURN(*cub_scan_thunk_proto->mutable_output_slice(),
                      output_slice_.ToProto());
  TF_ASSIGN_OR_RETURN(*cub_scan_thunk_proto->mutable_scratch_slice(),
                      scratch_slice_.ToProto());
  cub_scan_thunk_proto->set_num_elements(num_elements_);
  return proto;
}

absl::StatusOr<std::unique_ptr<CubScanThunk>> CubScanThunk::FromProto(
    ThunkInfo thunk_info, const CubScanThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice input_slice,
                      BufferAllocation::Slice::FromProto(proto.input_slice(),
                                                         buffer_allocations));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      BufferAllocation::Slice::FromProto(proto.output_slice(),
                                                         buffer_allocations));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scratch_slice,
                      BufferAllocation::Slice::FromProto(proto.scratch_slice(),
                                                         buffer_allocations));
  return std::make_unique<CubScanThunk>(
      std::move(thunk_info), /*runner=*/nullptr, proto.type(),
      proto.platform_name(), input_slice, output_slice, scratch_slice,
      proto.num_elements());
}

}  // namespace xla::gpu
