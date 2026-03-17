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

#include "xla/stream_executor/cuda/cub_scan_kernel_cuda.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/tsl/platform/statusor.h"

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(stream_executor::cuda::CubScanKind);

namespace stream_executor::cuda {

namespace {

absl::Status CubScanLaunchKernelFfiHandler(
    xla::ffi::AnyBuffer d_temp_storage, xla::ffi::AnyBuffer d_in,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_out, int64_t vector_length,
    int64_t row_length, int64_t column_length, CubScanKind kind,
    bool is_reverse, CUstream stream) {
  return CubScanLaunchKernel(d_in.element_type(), d_temp_storage.untyped_data(),
                             d_temp_storage.size_bytes(), d_in.untyped_data(),
                             d_out->untyped_data(), vector_length, row_length,
                             column_length, kind, is_reverse, stream);
}

absl::Status CubScanGetScratchSizeFfiHandler(
    xla::ffi::AnyBuffer d_in, size_t* temp_bytes, int64_t vector_length,
    int64_t row_length, int64_t column_length, CubScanKind kind,
    bool is_reverse) {
  TF_ASSIGN_OR_RETURN(
      *temp_bytes,
      CubScanGetScratchSize(d_in.element_type(), vector_length, row_length,
                            column_length, kind, is_reverse));
  return absl::OkStatus();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kCubScanExecute, CubScanLaunchKernelFfiHandler,
                       xla::ffi::Ffi::Bind()
                           .Arg<xla::ffi::AnyBuffer>()
                           .Arg<xla::ffi::AnyBuffer>()
                           .Ret<xla::ffi::AnyBuffer>()
                           .Attr<int64_t>("vector_length")
                           .Attr<int64_t>("row_length")
                           .Attr<int64_t>("column_length")
                           .Attr<CubScanKind>("kind")
                           .Attr<bool>("is_reverse")
                           .Ctx<xla::ffi::PlatformStream<CUstream>>());

XLA_FFI_DEFINE_HANDLER(
    kCubScanInitialize, CubScanGetScratchSizeFfiHandler,
    xla::ffi::Ffi::Bind<xla::ffi::ExecutionStage::kInitialize>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<xla::ffi::Pointer<size_t>>("temp_bytes")
        .Attr<int64_t>("vector_length")
        .Attr<int64_t>("row_length")
        .Attr<int64_t>("column_length")
        .Attr<CubScanKind>("kind")
        .Attr<bool>("is_reverse"));

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "xla.gpu.ext.cub_scan",
                         "CUDA",
                         {/* .instantiate = */ nullptr,
                          /* .prepare = */ nullptr,
                          /* .initialize = */ kCubScanInitialize,
                          /* .execute = */ kCubScanExecute});

}  // namespace stream_executor::cuda
