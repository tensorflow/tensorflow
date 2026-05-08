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
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/xla_data.pb.h"

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(stream_executor::cuda::CubScanKind);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(xla::PrimitiveType);

namespace stream_executor::cuda {

namespace {

absl::Status CubScanLaunchKernelFfiHandler(
    xla::ffi::AnyBuffer d_in, xla::ffi::Result<xla::ffi::AnyBuffer> d_out,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_temp_storage, int64_t vector_length,
    int64_t row_length, int64_t column_length, CubScanKind kind,
    bool is_reverse, CUstream stream) {
  return CubScanLaunchKernel(
      d_in.element_type(), d_temp_storage->untyped_data(),
      d_temp_storage->size_bytes(), d_in.untyped_data(), d_out->untyped_data(),
      vector_length, row_length, column_length, kind, is_reverse, stream);
}

absl::StatusOr<std::unique_ptr<int64_t>> CubScanGetScratchSizeFfiHandler(
    xla::PrimitiveType element_type, int64_t vector_length, int64_t row_length,
    int64_t column_length, CubScanKind kind, bool is_reverse) {
  ASSIGN_OR_RETURN(
      size_t temp_bytes,
      CubScanGetScratchSize(element_type, vector_length, row_length,
                            column_length, kind, is_reverse));
  return std::make_unique<int64_t>(temp_bytes);
}

absl::Status CubScanDummyExecuteFfiHandler() {
  return absl::InternalError("Dummy execute handler should not be called");
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kCubScanExecute, CubScanLaunchKernelFfiHandler,
                       xla::ffi::Ffi::Bind()
                           .Arg<xla::ffi::AnyBuffer>()  // d_in
                           .Ret<xla::ffi::AnyBuffer>()  // d_out
                           .Ret<xla::ffi::AnyBuffer>()  // d_temp_storage
                           .Attr<int64_t>("vector_length")
                           .Attr<int64_t>("row_length")
                           .Attr<int64_t>("column_length")
                           .Attr<CubScanKind>("kind")
                           .Attr<bool>("is_reverse")
                           .Ctx<xla::ffi::PlatformStream<CUstream>>(),
                       {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         xla::gpu::kCubDeviceScanTarget.data(), "CUDA",
                         {/*instantiate=*/nullptr, /*prepare=*/nullptr,
                          /*initialize=*/nullptr,
                          /*.execute=*/kCubScanExecute});

XLA_FFI_DEFINE_HANDLER(kCubScanInstantiate, CubScanGetScratchSizeFfiHandler,
                       xla::ffi::Ffi::BindInstantiate()
                           .Attr<xla::PrimitiveType>("element_type")
                           .Attr<int64_t>("vector_length")
                           .Attr<int64_t>("row_length")
                           .Attr<int64_t>("column_length")
                           .Attr<CubScanKind>("kind")
                           .Attr<bool>("is_reverse"));

XLA_FFI_DEFINE_HANDLER(kCubScanDummyExecute, CubScanDummyExecuteFfiHandler,
                       xla::ffi::Ffi::Bind());

XLA_FFI_REGISTER_HANDLER(
    xla::ffi::GetXlaFfiApi(),
    xla::gpu::kCubDeviceScanUnassignedScratchSizeTarget.data(), "CUDA",
    {/*.instantiate=*/kCubScanInstantiate, /*prepare=*/nullptr,
     /*initialize=*/nullptr, /*execute=*/kCubScanDummyExecute});

}  // namespace stream_executor::cuda
