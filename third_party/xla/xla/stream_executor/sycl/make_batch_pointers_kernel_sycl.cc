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

#include <cstddef>

// clang-format off
#include <sycl/sycl.hpp>
// clang-format on

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/make_batch_pointers_kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace {

namespace syclexp = ::sycl::ext::oneapi::experimental;
namespace syclext = ::sycl::ext::oneapi;

}  // namespace

// Free-function kernels require global scope since stream_executor::sycl
// namespace would shadow ::sycl in SYCL_EXT_ONEAPI_FUNCTION_PROPERTY's macro
// expansion. Also, the toolchain-generated integration header cannot resolve a
// namespaced free-function kernel.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void MakeBatchPointersSycl(char* base, size_t stride, size_t n,
                           void** ptrs_out) {
  size_t idx = syclext::this_work_item::get_nd_item<3>().get_global_linear_id();
  if (idx >= n) {
    return;
  }
  ptrs_out[idx] = base + idx * stride;
}

namespace stream_executor::sycl {

// Host-only: get_kernel_id<> needs is_kernel<> specialization from the
// integration header, which is unavailable during the device compilation pass.
#ifndef __SYCL_DEVICE_ONLY__

absl::StatusOr<::sycl::kernel_id> GetMakeBatchPointersKernelId() {
  try {
    return syclexp::get_kernel_id<MakeBatchPointersSycl>();
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(absl::StrCat(
        "GetMakeBatchPointersKernelId: Failed to get kernel id, got ",
        e.what()));
  }
}

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    MakeBatchPointersKernelSycl, stream_executor::gpu::MakeBatchPointersKernel,
    stream_executor::sycl::kSyclPlatformId, ([](size_t arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(
              &stream_executor::sycl::GetMakeBatchPointersKernelId),
          "MakeBatchPointersSycl", arity);
    }));

#endif  // __SYCL_DEVICE_ONLY__

}  // namespace stream_executor::sycl
