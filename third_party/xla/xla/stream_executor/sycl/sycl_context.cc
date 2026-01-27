/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_context.h"

namespace stream_executor::sycl {

absl::StatusOr<std::unique_ptr<SyclContext>> SyclContext::Create(
    int device_ordinal) {
  TF_ASSIGN_OR_RETURN(::sycl::context sycl_context,
                      SyclDevicePool::GetDeviceContext());
  return std::make_unique<SyclContext>(sycl_context, device_ordinal);
}

absl::StatusOr<uint64_t> SyclContext::GetDeviceTotalMemory(
    const ::sycl::device& device) {
  return device.get_info<::sycl::info::device::global_mem_size>();
}

absl::Status SyclContext::Synchronize() {
  return SyclStreamPool::SynchronizeStreamPool(device_ordinal_);
}

}  // namespace stream_executor::sycl
