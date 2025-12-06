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

#include "xla/pjrt/extensions/host_allocator/host_allocator_interface_impl.h"

#include <cstddef>

#include "absl/log/check.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/extensions/host_allocator/host_allocator_extension.h"

namespace xla {

HostAllocatorInterfaceImpl::HostAllocatorInterfaceImpl(
    PJRT_Client* client, PJRT_HostAllocator_Extension* extension)
    : client_(client), extension_(extension) {}

size_t HostAllocatorInterfaceImpl::GetPreferredAlignment() const {
  PJRT_HostAllocator_GetPreferredAlignment_Args args = {
      /*struct_size=*/sizeof(args),
      /*extension_start=*/&extension_->base,
      /*client=*/client_,
  };
  PJRT_Error* error = extension_->get_preferred_alignment(&args);
  CHECK_EQ(error, nullptr) << "Failed to get preferred alignment: " << error;
  return args.preferred_alignment;
}

void* HostAllocatorInterfaceImpl::Allocate(size_t size, size_t alignment) {
  PJRT_HostAllocator_Allocate_Args args = {
      /*struct_size=*/sizeof(args),
      /*extension_start=*/&extension_->base,
      /*client*/ client_,
      /*size =*/size,
      /*alignment =*/alignment,
  };
  PJRT_Error* error = extension_->allocate(&args);
  CHECK_EQ(error, nullptr) << "Failed to allocate memory: " << error;
  return args.ptr;
}

void HostAllocatorInterfaceImpl::Free(void* ptr) {
  PJRT_HostAllocator_Free_Args args = {
      /*struct_size=*/sizeof(args),
      /*extension_start=*/&extension_->base,
      /*client=*/client_,
      /*ptr=*/ptr,
  };
  PJRT_Error* error = extension_->free(&args);
  CHECK_EQ(error, nullptr) << "Failed to free memory: " << error;
}

}  // namespace xla
