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

#ifndef XLA_PJRT_EXTENSIONS_HOST_ALLOCATOR_HOST_ALLOCATOR_INTERFACE_IMPL_H_
#define XLA_PJRT_EXTENSIONS_HOST_ALLOCATOR_HOST_ALLOCATOR_INTERFACE_IMPL_H_

#include <cstddef>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/extensions/host_allocator/host_allocator_extension.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {

// An implementation of the PjRtClient::HostAllocator interface that uses the
// C Client API and HostAllocator extension.
class HostAllocatorInterfaceImpl : public PjRtClient::HostAllocator {
 public:
  // Constructs a HostAllocatorInterfaceImpl.
  //
  // NOTE: client and extension must outlive this object.
  HostAllocatorInterfaceImpl(PJRT_Client* client,
                             PJRT_HostAllocator_Extension* extension);

  // Returns the preferred alignment for allocations.
  size_t GetPreferredAlignment() const override;

  // Allocates `size` bytes of memory.
  void* Allocate(size_t size, size_t alignment) override;

  // Frees `ptr` allocated by this allocator.
  void Free(void* ptr) override;

 private:
  PJRT_Client* const client_;
  PJRT_HostAllocator_Extension* const extension_;
};

}  // namespace xla

#endif  // XLA_PJRT_EXTENSIONS_HOST_ALLOCATOR_HOST_ALLOCATOR_INTERFACE_IMPL_H_
