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

#ifndef XLA_PJRT_EXTENSIONS_HOST_MEMORY_ALLOCATOR_HOST_MEMORY_ALLOCATOR_INTERFACE_IMPL_H_
#define XLA_PJRT_EXTENSIONS_HOST_MEMORY_ALLOCATOR_HOST_MEMORY_ALLOCATOR_INTERFACE_IMPL_H_

#include <memory>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/extensions/host_memory_allocator/host_memory_allocator_extension.h"
#include "xla/pjrt/host_memory_allocator.h"

namespace pjrt {

std::unique_ptr<xla::HostMemoryAllocator> CreateHostMemoryAllocatorWrapper(
    PJRT_Client* c_client, const PJRT_HostMemoryAllocator_Extension* extension,
    const PJRT_Api* c_api);

}  // namespace pjrt

#endif  // XLA_PJRT_EXTENSIONS_HOST_MEMORY_ALLOCATOR_HOST_MEMORY_ALLOCATOR_INTERFACE_IMPL_H_
