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

#include "xla/pjrt/extensions/host_allocator/host_memory_allocator/host_memory_allocator_extension_internal.h"

#include "absl/status/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/extensions/host_allocator/host_memory_allocator/host_memory_allocator_extension.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"

namespace pjrt {

namespace {

PJRT_Error* HostMemoryAllocator_Allocate(
    PJRT_HostMemoryAllocator_Allocate_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_HostMemoryAllocator_Allocate_Args",
      PJRT_HostMemoryAllocator_Allocate_Args_STRUCT_SIZE, args->struct_size));
  if (args->client == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "Received null client in HostMemoryAllocator_Allocate"));
  }
  xla::HostMemoryAllocator* host_memory_allocator =
      args->client->client->GetHostMemoryAllocator();
  if (host_memory_allocator == nullptr) {
    return StatusToPjRtError(absl::UnimplementedError(
        "HostMemoryAllocator not implemented for client"));
  }
  xla::HostMemoryAllocator::AllocateOptions options;
  options.numa_node = args->numa_node;

  xla::HostMemoryAllocator::OwnedPtr owned_ptr =
      host_memory_allocator->Allocate(args->size, options);
  if (owned_ptr == nullptr) {
    args->ptr = nullptr;
    args->deleter = nullptr;
    args->deleter_arg = nullptr;
  } else {
    xla::HostMemoryAllocator::Deleter deleter = owned_ptr.get_deleter();
    args->deleter = deleter.deleter;
    args->deleter_arg = deleter.arg;
    args->ptr = owned_ptr.release();
  }
  return nullptr;
}

}  // namespace

PJRT_HostMemoryAllocator_Extension CreateHostMemoryAllocatorExtension(
    PJRT_Extension_Base* next) {
  return PJRT_HostMemoryAllocator_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_HostMemoryAllocator_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_HostMemoryAllocator,
          /*next=*/next,
      },
      /*allocate=*/HostMemoryAllocator_Allocate,
  };
}

}  // namespace pjrt
