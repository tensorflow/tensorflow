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

#include "xla/pjrt/extensions/host_allocator/host_allocator_extension.h"

#include "xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {

PJRT_HostAllocator_Extension CreateHostAllocatorExtension(
    PJRT_Extension_Base* next,
    PJRT_HostAllocator_GetPreferredAlignment get_preferred_alignment,
    PJRT_HostAllocator_Allocate allocate, PJRT_HostAllocator_Free free) {
  return PJRT_HostAllocator_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_HostAllocator_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_HostAllocator,
          /*next=*/next,
      },
      /*get_preferred_alignment=*/
      get_preferred_alignment,
      /*allocate=*/
      allocate,
      /*free=*/
      free,
  };
}

}  // namespace pjrt
