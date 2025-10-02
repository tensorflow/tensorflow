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

#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/extensions/host_allocator/host_allocator_extension.h"

namespace xla {
namespace {

// Returns a dummy client for testing.
PJRT_Client* ClientForTest() {
  PJRT_Client* client = reinterpret_cast<PJRT_Client*>(0xcafebabe);
  return client;
}

TEST(HostAllocatorInterfaceImplTest, GetPreferredAlignment) {
  PJRT_HostAllocator_Extension extension = {
      /*base=*/{0},
      /*get_preferred_alignment=*/
      +[](PJRT_HostAllocator_GetPreferredAlignment_Args* args) -> PJRT_Error* {
        EXPECT_EQ(args->client, ClientForTest());
        args->preferred_alignment = 64;
        return nullptr;
      },
      /*allocate=*/nullptr,
      /*free=*/nullptr,
  };
  HostAllocatorInterfaceImpl host_allocator_interface_impl(ClientForTest(),
                                                           &extension);
  EXPECT_EQ(host_allocator_interface_impl.GetPreferredAlignment(), 64);
}

TEST(HostAllocatorInterfaceImplTest, Allocate) {
  PJRT_HostAllocator_Extension extension = {
      /*base=*/{0},
      /*get_preferred_alignment=*/nullptr,
      /*allocate=*/
      +[](PJRT_HostAllocator_Allocate_Args* args) -> PJRT_Error* {
        EXPECT_EQ(args->client, ClientForTest());
        EXPECT_EQ(args->size, 1024);
        EXPECT_EQ(args->alignment, 64);
        args->ptr = reinterpret_cast<void*>(0xdeadbeef);
        return nullptr;
      },
      /*free=*/nullptr,
  };
  HostAllocatorInterfaceImpl host_allocator_interface_impl(ClientForTest(),
                                                           &extension);
  EXPECT_EQ(host_allocator_interface_impl.Allocate(1024, 64),
            reinterpret_cast<void*>(0xdeadbeef));
}

TEST(HostAllocatorInterfaceImplTest, Free) {
  PJRT_HostAllocator_Extension extension = {
      /*base=*/{0},
      /*get_preferred_alignment=*/nullptr,
      /*allocate=*/
      +[](PJRT_HostAllocator_Allocate_Args* args) -> PJRT_Error* {
        EXPECT_EQ(args->client, ClientForTest());
        EXPECT_EQ(args->size, 1024);
        EXPECT_EQ(args->alignment, 64);
        args->ptr = reinterpret_cast<void*>(0xdeadbeef);
        return nullptr;
      },
      /*free=*/
      +[](PJRT_HostAllocator_Free_Args* args) -> PJRT_Error* {
        EXPECT_EQ(args->client, ClientForTest());
        EXPECT_EQ(args->ptr, reinterpret_cast<void*>(0xdeadbeef));
        return nullptr;
      },
  };
  HostAllocatorInterfaceImpl host_allocator_interface_impl(ClientForTest(),
                                                           &extension);
  host_allocator_interface_impl.Free(reinterpret_cast<void*>(0xdeadbeef));
}

}  // namespace
}  // namespace xla
