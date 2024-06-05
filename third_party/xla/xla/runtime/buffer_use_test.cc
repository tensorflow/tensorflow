/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/runtime/buffer_use.h"

#include "xla/service/buffer_assignment.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(BufferUseTest, Equality) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);

  BufferUse use0(slice0, BufferUse::MemoryAccess::kRead);
  BufferUse use1(slice0, BufferUse::MemoryAccess::kWrite);
  BufferUse use2(slice0, BufferUse::MemoryAccess::kRead);

  EXPECT_NE(use0, use1);
  EXPECT_EQ(use0, use2);
}

TEST(BufferUseTest, ReadWriteSet) {
  BufferUse::ReadWriteSet rwset;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, 0, 10);
  BufferAllocation::Slice slice1(&alloc, 5, 10);
  BufferAllocation::Slice slice2(&alloc, 10, 10);

  rwset.Add(BufferUse::Read(slice0));
  EXPECT_FALSE(rwset.HasConflicts({BufferUse::Read(slice1)}));
  EXPECT_TRUE(rwset.HasConflicts({BufferUse::Write(slice1)}));
  EXPECT_FALSE(rwset.HasConflicts({BufferUse::Write(slice2)}));

  rwset.Add(BufferUse::Read(slice1));
  EXPECT_TRUE(rwset.HasConflicts({BufferUse::Write(slice2)}));
}

}  // namespace
}  // namespace xla
