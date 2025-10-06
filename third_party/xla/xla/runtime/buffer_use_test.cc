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

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/service/buffer_assignment.h"

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

TEST(BufferUseTest, HasReadWriteAccess) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, 10);

  BufferUse read = BufferUse::Read(slice);
  EXPECT_TRUE(read.HasReadAccess());
  EXPECT_FALSE(read.HasWriteAccess());

  BufferUse write = BufferUse::Write(slice);
  EXPECT_FALSE(write.HasReadAccess());
  EXPECT_TRUE(write.HasWriteAccess());

  BufferUse read_write = BufferUse::ReadWrite(slice);
  EXPECT_TRUE(read_write.HasReadAccess());
  EXPECT_TRUE(read_write.HasWriteAccess());
}

TEST(BufferUseTest, AbslStringify) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, 10);

  EXPECT_EQ(absl::StrCat(BufferUse::Read(slice)),
            "slice: {index:0, offset:0, size:10}, access: R");
  EXPECT_EQ(absl::StrCat(BufferUse::Write(slice)),
            "slice: {index:0, offset:0, size:10}, access: W");
  EXPECT_EQ(absl::StrCat(BufferUse::ReadWrite(slice)),
            "slice: {index:0, offset:0, size:10}, access: RW");
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
