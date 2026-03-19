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
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

TEST(BufferUseTest, Equality) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  Shape slice0_shape = ShapeUtil::MakeShape(F32, {2});
  BufferAllocation::Slice slice0(&alloc, 0, 8);

  BufferUse use_read0 = BufferUse::Read(slice0, slice0_shape);
  BufferUse use_read1 = BufferUse::Read(slice0, slice0_shape);
  BufferUse use_write = BufferUse::Write(slice0, slice0_shape);
  BufferUse use_scratch = BufferUse::Scratch(slice0, slice0_shape);
  BufferUse use_consume = BufferUse::Consume(slice0, slice0_shape);

  EXPECT_EQ(use_read0, use_read1);
  EXPECT_NE(use_read0, use_write);
  EXPECT_NE(use_read0, use_scratch);
  EXPECT_NE(use_read0, use_consume);

  EXPECT_NE(use_write, use_scratch);
  EXPECT_NE(use_write, use_consume);

  EXPECT_NE(use_scratch, use_consume);
}

TEST(BufferUseTest, HasDefinedContents) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  Shape slice_shape = ShapeUtil::MakeShape(F32, {2});
  BufferAllocation::Slice slice(&alloc, 0, 8);

  BufferUse read = BufferUse::Read(slice, slice_shape);
  EXPECT_TRUE(read.HasDefinedContentsOnInput());
  EXPECT_TRUE(read.HasDefinedContentsOnOutput());

  BufferUse write = BufferUse::Write(slice, slice_shape);
  EXPECT_FALSE(write.HasDefinedContentsOnInput());
  EXPECT_TRUE(write.HasDefinedContentsOnOutput());

  BufferUse scratch = BufferUse::Scratch(slice, slice_shape);
  EXPECT_FALSE(scratch.HasDefinedContentsOnInput());
  EXPECT_FALSE(scratch.HasDefinedContentsOnOutput());

  BufferUse consume = BufferUse::Consume(slice, slice_shape);
  EXPECT_TRUE(consume.HasDefinedContentsOnInput());
  EXPECT_FALSE(consume.HasDefinedContentsOnOutput());
}

TEST(BufferUseTest, AbslStringify) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  Shape slice_shape = ShapeUtil::MakeShape(F32, {2});
  BufferAllocation::Slice slice(&alloc, 0, 8);

  EXPECT_EQ(
      absl::StrCat(BufferUse::Read(slice, slice_shape)),
      "{slice: {index:0, offset:0, size:8}, access: R, content_validity: IO}");
  EXPECT_EQ(
      absl::StrCat(BufferUse::Write(slice, slice_shape)),
      "{slice: {index:0, offset:0, size:8}, access: W, content_validity: O}");
  EXPECT_EQ(
      absl::StrCat(BufferUse::Scratch(slice, slice_shape)),
      "{slice: {index:0, offset:0, size:8}, access: W, content_validity: }");
  EXPECT_EQ(
      absl::StrCat(BufferUse::Consume(slice, slice_shape)),
      "{slice: {index:0, offset:0, size:8}, access: W, content_validity: I}");
}

TEST(BufferUseTest, ReadWriteSet) {
  BufferUse::ReadWriteSet rwset;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);

  Shape slice_shape = ShapeUtil::MakeShape(F32, {2});
  BufferAllocation::Slice slice0(&alloc, 0, 8);
  BufferAllocation::Slice slice1(&alloc, 4, 8);
  BufferAllocation::Slice slice2(&alloc, 8, 8);

  rwset.Add(BufferUse::Read(slice0, slice_shape));
  EXPECT_FALSE(rwset.HasConflicts({BufferUse::Read(slice1, slice_shape)}));
  EXPECT_TRUE(rwset.HasConflicts({BufferUse::Write(slice1, slice_shape)}));
  EXPECT_FALSE(rwset.HasConflicts({BufferUse::Write(slice2, slice_shape)}));

  rwset.Add(BufferUse::Read(slice1, slice_shape));
  EXPECT_TRUE(rwset.HasConflicts({BufferUse::Write(slice2, slice_shape)}));
}

}  // namespace
}  // namespace xla
