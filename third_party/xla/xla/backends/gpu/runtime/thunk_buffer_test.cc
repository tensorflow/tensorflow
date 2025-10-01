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

#include "xla/backends/gpu/runtime/thunk_buffer.h"

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/service/buffer_assignment.h"

namespace xla::gpu {
namespace {

TEST(ThunkBufferTest, AbslStringify) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/123, /*size=*/456);

  const ThunkBuffer buffer{
      /*slice=*/slice,
      /*is_content_defined_on_input=*/true,
      /*is_content_defined_on_output=*/false,
  };

  EXPECT_EQ(absl::StrCat(buffer),
            "{slice:{index:0, offset:123, size:456}, "
            "is_content_defined_on_input:true, "
            "is_content_defined_on_output:false}");
}

}  // namespace
}  // namespace xla::gpu
