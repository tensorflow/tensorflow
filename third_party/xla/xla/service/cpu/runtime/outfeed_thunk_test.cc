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

#include "xla/service/cpu/runtime/outfeed_thunk.h"

#include <memory>

#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(OutfeedThunkTest, BufferUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice outfeed_slice(&alloc, 10, 40);

  OutfeedThunk::OutfeedBuffer outfeed_buffer = {
      outfeed_slice,
      ShapeUtil::MakeShape(F32, {10}),
  };

  TF_ASSERT_OK_AND_ASSIGN(auto thunk,
                          OutfeedThunk::Create({"outfeed"}, {outfeed_buffer}));

  EXPECT_EQ(thunk->buffer_uses().size(), 2);
  EXPECT_EQ(thunk->buffer_uses()[0], BufferUse::Read(outfeed_slice));

  BufferAllocation::Slice side_effect_slice(&alloc, 0, 1);
  EXPECT_EQ(thunk->buffer_uses()[1], BufferUse::Write(side_effect_slice));
}

}  // namespace
}  // namespace xla::cpu
