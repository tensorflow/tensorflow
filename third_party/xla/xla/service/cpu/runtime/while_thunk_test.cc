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

#include "xla/service/cpu/runtime/while_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime/thunk_testlib.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(WhileThunkTest, BufferUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice predicate_slice(&alloc, 0, sizeof(int32_t));
  BufferAllocation::Slice cond_read_slice(&alloc, 10, 10);
  BufferAllocation::Slice body_read_slice(&alloc, 20, 10);

  ThunkSequence cond_sequence;
  cond_sequence.push_back(
      std::make_unique<BufferUseThunk>(BufferUse::Read(cond_read_slice)));

  ThunkSequence body_sequence;
  body_sequence.push_back(
      std::make_unique<BufferUseThunk>(BufferUse::Read(body_read_slice)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      WhileThunk::Create({"while"}, predicate_slice, std::move(cond_sequence),
                         std::move(body_sequence)));

  EXPECT_EQ(thunk->buffer_uses().size(), 3);
  EXPECT_EQ(thunk->buffer_uses()[0], BufferUse::Write(predicate_slice));
  EXPECT_EQ(thunk->buffer_uses()[1], BufferUse::Read(cond_read_slice));
  EXPECT_EQ(thunk->buffer_uses()[2], BufferUse::Read(body_read_slice));
}

}  // namespace
}  // namespace xla::cpu
