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

#include "xla/backends/cpu/runtime/rng_seed_thunk.h"

#include <cstdint>

#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(RngSeedThunkTest, SetProvidedSeed) {
  auto dst = LiteralUtil::CreateR0<uint64_t>(0);

  BufferAllocations allocations = CreateBufferAllocations(dst);
  auto [dst_alloc] = CreateBufferAllocation(dst);
  auto dst_slice = CreateBufferAllocationSlice(dst_alloc);

  TF_ASSERT_OK_AND_ASSIGN(auto thunk,
                          RngSeedThunk::Create({"rng_seed"}, dst_slice));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.rng_seed = 42;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(dst, LiteralUtil::CreateR0<uint64_t>(42));
}

TEST(RngSeedThunkTest, GenerateRandomSeed) {
  auto dst = LiteralUtil::CreateR0<uint64_t>(0);

  BufferAllocations allocations = CreateBufferAllocations(dst);
  auto [dst_alloc] = CreateBufferAllocation(dst);
  auto dst_slice = CreateBufferAllocationSlice(dst_alloc);

  TF_ASSERT_OK_AND_ASSIGN(auto thunk,
                          RngSeedThunk::Create({"rng_seed"}, dst_slice));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.rng_seed = 0;  // Triggers random generation.

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_NE(dst, LiteralUtil::CreateR0<uint64_t>(0));
}

TEST(RngSeedThunkTest, BufferUses) {
  auto dst = LiteralUtil::CreateR0<uint64_t>(0);
  auto [dst_alloc] = CreateBufferAllocation(dst);
  auto dst_slice = CreateBufferAllocationSlice(dst_alloc);

  TF_ASSERT_OK_AND_ASSIGN(auto thunk,
                          RngSeedThunk::Create({"rng_seed"}, dst_slice));

  auto uses = thunk->buffer_uses();
  ASSERT_EQ(uses.size(), 1);
  EXPECT_EQ(uses[0].slice(), dst_slice);
  EXPECT_EQ(uses[0].access(), BufferUse::MemoryAccess::kWrite);
}

}  // namespace
}  // namespace xla::cpu
