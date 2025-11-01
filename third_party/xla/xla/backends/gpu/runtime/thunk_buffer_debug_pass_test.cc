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

#include "xla/backends/gpu/runtime/thunk_buffer_debug_pass.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/buffers_checksum_thunk.h"
#include "xla/backends/gpu/runtime/buffers_nan_count_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_buffer_id.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using testing::ElementsAre;
using testing::Pair;
using testing::Pointer;
using testing::SizeIs;
using testing::UnorderedElementsAre;

MATCHER_P(IsUniquePointerTo, ptr, "") { return arg.get() == ptr; }

class FakeThunkPassBufferAllocator : public ThunkPassBufferAllocator {
 public:
  absl::StatusOr<BufferAllocation*> NewEmptyAllocation(int64_t size) override {
    if (CreatedAlloc()) {
      return absl::InvalidArgumentError("Expected only one allocation");
    }
    alloc_ = std::make_unique<BufferAllocation>(0, size, 0);
    return alloc_.get();
  }

  bool CreatedAlloc() { return alloc_ != nullptr; }

 private:
  std::unique_ptr<BufferAllocation> alloc_;
};

class FakeThunk : public Thunk {
 public:
  explicit FakeThunk(ThunkInfo info, BufferUses buffer_uses)
      : Thunk(Thunk::Kind::kGemm, std::move(info)),
        buffer_uses_(std::move(buffer_uses)) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }

  BufferUses buffer_uses() const override { return buffer_uses_; }

 private:
  BufferUses buffer_uses_;
};

TEST(ThunkBufferDebugPassTest, IsNoOpWhenHloModuleIsNull) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(
      true);
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice slice(&alloc, 0, 1);
  auto fake_thunk = std::make_unique<FakeThunk>(
      Thunk::ThunkInfo(), Thunk::BufferUses{BufferUse::Read(slice)});
  Thunk* fake_thunk_ptr = fake_thunk.get();
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(fake_thunk));
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));

  ThunkBufferDebugPass pass(ThunkBufferDebugPass::Mode::kChecksum);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options,
                             /*hlo_module=*/nullptr, device_info, allocator));
  EXPECT_FALSE(changed);
  EXPECT_THAT(root_thunk->thunks(), ElementsAre(Pointer(fake_thunk_ptr)));
}

TEST(ThunkBufferDebugPassTest, InsertsBuffersDebugChecksumThunks) {
  static constexpr ThunkId kTestThunkId = ThunkId(123);
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(
      true);
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  // The callbacks created by ThunkBufferDebugPass require a HloModule with
  // a non-null entry computation.
  auto builder = HloComputation::Builder("entry");
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(1)));
  std::unique_ptr<HloComputation> entry_computation = builder.Build(root);
  HloModule hlo_module("test_module", HloModuleConfig());
  hlo_module.AddEntryComputation(std::move(entry_computation));
  // Create a fake thunk with a few different buffer uses.
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice slice_i(&alloc, 0, 1);
  BufferAllocation::Slice slice_o(&alloc, 1, 1);
  BufferAllocation::Slice slice_io(&alloc, 2, 1);
  BufferAllocation::Slice slice_scratch(&alloc, 3, 1);
  Thunk::ThunkInfo fake_thunk_info;
  fake_thunk_info.thunk_id = ThunkId(kTestThunkId);
  auto fake_thunk = std::make_unique<FakeThunk>(
      fake_thunk_info,
      Thunk::BufferUses{
          // Consume means the thunk can reuse the buffer for scratch space, so
          // only check it on input.
          BufferUse::Consume(slice_i),
          // Write is undefined on input, but defined on output.
          BufferUse::Write(slice_o),
          // Unlike Consume, Read is supposed to preserve the contents of the
          // buffer, so we check it on input *and* output.
          BufferUse::Read(slice_io),
          // Scratch buffers are not checked at all.
          BufferUse::Scratch(slice_scratch),
      });
  Thunk* fake_thunk_ptr = fake_thunk.get();
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(fake_thunk));
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));

  ThunkBufferDebugPass pass(ThunkBufferDebugPass::Mode::kChecksum);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          pass.Run(root_thunk.get(), debug_options, &hlo_module,
                                   device_info, allocator));
  EXPECT_TRUE(changed);

  // Expected thunk structure after the pass:
  // 1. CustomCallThunk (buffer debug log init)
  // 2. SequentialThunk
  //    1. BuffersDebugChecksumThunk (checksum input buffers)
  //    2. FakeThunk
  //    3. BuffersDebugChecksumThunk (checksum output buffers)
  // 3. CustomCallThunk (buffer debug log dump)
  const std::vector<std::unique_ptr<Thunk>>& new_thunks = root_thunk->thunks();
  EXPECT_THAT(new_thunks, SizeIs(3));
  EXPECT_EQ(new_thunks[0]->kind(), Thunk::Kind::kCustomCall);
  EXPECT_EQ(new_thunks[1]->kind(), Thunk::Kind::kSequential);
  EXPECT_EQ(new_thunks[2]->kind(), Thunk::Kind::kCustomCall);

  const CustomCallThunk& buffer_debug_init_thunk =
      static_cast<const CustomCallThunk&>(*new_thunks[0]);
  EXPECT_EQ(buffer_debug_init_thunk.target_name(),
            "xla_gpu_buffer_debug_log_init");

  const CustomCallThunk& buffer_debug_dump_thunk =
      static_cast<const CustomCallThunk&>(*new_thunks[2]);
  EXPECT_EQ(buffer_debug_dump_thunk.target_name(),
            "xla_gpu_buffer_debug_log_dump");

  const std::vector<std::unique_ptr<Thunk>>& sub_thunks =
      static_cast<const SequentialThunk&>(*new_thunks[1]).thunks();
  EXPECT_THAT(sub_thunks, SizeIs(3));
  EXPECT_EQ(sub_thunks[0]->kind(), Thunk::Kind::kBuffersDebugChecksum);
  EXPECT_THAT(sub_thunks[1], Pointer(fake_thunk_ptr));
  EXPECT_EQ(sub_thunks[2]->kind(), Thunk::Kind::kBuffersDebugChecksum);

  const BuffersDebugChecksumThunk& buffer_debug_before_fake_thunk =
      static_cast<const BuffersDebugChecksumThunk&>(*sub_thunks[0]);
  EXPECT_THAT(
      buffer_debug_before_fake_thunk.buffer_slices(),
      UnorderedElementsAre(
          Pair(ThunkBufferId::Create(kTestThunkId, 0).value(), slice_i),
          Pair(ThunkBufferId::Create(kTestThunkId, 2).value(), slice_io)));

  const BuffersDebugChecksumThunk& buffer_debug_after_fake_thunk =
      static_cast<const BuffersDebugChecksumThunk&>(*sub_thunks[2]);
  EXPECT_THAT(
      buffer_debug_after_fake_thunk.buffer_slices(),
      UnorderedElementsAre(
          Pair(ThunkBufferId::Create(kTestThunkId, 1).value(), slice_o),
          Pair(ThunkBufferId::Create(kTestThunkId, 2).value(), slice_io)));
}

TEST(ThunkBufferDebugPassTest, InsertsBuffersDebugNanCounterThunks) {
  static constexpr ThunkId kTestThunkId = ThunkId(123);
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_nan_counter_on_thunks(true);
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  // The callbacks created by ThunkBufferDebugPass require a HloModule with
  // a non-null entry computation.
  auto builder = HloComputation::Builder("entry");
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(1.0f)));
  std::unique_ptr<HloComputation> entry_computation = builder.Build(root);
  HloModule hlo_module("test_module", HloModuleConfig());
  hlo_module.AddEntryComputation(std::move(entry_computation));
  // Create a fake thunk with a few different buffer uses.
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice slice_i(&alloc, 0, 1, PrimitiveType::F32);
  BufferAllocation::Slice slice_o(&alloc, 1, 1, PrimitiveType::F32);
  BufferAllocation::Slice slice_io(&alloc, 2, 1, PrimitiveType::F32);
  BufferAllocation::Slice slice_scratch(&alloc, 3, 1, PrimitiveType::F32);
  Thunk::ThunkInfo fake_thunk_info;
  fake_thunk_info.thunk_id = ThunkId(kTestThunkId);
  auto fake_thunk = std::make_unique<FakeThunk>(
      fake_thunk_info,
      Thunk::BufferUses{
          // Consume means the thunk can reuse the buffer for scratch space, so
          // only check it on input.
          BufferUse::Consume(slice_i),
          // Write is undefined on input, but defined on output.
          BufferUse::Write(slice_o),
          // Unlike Consume, Read is supposed to preserve the contents of the
          // buffer, so we check it on input *and* output.
          BufferUse::Read(slice_io),
          // Scratch buffers are not checked at all.
          BufferUse::Scratch(slice_scratch),
      });
  Thunk* fake_thunk_ptr = fake_thunk.get();
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(fake_thunk));
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));

  ThunkBufferDebugPass pass(ThunkBufferDebugPass::Mode::kNanCounter);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          pass.Run(root_thunk.get(), debug_options, &hlo_module,
                                   device_info, allocator));
  EXPECT_TRUE(changed);

  // Expected thunk structure after the pass:
  // 1. CustomCallThunk (buffer debug log init)
  // 2. SequentialThunk
  //    1. FakeThunk
  //    2. BuffersDebugNanCountThunk (nan counter output buffers)
  // 3. CustomCallThunk (buffer debug log dump)
  const std::vector<std::unique_ptr<Thunk>>& new_thunks = root_thunk->thunks();
  EXPECT_THAT(new_thunks, SizeIs(3));
  EXPECT_EQ(new_thunks[0]->kind(), Thunk::Kind::kCustomCall);
  EXPECT_EQ(new_thunks[1]->kind(), Thunk::Kind::kSequential);
  EXPECT_EQ(new_thunks[2]->kind(), Thunk::Kind::kCustomCall);

  const CustomCallThunk& buffer_debug_init_thunk =
      static_cast<const CustomCallThunk&>(*new_thunks[0]);
  EXPECT_EQ(buffer_debug_init_thunk.target_name(),
            "xla_gpu_buffer_debug_log_init");

  const CustomCallThunk& buffer_debug_dump_thunk =
      static_cast<const CustomCallThunk&>(*new_thunks[2]);
  EXPECT_EQ(buffer_debug_dump_thunk.target_name(),
            "xla_gpu_buffer_debug_log_dump");

  const std::vector<std::unique_ptr<Thunk>>& sub_thunks =
      static_cast<const SequentialThunk&>(*new_thunks[1]).thunks();
  EXPECT_THAT(sub_thunks, SizeIs(2));
  EXPECT_THAT(sub_thunks[0], Pointer(fake_thunk_ptr));
  EXPECT_EQ(sub_thunks[1]->kind(), Thunk::Kind::kBuffersDebugNanCount);

  const BuffersDebugNanCountThunk& buffer_debug_after_fake_thunk =
      static_cast<const BuffersDebugNanCountThunk&>(*sub_thunks[1]);
  EXPECT_THAT(
      buffer_debug_after_fake_thunk.buffer_slices(),
      UnorderedElementsAre(
          Pair(ThunkBufferId::Create(kTestThunkId, 1).value(), slice_o),
          Pair(ThunkBufferId::Create(kTestThunkId, 2).value(), slice_io)));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
