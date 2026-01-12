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

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/buffers_checksum_thunk.h"
#include "xla/backends/gpu/runtime/buffers_float_check_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/runtime_intrinsics.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_saver_inserter.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using testing::ElementsAre;
using testing::Eq;
using testing::IsEmpty;
using testing::Pair;
using testing::Pointer;
using testing::SizeIs;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

MATCHER_P(IsUniquePointerTo, ptr, "") { return arg.get() == ptr; }

MATCHER_P(ThunkKindIs, kind, "") {
  return ExplainMatchResult(Eq(kind), arg->kind(), result_listener);
}

MATCHER_P(IsCustomCallThunkWithTargetName, target_name, "") {
  return ExplainMatchResult(Eq(Thunk::Kind::kCustomCall), arg->kind(),
                            result_listener) &&
         ExplainMatchResult(
             Eq(target_name),
             static_cast<const CustomCallThunk&>(*arg).target_name(),
             result_listener);
}

MATCHER_P(IsChecksumThunkChecking, slice, "") {
  return ExplainMatchResult(Eq(Thunk::Kind::kBuffersDebugChecksum), arg->kind(),
                            result_listener) &&
         ExplainMatchResult(UnorderedElementsAreArray(slice),
                            static_cast<const BuffersDebugChecksumThunk&>(*arg)
                                .buffer_slices(),
                            result_listener);
}

MATCHER_P(IsSequentialThunkWith, thunk_matcher, "") {
  return ExplainMatchResult(Eq(Thunk::Kind::kSequential), arg->kind(),
                            result_listener) &&
         ExplainMatchResult(thunk_matcher,
                            static_cast<const SequentialThunk&>(*arg).thunks(),
                            result_listener);
}

using SliceList =
    std::initializer_list<std::pair<size_t, BufferAllocation::Slice>>;

class FakeThunkPassBufferAllocator : public ThunkPassBufferAllocator {
 public:
  absl::StatusOr<BufferAllocation*> NewEmptyAllocation(int64_t size) override {
    allocs_.push_back(std::make_unique<BufferAllocation>(0, size, 0));
    return allocs_.back().get();
  }

  const std::vector<std::unique_ptr<BufferAllocation>>& allocs() const {
    return allocs_;
  }

 private:
  std::vector<std::unique_ptr<BufferAllocation>> allocs_;
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

class ThunkBufferDebugPassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // The callbacks created by ThunkBufferDebugPass require a HloModule
    // with a non-null entry computation.
    auto builder = HloComputation::Builder("entry");
    HloInstruction* root = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(1)));
    std::unique_ptr<HloComputation> entry_computation = builder.Build(root);
    fake_hlo_module_ =
        std::make_unique<HloModule>("test_module", HloModuleConfig());
    fake_hlo_module_->AddEntryComputation(std::move(entry_computation));
  }

  Thunk::ThunkInfo ThunkInfoWithId(ThunkId thunk_id) {
    Thunk::ThunkInfo info;
    info.thunk_id = thunk_id;
    return info;
  }

  // Create a new, unique, non-null slice backed by `alloc_`.
  BufferAllocation::Slice CreateSlice() {
    BufferAllocation::Slice slice(&alloc_, used_alloc_size_, 1);
    used_alloc_size_ += slice.size();
    return slice;
  }

  BufferAllocation alloc_ = BufferAllocation(0, 1024, 0);
  size_t used_alloc_size_ = 0;
  std::unique_ptr<HloModule> fake_hlo_module_;
};

TEST_F(ThunkBufferDebugPassTest, IsNoOpWhenHloModuleIsNull) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(
      true);
  se::DeviceDescription device_info;

  FakeThunkPassBufferAllocator allocator;
  BufferAllocation alloc(0, 1024, 0);
  Shape arg_shape = ShapeUtil::MakeShape(U8, {1});
  BufferAllocation::Slice slice(&alloc, 0, 1);

  auto fake_thunk = std::make_unique<FakeThunk>(
      Thunk::ThunkInfo(), Thunk::BufferUses{BufferUse::Read(slice, arg_shape)});
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
  EXPECT_THAT(allocator.allocs(), IsEmpty());
}

TEST_F(ThunkBufferDebugPassTest, InsertsBuffersDebugChecksumThunks) {
  static constexpr ThunkId kTestThunkId = ThunkId(123);
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(
      true);
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  // Create a fake thunk with a few different buffer uses.
  BufferAllocation alloc(0, 1024, 0);

  Shape arg_shape = ShapeUtil::MakeShape(U8, {1});
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
          BufferUse::Consume(slice_i, arg_shape),
          // Write is undefined on input, but defined on output.
          BufferUse::Write(slice_o, arg_shape),
          // Unlike Consume, Read is supposed to preserve the contents of the
          // buffer, so we check it on input *and* output.
          BufferUse::Read(slice_io, arg_shape),
          // Scratch buffers are not checked at all.
          BufferUse::Scratch(slice_scratch, arg_shape),
      });
  Thunk* fake_thunk_ptr = fake_thunk.get();
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(fake_thunk));
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));

  ThunkBufferDebugPass pass(ThunkBufferDebugPass::Mode::kChecksum);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options,
                             fake_hlo_module_.get(), device_info, allocator));
  EXPECT_TRUE(changed);

  // Expected thunk structure after the pass:
  // 1. CustomCallThunk (buffer debug log init)
  // 2. SequentialThunk [
  //    1. BuffersDebugChecksumThunk (checksum input buffers)
  //    2. FakeThunk
  //    3. BuffersDebugChecksumThunk (checksum output buffers)
  // ]
  // 3. CustomCallThunk (buffer debug log dump)
  const std::vector<std::unique_ptr<Thunk>>& new_thunks = root_thunk->thunks();
  EXPECT_THAT(
      new_thunks,
      ElementsAre(
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_init"),
          IsSequentialThunkWith(ElementsAre(IsChecksumThunkChecking(SliceList{
                                                {0, slice_i},
                                                {2, slice_io},
                                            }),
                                            Pointer(fake_thunk_ptr),
                                            IsChecksumThunkChecking(SliceList{
                                                {1, slice_o},
                                                {2, slice_io},
                                            }))),
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_dump")));

  EXPECT_THAT(allocator.allocs(), SizeIs(1));
}

TEST_F(ThunkBufferDebugPassTest, RecursivelyInsertsBuffersDebugChecksumThunks) {
  static constexpr ThunkId kWhileConditionFakeThunkId = ThunkId(100);
  static constexpr ThunkId kWhileBodyId = ThunkId(101);
  static constexpr ThunkId kBranch0ThunkId = ThunkId(102);
  static constexpr ThunkId kBranch1ThunkId = ThunkId(103);
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(
      true);
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  // Create a fake thunk with a few different buffer uses.
  Shape arg_shape = ShapeUtil::MakeShape(U8, {1});
  BufferAllocation::Slice slice_while_condition = CreateSlice();
  BufferAllocation::Slice slice_while_body = CreateSlice();
  BufferAllocation::Slice slice_branch0 = CreateSlice();
  BufferAllocation::Slice slice_branch1 = CreateSlice();
  // Setup a thunk tree.
  auto while_condition_fake_thunk = std::make_unique<FakeThunk>(
      ThunkInfoWithId(kWhileConditionFakeThunkId),
      Thunk::BufferUses{BufferUse::Read(slice_while_condition, arg_shape)});
  const Thunk* const while_condition_fake_thunk_ptr =
      while_condition_fake_thunk.get();
  auto while_body_fake_thunk = std::make_unique<FakeThunk>(
      ThunkInfoWithId(kWhileBodyId),
      Thunk::BufferUses{BufferUse::Read(slice_while_body, arg_shape)});
  const Thunk* const while_body_fake_thunk_ptr = while_body_fake_thunk.get();
  auto conditional_branch0_thunk = std::make_unique<FakeThunk>(
      ThunkInfoWithId(kBranch0ThunkId),
      Thunk::BufferUses{BufferUse::Read(slice_branch0, arg_shape)});
  const Thunk* const branch0_thunk_ptr = conditional_branch0_thunk.get();
  auto conditional_branch1_thunk = std::make_unique<FakeThunk>(
      ThunkInfoWithId(kBranch1ThunkId),
      Thunk::BufferUses{BufferUse::Read(slice_branch1, arg_shape)});
  const Thunk* const branch1_thunk_ptr = conditional_branch1_thunk.get();
  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks;
  branch_thunks.push_back(
      SequentialThunk::FromThunk(std::move(conditional_branch0_thunk)));
  branch_thunks.push_back(
      SequentialThunk::FromThunk(std::move(conditional_branch1_thunk)));

  Shape condition_shape = ShapeUtil::MakeShape(PRED, {});
  BufferAllocation::Slice condition_slice = CreateSlice();

  auto conditional_thunk = std::make_unique<ConditionalThunk>(
      Thunk::ThunkInfo(), ShapedSlice{condition_slice, condition_shape},
      std::move(branch_thunks));
  const Thunk* const conditional_thunk_ptr = conditional_thunk.get();
  std::vector<std::unique_ptr<Thunk>> while_body_thunks;
  while_body_thunks.push_back(std::move(while_body_fake_thunk));
  while_body_thunks.push_back(std::move(conditional_thunk));
  auto while_thunk = std::make_unique<WhileThunk>(
      Thunk::ThunkInfo(), /*loop=*/nullptr,
      /*condition_result_buffer_index=*/BufferAllocation::Slice(),
      /*condition_thunk_sequence=*/
      SequentialThunk::FromThunk(std::move(while_condition_fake_thunk)),
      /*body_thunk_sequence=*/
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                        std::move(while_body_thunks)));
  std::unique_ptr<SequentialThunk> root_thunk =
      SequentialThunk::FromThunk(std::move(while_thunk));

  // Thunk structure before the pass:
  // 1. WhileThunk
  //    Condition: SequentialThunk [
  //       FakeThunk (kWhileConditionFakeThunkId)
  //    ]
  //    Body: SequentialThunk [
  //       FakeThunk (kWhileBodyId)
  //       ConditionalThunk [
  //          Branch 0: SequentialThunk [
  //             FakeThunk (kBranch0ThunkId)
  //          ]
  //          Branch 1: SequentialThunk [
  //             FakeThunk (kBranch1ThunkId)
  //          ]
  //       ]
  //    ]

  ThunkBufferDebugPass pass(ThunkBufferDebugPass::Mode::kChecksum);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options,
                             fake_hlo_module_.get(), device_info, allocator));
  EXPECT_TRUE(changed);

  // Each FakeThunk is supposed to be transformed into a SequentialThunk
  // containing the original FakeThunk sandwiched between two
  // BuffersDebugChecksumThunk thunks.
  //
  // Thunk structure after the pass:
  // 1. CustomCallThunk (buffer debug log init)
  // 2. WhileThunk
  //    1. Condition: SequentialThunk [
  //       1. SequentialThunk [
  //          1. BuffersDebugChecksumThunk (checksum input buffers)
  //          2. FakeThunk (kWhileConditionFakeThunkId)
  //          3. BuffersDebugChecksumThunk (checksum output buffers)
  //       ]
  //    ]
  //    2. Body: SequentialThunk [
  //       1. SequentialThunk [
  //          1. BuffersDebugChecksumThunk (checksum input buffers)
  //          2. FakeThunk (kWhileBodyId)
  //          3. BuffersDebugChecksumThunk (checksum output buffers)
  //       ]
  //       2. ConditionalThunk [
  //          Branch 0: SequentialThunk [
  //             1. SequentialThunk [
  //                1. BuffersDebugChecksumThunk (checksum input buffers)
  //                2. FakeThunk (kBranch0ThunkId)
  //                3. BuffersDebugChecksumThunk (checksum output buffers)
  //             ]
  //          ]
  //          Branch 1: SequentialThunk [
  //             1. SequentialThunk [
  //                1. BuffersDebugChecksumThunk (checksum input buffers)
  //                2. FakeThunk (kBranch1ThunkId)
  //                3. BuffersDebugChecksumThunk (checksum output buffers)
  //             ]
  //          ]
  //       ]
  //    ]
  // 3. CustomCallThunk (buffer debug log dump)

  const std::vector<std::unique_ptr<Thunk>>& new_thunks = root_thunk->thunks();
  EXPECT_THAT(
      new_thunks,
      ElementsAre(
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_init"),
          ThunkKindIs(Thunk::Kind::kSequential),
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_dump")));

  {
    ASSERT_EQ(new_thunks[1]->kind(), Thunk::Kind::kSequential);
    const SequentialThunk& top_seq_thunk =
        static_cast<const SequentialThunk&>(*new_thunks[1]);

    ASSERT_EQ(top_seq_thunk.thunks()[1]->kind(), Thunk::Kind::kWhile);
    const WhileThunk& while_thunk =
        static_cast<const WhileThunk&>(*top_seq_thunk.thunks()[1]);

    EXPECT_THAT(while_thunk.body_thunk_sequence()->thunks(),
                ElementsAre(ThunkKindIs(Thunk::Kind::kSequential),
                            ThunkKindIs(Thunk::Kind::kSequential)));
    const SequentialThunk& condition_fake_thunk_sequence =
        static_cast<const SequentialThunk&>(
            *while_thunk.condition_thunk_sequence()->thunks()[0]);
    EXPECT_THAT(
        condition_fake_thunk_sequence.thunks(),
        ElementsAre(
            IsChecksumThunkChecking(SliceList{{0, slice_while_condition}}),
            Pointer(while_condition_fake_thunk_ptr),
            IsChecksumThunkChecking(SliceList{{0, slice_while_condition}})));

    const SequentialThunk& body_fake_thunk_sequence =
        static_cast<const SequentialThunk&>(
            *while_thunk.body_thunk_sequence()->thunks()[0]);
    EXPECT_THAT(
        body_fake_thunk_sequence.thunks(),
        ElementsAre(IsChecksumThunkChecking(SliceList{{0, slice_while_body}}),
                    Pointer(while_body_fake_thunk_ptr),
                    IsChecksumThunkChecking(SliceList{{0, slice_while_body}})));

    ASSERT_EQ(while_thunk.body_thunk_sequence()->thunks()[1]->kind(),
              Thunk::Kind::kSequential);
    const SequentialThunk& condition_warpper_thunk =
        static_cast<const SequentialThunk&>(
            *while_thunk.body_thunk_sequence()->thunks()[1]);

    ASSERT_EQ(condition_warpper_thunk.thunks()[1]->kind(),
              Thunk::Kind::kConditional);
    const ConditionalThunk& conditional_thunk =
        static_cast<const ConditionalThunk&>(
            *condition_warpper_thunk.thunks()[1]);
    EXPECT_EQ(&conditional_thunk, conditional_thunk_ptr);

    EXPECT_THAT(conditional_thunk.branch_thunks(),
                ElementsAre(ThunkKindIs(Thunk::Kind::kSequential),
                            ThunkKindIs(Thunk::Kind::kSequential)));

    const SequentialThunk& branch0_thunk = static_cast<const SequentialThunk&>(
        *conditional_thunk.branch_thunks()[0]);
    EXPECT_THAT(branch0_thunk.thunks(),
                ElementsAre(ThunkKindIs(Thunk::Kind::kSequential)));

    const SequentialThunk& branch0_fake_thunk_sequence =
        static_cast<const SequentialThunk&>(*branch0_thunk.thunks()[0]);
    EXPECT_THAT(
        branch0_fake_thunk_sequence.thunks(),
        ElementsAre(IsChecksumThunkChecking(SliceList{{0, slice_branch0}}),
                    Pointer(branch0_thunk_ptr),
                    IsChecksumThunkChecking(SliceList{{0, slice_branch0}})));

    const SequentialThunk& branch1_thunk = static_cast<const SequentialThunk&>(
        *conditional_thunk.branch_thunks()[1]);
    EXPECT_THAT(branch1_thunk.thunks(),
                ElementsAre(ThunkKindIs(Thunk::Kind::kSequential)));

    const SequentialThunk& branch1_fake_thunk_sequence =
        static_cast<const SequentialThunk&>(*branch1_thunk.thunks()[0]);
    EXPECT_THAT(
        branch1_fake_thunk_sequence.thunks(),
        ElementsAre(IsChecksumThunkChecking(SliceList{{0, slice_branch1}}),
                    Pointer(branch1_thunk_ptr),
                    IsChecksumThunkChecking(SliceList{{0, slice_branch1}})));
  }

  EXPECT_THAT(allocator.allocs(), SizeIs(1));
}

TEST_F(ThunkBufferDebugPassTest, InsertsBuffersDebugFloatCheckThunks) {
  static constexpr ThunkId kTestThunkId = ThunkId(123);
  DebugOptions debug_options;
  debug_options.set_xla_gpu_detect_nan(DebugOptions::DETECTION_MODE_WARNING);
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
  Shape arg_shape = ShapeUtil::MakeShape(F32, {1});
  BufferAllocation::Slice slice_i(&alloc, 0, 4, PrimitiveType::F32);
  BufferAllocation::Slice slice_o(&alloc, 4, 4, PrimitiveType::F32);
  BufferAllocation::Slice slice_io(&alloc, 8, 4, PrimitiveType::F32);
  BufferAllocation::Slice slice_scratch(&alloc, 12, 4, PrimitiveType::F32);
  Thunk::ThunkInfo fake_thunk_info;
  fake_thunk_info.thunk_id = ThunkId(kTestThunkId);
  auto fake_thunk = std::make_unique<FakeThunk>(
      fake_thunk_info,
      Thunk::BufferUses{
          // Consume means the thunk can reuse the buffer for scratch space, so
          // only check it on input.
          BufferUse::Consume(slice_i, arg_shape),
          // Write is undefined on input, but defined on output.
          BufferUse::Write(slice_o, arg_shape),
          // Unlike Consume, Read is supposed to preserve the contents of the
          // buffer, so we check it on input *and* output.
          BufferUse::Read(slice_io, arg_shape),
          // Scratch buffers are not checked at all.
          BufferUse::Scratch(slice_scratch, arg_shape),
      });
  Thunk* fake_thunk_ptr = fake_thunk.get();
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(fake_thunk));
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));

  ThunkBufferDebugPass pass(ThunkBufferDebugPass::Mode::kFloatChecker);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          pass.Run(root_thunk.get(), debug_options, &hlo_module,
                                   device_info, allocator));
  EXPECT_TRUE(changed);

  // Expected thunk structure after the pass:
  // 1. CustomCallThunk (buffer debug log init)
  // 2. SequentialThunk
  //    1. FakeThunk
  //    2. BuffersDebugFloatCheckThunk (float check output buffers)
  // 3. CustomCallThunk (buffer debug log dump)
  const std::vector<std::unique_ptr<Thunk>>& new_thunks = root_thunk->thunks();
  EXPECT_THAT(new_thunks, SizeIs(3));
  EXPECT_EQ(new_thunks[0]->kind(), Thunk::Kind::kCustomCall);
  EXPECT_EQ(new_thunks[1]->kind(), Thunk::Kind::kSequential);
  EXPECT_EQ(new_thunks[2]->kind(), Thunk::Kind::kCustomCall);

  const CustomCallThunk& buffer_debug_init_thunk =
      static_cast<const CustomCallThunk&>(*new_thunks[0]);
  EXPECT_EQ(buffer_debug_init_thunk.target_name(),
            "xla_gpu_buffer_debug_float_check_init");

  const CustomCallThunk& buffer_debug_dump_thunk =
      static_cast<const CustomCallThunk&>(*new_thunks[2]);
  EXPECT_EQ(buffer_debug_dump_thunk.target_name(),
            "xla_gpu_buffer_debug_float_check");

  const std::vector<std::unique_ptr<Thunk>>& sub_thunks =
      static_cast<const SequentialThunk&>(*new_thunks[1]).thunks();
  EXPECT_THAT(sub_thunks, SizeIs(2));
  EXPECT_THAT(sub_thunks[0], Pointer(fake_thunk_ptr));
  EXPECT_EQ(sub_thunks[1]->kind(), Thunk::Kind::kBuffersDebugFloatCheck);

  const BuffersDebugFloatCheckThunk& buffer_debug_after_fake_thunk =
      static_cast<const BuffersDebugFloatCheckThunk&>(*sub_thunks[1]);
  EXPECT_THAT(buffer_debug_after_fake_thunk.buffer_slices(),
              UnorderedElementsAre(Pair(1, slice_o), Pair(2, slice_io)));

  // 1 for the log buffer, 1 per wrapped thunk for the temp buffer
  EXPECT_THAT(allocator.allocs(), SizeIs(2));
}

TEST_F(ThunkBufferDebugPassTest, BufferSaverInserter) {
  static constexpr ThunkId kTestThunkId = ThunkId(123);
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
  Shape arg_shape = ShapeUtil::MakeShape(U8, {1});
  BufferAllocation::Slice slice_o(&alloc, 1, 1, PrimitiveType::F32);
  BufferAllocation::Slice slice_io(&alloc, 2, 1, PrimitiveType::F32);
  Thunk::ThunkInfo fake_thunk_info;
  fake_thunk_info.thunk_id = kTestThunkId;

  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::make_unique<FakeThunk>(
      fake_thunk_info,
      Thunk::BufferUses{
          // Write is undefined on input, but defined on output.
          BufferUse::Write(slice_o, arg_shape),
          // Unlike Consume, Read is supposed to preserve the contents of the
          // buffer, so we check it on input *and* output.
          BufferUse::Read(slice_io, arg_shape),
      }));
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));

  DebugOptions debug_options =
      tsl::proto_testing::ParseTextProtoOrDie<DebugOptions>(R"pb(
        xla_dump_to: "/tmp/123"
        xla_gpu_experimental_enable_buffer_saver_on_thunks: true
      )pb");

  EXPECT_OK(RunDebugSaverInserter(*root_thunk, debug_options, hlo_module));

  // Expected thunk structure after the pass:
  // 1. SequentialThunk
  //    1. FakeThunk
  //    2. CustomCall (buffer saver)
  const std::vector<std::unique_ptr<Thunk>>& new_thunks = root_thunk->thunks();
  EXPECT_THAT(
      new_thunks,
      ElementsAre(IsSequentialThunkWith(ElementsAre(
          ThunkKindIs(Thunk::Kind::kGemm),
          IsCustomCallThunkWithTargetName(kXlaGpuAppendToFileCustomCallTag)))));
}

TEST_F(ThunkBufferDebugPassTest, FiltersThunksByIdRanges) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(
      true);
  IntRangeInclusive* range =
      debug_options.mutable_xla_gpu_experimental_thunk_buffer_debug_filter()
          ->add_thunk_id_ranges();
  range->set_first(2);
  range->set_last(2);
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  // Create a fake thunk with a few different buffer uses.
  BufferAllocation alloc(0, 1024, 0);
  Shape slice_shape = ShapeUtil::MakeShape(U8, {1});
  BufferAllocation::Slice slice1_io(&alloc, 0, 1);
  BufferAllocation::Slice slice2_io(&alloc, 1, 1);
  Thunk::ThunkInfo fake_thunk1_info;
  fake_thunk1_info.thunk_id = ThunkId(1);
  auto fake_thunk1 = std::make_unique<FakeThunk>(
      fake_thunk1_info,
      Thunk::BufferUses{BufferUse::Read(slice1_io, slice_shape)});
  Thunk::ThunkInfo fake_thunk2_info;
  fake_thunk2_info.thunk_id = ThunkId(2);
  auto fake_thunk2 = std::make_unique<FakeThunk>(
      fake_thunk2_info,
      Thunk::BufferUses{BufferUse::Read(slice2_io, slice_shape)});
  Thunk* fake_thunk1_ptr = fake_thunk1.get();
  Thunk* fake_thunk2_ptr = fake_thunk2.get();
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(fake_thunk1));
  thunks.push_back(std::move(fake_thunk2));
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));

  ThunkBufferDebugPass pass(ThunkBufferDebugPass::Mode::kChecksum);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options,
                             fake_hlo_module_.get(), device_info, allocator));
  EXPECT_TRUE(changed);

  // Expected thunk structure after the pass:
  // 1. CustomCallThunk (buffer debug log init)
  // 2. FakeThunk1 (not instrumented due to filter)
  // 3. SequentialThunk [
  //    1. BuffersDebugChecksumThunk (checksum input buffers)
  //    2. FakeThunk2
  //    3. BuffersDebugChecksumThunk (checksum output buffers)
  // 4. CustomCallThunk (buffer debug log dump)
  const std::vector<std::unique_ptr<Thunk>>& new_thunks = root_thunk->thunks();
  EXPECT_THAT(
      new_thunks,
      ElementsAre(
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_init"),
          Pointer(fake_thunk1_ptr),
          IsSequentialThunkWith(ElementsAre(IsChecksumThunkChecking(SliceList{
                                                {0, slice2_io},
                                            }),
                                            Pointer(fake_thunk2_ptr),
                                            IsChecksumThunkChecking(SliceList{
                                                {0, slice2_io},
                                            }))),
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_dump")));
}

TEST_F(ThunkBufferDebugPassTest, FiltersThunksByProfileAnnotationRegexes) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(
      true);
  debug_options.mutable_xla_gpu_experimental_thunk_buffer_debug_filter()
      ->add_profile_annotation_regexes("thunk1");
  debug_options.mutable_xla_gpu_experimental_thunk_buffer_debug_filter()
      ->add_profile_annotation_regexes("^fake.*2$");
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  // Create a fake thunk with a few different buffer uses.
  BufferAllocation alloc(0, 1024, 0);
  Shape slice_shape = ShapeUtil::MakeShape(U8, {1});
  BufferAllocation::Slice slice1_io(&alloc, 0, 1);
  BufferAllocation::Slice slice2_io(&alloc, 1, 1);
  Thunk::ThunkInfo fake_thunk1_info;
  fake_thunk1_info.thunk_id = ThunkId(1);
  fake_thunk1_info.profile_annotation = "fake_thunk1";
  auto fake_thunk1 = std::make_unique<FakeThunk>(
      fake_thunk1_info,
      Thunk::BufferUses{BufferUse::Read(slice1_io, slice_shape)});
  Thunk::ThunkInfo fake_thunk2_info;
  fake_thunk2_info.profile_annotation = "fake_thunk2";
  fake_thunk2_info.thunk_id = ThunkId(2);
  auto fake_thunk2 = std::make_unique<FakeThunk>(
      fake_thunk2_info,
      Thunk::BufferUses{BufferUse::Read(slice2_io, slice_shape)});
  Thunk::ThunkInfo fake_thunk3_info;
  fake_thunk3_info.profile_annotation = "fake_thunk3";
  fake_thunk3_info.thunk_id = ThunkId(3);
  auto fake_thunk3 = std::make_unique<FakeThunk>(
      fake_thunk3_info,
      Thunk::BufferUses{BufferUse::Read(slice2_io, slice_shape)});
  Thunk* fake_thunk1_ptr = fake_thunk1.get();
  Thunk* fake_thunk2_ptr = fake_thunk2.get();
  Thunk* fake_thunk3_ptr = fake_thunk3.get();
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(fake_thunk1));
  thunks.push_back(std::move(fake_thunk2));
  thunks.push_back(std::move(fake_thunk3));
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));

  ThunkBufferDebugPass pass(ThunkBufferDebugPass::Mode::kChecksum);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options,
                             fake_hlo_module_.get(), device_info, allocator));
  EXPECT_TRUE(changed);

  // Expected thunk structure after the pass:
  // 1. CustomCallThunk (buffer debug log init)
  // 2. SequentialThunk [
  //    1. BuffersDebugChecksumThunk (checksum input buffers)
  //    2. FakeThunk1 (instrumented due to thunk1)
  //    3. BuffersDebugChecksumThunk (checksum output buffers)
  // ]
  // 3. SequentialThunk [
  //    4. BuffersDebugChecksumThunk (checksum input buffers)
  //    5. FakeThunk2 (instrumented due to 2$)
  //    6. BuffersDebugChecksumThunk (checksum output buffers)
  // ]
  // 3. FakeThunk3 (not instrumented)
  // 4. CustomCallThunk (buffer debug log dump)
  const std::vector<std::unique_ptr<Thunk>>& new_thunks = root_thunk->thunks();
  EXPECT_THAT(
      new_thunks,
      ElementsAre(
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_init"),
          IsSequentialThunkWith(ElementsAre(IsChecksumThunkChecking(SliceList{
                                                {0, slice1_io},
                                            }),
                                            Pointer(fake_thunk1_ptr),
                                            IsChecksumThunkChecking(SliceList{
                                                {0, slice1_io},
                                            }))),
          IsSequentialThunkWith(ElementsAre(IsChecksumThunkChecking(SliceList{
                                                {0, slice2_io},
                                            }),
                                            Pointer(fake_thunk2_ptr),
                                            IsChecksumThunkChecking(SliceList{
                                                {0, slice2_io},
                                            }))),
          Pointer(fake_thunk3_ptr),
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_dump")));
}

TEST_F(ThunkBufferDebugPassTest,
       FiltersThunksByIdRangesAndProfileAnnotationRegexes) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(
      true);
  IntRangeInclusive* range =
      debug_options.mutable_xla_gpu_experimental_thunk_buffer_debug_filter()
          ->add_thunk_id_ranges();
  range->set_first(2);
  range->set_last(3);
  debug_options.mutable_xla_gpu_experimental_thunk_buffer_debug_filter()
      ->add_profile_annotation_regexes("instrument_me");
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  // Create a fake thunk with a few different buffer uses.
  BufferAllocation alloc(0, 1024, 0);
  Shape slice_shape = ShapeUtil::MakeShape(U8, {1});
  BufferAllocation::Slice slice1_io(&alloc, 0, 1);
  BufferAllocation::Slice slice2_io(&alloc, 1, 1);
  BufferAllocation::Slice slice3_io(&alloc, 2, 1);
  Thunk::ThunkInfo fake_thunk1_info;
  fake_thunk1_info.thunk_id = ThunkId(1);
  fake_thunk1_info.profile_annotation = "instrument_me";
  auto fake_thunk1 = std::make_unique<FakeThunk>(
      fake_thunk1_info,
      Thunk::BufferUses{BufferUse::Read(slice1_io, slice_shape)});
  Thunk::ThunkInfo fake_thunk2_info;
  fake_thunk2_info.thunk_id = ThunkId(2);
  fake_thunk2_info.profile_annotation = "ignore_me";
  auto fake_thunk2 = std::make_unique<FakeThunk>(
      fake_thunk2_info,
      Thunk::BufferUses{BufferUse::Read(slice2_io, slice_shape)});
  Thunk::ThunkInfo fake_thunk3_info;
  fake_thunk3_info.thunk_id = ThunkId(3);
  fake_thunk3_info.profile_annotation = "instrument_me";
  auto fake_thunk3 = std::make_unique<FakeThunk>(
      fake_thunk3_info,
      Thunk::BufferUses{BufferUse::Read(slice3_io, slice_shape)});
  Thunk* fake_thunk1_ptr = fake_thunk1.get();
  Thunk* fake_thunk2_ptr = fake_thunk2.get();
  Thunk* fake_thunk3_ptr = fake_thunk3.get();
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(fake_thunk1));
  thunks.push_back(std::move(fake_thunk2));
  thunks.push_back(std::move(fake_thunk3));
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));

  ThunkBufferDebugPass pass(ThunkBufferDebugPass::Mode::kChecksum);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options,
                             fake_hlo_module_.get(), device_info, allocator));
  EXPECT_TRUE(changed);

  // Expected thunk structure after the pass:
  // 1. CustomCallThunk (buffer debug log init)
  // 2. FakeThunk1 (not instrumented due to thunk ID filter)
  // 3. FakeThunk2 (not instrumented due to profile annotation regex filter)
  // 4. SequentialThunk [
  //    1. BuffersDebugChecksumThunk (checksum input buffers)
  //    2. FakeThunk3
  //    3. BuffersDebugChecksumThunk (checksum output buffers)
  // 5. CustomCallThunk (buffer debug log dump)
  const std::vector<std::unique_ptr<Thunk>>& new_thunks = root_thunk->thunks();
  EXPECT_THAT(
      new_thunks,
      ElementsAre(
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_init"),
          Pointer(fake_thunk1_ptr), Pointer(fake_thunk2_ptr),
          IsSequentialThunkWith(ElementsAre(IsChecksumThunkChecking(SliceList{
                                                {0, slice3_io},
                                            }),
                                            Pointer(fake_thunk3_ptr),
                                            IsChecksumThunkChecking(SliceList{
                                                {0, slice3_io},
                                            }))),
          IsCustomCallThunkWithTargetName("xla_gpu_buffer_debug_log_dump")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
