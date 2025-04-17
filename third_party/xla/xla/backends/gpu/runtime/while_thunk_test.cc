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

#include "xla/backends/gpu/runtime/while_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::tsl::testing::IsOk;

class IterationLoggerThunk : public Thunk {
 public:
  explicit IterationLoggerThunk(const HloInstruction* loop)
      : Thunk(Thunk::Kind::kKernel, Thunk::ThunkInfo()), loop_(loop) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    auto iter = WhileThunk::CurrentLoopIteration(loop_);
    if (iter.ok()) {
      iteration_counters_.push_back(*iter);
    } else {
      iteration_counters_.push_back(std::nullopt);
    }
    return absl::OkStatus();
  }

  const std::vector<std::optional<int64_t>>& logged_counters() const {
    return iteration_counters_;
  }

 private:
  const HloInstruction* loop_;
  std::vector<std::optional<int64_t>> iteration_counters_;
};

// Non-known trip count while thunks are difficult to unit test, so we only have
// a unit test for the known trip count case.
class KnownTripCountWhileThunkTest : public HloPjRtTestBase {
 protected:
  absl::StatusOr<const HloInstruction*> CreateFakeWhileInstruction() {
    constexpr absl::string_view kDummyModule = R"(
        body {
          ROOT r = (pred[]) parameter(0)
        }
        cond {
          param = (pred[]) parameter(0)
          ROOT r = pred[] get-tuple-element(param), index=0
        }
        ENTRY main {
          p = (pred[]) parameter(0)
          ROOT while = (pred[]) while(p), condition=cond, body=body
        })";

    TF_ASSIGN_OR_RETURN(owned_modules_.emplace_back(),
                        ParseAndReturnVerifiedModule(kDummyModule));
    return owned_modules_.back()->entry_computation()->root_instruction();
  }

  absl::Status ExecuteThunk(Thunk& thunk) {
    TF_ASSIGN_OR_RETURN(auto name, PlatformUtil::CanonicalPlatformName("gpu"));
    TF_ASSIGN_OR_RETURN(auto* platform,
                        se::PlatformManager::PlatformWithName(name));
    TF_ASSIGN_OR_RETURN(auto* executor, platform->ExecutorForDevice(0));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                        executor->CreateStream());
    se::StreamExecutorMemoryAllocator allocator(executor);
    Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
        ServiceExecutableRunOptions(), BufferAllocations({}, 0, &allocator),
        stream.get(), stream.get(), nullptr, nullptr);
    return thunk.ExecuteOnStream(Thunk::ExecuteParams(params));
  }

  std::pair<std::unique_ptr<SequentialThunk>, IterationLoggerThunk*>
  CreateLoggingSequentialThunk(const HloInstruction* loop) {
    auto owned_logger = std::make_unique<IterationLoggerThunk>(loop);
    auto* logger = owned_logger.get();
    ThunkSequence sequence;
    sequence.push_back(std::move(owned_logger));
    auto thunk = std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                                   std::move(sequence));
    return std::make_pair(std::move(thunk), logger);
  }

 private:
  std::vector<std::unique_ptr<VerifiedHloModule>> owned_modules_;
};

TEST_F(KnownTripCountWhileThunkTest, CurrentLoopIterationKnownTripCountTest) {
  TF_ASSERT_OK_AND_ASSIGN(const HloInstruction* loop,
                          CreateFakeWhileInstruction());

  auto [body_thunk, logger] = CreateLoggingSequentialThunk(loop);
  auto condition_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), ThunkSequence());

  BufferAllocation::Slice slice;
  WhileThunk while_thunk(
      Thunk::ThunkInfo(), loop,
      /*condition_result_buffer_index=*/slice,
      /*condition_thunk_sequence=*/std::move(condition_thunk),
      /*body_thunk_sequence_=*/std::move(body_thunk),
      /*trip_count=*/5);

  EXPECT_THAT(ExecuteThunk(while_thunk), IsOk());
  EXPECT_THAT(logger->logged_counters(), ElementsAre(0, 1, 2, 3, 4));
}

TEST_F(KnownTripCountWhileThunkTest, CurrentLoopIterationNestedTest) {
  TF_ASSERT_OK_AND_ASSIGN(const HloInstruction* outer_loop,
                          CreateFakeWhileInstruction());
  TF_ASSERT_OK_AND_ASSIGN(const HloInstruction* inner_loop,
                          CreateFakeWhileInstruction());

  auto [body_thunk, logger] = CreateLoggingSequentialThunk(outer_loop);
  auto inner_condition_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), ThunkSequence());
  auto outer_condition_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), ThunkSequence());

  BufferAllocation::Slice slice;
  auto inner_while_thunk = std::make_unique<WhileThunk>(
      Thunk::ThunkInfo(), inner_loop,
      /*condition_result_buffer_index=*/slice,
      /*condition_thunk_sequence=*/std::move(inner_condition_thunk),
      /*body_thunk_sequence_=*/std::move(body_thunk),
      /*trip_count=*/2);

  ThunkSequence outer_body_sequence;
  outer_body_sequence.push_back(std::move(inner_while_thunk));
  auto outer_body_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(outer_body_sequence));

  WhileThunk outer_while_thunk(
      Thunk::ThunkInfo(), outer_loop,
      /*condition_result_buffer_index=*/slice,
      /*condition_thunk_sequence=*/std::move(outer_condition_thunk),
      /*body_thunk_sequence_=*/std::move(outer_body_thunk),
      /*trip_count=*/3);

  EXPECT_THAT(ExecuteThunk(outer_while_thunk), IsOk());
  EXPECT_THAT(logger->logged_counters(), ElementsAre(0, 0, 1, 1, 2, 2));
}

TEST_F(KnownTripCountWhileThunkTest, CurrentLoopIterationUnknownLoopTest) {
  TF_ASSERT_OK_AND_ASSIGN(const HloInstruction* loop,
                          CreateFakeWhileInstruction());
  TF_ASSERT_OK_AND_ASSIGN(const HloInstruction* not_running_loop,
                          CreateFakeWhileInstruction());

  auto [body_thunk, logger] = CreateLoggingSequentialThunk(not_running_loop);
  auto condition_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), ThunkSequence());

  BufferAllocation::Slice slice;
  WhileThunk while_thunk(
      Thunk::ThunkInfo(), loop,
      /*condition_result_buffer_index=*/slice,
      /*condition_thunk_sequence=*/std::move(condition_thunk),
      /*body_thunk_sequence_=*/std::move(body_thunk),
      /*trip_count=*/3);

  EXPECT_THAT(ExecuteThunk(while_thunk), IsOk());
  EXPECT_THAT(logger->logged_counters(),
              ElementsAre(std::nullopt, std::nullopt, std::nullopt));
}

}  // namespace
}  // namespace xla::gpu
