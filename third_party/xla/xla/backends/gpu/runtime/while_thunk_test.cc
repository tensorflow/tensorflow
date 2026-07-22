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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/mock_command_buffer.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using Kind = Thunk::Kind;

// A dummy `Thunk` that does nothing.
struct DummyThunk : public Thunk {
  explicit DummyThunk(Thunk::Kind kind, Thunk::ThunkInfo thunk_info)
      : Thunk(kind, std::move(thunk_info)) {}
  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
  BufferUses buffer_uses() const override { return {}; }
  static absl::StatusOr<std::unique_ptr<DummyThunk>> FromProto(
      const ThunkProto& thunk_proto, Thunk::Kind kind) {
    ASSIGN_OR_RETURN(Thunk::ThunkInfo thunk_info,
                     Thunk::ThunkInfo::FromProto(thunk_proto.thunk_info()));
    return std::make_unique<DummyThunk>(kind, std::move(thunk_info));
  }

  absl::StatusOr<ThunkProto> ToProto() const override {
    ThunkProto proto;
    *proto.mutable_thunk_info() = thunk_info().ToProto();
    return proto;
  }
};

WhileThunk CreateWhileThunk(
    const Thunk::ThunkInfo& thunk_info,
    const BufferAllocation::Slice& condition_result_buffer_index,
    ThunkSequence condition_thunks, ThunkSequence body_thunks,
    std::optional<int64_t> trip_count) {
  return WhileThunk(thunk_info, condition_result_buffer_index,
                    std::move(condition_thunks), std::move(body_thunks),
                    trip_count);
}

struct CommandRecordCounts {
  int prepares = 0;
  int creates = 0;
  int updates = 0;
};

struct FakeSeCommand : public se::CommandBuffer::Command {};

class RecordingCommand : public Command {
 public:
  explicit RecordingCommand(CommandRecordCounts* counts)
      : Command(Thunk::Kind::kCommand), counts_(counts) {}

  absl::Status Prepare(const PrepareParams&) override {
    ++counts_->prepares;
    return absl::OkStatus();
  }

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams&, const RecordParams&, RecordAction action,
      se::CommandBuffer*) override {
    if (std::get_if<Command::RecordCreate>(&action) != nullptr) {
      ++counts_->creates;
      return &recorded_command_;
    }

    auto* update = std::get_if<Command::RecordUpdate>(&action);
    if (update == nullptr) {
      return absl::InternalError("unexpected record action");
    }
    if (update->command != &recorded_command_) {
      return absl::InternalError("unexpected recorded command");
    }
    ++counts_->updates;
    return &recorded_command_;
  }

  BufferUses buffer_uses() const override { return {}; }

 private:
  CommandRecordCounts* counts_;
  FakeSeCommand recorded_command_;
};

absl::StatusOr<CommandExecutor> MakeCommandExecutor(
    CommandRecordCounts* counts) {
  CommandSequence commands;
  commands.Append(std::make_unique<RecordingCommand>(counts));
  return CommandExecutor::Create(
      std::move(commands), CommandExecutor::SynchronizationMode::kSerialize);
}

struct NestedCommandBuffer {
  std::unique_ptr<::testing::NiceMock<se::MockCommandBuffer>> command_buffer =
      std::make_unique<::testing::NiceMock<se::MockCommandBuffer>>();
  se::CommandBuffer::State state = se::CommandBuffer::State::kCreate;
};

void ConfigureNestedCommandBuffer(NestedCommandBuffer* nested) {
  using Mode = se::CommandBuffer::Mode;
  using State = se::CommandBuffer::State;

  ON_CALL(*nested->command_buffer, mode())
      .WillByDefault(::testing::Return(Mode::kNested));
  ON_CALL(*nested->command_buffer, state()).WillByDefault([nested] {
    return nested->state;
  });
  ON_CALL(*nested->command_buffer, Finalize()).WillByDefault([nested] {
    if (nested->state != State::kCreate && nested->state != State::kUpdate) {
      return absl::FailedPreconditionError(
          "command buffer is not in create/update state");
    }
    nested->state = State::kFinalized;
    return absl::OkStatus();
  });
  ON_CALL(*nested->command_buffer, Update()).WillByDefault([nested] {
    if (nested->state != State::kFinalized) {
      return absl::FailedPreconditionError(
          "command buffer is not in finalized state");
    }
    nested->state = State::kUpdate;
    return absl::OkStatus();
  });
}

class IterationLoggerThunk : public Thunk {
 public:
  explicit IterationLoggerThunk()
      : Thunk(Thunk::Kind::kKernel, Thunk::ThunkInfo()) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    if (const WhileLoopState* state = IsInsideWhileLoop()) {
      iteration_counters_.push_back(state->loop_iteration);
    } else {
      iteration_counters_.push_back(std::nullopt);
    }
    return absl::OkStatus();
  }

  BufferUses buffer_uses() const override { return {}; }

  const std::vector<std::optional<int64_t>>& logged_counters() const {
    return iteration_counters_;
  }

  absl::StatusOr<ThunkProto> ToProto() const override {
    return absl::UnimplementedError("Not implemented");
  }

 private:
  std::vector<std::optional<int64_t>> iteration_counters_;
};

// Non-known trip count while thunks are difficult to unit test, so we only have
// a unit test for the known trip count case.
class KnownTripCountWhileThunkTest : public HloTestBase {
 protected:
  absl::Status ExecuteThunk(Thunk& thunk) {
    ASSIGN_OR_RETURN(auto name, PlatformUtil::CanonicalPlatformName("gpu"));
    ASSIGN_OR_RETURN(auto* platform,
                     se::PlatformManager::PlatformWithName(name));
    ASSIGN_OR_RETURN(auto* executor, platform->ExecutorForDevice(0));
    ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                     executor->CreateStream());
    stream_executor::StreamExecutorAddressAllocator allocator(executor);
    Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
        ServiceExecutableRunOptions(), BufferAllocations({}, 0, &allocator),
        stream.get(), stream.get(), nullptr, nullptr, nullptr);
    return thunk.ExecuteOnStream(Thunk::ExecuteParams(params));
  }

  std::pair<ThunkSequence, IterationLoggerThunk*> CreateLoggingThunkSequence() {
    auto owned_logger = std::make_unique<IterationLoggerThunk>();
    auto* logger = owned_logger.get();
    ThunkSequence sequence;
    sequence.push_back(std::move(owned_logger));
    return std::make_pair(std::move(sequence), logger);
  }

 private:
  std::vector<std::unique_ptr<VerifiedHloModule>> owned_modules_;
};

TEST_F(KnownTripCountWhileThunkTest, CurrentLoopIterationKnownTripCountTest) {
  auto [body_thunks, logger] = CreateLoggingThunkSequence();

  BufferAllocation::Slice slice;
  WhileThunk while_thunk(Thunk::ThunkInfo(),
                         /*condition_result_buffer_index=*/slice,
                         /*condition_thunks=*/ThunkSequence(),
                         /*body_thunks=*/std::move(body_thunks),
                         /*trip_count=*/5);

  EXPECT_THAT(ExecuteThunk(while_thunk), absl_testing::IsOk());
  EXPECT_THAT(logger->logged_counters(), ElementsAre(0, 1, 2, 3, 4));
}
TEST_F(KnownTripCountWhileThunkTest, CurrentLoopIterationNestedTest) {
  auto [body_thunks, logger] = CreateLoggingThunkSequence();

  BufferAllocation::Slice slice;
  auto inner_while_thunk =
      std::make_unique<WhileThunk>(Thunk::ThunkInfo(),
                                   /*condition_result_buffer_index=*/slice,
                                   /*condition_thunks=*/ThunkSequence(),
                                   /*body_thunks=*/std::move(body_thunks),
                                   /*trip_count=*/2);

  ThunkSequence outer_body_sequence;
  outer_body_sequence.push_back(std::move(inner_while_thunk));

  WhileThunk outer_while_thunk(Thunk::ThunkInfo(),
                               /*condition_result_buffer_index=*/slice,
                               /*condition_thunks=*/ThunkSequence(),
                               /*body_thunks=*/std::move(outer_body_sequence),
                               /*trip_count=*/3);

  EXPECT_THAT(ExecuteThunk(outer_while_thunk), absl_testing::IsOk());
  EXPECT_THAT(logger->logged_counters(), ElementsAre(0, 1, 0, 1, 0, 1));
}

TEST(WhileThunkTest, PreparePropagatesToCommandBufferExecutors) {
  BufferAllocation pred_alloc(/*index=*/0, /*size=*/sizeof(bool), /*color=*/0);
  BufferAllocation::Slice pred_slice(&pred_alloc, /*offset=*/0,
                                     /*size=*/sizeof(bool));
  WhileThunk thunk(Thunk::ThunkInfo(), pred_slice,
                   /*condition_thunks=*/ThunkSequence(),
                   /*body_thunks=*/ThunkSequence());

  CommandRecordCounts cond_counts;
  CommandRecordCounts body_counts;
  ASSERT_OK_AND_ASSIGN(CommandExecutor cond_executor,
                       MakeCommandExecutor(&cond_counts));
  ASSERT_OK_AND_ASSIGN(CommandExecutor body_executor,
                       MakeCommandExecutor(&body_counts));
  ASSERT_OK(thunk.SetOrUpdateCommandBufferExecutors(
      std::move(cond_executor), std::move(body_executor),
      /*enable_loop_unroll=*/false));

  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(platform_name));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  std::vector<se::DeviceAddressBase> buffers;
  BufferAllocations allocations(buffers, /*device_ordinal=*/0,
                                /*memory_allocator=*/nullptr);

  Thunk::PrepareParams prepare_params{/*collective_params=*/nullptr,
                                      /*collective_clique_requests=*/nullptr,
                                      /*collective_memory_requests=*/nullptr,
                                      /*executor=*/executor,
                                      /*buffer_allocations=*/&allocations};
  ASSERT_OK(thunk.Prepare(prepare_params));

  EXPECT_EQ(cond_counts.prepares, 1);
  EXPECT_EQ(body_counts.prepares, 1);
}

TEST(WhileThunkTest, RecordCreatesAndUpdatesCommandBufferWhile) {
  BufferAllocation pred_alloc(/*index=*/0, /*size=*/sizeof(bool), /*color=*/0);
  BufferAllocation::Slice pred_slice(&pred_alloc, /*offset=*/0,
                                     /*size=*/sizeof(bool));
  WhileThunk thunk(Thunk::ThunkInfo(), pred_slice,
                   /*condition_thunks=*/ThunkSequence(),
                   /*body_thunks=*/ThunkSequence());

  CommandRecordCounts cond_counts;
  CommandRecordCounts body_counts;
  ASSERT_OK_AND_ASSIGN(CommandExecutor cond_executor,
                       MakeCommandExecutor(&cond_counts));
  ASSERT_OK_AND_ASSIGN(CommandExecutor body_executor,
                       MakeCommandExecutor(&body_counts));
  ASSERT_OK(thunk.SetOrUpdateCommandBufferExecutors(
      std::move(cond_executor), std::move(body_executor),
      /*enable_loop_unroll=*/false));

  bool pred = true;
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(&pred, sizeof(pred))};
  BufferAllocations allocations(buffers, /*device_ordinal=*/0,
                                /*memory_allocator=*/nullptr);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, allocations, /*stream=*/nullptr,
      /*command_buffer_trace_stream=*/nullptr, /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

  CommandStateManager state_manager;
  Command::RecordParams record_params{state_manager};
  ::testing::NiceMock<se::MockCommandBuffer> command_buffer;
  se::CommandBuffer::State command_buffer_state =
      se::CommandBuffer::State::kCreate;
  ON_CALL(command_buffer, mode())
      .WillByDefault(::testing::Return(se::CommandBuffer::Mode::kPrimary));
  ON_CALL(command_buffer, state()).WillByDefault([&] {
    return command_buffer_state;
  });

  FakeSeCommand while_se_command;
  FakeSeCommand dependency_command;
  auto body_command_buffer = std::make_unique<NestedCommandBuffer>();
  ConfigureNestedCommandBuffer(body_command_buffer.get());
  int create_while_calls = 0;
  int update_while_calls = 0;

  std::vector<const se::CommandBuffer::Command*> dependencies = {
      &dependency_command};
  EXPECT_CALL(command_buffer,
              CreateWhile(::testing::A<se::DeviceAddress<bool>>(), ::testing::_,
                          ::testing::_, ::testing::_))
      .WillOnce([&](se::DeviceAddress<bool>,
                    se::CommandBuffer::CreateCommands create_cond,
                    se::CommandBuffer::CreateCommands create_body,
                    absl::Span<const se::CommandBuffer::Command* const>
                        create_dependencies)
                    -> absl::StatusOr<const se::CommandBuffer::Command*> {
        ++create_while_calls;
        if (create_dependencies.size() != dependencies.size()) {
          return absl::InternalError("unexpected dependency count");
        }
        for (size_t i = 0; i < dependencies.size(); ++i) {
          if (create_dependencies[i] != dependencies[i]) {
            return absl::InternalError("unexpected dependency");
          }
        }

        RETURN_IF_ERROR(
            create_cond(&command_buffer, create_dependencies).status());
        ASSIGN_OR_RETURN(
            std::vector<const se::CommandBuffer::Command*> body_commands,
            create_body(body_command_buffer->command_buffer.get(),
                        /*dependencies=*/{}));
        RETURN_IF_ERROR(create_cond(body_command_buffer->command_buffer.get(),
                                    body_commands)
                            .status());
        RETURN_IF_ERROR(body_command_buffer->command_buffer->Finalize());
        return &while_se_command;
      });

  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* while_command,
      thunk.Record(execute_params, record_params,
                   Command::RecordCreate{dependencies}, &command_buffer));

  EXPECT_EQ(while_command, &while_se_command);
  EXPECT_EQ(create_while_calls, 1);
  EXPECT_EQ(cond_counts.creates, 2);
  EXPECT_EQ(body_counts.creates, 1);
  EXPECT_EQ(cond_counts.updates, 0);
  EXPECT_EQ(body_counts.updates, 0);

  command_buffer_state = se::CommandBuffer::State::kUpdate;
  EXPECT_CALL(
      command_buffer,
      UpdateWhile(&while_se_command, ::testing::A<se::DeviceAddress<bool>>(),
                  ::testing::_, ::testing::_))
      .WillOnce([&](const se::CommandBuffer::Command* command,
                    se::DeviceAddress<bool>,
                    se::CommandBuffer::UpdateCommands update_cond,
                    se::CommandBuffer::UpdateCommands update_body)
                    -> absl::Status {
        ++update_while_calls;
        if (command != &while_se_command) {
          return absl::InternalError("unexpected while command");
        }

        RETURN_IF_ERROR(update_cond(&command_buffer));
        RETURN_IF_ERROR(body_command_buffer->command_buffer->Update());
        RETURN_IF_ERROR(update_body(body_command_buffer->command_buffer.get()));
        RETURN_IF_ERROR(update_cond(body_command_buffer->command_buffer.get()));
        RETURN_IF_ERROR(body_command_buffer->command_buffer->Finalize());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_while_command,
      thunk.Record(execute_params, record_params,
                   Command::RecordUpdate{while_command}, &command_buffer));

  EXPECT_EQ(updated_while_command, while_command);
  EXPECT_EQ(update_while_calls, 1);
  EXPECT_EQ(cond_counts.creates, 2);
  EXPECT_EQ(body_counts.creates, 1);
  EXPECT_EQ(cond_counts.updates, 2);
  EXPECT_EQ(body_counts.updates, 1);
}

TEST(WhileThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);

  ThunkSequence condition_thunks;
  condition_thunks.Emplace<DummyThunk>(Kind::kConditional, thunk_info);
  condition_thunks.Emplace<DummyThunk>(Kind::kConditional, thunk_info);

  ThunkSequence body_thunks;
  body_thunks.Emplace<DummyThunk>(Kind::kGemm, thunk_info);
  body_thunks.Emplace<DummyThunk>(Kind::kCustomCall, thunk_info);

  WhileThunk thunk =
      CreateWhileThunk(thunk_info, slice, std::move(condition_thunks),
                       std::move(body_thunks), /*trip_count=*/10);
  ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());

  EXPECT_THAT(
      proto, EqualsProto(R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        while_thunk {
          condition_result_buffer_index { size: 256 }
          condition_thunk_sequence {
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
          }
          body_thunk_sequence {
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
          }
          trip_count: 10
        }
      )pb"));
}

TEST(WhileThunkTest, FromProto) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        while_thunk {
          condition_result_buffer_index {
            buffer_allocation_index: 1
            offset: 16
            size: 256
          }
          condition_thunk_sequence {
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
          }
          body_thunk_sequence {
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
          }
          trip_count: 10
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WhileThunk> thunk,
      WhileThunk::FromProto(thunk_info, proto.while_thunk(), buffer_allocations,
                            [](const ThunkProto& proto)
                                -> absl::StatusOr<std::unique_ptr<DummyThunk>> {
                              return DummyThunk::FromProto(proto,
                                                           Kind::kCustomCall);
                            }));
  ASSERT_NE(thunk, nullptr);
  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(WhileThunkTest, TransformNested) {
  Thunk::ThunkInfo thunk_info;
  BufferAllocation::Slice slice;

  ThunkSequence condition_thunks =
      ThunkSequence::Of<DummyThunk>(Kind::kGemm, thunk_info);
  ThunkSequence body_thunks =
      ThunkSequence::Of<DummyThunk>(Kind::kGemm, thunk_info);

  auto while_thunk = std::make_unique<WhileThunk>(
      Thunk::ThunkInfo(),
      /*condition_result_buffer_index=*/slice,
      /*condition_thunks=*/std::move(condition_thunks),
      /*body_thunks=*/std::move(body_thunks),
      /*trip_count=*/3);

  EXPECT_OK(while_thunk->TransformNested([](auto) {
    return std::make_unique<DummyThunk>(Kind::kCustomCall, Thunk::ThunkInfo());
  }));

  EXPECT_THAT(while_thunk->condition_executor().thunks(), SizeIs(1));
  EXPECT_THAT(while_thunk->condition_executor().thunks()[0]->kind(),
              Kind::kCustomCall);
  EXPECT_THAT(while_thunk->body_executor().thunks(), SizeIs(1));
  EXPECT_THAT(while_thunk->body_executor().thunks()[0]->kind(),
              Kind::kCustomCall);
}

}  // namespace
}  // namespace xla::gpu
