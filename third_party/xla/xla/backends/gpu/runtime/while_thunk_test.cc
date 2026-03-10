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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::NotNull;
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
  static absl::StatusOr<std::unique_ptr<DummyThunk>> FromProto(
      const ThunkProto& thunk_proto, Thunk::Kind kind) {
    TF_ASSIGN_OR_RETURN(Thunk::ThunkInfo thunk_info,
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

  const std::vector<std::optional<int64_t>>& logged_counters() const {
    return iteration_counters_;
  }

 private:
  std::vector<std::optional<int64_t>> iteration_counters_;
};

// Non-known trip count while thunks are difficult to unit test, so we only have
// a unit test for the known trip count case.
class KnownTripCountWhileThunkTest : public HloPjRtTestBase {
 protected:
  absl::Status ExecuteThunk(Thunk& thunk) {
    TF_ASSIGN_OR_RETURN(auto name, PlatformUtil::CanonicalPlatformName("gpu"));
    TF_ASSIGN_OR_RETURN(auto* platform,
                        se::PlatformManager::PlatformWithName(name));
    TF_ASSIGN_OR_RETURN(auto* executor, platform->ExecutorForDevice(0));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
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

TEST(WhileThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);

  ThunkSequence condition_thunks;
  condition_thunks.push_back(
      std::make_unique<DummyThunk>(Kind::kConditional, thunk_info));
  condition_thunks.push_back(
      std::make_unique<DummyThunk>(Kind::kConditional, thunk_info));

  ThunkSequence body_thunks;
  body_thunks.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  body_thunks.push_back(
      std::make_unique<DummyThunk>(Kind::kCustomCall, thunk_info));

  WhileThunk thunk =
      CreateWhileThunk(thunk_info, slice, std::move(condition_thunks),
                       std::move(body_thunks), /*trip_count=*/10);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());

  EXPECT_THAT(proto, EqualsProto(R"pb(
                thunk_info {
                  profile_annotation: "profile_annotation"
                  execution_stream_id: 123
                }
                while_thunk {
                  condition_result_buffer_index { size: 256 }
                  condition_thunk_sequence {
                    thunks {
                      thunk_info {
                        profile_annotation: "profile_annotation"
                        execution_stream_id: 123
                      }
                    }
                    thunks {
                      thunk_info {
                        profile_annotation: "profile_annotation"
                        execution_stream_id: 123
                      }
                    }
                  }
                  body_thunk_sequence {
                    thunks {
                      thunk_info {
                        profile_annotation: "profile_annotation"
                        execution_stream_id: 123
                      }
                    }
                    thunks {
                      thunk_info {
                        profile_annotation: "profile_annotation"
                        execution_stream_id: 123
                      }
                    }
                  }
                  trip_count: 10
                }
              )pb"));
}

TEST(WhileThunkTest, FromProto) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        while_thunk {
          condition_result_buffer_index {
            buffer_allocation_index: 1
            offset: 16
            size: 256
          }
          condition_thunk_sequence {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
            }
          }
          body_thunk_sequence {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
            }
          }
          trip_count: 10
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WhileThunk> thunk,
      WhileThunk::FromProto(thunk_info, proto.while_thunk(), buffer_allocations,
                            [](const ThunkProto& proto)
                                -> absl::StatusOr<std::unique_ptr<DummyThunk>> {
                              return DummyThunk::FromProto(proto,
                                                           Kind::kCustomCall);
                            }));
  ASSERT_NE(thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(WhileThunkTest, TransformNested) {
  Thunk::ThunkInfo thunk_info;
  BufferAllocation::Slice slice;

  ThunkSequence condition_thunks;
  condition_thunks.push_back(
      std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  ThunkSequence body_thunks;
  body_thunks.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  auto while_thunk = std::make_unique<WhileThunk>(
      Thunk::ThunkInfo(),
      /*condition_result_buffer_index=*/slice,
      /*condition_thunks=*/std::move(condition_thunks),
      /*body_thunks=*/std::move(body_thunks),
      /*trip_count=*/3);

  TF_EXPECT_OK(while_thunk->TransformNested([](auto) {
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
