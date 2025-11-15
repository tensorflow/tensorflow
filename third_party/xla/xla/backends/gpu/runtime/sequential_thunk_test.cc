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

#include "xla/backends/gpu/runtime/sequential_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class DummyThunk : public Thunk {
 public:
  explicit DummyThunk(Thunk::Kind kind, Thunk::ThunkInfo thunk_info)
      : Thunk(kind, std::move(thunk_info)) {}
  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
};

using ::testing::IsEmpty;

constexpr ExecutionStreamId kExecutionStreamId{123};
constexpr absl::string_view kProfileAnnotation = "profile_annotation";

Thunk::ThunkInfo GetExampleThunkInfo() {
  Thunk::ThunkInfo thunk_info{};
  thunk_info.execution_stream_id = kExecutionStreamId;
  thunk_info.profile_annotation = kProfileAnnotation;
  thunk_info.thunk_id = ThunkId(1);
  return thunk_info;
}

TEST(SequentialThunkTest, EmptySequentialThunkToProto) {
  SequentialThunk thunk{GetExampleThunkInfo(), {}};
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  ASSERT_TRUE(proto.has_sequential_thunk());
  EXPECT_EQ(proto.sequential_thunk().thunks_size(), 0);

  ASSERT_TRUE(proto.has_thunk_info());
  EXPECT_EQ(proto.thunk_info().execution_stream_id(), kExecutionStreamId);
  EXPECT_EQ(proto.thunk_info().profile_annotation(), kProfileAnnotation);
}

TEST(SequentialThunkTest, EmptySequentialThunkFromProto) {
  SequentialThunkProto proto;

  Thunk::Deserializer deserializer =
      [](const ThunkProto&) -> absl::StatusOr<std::unique_ptr<Thunk>> {
    return absl::InternalError("This should never be called");
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SequentialThunk> sequential_thunk,
      SequentialThunk::FromProto(GetExampleThunkInfo(), proto, deserializer));

  ASSERT_NE(sequential_thunk, nullptr);
  EXPECT_EQ(sequential_thunk->execution_stream_id(), kExecutionStreamId);
  EXPECT_EQ(sequential_thunk->profile_annotation(), kProfileAnnotation);
  EXPECT_THAT(sequential_thunk->thunks(), IsEmpty());
}

TEST(SequentialThunkTest, SequentialThunkChainFromProto) {
  SequentialThunkProto outer_proto;
  // This adds an inner SequentialThunk into the ThunkSequence of the outer
  // sequential thunk.
  ThunkProto* inner_proto = outer_proto.add_thunks();
  inner_proto->mutable_sequential_thunk();
  inner_proto->mutable_thunk_info()->set_profile_annotation(kProfileAnnotation);
  inner_proto->mutable_thunk_info()->set_execution_stream_id(
      kExecutionStreamId.value());

  Thunk::Deserializer always_fail_deserializer = [](const ThunkProto&) {
    return absl::InternalError("This should never be called.");
  };

  Thunk::Deserializer only_supports_sequential_thunk_deserializer =
      [&](const ThunkProto& proto) -> absl::StatusOr<std::unique_ptr<Thunk>> {
    if (!proto.has_sequential_thunk()) {
      return absl::InvalidArgumentError("This should be a sequential thunk!");
    }

    return SequentialThunk::FromProto(GetExampleThunkInfo(),
                                      proto.sequential_thunk(),
                                      always_fail_deserializer);
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SequentialThunk> outer_thunk,
      SequentialThunk::FromProto(GetExampleThunkInfo(), outer_proto,
                                 only_supports_sequential_thunk_deserializer));

  ASSERT_NE(outer_thunk, nullptr);
  EXPECT_EQ(outer_thunk->execution_stream_id(), kExecutionStreamId);
  EXPECT_EQ(outer_thunk->profile_annotation(), kProfileAnnotation);

  ASSERT_EQ(outer_thunk->thunks().size(), 1);
  const SequentialThunk* inner_thunk =
      dynamic_cast<const SequentialThunk*>(outer_thunk->thunks().front().get());
  ASSERT_NE(inner_thunk, nullptr);
  EXPECT_THAT(inner_thunk->thunks(), IsEmpty());
  EXPECT_EQ(inner_thunk->execution_stream_id(), kExecutionStreamId);
  EXPECT_EQ(inner_thunk->profile_annotation(), kProfileAnnotation);
}

TEST(SequentialThunkTest, ToString) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;
  thunk_info.thunk_id = ThunkId(1);

  ThunkSequence thunks;
  thunks.push_back(
      std::make_unique<DummyThunk>(Thunk::Kind::kGemm, thunk_info));

  thunk_info.thunk_id = ThunkId(2);
  thunks.push_back(
      std::make_unique<DummyThunk>(Thunk::Kind::kGemm, thunk_info));

  thunk_info.thunk_id = ThunkId(3);
  thunks.push_back(
      std::make_unique<DummyThunk>(Thunk::Kind::kGemm, thunk_info));

  thunk_info.thunk_id = ThunkId(4);
  SequentialThunk sequential_thunk(thunk_info, std::move(thunks));
  EXPECT_EQ(sequential_thunk.ToString(/*indent=*/0),
            "001: kGemm\t\n"
            "002: kGemm\t\n"
            "003: kGemm\t\n");
  EXPECT_EQ(sequential_thunk.ToString(/*indent=*/1),
            "  001: kGemm\t\n"
            "  002: kGemm\t\n"
            "  003: kGemm\t\n");
}

TEST(SequentialThunkTest, TransformAllNestedThunks) {
  auto make_info = [](uint64_t id) {
    Thunk::ThunkInfo info;
    info.thunk_id = ThunkId(id);
    return info;
  };
  ThunkSequence thunks;
  thunks.push_back(
      std::make_unique<DummyThunk>(Thunk::Kind::kGemm, make_info(1)));
  thunks.push_back(
      std::make_unique<DummyThunk>(Thunk::Kind::kGemm, make_info(2)));
  thunks.push_back(
      std::make_unique<DummyThunk>(Thunk::Kind::kGemm, make_info(3)));
  SequentialThunk sequential_thunk(Thunk::ThunkInfo(), std::move(thunks));

  TF_EXPECT_OK(sequential_thunk.TransformAllNestedThunks(
      [&](std::unique_ptr<Thunk> thunk) -> std::unique_ptr<Thunk> {
        return std::make_unique<DummyThunk>(
            Thunk::Kind::kCopy,
            make_info(thunk->thunk_info().thunk_id.value() + 10));
      }));

  EXPECT_EQ(sequential_thunk.thunks().size(), 3);
  EXPECT_EQ(sequential_thunk.thunks()[0]->kind(), Thunk::Kind::kCopy);
  EXPECT_EQ(sequential_thunk.thunks()[0]->thunk_info().thunk_id, ThunkId(11));
  EXPECT_EQ(sequential_thunk.thunks()[1]->kind(), Thunk::Kind::kCopy);
  EXPECT_EQ(sequential_thunk.thunks()[1]->thunk_info().thunk_id, ThunkId(12));
  EXPECT_EQ(sequential_thunk.thunks()[2]->kind(), Thunk::Kind::kCopy);
  EXPECT_EQ(sequential_thunk.thunks()[2]->thunk_info().thunk_id, ThunkId(13));
}

}  // namespace
}  // namespace xla::gpu
