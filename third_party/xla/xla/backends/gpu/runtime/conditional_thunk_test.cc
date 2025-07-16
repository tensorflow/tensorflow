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

#include "xla/backends/gpu/runtime/conditional_thunk.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu {
namespace {

using ::testing::Pointee;
using ::testing::Property;
using ::tsl::proto_testing::EqualsProto;
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
    TF_ASSIGN_OR_RETURN(*proto.mutable_thunk_info(), GetThunkInfoProto());
    return proto;
  }
};

ConditionalThunk CreateConditionalThunk(
    const Thunk::ThunkInfo& thunk_info,
    const BufferAllocation::Slice& branch_index_buffer_index,
    std::vector<ThunkSequence> branch_thunk_sequences,
    bool kBranchIndexIsBool) {
  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks;
  for (auto& thunk_sequence : branch_thunk_sequences) {
    branch_thunks.push_back(std::make_unique<SequentialThunk>(
        thunk_info, std::move(thunk_sequence)));
  }

  return ConditionalThunk(thunk_info, branch_index_buffer_index,
                          std::move(branch_thunks), kBranchIndexIsBool);
}

TEST(ConditionalThunkTest, BufferUses) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);

  ThunkSequence false_seq;
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  ThunkSequence true_seq;
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  std::vector<ThunkSequence> branch_thunk_sequences;
  branch_thunk_sequences.push_back(std::move(false_seq));
  branch_thunk_sequences.push_back(std::move(true_seq));

  constexpr bool kBranchIndexIsBool = true;
  ConditionalThunk thunk = CreateConditionalThunk(
      thunk_info, slice, std::move(branch_thunk_sequences), kBranchIndexIsBool);

  EXPECT_EQ(thunk.branch_index_is_bool(), kBranchIndexIsBool);
  EXPECT_EQ(thunk.branch_index_buffer(), slice);

  auto thunk_matcher = Pointee(Property(&Thunk::kind, Thunk::Kind::kGemm));
  auto branch_matcher = Pointee(Property(
      &SequentialThunk::thunks, ElementsAre(thunk_matcher, thunk_matcher)));
  EXPECT_THAT(thunk.branch_thunks(),
              ElementsAre(branch_matcher, branch_matcher));
}

TEST(ConditionalThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);

  ThunkSequence false_seq;
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  ThunkSequence true_seq;
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  std::vector<ThunkSequence> branch_thunk_seq;
  branch_thunk_seq.push_back(std::move(false_seq));
  branch_thunk_seq.push_back(std::move(true_seq));

  constexpr bool kBranchIndexIsBool = true;
  ConditionalThunk thunk = CreateConditionalThunk(
      thunk_info, slice, std::move(branch_thunk_seq), kBranchIndexIsBool);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());

  std::string expected = R"pb(
    thunk_info {
      profile_annotation: "profile_annotation"
      execution_stream_id: 123
    }
    conditional_thunk {
      branch_index_buffer { size: 256 }
      branch_thunks {
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
      branch_thunks {
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
      branch_index_is_bool: true
    }
  )pb";
  EXPECT_THAT(proto, EqualsProto(expected));
}

TEST(ConditionalThunkTest, FromProto) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        conditional_thunk {
          branch_index_buffer { offset: 8 size: 256 buffer_allocation_index: 0 }
          branch_thunks {
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
          branch_thunks {
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
          branch_index_is_bool: true
        }
      )pb",
      &proto));

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ConditionalThunk> thunk,
      ConditionalThunk::FromProto(
          thunk_info, proto.conditional_thunk(), buffer_allocations,
          [](const ThunkProto& proto)
              -> absl::StatusOr<std::unique_ptr<DummyThunk>> {
            return DummyThunk::FromProto(proto, Kind::kCustomCall);
          }));
  ASSERT_NE(thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
