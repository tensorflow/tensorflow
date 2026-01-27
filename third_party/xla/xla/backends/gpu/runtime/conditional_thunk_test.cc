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
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::Pointee;
using ::testing::Property;
using ::testing::SizeIs;
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
    *proto.mutable_thunk_info() = thunk_info().ToProto();
    return proto;
  }
};

std::unique_ptr<ConditionalThunk> CreateConditionalThunk(
    const Thunk::ThunkInfo& thunk_info,
    const ShapedSlice& branch_index_buffer_index,
    std::vector<ThunkSequence> branch_thunk_sequences) {
  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks;
  for (auto& thunk_sequence : branch_thunk_sequences) {
    branch_thunks.push_back(std::make_unique<SequentialThunk>(
        thunk_info, std::move(thunk_sequence)));
  }

  return std::make_unique<ConditionalThunk>(
      thunk_info, branch_index_buffer_index, std::move(branch_thunks));
}

TEST(ConditionalThunkTest, BufferUses) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);

  constexpr bool kBranchIndexIsBool = true;
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);
  Shape shape = ShapeUtil::MakeShape(PRED, {});

  ThunkSequence false_seq;
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  ThunkSequence true_seq;
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  std::vector<ThunkSequence> branch_thunk_sequences;
  branch_thunk_sequences.push_back(std::move(false_seq));
  branch_thunk_sequences.push_back(std::move(true_seq));

  std::unique_ptr<ConditionalThunk> thunk = CreateConditionalThunk(
      thunk_info, {slice, shape}, std::move(branch_thunk_sequences));

  EXPECT_EQ(thunk->branch_index_is_bool(), kBranchIndexIsBool);
  EXPECT_EQ(thunk->branch_index_buffer().slice, slice);

  auto thunk_matcher = Pointee(Property(&Thunk::kind, Thunk::Kind::kGemm));
  auto branch_matcher = Pointee(Property(
      &SequentialThunk::thunks, ElementsAre(thunk_matcher, thunk_matcher)));
  EXPECT_THAT(thunk->branch_thunks(),
              ElementsAre(branch_matcher, branch_matcher));
}

TEST(ConditionalThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);
  Shape shape = ShapeUtil::MakeShape(PRED, {});

  ThunkSequence false_seq;
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  ThunkSequence true_seq;
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  std::vector<ThunkSequence> branch_thunk_seq;
  branch_thunk_seq.push_back(std::move(false_seq));
  branch_thunk_seq.push_back(std::move(true_seq));

  std::unique_ptr<ConditionalThunk> thunk = CreateConditionalThunk(
      thunk_info, {slice, shape}, std::move(branch_thunk_seq));
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk->ToProto());

  std::string expected = R"pb(
    thunk_info {
      profile_annotation: "profile_annotation"
      execution_stream_id: 123
    }
    conditional_thunk {
      branch_index_buffer {
        slice { size: 256 }
        shape {
          element_type: PRED
          layout { tail_padding_alignment_in_elements: 1 }
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
          branch_index_buffer {
            slice { offset: 8 size: 256 buffer_allocation_index: 0 }
            shape {
              element_type: PRED
              layout { tail_padding_alignment_in_elements: 1 }
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

TEST(ConditionalThunkTest, ToString) {
  Thunk::ThunkInfo thunk_info;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);
  Shape bool_shape = ShapeUtil::MakeShape(PRED, {});
  Shape int_shape = ShapeUtil::MakeShape(S32, {});

  auto create_branch_thunk_sequences = [&]() -> std::vector<ThunkSequence> {
    ThunkSequence false_seq;
    false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

    ThunkSequence true_seq;
    true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
    true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

    std::vector<ThunkSequence> branch_thunk_sequences;
    branch_thunk_sequences.push_back(std::move(false_seq));
    branch_thunk_sequences.push_back(std::move(true_seq));
    return branch_thunk_sequences;
  };

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(CreateConditionalThunk(
      thunk_info, {slice, bool_shape}, create_branch_thunk_sequences()));
  auto sequential_thunk =
      std::make_unique<SequentialThunk>(thunk_info, std::move(thunk_sequence));
  EXPECT_EQ(sequential_thunk->ToString(/*indent=*/0),
            "000: kConditional\t  \n"
            "  false_branch:\n"
            "    000: kGemm\t\n"
            "  true_branch:\n"
            "    000: kGemm\t\n"
            "    000: kGemm\t\n\n");

  std::unique_ptr<ConditionalThunk> thunk = CreateConditionalThunk(
      thunk_info, {slice, int_shape}, create_branch_thunk_sequences());

  EXPECT_EQ(thunk->ToString(/*indent=*/0),
            "\n"
            "branch_0:\n"
            "  000: kGemm\t\n"
            "branch_1:\n"
            "  000: kGemm\t\n"
            "  000: kGemm\t\n");
}

TEST(ConditionalThunkTest, TransformAllNestedThunks) {
  BufferAllocation::Slice slice;
  Shape shape = ShapeUtil::MakeShape(S32, {});

  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks;
  branch_thunks.push_back(
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), ThunkSequence()));
  branch_thunks.push_back(
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), ThunkSequence()));
  auto conditional_thunk = std::make_unique<ConditionalThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice, shape}, std::move(branch_thunks));

  TF_EXPECT_OK(conditional_thunk->TransformAllNestedThunks([](auto) {
    return std::make_unique<DummyThunk>(Kind::kCustomCall, Thunk::ThunkInfo());
  }));

  EXPECT_THAT(conditional_thunk->branch_thunks(), SizeIs(2));
  EXPECT_THAT(conditional_thunk->branch_thunks()[0]->thunks(), SizeIs(1));
  EXPECT_THAT(conditional_thunk->branch_thunks()[0]->thunks()[0]->kind(),
              Kind::kCustomCall);
  EXPECT_THAT(conditional_thunk->branch_thunks()[1]->thunks(), SizeIs(1));
  EXPECT_THAT(conditional_thunk->branch_thunks()[1]->thunks()[0]->kind(),
              Kind::kCustomCall);
}

}  // namespace
}  // namespace xla::gpu
