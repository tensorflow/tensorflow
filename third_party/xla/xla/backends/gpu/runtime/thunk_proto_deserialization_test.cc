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

#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pointer;
using ::testing::Property;
using ::testing::WhenDynamicCastTo;
using ::tsl::proto_testing::EqualsProto;
using Kind = Thunk::Kind;

TEST(ThunkProtoDeserializationTest, SequentialThunkChain) {
  constexpr ExecutionStreamId kExecutionStreamId{123};
  constexpr absl::string_view kProfileAnnotation = "profile_annotation";

  Thunk::ThunkInfo thunk_info{};
  thunk_info.execution_stream_id = kExecutionStreamId;
  thunk_info.profile_annotation = kProfileAnnotation;

  // This constructs the following thunk tree:
  // `SequentialThunk{SequentialThunk{}}`
  std::unique_ptr<Thunk> inner_thunk =
      std::make_unique<SequentialThunk>(thunk_info, ThunkSequence{});
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(inner_thunk));
  SequentialThunk outer_thunk(thunk_info, std::move(thunk_sequence));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, outer_thunk.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> new_thunk,
                          DeserializeThunkProto(proto, {}));

  EXPECT_THAT(new_thunk.get(),
              WhenDynamicCastTo<const SequentialThunk*>(Property(
                  &SequentialThunk::thunks,
                  ElementsAre(Pointer(WhenDynamicCastTo<const SequentialThunk*>(
                      Property(&SequentialThunk::thunks, IsEmpty())))))));
}

TEST(ThunkProtoDeserializationTest, CopyThunk) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        copy_thunk {
          source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
          destination_buffer { offset: 0 size: 256 buffer_allocation_index: 1 }
          mem_size: 256
        }
      )pb",
      &proto));

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> thunk,
                          DeserializeThunkProto(proto, buffer_allocations));
  auto* copy_thunk = dynamic_cast<CopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, DeviceToHostCopyThunk) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        device_to_host_copy_thunk {
          copy_thunk {
            source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
            destination_buffer {
              offset: 0
              size: 256
              buffer_allocation_index: 1
            }
            mem_size: 256
          }
        }
      )pb",
      &proto));

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> thunk,
                          DeserializeThunkProto(proto, buffer_allocations));
  auto* copy_thunk = dynamic_cast<DeviceToHostCopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, HostToDeviceCopyThunk) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        host_to_device_copy_thunk {
          copy_thunk {
            source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
            destination_buffer {
              offset: 0
              size: 256
              buffer_allocation_index: 1
            }
            mem_size: 256
          }
        }
      )pb",
      &proto));

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> thunk,
                          DeserializeThunkProto(proto, buffer_allocations));
  auto* copy_thunk = dynamic_cast<HostToDeviceCopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk, nullptr);  // Check the cast succeeded
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, copy_thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, WhileThunk) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        while_thunk {
          condition_result_buffer_index {
            buffer_allocation_index: 5
            offset: 16
            size: 256
          }
          condition_thunk_sequence {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { buffer_allocation_index: 0 }
                destination_buffer { buffer_allocation_index: 1 }
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { buffer_allocation_index: 1 }
                destination_buffer { buffer_allocation_index: 2 }
              }
            }
          }
          body_thunk_sequence {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { buffer_allocation_index: 2 }
                destination_buffer { buffer_allocation_index: 3 }
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { buffer_allocation_index: 3 }
                destination_buffer { buffer_allocation_index: 4 }
              }
            }
          }
          trip_count: 10
        }
      )pb",
      &proto));

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/3, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/4, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/5, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> athunk,
                          DeserializeThunkProto(proto, buffer_allocations));
  auto* thunk = dynamic_cast<WhileThunk*>(athunk.get());
  ASSERT_NE(thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ThunkProtoDeserializationTest, ConditionalThunk) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        conditional_thunk {
          branch_index_buffer { offset: 8 size: 256 buffer_allocation_index: 5 }
          branch_thunks {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { offset: 0 size: 256 buffer_allocation_index: 0 }
                destination_buffer {
                  offset: 1
                  size: 257
                  buffer_allocation_index: 1
                }
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { offset: 2 size: 258 buffer_allocation_index: 1 }
                destination_buffer {
                  offset: 3
                  size: 259
                  buffer_allocation_index: 2
                }
              }
            }
          }
          branch_thunks {
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { offset: 4 size: 260 buffer_allocation_index: 2 }
                destination_buffer {
                  offset: 5
                  size: 261
                  buffer_allocation_index: 3
                }
              }
            }
            thunks {
              thunk_info {
                profile_annotation: "profile_annotation"
                execution_stream_id: 123
              }
              copy_thunk {
                source_buffer { offset: 6 size: 262 buffer_allocation_index: 3 }
                destination_buffer {
                  offset: 7
                  size: 263
                  buffer_allocation_index: 4
                }
              }
            }
          }
          branch_index_is_bool: true
        }
      )pb",
      &proto));

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/3, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/4, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/5, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> athunk,
                          DeserializeThunkProto(proto, buffer_allocations));
  auto* thunk = dynamic_cast<ConditionalThunk*>(athunk.get());
  ASSERT_NE(thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
