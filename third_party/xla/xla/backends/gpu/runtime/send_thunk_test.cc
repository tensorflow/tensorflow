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

#include "xla/backends/gpu/runtime/send_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

TEST(CollectiveThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        send_thunk {
          buffer {
            element_count: 64
            source_buffer {
              slice { offset: 128 size: 256 buffer_allocation_index: 0 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
              }
            }
            destination_buffer {
              slice { offset: 0 size: 256 buffer_allocation_index: 1 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
              }
            }
          }
          collective_config {}
          source_target_pairs: { source: 1 target: 2 }
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SendThunk> thunk,
      SendThunk::FromProto(thunk_info, proto.send_thunk(), buffer_allocations));

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
