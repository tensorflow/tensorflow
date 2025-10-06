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

#include "xla/backends/gpu/runtime/outfeed_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(OutfeedThunkTest, ProtoRoundTrip) {
  auto thunk_proto = ParseTextProtoOrDie<ThunkProto>(R"pb(
    thunk_info { profile_annotation: "outfeed" execution_stream_id: 2 }
    outfeed_thunk {
      source_slices {
        slice { offset: 0 size: 4 buffer_allocation_index: 0 }
        shape { dimensions: 8 element_type: F32 is_dynamic_dimension: false }
      }
    }
  )pb");
  TF_ASSERT_OK_AND_ASSIGN(
      Thunk::ThunkInfo thunk_info,
      Thunk::ThunkInfo::FromProto(thunk_proto.thunk_info()));
  std::vector<BufferAllocation> source_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OutfeedThunk> thunk,
      OutfeedThunk::FromProto(thunk_info, thunk_proto.outfeed_thunk(),
                              source_allocations));

  EXPECT_THAT(thunk->ToProto(), IsOkAndHolds(EqualsProto(thunk_proto)));
}

}  // namespace
}  // namespace xla::gpu
