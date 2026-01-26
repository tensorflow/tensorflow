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

#include "xla/backends/gpu/runtime/cub_sort_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/ffi/ffi.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/platform_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(CubSortThunkTest, ProtoRoundTrip) {
  TF_ASSERT_OK_AND_ASSIGN(absl::string_view name,
                          PlatformUtil::CanonicalPlatformName("gpu"));
  auto proto = ParseTextProtoOrDie<ThunkProto>(R"pb(
    thunk_info {
      profile_annotation: "cub_sort_thunk_profile"
      execution_stream_id: 1
    }
    cub_sort_thunk {
      operands {
        slice { offset: 0 size: 4 buffer_allocation_index: 0 }
        shape { element_type: F32 dimensions: 1 is_dynamic_dimension: false }
      }
      results {
        slice { offset: 0 size: 4 buffer_allocation_index: 1 }
        shape { element_type: F32 dimensions: 1 is_dynamic_dimension: false }
      }
      scratch { offset: 0 size: 1024 buffer_allocation_index: 2 }
      descending: true
      batch_size: 1
    }
  )pb");

  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/4, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/1, /*size=*/4, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/2, /*size=*/1024, /*color=*/0);

  TF_ASSERT_OK_AND_ASSIGN(Thunk::ThunkInfo thunk_info,
                          Thunk::ThunkInfo::FromProto(proto.thunk_info()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CubSortThunk> thunk,
      CubSortThunk::FromProto(thunk_info, proto.cub_sort_thunk(),
                              buffer_allocations, name));
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
