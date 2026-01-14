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

#include "xla/backends/gpu/runtime/convolution_reorder_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(ConvolutionReorderThunkTest, ProtoRoundTrip) {
  auto proto = ParseTextProtoOrDie<ThunkProto>(R"pb(
    thunk_info { profile_annotation: "test" execution_stream_id: 0 }
    convolution_reorder_thunk {
      filter_input {
        slice { buffer_allocation_index: 0 offset: 0 size: 1024 }
        shape {}
      }
      filter_output {
        slice { buffer_allocation_index: 1 offset: 0 size: 512 }
        shape {
          element_type: F32
          dimensions: 1
          dimensions: 2
          dimensions: 3
          dimensions: 4
          dimensions: 32
          is_dynamic_dimension: false
          is_dynamic_dimension: false
          is_dynamic_dimension: false
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
      }
      biases {
        bias_input {
          slice { buffer_allocation_index: 2 offset: 0 size: 256 }
          shape {}
        }
        bias_output {
          slice { buffer_allocation_index: 3 offset: 0 size: 128 }
          shape {}
        }
      }
    }
  )pb");

  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/1024, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/1, /*size=*/512, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/2, /*size=*/256, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/3, /*size=*/128, /*color=*/0);

  TF_ASSERT_OK_AND_ASSIGN(Thunk::ThunkInfo thunk_info,
                          Thunk::ThunkInfo::FromProto(proto.thunk_info()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ConvolutionReorderThunk> thunk,
      ConvolutionReorderThunk::FromProto(
          thunk_info, proto.convolution_reorder_thunk(), buffer_allocations));
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
