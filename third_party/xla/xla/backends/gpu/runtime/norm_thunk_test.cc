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

#include "xla/backends/gpu/runtime/norm_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(NormThunkTest, ProtoRoundTrip) {
  auto proto = ParseTextProtoOrDie<ThunkProto>(R"pb(
    thunk_info {
      profile_annotation: "norm_thunk_profile"
      execution_stream_id: 0
    }
    norm_thunk {
      norm_descriptor {
        backend_config {
          epsilon: 0.001
          kind: LAYER_FWD_INFER
          algorithm { algo_id: 0 is_cudnn_frontend: true }
        }
        x_shape {
          element_type: F32
          dimensions: [ 2, 3 ]
          layout {
            minor_to_major: [ 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false ]
        }
        scale_shape {
          element_type: F32
          dimensions: [ 3 ]
          layout {
            minor_to_major: [ 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: false
        }
        bias_shape {
          element_type: F32
          dimensions: [ 3 ]
          layout {
            minor_to_major: [ 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: false
        }
        y_or_dx_shape {
          element_type: F32
          dimensions: [ 2, 3 ]
          layout {
            minor_to_major: [ 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false ]
        }
        scratch_shape: {
          element_type: S8
          dimensions: 1024
          is_dynamic_dimension: false
        }
      }
      x { offset: 0 size: 24 buffer_allocation_index: 0 }
      scale { offset: 0 size: 12 buffer_allocation_index: 1 }
      y_or_dx { offset: 0 size: 24 buffer_allocation_index: 2 }
      bias { offset: 0 size: 12 buffer_allocation_index: 3 }
      scratch { offset: 0 size: 1024 buffer_allocation_index: 4 }
    }
  )pb");

  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/24, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/1, /*size=*/12, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/2, /*size=*/24, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/3, /*size=*/12, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/4, /*size=*/1024, /*color=*/0);

  TF_ASSERT_OK_AND_ASSIGN(Thunk::ThunkInfo thunk_info,
                          Thunk::ThunkInfo::FromProto(proto.thunk_info()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<NormThunk> thunk,
      NormThunk::FromProto(thunk_info, proto.norm_thunk(), buffer_allocations));
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
