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

#include "xla/backends/gpu/runtime/convolution_thunk.h"

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

TEST(ConvolutionThunkTest, ProtoRoundTrip) {
  auto proto = ParseTextProtoOrDie<ThunkProto>(R"pb(
    thunk_info {
      profile_annotation: "conv_thunk_profile"
      execution_stream_id: 0
    }
    convolution_thunk {
      conv_descriptor {
        kind: FORWARD
        backend_config {
          conv_result_scale: 1
          activation_mode: 0
          side_input_scale: 0
          leakyrelu_alpha: 0
        }
        window {
          dimensions {
            size: 1
            stride: 1
            padding_low: 0
            padding_high: 0
            window_dilation: 1
            base_dilation: 1
            window_reversal: false
          }
          dimensions {
            size: 1
            stride: 1
            padding_low: 0
            padding_high: 0
            window_dilation: 1
            base_dilation: 1
            window_reversal: false
          }
        }
        operand0_shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
        operand1_shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
        result_shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
        scratch_size: 1024
        dnums {
          input_batch_dimension: 0
          input_feature_dimension: 1
          input_spatial_dimensions: [ 2, 3 ]
          kernel_input_feature_dimension: 1
          kernel_output_feature_dimension: 0
          kernel_spatial_dimensions: [ 2, 3 ]
          output_batch_dimension: 0
          output_feature_dimension: 1
          output_spatial_dimensions: [ 2, 3 ]
        }
      }
      operand_buffers {
        slice { offset: 0 size: 4 buffer_allocation_index: 0 }
        shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
      }
      operand_buffers {
        slice { offset: 0 size: 4 buffer_allocation_index: 1 }
        shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
      }
      result_buffers {
        slice { offset: 0 size: 4 buffer_allocation_index: 2 }
        shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
      }
      scratch_buffer { offset: 0 size: 1024 buffer_allocation_index: 3 }
    }
  )pb");

  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/4, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/1, /*size=*/4, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/2, /*size=*/4, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/3, /*size=*/1024, /*color=*/0);

  TF_ASSERT_OK_AND_ASSIGN(Thunk::ThunkInfo thunk_info,
                          Thunk::ThunkInfo::FromProto(proto.thunk_info()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ConvolutionThunk> thunk,
      ConvolutionThunk::FromProto(thunk_info, proto.convolution_thunk(),
                                  buffer_allocations));
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
