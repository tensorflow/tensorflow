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

#include "xla/service/gpu/gpu_conv_runner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_conv_runner.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(GpuConvDescriptorTest, ProtoRoundTrip) {
  auto proto = ParseTextProtoOrDie<GpuConvDescriptorProto>(R"pb(
    kind: FORWARD
    backend_config {
      algorithm { algo_id: 1 }
      conv_result_scale: 1.0
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
    }
    dnums {
      input_batch_dimension: 0
      input_feature_dimension: 1
      input_spatial_dimensions: 2
      kernel_input_feature_dimension: 1
      kernel_output_feature_dimension: 0
      kernel_spatial_dimensions: [ 2 ]
      output_batch_dimension: 0
      output_feature_dimension: 1
      output_spatial_dimensions: 2
    }
    feature_group_count: 1
  )pb");

  TF_ASSERT_OK_AND_ASSIGN(GpuConvDescriptor desc,
                          GpuConvDescriptor::FromProto(proto));

  EXPECT_THAT(desc.ToProto(), EqualsProto(proto));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
