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

#include "xla/service/gpu/gpu_norm_runner.h"

#include "xla/service/gpu/gpu_norm_runner.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(GpuNormRunnerTest, GpuNormDescriptorToFromProto) {
  auto descriptor_proto = ParseTextProtoOrDie<GpuNormDescriptorProto>(R"pb(
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
  )pb");

  TF_ASSERT_OK_AND_ASSIGN(GpuNormDescriptor descriptor,
                          GpuNormDescriptor::FromProto(descriptor_proto));
  EXPECT_THAT(descriptor.ToProto(), EqualsProto(descriptor_proto));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
