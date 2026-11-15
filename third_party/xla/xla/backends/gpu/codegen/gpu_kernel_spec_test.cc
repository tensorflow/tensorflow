/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/codegen/gpu_kernel_spec.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(GpuKernelSpecTest, ProtoRoundTrip) {
  GpuKernelSpecProto proto = ParseTextProtoOrDie<GpuKernelSpecProto>(
      R"pb(
        name: "test_kernel"
        launch_dimensions {
          block_counts { coordinates { x: 32 y: 31 z: 30 } }
          thread_counts_per_block { coordinates { x: 256 y: 255 z: 254 } }
        }
        kernel_arguments {
          args {
            shape {
              dimensions: 1
              element_type: F32
              is_dynamic_dimension: false
            }
            slice { offset: 0 size: 4 buffer_allocation_index: 0 }
            aliased: true
            alignment: 16
            written: true
            invariant: false
            slice_index: 0
          }
          positions {
            operands: 0
            results {
              shape_index {}
              position: 0
            }
          }
        }
        scratch_bytes: 1024
        use_pdl: true
        tma_metadata {
          arg_index_to_tma_info {
            key: 0
            value {
              element_size: 4
              global_dims: 1024
              global_dims: 1024
              global_strides: 1024
              box_dims: 128
              box_dims: 128
              element_strides: 1
              element_strides: 1
              interleave: INTERLEAVE_NONE
              swizzle: SWIZZLE_NONE
              l2_promotion: L2_PROMOTION_NONE
              float_oob_fill: FLOAT_OOB_FILL_NONE
            }
          }
        }
        cluster_dim { coordinates { x: 8 y: 7 z: 6 } }
      )pb");
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(GpuKernelSpec spec,
                       GpuKernelSpec::FromProto(proto, buffer_allocations));

  ASSERT_OK_AND_ASSIGN(GpuKernelSpecProto round_trip_proto, spec.ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
