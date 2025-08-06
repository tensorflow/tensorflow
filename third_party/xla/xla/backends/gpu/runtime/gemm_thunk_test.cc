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

#include "xla/backends/gpu/runtime/gemm_thunk.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/protobuf.h"

using ::tsl::proto_testing::EqualsProto;

namespace xla::gpu {
namespace {

TEST(GemmThunkTest, ProtoRoundTrip) {
  constexpr absl::string_view kProtoText = R"pb(
    thunk_info {
      profile_annotation: "gemm_thunk_test_profile"
      execution_stream_id: 0
    }
    gemm_thunk {
      gemm_config {
        lhs_layout {
          order: ORDER_ROW_MAJOR
          num_rows: 2
          num_cols: 3
          batch_size: 1
          leading_dim_stride: 3
          transpose: BLAS_NO_TRANSPOSE
          dtype: F32
        }
        rhs_layout {
          order: ORDER_ROW_MAJOR
          num_rows: 3
          num_cols: 4
          batch_size: 1
          leading_dim_stride: 4
          transpose: BLAS_NO_TRANSPOSE
          dtype: F32
        }
        c_layout {
          order: ORDER_ROW_MAJOR
          num_rows: 2
          num_cols: 4
          batch_size: 1
          leading_dim_stride: 4
          transpose: BLAS_NO_TRANSPOSE
          dtype: F32
        }
        output_layout {
          order: ORDER_ROW_MAJOR
          num_rows: 2
          num_cols: 4
          batch_size: 1
          leading_dim_stride: 4
          transpose: BLAS_NO_TRANSPOSE
          dtype: F32
        }
        alpha_real: 1.0
        alpha_imag: 0.0
        beta: 0.5
        precision_algorithm: ALG_UNSET
        compute_type: BLAS_COMPUTATION_TYPE_F32
      }
      lhs_buffer { offset: 10 size: 24 buffer_allocation_index: 0 }
      rhs_buffer { offset: 20 size: 48 buffer_allocation_index: 1 }
      output_buffer { offset: 30 size: 32 buffer_allocation_index: 2 }
      workspace { offset: 40 size: 1024 buffer_allocation_index: 3 }
      deterministic: true
    }
  )pb";

  ThunkProto original_thunk_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      std::string(kProtoText),  // NOLINT -- openxla protobuf version requires
                                // it to be a string
      &original_thunk_proto));

  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/100, /*color=*/10);
  buffer_allocations.emplace_back(/*index=*/1, /*size=*/200, /*color=*/11);
  buffer_allocations.emplace_back(/*index=*/2, /*size=*/150, /*color=*/12);
  buffer_allocations.emplace_back(/*index=*/3, /*size=*/2048, /*color=*/13);

  const GemmThunkProto& original_gemm_thunk_proto =
      original_thunk_proto.gemm_thunk();

  TF_ASSERT_OK_AND_ASSIGN(
      Thunk::ThunkInfo thunk_info_from_proto,
      Thunk::ThunkInfo::FromProto(original_thunk_proto.thunk_info()));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GemmThunk> gemm_thunk,
      GemmThunk::FromProto(thunk_info_from_proto, original_gemm_thunk_proto,
                           buffer_allocations));
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_tripped_thunk_proto,
                          gemm_thunk->ToProto());
  EXPECT_THAT(round_tripped_thunk_proto, EqualsProto(original_thunk_proto));
}
}  // namespace
}  // namespace xla::gpu
