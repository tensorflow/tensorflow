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

#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.h"

#include <cstdint>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace stream_executor::cuda {
namespace {

using ::absl_testing::StatusIs;
using ::tsl::proto_testing::EqualsProto;

CublasLtMatmulAlgorithm SampleAlgorithm() {
  CublasLtMatmulAlgorithm algo;
  algo.algo_id = 23;
  algo.tile_id = 15;
  algo.split_k = -1;
  algo.reduction_scheme = 2;
  algo.cta_swizzling = 1;
  algo.custom_option = 3;
  algo.stages_id = 19;
  algo.inner_shape_id = 4;
  algo.cluster_shape_id = 1;
  algo.numerical_impl_flags = 0x101;
  return algo;
}

TEST(CublasLtMatmulAlgorithmTest, ProtoConversionAndEquality) {
  CublasLtMatmulAlgorithm original = SampleAlgorithm();

  CublasLtMatmulAlgorithmProto proto = original.ToProto();
  ASSERT_OK_AND_ASSIGN(CublasLtMatmulAlgorithm round_tripped,
                       CublasLtMatmulAlgorithm::FromProto(proto));
  EXPECT_EQ(original, round_tripped);
  EXPECT_THAT(round_tripped.ToProto(), EqualsProto(proto));

  CublasLtMatmulAlgorithm modified = round_tripped;
  modified.tile_id = 16;
  EXPECT_NE(original, modified);
}

TEST(CublasLtMatmulAlgorithmTest, NegativeUnsignedAttributesRejected) {
  const CublasLtMatmulAlgorithmProto valid_proto = SampleAlgorithm().ToProto();

  CublasLtMatmulAlgorithmProto bad_tile = valid_proto;
  bad_tile.set_tile_id(-1);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_tile),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CublasLtMatmulAlgorithmProto bad_reduction = valid_proto;
  bad_reduction.set_reduction_scheme(-1);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_reduction),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CublasLtMatmulAlgorithmProto bad_swizzling = valid_proto;
  bad_swizzling.set_cta_swizzling(-1);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_swizzling),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CublasLtMatmulAlgorithmProto bad_custom_option = valid_proto;
  bad_custom_option.set_custom_option(-1);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_custom_option),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CublasLtMatmulAlgorithmProto bad_stages = valid_proto;
  bad_stages.set_stages_id(-1);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_stages),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CublasLtMatmulAlgorithmProto bad_inner_shape = valid_proto;
  bad_inner_shape.set_inner_shape_id(-1);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_inner_shape),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CublasLtMatmulAlgorithmProto bad_cluster_shape = valid_proto;
  bad_cluster_shape.set_cluster_shape_id(-1);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_cluster_shape),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CublasLtMatmulAlgorithmProto bad_flags = valid_proto;
  bad_flags.set_numerical_impl_flags(-1);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_flags),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CublasLtMatmulAlgorithmTest, ShapeIdOverflowRejected) {
  const CublasLtMatmulAlgorithmProto valid_proto = SampleAlgorithm().ToProto();
  constexpr int32_t kOverflow =
      static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1;

  CublasLtMatmulAlgorithmProto bad_inner_shape = valid_proto;
  bad_inner_shape.set_inner_shape_id(kOverflow);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_inner_shape),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CublasLtMatmulAlgorithmProto bad_cluster_shape = valid_proto;
  bad_cluster_shape.set_cluster_shape_id(kOverflow);
  EXPECT_THAT(CublasLtMatmulAlgorithm::FromProto(bad_cluster_shape),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace stream_executor::cuda
