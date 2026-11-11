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

#include "xla/stream_executor/gpu/blas_lt_matmul_algorithm.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.h"
#include "xla/stream_executor/gpu/blas_lt_matmul_algorithm.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace stream_executor::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::tsl::proto_testing::EqualsProto;

TEST(BlasLtMatmulAlgorithmTest, ProtoConversionAndEquality) {
  cuda::CublasLtMatmulAlgorithm cublaslt;
  cublaslt.algo_id = 23;
  cublaslt.tile_id = 15;
  cublaslt.split_k = 1;
  cublaslt.reduction_scheme = 2;
  cublaslt.cta_swizzling = 0;
  cublaslt.custom_option = 3;
  cublaslt.stages_id = 19;
  cublaslt.inner_shape_id = 4;
  cublaslt.cluster_shape_id = 1;
  cublaslt.numerical_impl_flags = 0x101;

  BlasLtMatmulAlgorithm original{
      /*library_version=*/120800,
      /*algorithm=*/cublaslt,
  };

  BlasLtMatmulAlgorithmProto proto = original.ToProto();
  ASSERT_OK_AND_ASSIGN(BlasLtMatmulAlgorithm round_tripped,
                       BlasLtMatmulAlgorithm::FromProto(proto));
  EXPECT_EQ(original, round_tripped);
  EXPECT_THAT(round_tripped.ToProto(), EqualsProto(proto));

  BlasLtMatmulAlgorithm modified = round_tripped;
  modified.library_version = 120900;
  EXPECT_NE(original, modified);
}

TEST(BlasLtMatmulAlgorithmTest, EmptyAlgorithmOneofRejected) {
  BlasLtMatmulAlgorithmProto empty_proto;
  empty_proto.set_library_version(120800);
  EXPECT_THAT(BlasLtMatmulAlgorithm::FromProto(empty_proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace stream_executor::gpu
