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

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "google/protobuf/unknown_field_set.h"
#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.h"
#include "xla/stream_executor/gpu/blas_lt_matmul_algorithm.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace stream_executor::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
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
  ASSERT_OK_AND_ASSIGN(std::optional<BlasLtMatmulAlgorithm> round_tripped,
                       BlasLtMatmulAlgorithm::FromProto(proto));
  ASSERT_TRUE(round_tripped.has_value());
  EXPECT_EQ(original, *round_tripped);
  EXPECT_THAT(round_tripped->ToProto(), EqualsProto(proto));

  BlasLtMatmulAlgorithm modified = *round_tripped;
  modified.library_version = 120900;
  EXPECT_NE(original, modified);
}

TEST(BlasLtMatmulAlgorithmTest, EmptyAlgorithmOneofIsNoAlgorithm) {
  BlasLtMatmulAlgorithmProto empty_proto;
  empty_proto.set_library_version(120800);
  EXPECT_THAT(BlasLtMatmulAlgorithm::FromProto(empty_proto),
              IsOkAndHolds(std::nullopt));
}

// An algorithm kind added to the oneof by a newer XLA reaches this reader as an
// unknown field. It must not fail the load: the caller falls back to the
// heuristic algorithm index or, under require-exact, reports the missing
// algorithm.
TEST(BlasLtMatmulAlgorithmTest, UnknownAlgorithmOneofIsNoAlgorithm) {
  BlasLtMatmulAlgorithmProto writer_proto;
  writer_proto.set_library_version(120800);
  writer_proto.GetReflection()
      ->MutableUnknownFields(&writer_proto)
      ->AddLengthDelimited(/*number=*/15, /*value=*/"\x08\x05");

  BlasLtMatmulAlgorithmProto reader_proto;
  ASSERT_TRUE(reader_proto.ParseFromString(writer_proto.SerializeAsString()));
  ASSERT_EQ(reader_proto.algorithm_case(),
            BlasLtMatmulAlgorithmProto::ALGORITHM_NOT_SET);
  EXPECT_THAT(BlasLtMatmulAlgorithm::FromProto(reader_proto),
              IsOkAndHolds(std::nullopt));
}

}  // namespace
}  // namespace stream_executor::gpu
