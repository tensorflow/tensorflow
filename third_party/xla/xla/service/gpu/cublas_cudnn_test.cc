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

#include "xla/service/gpu/cublas_cudnn.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/service/gpu/cublas_cudnn.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(CublasCudnnTest, CudnnConvKindToProto) {
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kForward),
            CUDNN_CONV_KIND_FORWARD);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kBackwardInput),
            CUDNN_CONV_KIND_BACKWARD_INPUT);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kBackwardFilter),
            CUDNN_CONV_KIND_BACKWARD_FILTER);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kForwardActivation),
            CUDNN_CONV_KIND_FORWARD_ACTIVATION);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kForwardGraph),
            CUDNN_CONV_KIND_FORWARD_GRAPH);
}

TEST(CublasCudnnTest, CudnnConvKindFromProtoKnownValue) {
  EXPECT_THAT(CudnnConvKindFromProto(CUDNN_CONV_KIND_FORWARD),
              IsOkAndHolds(CudnnConvKind::kForward));
  EXPECT_THAT(CudnnConvKindFromProto(CUDNN_CONV_KIND_BACKWARD_INPUT),
              IsOkAndHolds(CudnnConvKind::kBackwardInput));
  EXPECT_THAT(CudnnConvKindFromProto(CUDNN_CONV_KIND_BACKWARD_FILTER),
              IsOkAndHolds(CudnnConvKind::kBackwardFilter));
  EXPECT_THAT(CudnnConvKindFromProto(CUDNN_CONV_KIND_FORWARD_GRAPH),
              IsOkAndHolds(CudnnConvKind::kForwardGraph));
}

TEST(CublasCudnnTest, CudnnConvKindFromProtoUnknownValue) {
  EXPECT_THAT(CudnnConvKindFromProto(CUDNN_CONV_KIND_UNSPECIFIED),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
