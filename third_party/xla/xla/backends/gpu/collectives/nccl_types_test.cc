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

#include "xla/backends/gpu/collectives/nccl_types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "third_party/nccl/nccl.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = stream_executor;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(NcclTypesTest, ToNcclCountScalar) {
  EXPECT_EQ(ToNcclCount(F32, 10), 10);
  EXPECT_EQ(ToNcclCount(S32, 5), 5);
}

TEST(NcclTypesTest, ToNcclCountComplex) {
  EXPECT_EQ(ToNcclCount(C64, 10), 20);
  EXPECT_EQ(ToNcclCount(C128, 5), 10);
}

TEST(NcclTypesTest, ToNcclDataTypeBasicTypes) {
  se::CudaComputeCapability cc(8, 0);
  EXPECT_THAT(ToNcclDataType(F32, false, cc), IsOkAndHolds(ncclFloat32));
  EXPECT_THAT(ToNcclDataType(F64, false, cc), IsOkAndHolds(ncclFloat64));
  EXPECT_THAT(ToNcclDataType(F16, false, cc), IsOkAndHolds(ncclFloat16));
  EXPECT_THAT(ToNcclDataType(BF16, false, cc), IsOkAndHolds(ncclBfloat16));
  EXPECT_THAT(ToNcclDataType(S32, false, cc), IsOkAndHolds(ncclInt32));
  EXPECT_THAT(ToNcclDataType(U32, false, cc), IsOkAndHolds(ncclUint32));
  EXPECT_THAT(ToNcclDataType(S64, false, cc), IsOkAndHolds(ncclInt64));
  EXPECT_THAT(ToNcclDataType(U64, false, cc), IsOkAndHolds(ncclUint64));
  EXPECT_THAT(ToNcclDataType(S8, false, cc), IsOkAndHolds(ncclInt8));
  EXPECT_THAT(ToNcclDataType(U8, false, cc), IsOkAndHolds(ncclUint8));
  EXPECT_THAT(ToNcclDataType(PRED, false, cc), IsOkAndHolds(ncclUint8));
}

TEST(NcclTypesTest, ToNcclDataTypeComplexUsesReal) {
  se::CudaComputeCapability cc(8, 0);
  EXPECT_THAT(ToNcclDataType(C64, false, cc), IsOkAndHolds(ncclFloat32));
  EXPECT_THAT(ToNcclDataType(C128, false, cc), IsOkAndHolds(ncclFloat64));
}

TEST(NcclTypesTest, ToNcclDataType16BitIntNonReduction) {
  se::CudaComputeCapability cc(8, 0);
  EXPECT_THAT(ToNcclDataType(S16, false, cc), IsOkAndHolds(ncclFloat16));
  EXPECT_THAT(ToNcclDataType(U16, false, cc), IsOkAndHolds(ncclFloat16));
}

TEST(NcclTypesTest, ToNcclDataType16BitIntReductionFails) {
  se::CudaComputeCapability cc(8, 0);
  EXPECT_THAT(ToNcclDataType(S16, true, cc),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ToNcclDataType(U16, true, cc),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NcclTypesTest, ToNcclDataTypeFp8PreHopper) {
  se::CudaComputeCapability cc(8, 0);
  EXPECT_THAT(ToNcclDataType(F8E5M2, false, cc), IsOkAndHolds(ncclInt8));
  EXPECT_THAT(ToNcclDataType(F8E4M3FN, false, cc), IsOkAndHolds(ncclInt8));
}

TEST(NcclTypesTest, ToNcclDataTypeFp8Hopper) {
  se::CudaComputeCapability cc(9, 0);
  EXPECT_THAT(ToNcclDataType(F8E5M2, false, cc), IsOkAndHolds(ncclFloat8e5m2));
  EXPECT_THAT(ToNcclDataType(F8E4M3FN, false, cc),
              IsOkAndHolds(ncclFloat8e4m3));
}

TEST(NcclTypesTest, ToNcclDataTypeUnsupported) {
  se::CudaComputeCapability cc(8, 0);
  EXPECT_THAT(ToNcclDataType(TOKEN, false, cc),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NcclTypesTest, ToNcclReduction) {
  EXPECT_EQ(ToNcclReduction(ReductionKind::SUM), ncclSum);
  EXPECT_EQ(ToNcclReduction(ReductionKind::PRODUCT), ncclProd);
  EXPECT_EQ(ToNcclReduction(ReductionKind::MIN), ncclMin);
  EXPECT_EQ(ToNcclReduction(ReductionKind::MAX), ncclMax);
}

}  // namespace
}  // namespace xla::gpu
