/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/platform_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(PlatformUtilTest, GetPlatformIdFromCanonicalName) {
  EXPECT_THAT(PlatformUtil::GetPlatformIdFromCanonicalName("host"),
              IsOkAndHolds(stream_executor::host::kHostPlatformId));
  EXPECT_THAT(PlatformUtil::GetPlatformIdFromCanonicalName("Host"),
              IsOkAndHolds(stream_executor::host::kHostPlatformId));

  EXPECT_THAT(PlatformUtil::GetPlatformIdFromCanonicalName("cuda"),
              IsOkAndHolds(stream_executor::cuda::kCudaPlatformId));
  EXPECT_THAT(PlatformUtil::GetPlatformIdFromCanonicalName("CUDA"),
              IsOkAndHolds(stream_executor::cuda::kCudaPlatformId));

  EXPECT_THAT(PlatformUtil::GetPlatformIdFromCanonicalName("rocm"),
              IsOkAndHolds(stream_executor::rocm::kROCmPlatformId));
  EXPECT_THAT(PlatformUtil::GetPlatformIdFromCanonicalName("ROCM"),
              IsOkAndHolds(stream_executor::rocm::kROCmPlatformId));

  EXPECT_THAT(PlatformUtil::GetPlatformIdFromCanonicalName("sycl"),
              IsOkAndHolds(stream_executor::sycl::kSyclPlatformId));
  EXPECT_THAT(PlatformUtil::GetPlatformIdFromCanonicalName("SYCL"),
              IsOkAndHolds(stream_executor::sycl::kSyclPlatformId));

  EXPECT_THAT(PlatformUtil::GetPlatformIdFromCanonicalName("unknown"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
