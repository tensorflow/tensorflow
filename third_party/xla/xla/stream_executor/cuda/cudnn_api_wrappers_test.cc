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

#include "xla/stream_executor/cuda/cudnn_api_wrappers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "third_party/gpus/cudnn/cudnn_graph.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace cuda {
namespace {
using absl_testing::IsOk;
using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using testing::Ge;
using testing::HasSubstr;

TEST(CudnnApiWrappersTest, GetCudnnProperty) {
  EXPECT_THAT(GetCudnnProperty(CudnnProperty::kMajorVersion),
              IsOkAndHolds(Ge(8)));
}

TEST(CudnnApiWrappersTest, GetLoadedCudnnVersion) {
  // This test makes sure we can determine the version of cuDNN without an
  // accelerator present and without initializing cuDNN.
  TF_ASSERT_OK_AND_ASSIGN(SemanticVersion version,
                          stream_executor::cuda::GetLoadedCudnnVersion());

  // As the time of writing this test, the oldest supported version of cuDNN
  // is 8.9.4. So we expect the version to be at least this.
  EXPECT_GE(version, SemanticVersion(8, 9, 4));

  // We don't link in the CUDA platform intentionally, to make sure the above
  // version query works without any additional dependencies. This assertion
  // ensures that the dependency on the CUDA platform will not be accidentally
  // introduced in the future.
  EXPECT_THAT(stream_executor::PlatformManager::PlatformWithId(kCudaPlatformId),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(CudnnApiWrappersTest, ToStatus) {
  EXPECT_THAT(ToStatus(CUDNN_STATUS_SUCCESS), IsOk());
  EXPECT_THAT(ToStatus(CUDNN_STATUS_NOT_INITIALIZED),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("CUDNN_STATUS_NOT_INITIALIZED")));
  EXPECT_THAT(ToStatus(CUDNN_STATUS_NOT_SUPPORTED, "some additional detail"),
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     HasSubstr("some additional detail")));
}

}  // namespace
}  // namespace cuda
}  // namespace stream_executor
