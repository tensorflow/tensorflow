/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/cuda/cudnn_version.h"

#include "testing/base/public/gunit.h"
#include "tensorflow/core/platform/test.h"

namespace perftools {
namespace gputools {
namespace cuda {
namespace {

TEST(CuDNNVersion, ToString) {
  CudnnVersion version(7, 0, 12);
  EXPECT_EQ(version.ToString(), "7.0.12");
}

TEST(IsSourceCompatibleWithCudnnLibraryTest, Basic) {
  // Returns true if both major and minor versions are matching and even if the
  // patch versions are not matching.
  EXPECT_TRUE(IsSourceCompatibleWithCudnnLibrary(
      /*source_version=*/CudnnVersion(7, 0, 12),
      /*loaded_version=*/CudnnVersion(7, 0, 14)));
  EXPECT_TRUE(IsSourceCompatibleWithCudnnLibrary(
      /*source_version=*/CudnnVersion(6, 1, 14),
      /*loaded_version=*/CudnnVersion(6, 1, 00)));

  // Returns false if major versions are not matching as they are neither
  // forward or backward compatible.
  EXPECT_FALSE(IsSourceCompatibleWithCudnnLibrary(
      /*source_version=*/CudnnVersion(7, 0, 12),
      /*loaded_version=*/CudnnVersion(6, 1, 14)));
  EXPECT_FALSE(IsSourceCompatibleWithCudnnLibrary(
      /*source_version=*/CudnnVersion(8, 1, 15),
      /*loaded_version=*/CudnnVersion(7, 0, 14)));

  // Returns true if the loaded version is equal or higher because minor version
  // are backward compatible with CuDNN version 7.
  EXPECT_TRUE(IsSourceCompatibleWithCudnnLibrary(
      /*source_version=*/CudnnVersion(7, 0, 14),
      /*loaded_version=*/CudnnVersion(7, 1, 14)));
  EXPECT_TRUE(IsSourceCompatibleWithCudnnLibrary(
      /*source_version=*/CudnnVersion(7, 0, 14),
      /*loaded_version=*/CudnnVersion(7, 1, 15)));
  EXPECT_FALSE(IsSourceCompatibleWithCudnnLibrary(
      /*source_version=*/CudnnVersion(7, 1, 15),
      /*loaded_version=*/CudnnVersion(7, 0, 14)));

  // Returns false if minor versions are not matching for version 6. Before
  // version 7, minor versions are also neither forward or backward compatible.
  EXPECT_FALSE(IsSourceCompatibleWithCudnnLibrary(
      /*source_version=*/CudnnVersion(6, 0, 14),
      /*loaded_version=*/CudnnVersion(6, 1, 15)));
  EXPECT_FALSE(IsSourceCompatibleWithCudnnLibrary(
      /*source_version=*/CudnnVersion(6, 1, 14),
      /*loaded_version=*/CudnnVersion(6, 0, 14)));
}

}  // namespace
}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
