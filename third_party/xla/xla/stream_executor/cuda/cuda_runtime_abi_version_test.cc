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

#include "xla/stream_executor/cuda/cuda_runtime_abi_version.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/stream_executor/cuda/cuda_runtime_abi_version.pb.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::cuda {
namespace {

TEST(CudaRuntimeAbiVersionTest, Construction) {
  CudaRuntimeAbiVersion runtime_abi_version(SemanticVersion(12, 0, 0),
                                            SemanticVersion(9, 0, 0),
                                            SemanticVersion(2, 0, 0));
  EXPECT_EQ(runtime_abi_version.cuda_toolkit_version(),
            SemanticVersion(12, 0, 0));
  EXPECT_EQ(runtime_abi_version.cudnn_version(), SemanticVersion(9, 0, 0));
  EXPECT_EQ(runtime_abi_version.cub_version(), SemanticVersion(2, 0, 0));
}

TEST(CudaRuntimeAbiVersionTest, ToProto) {
  CudaRuntimeAbiVersion runtime_abi_version(SemanticVersion(12, 0, 0),
                                            SemanticVersion(9, 0, 0),
                                            SemanticVersion(2, 0, 0));
  ASSERT_OK_AND_ASSIGN(RuntimeAbiVersionProto proto,
                       runtime_abi_version.ToProto());
  EXPECT_EQ(proto.platform_name(), "CUDA");
  CudaRuntimeAbiVersionProto cuda_proto;
  ASSERT_TRUE(cuda_proto.ParseFromString(proto.platform_specific_version()));
  EXPECT_EQ(cuda_proto.cuda_toolkit_version(), "12.0.0");
  EXPECT_EQ(cuda_proto.cudnn_version(), "9.0.0");
  EXPECT_EQ(cuda_proto.cub_version(), "2.0.0");
}

TEST(CudaRuntimeAbiVersionTest, FromProto) {
  CudaRuntimeAbiVersionProto proto;
  proto.set_cuda_toolkit_version("12.0.0");
  proto.set_cudnn_version("9.0.0");
  proto.set_cub_version("2.0.0");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CudaRuntimeAbiVersion> runtime_abi_version,
      CudaRuntimeAbiVersion::FromProto(proto));
  EXPECT_EQ(runtime_abi_version->cuda_toolkit_version(),
            SemanticVersion(12, 0, 0));
  EXPECT_EQ(runtime_abi_version->cudnn_version(), SemanticVersion(9, 0, 0));
  EXPECT_EQ(runtime_abi_version->cub_version(), SemanticVersion(2, 0, 0));
}

TEST(CudaRuntimeAbiVersionTest, IsCompatibleWith) {
  CudaRuntimeAbiVersion current_runtime_abi_version(SemanticVersion(12, 0, 0),
                                                    SemanticVersion(9, 0, 0),
                                                    SemanticVersion(2, 0, 0));

  ASSERT_OK_AND_ASSIGN(ExecutableAbiVersion current_executable_abi_version, [] {
    ExecutableAbiVersionProto proto;
    proto.set_platform_name("CUDA");
    proto.mutable_cuda_platform_version()->set_cuda_toolkit_version("12.0.0");
    proto.mutable_cuda_platform_version()->set_cudnn_version("9.0.0");
    proto.mutable_cuda_platform_version()->set_cub_version("2.0.0");
    return ExecutableAbiVersion::FromProto(proto);
  }());

  ASSERT_OK_AND_ASSIGN(ExecutableAbiVersion older_executable_abi_version, [] {
    ExecutableAbiVersionProto proto;
    proto.set_platform_name("CUDA");
    proto.mutable_cuda_platform_version()->set_cuda_toolkit_version("11.0.0");
    proto.mutable_cuda_platform_version()->set_cudnn_version("8.0.0");
    proto.mutable_cuda_platform_version()->set_cub_version("1.0.0");
    return ExecutableAbiVersion::FromProto(proto);
  }());

  ASSERT_OK_AND_ASSIGN(ExecutableAbiVersion newer_executable_abi_version, [] {
    ExecutableAbiVersionProto proto;
    proto.set_platform_name("CUDA");
    proto.mutable_cuda_platform_version()->set_cuda_toolkit_version("13.0.0");
    proto.mutable_cuda_platform_version()->set_cudnn_version("10.0.0");
    proto.mutable_cuda_platform_version()->set_cub_version("3.0.0");
    return ExecutableAbiVersion::FromProto(proto);
  }());

  // Current runtime ABI version should be compatible with current executable
  // ABI versions.
  EXPECT_OK(current_runtime_abi_version.IsCompatibleWith(
      current_executable_abi_version));

  // Older executable ABI version should be compatible with current runtime ABI
  // version.
  EXPECT_OK(current_runtime_abi_version.IsCompatibleWith(
      older_executable_abi_version));

  // Newer executable ABI version should not be compatible with current runtime
  // ABI version.
  EXPECT_THAT(current_runtime_abi_version.IsCompatibleWith(
                  newer_executable_abi_version),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace stream_executor::cuda
