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

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/pjrt/gpu/abi_helpers.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/python/pjrt_ifrt/gpu_xla_executable_abi_version.h"
#include "xla/python/pjrt_ifrt/xla_executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_runtime_abi_version.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;

stream_executor::cuda::CudaRuntimeAbiVersion GetTestRuntimeAbi() {
  return stream_executor::cuda::CudaRuntimeAbiVersion(
      stream_executor::SemanticVersion(1, 2, 3),
      stream_executor::SemanticVersion(4, 5, 6),
      stream_executor::SemanticVersion(7, 8, 9));
}

absl::StatusOr<stream_executor::ExecutableAbiVersion>
GetTestExecutableAbiVersion(
    stream_executor::SemanticVersion cuda_toolkit_version) {
  stream_executor::ExecutableAbiVersionProto proto;
  proto.set_platform_name(stream_executor::cuda::kCudaPlatformId->ToName());
  proto.mutable_cuda_platform_version()->set_cuda_toolkit_version(
      cuda_toolkit_version.ToString());
  proto.mutable_cuda_platform_version()->set_cudnn_version("4.5.6");
  proto.mutable_cuda_platform_version()->set_cub_version("7.8.9");
  return stream_executor::ExecutableAbiVersion::FromProto(proto);
}

absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>> GetPjrtAbiVersion(
    const stream_executor::RuntimeAbiVersion& abi) {
  xla::PjRtRuntimeAbiVersionProto proto;
  ASSIGN_OR_RETURN(auto abi_proto, abi.ToProto());
  proto.set_version(abi_proto.SerializeAsString());
  proto.set_platform(xla::CudaId());
  return xla::gpu::PjRtRuntimeAbiVersionFromProto(proto);
}

absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>
GetPjrtExecutableAbiVersion(const stream_executor::ExecutableAbiVersion& abi) {
  xla::PjRtExecutableAbiVersionProto proto;
  proto.set_version(abi.proto().SerializeAsString());
  proto.set_platform(xla::CudaId());
  return xla::gpu::PjRtExecutableAbiVersionFromProto(proto);
}

absl::StatusOr<std::unique_ptr<xla::ifrt::XlaExecutableAbiVersion>>
CreateXlaExecutableAbiVersion(
    std::unique_ptr<xla::PjRtExecutableAbiVersion> executable_abi_version) {
  return std::make_unique<xla::GpuXlaExecutableAbiVersion>(
      std::move(executable_abi_version));
}

TEST(GpuXlaExecutableAbiVersionTest, SerializeDeserialize) {
  ASSERT_OK_AND_ASSIGN(auto runtime_abi_version,
                       GetPjrtAbiVersion(GetTestRuntimeAbi()));
  ASSERT_OK_AND_ASSIGN(
      auto test_executable_abi_version,
      GetTestExecutableAbiVersion(stream_executor::SemanticVersion(1, 2, 3)));
  ASSERT_OK_AND_ASSIGN(
      auto executable_abi_version,
      GetPjrtExecutableAbiVersion(test_executable_abi_version));
  ASSERT_OK_AND_ASSIGN(
      auto xla_executable_abi_version,
      CreateXlaExecutableAbiVersion(std::move(executable_abi_version)));

  ASSERT_OK_AND_ASSIGN(auto serialized_version,
                       xla_executable_abi_version->Serialize());
  ASSERT_OK_AND_ASSIGN(
      auto deserialized_version,
      xla::ifrt::XlaExecutableAbiVersion::Deserialize(serialized_version));

  EXPECT_THAT(runtime_abi_version->IsCompatibleWith(
                  deserialized_version->ExecutableAbiVersion()),
              IsOk());
}

}  // namespace
}  // namespace xla
