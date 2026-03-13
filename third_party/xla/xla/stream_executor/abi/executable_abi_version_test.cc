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

#include "xla/stream_executor/abi/executable_abi_version.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace stream_executor {
namespace {

using tsl::proto_testing::EqualsProto;

TEST(ExecutableAbiVersionTest, FromProto) {
  ExecutableAbiVersionProto proto;
  proto.set_platform_name("CUDA");
  proto.mutable_cuda_platform_version()->set_cuda_toolkit_version("12.0.0");
  proto.mutable_cuda_platform_version()->set_cudnn_version("9.0.0");
  proto.mutable_cuda_platform_version()->set_cub_version("2.0.0");
  ASSERT_OK_AND_ASSIGN(ExecutableAbiVersion executable_abi_version,
                       ExecutableAbiVersion::FromProto(proto));
  EXPECT_THAT(executable_abi_version.platform_name(), "CUDA");
  EXPECT_THAT(executable_abi_version.proto(), EqualsProto(proto));
}

TEST(ExecutableAbiVersionTest, FromDeviceDescription) {
  DeviceDescription device_description;
  device_description.set_gpu_compute_capability(CudaComputeCapability::Volta());
  device_description.set_runtime_version(SemanticVersion(12, 0, 0));
  device_description.set_dnn_version(SemanticVersion(9, 0, 0));
  device_description.set_cub_version(SemanticVersion(2, 0, 0));

  ASSERT_OK_AND_ASSIGN(
      ExecutableAbiVersion executable_abi_version,
      ExecutableAbiVersion::FromDeviceDescription(device_description));
  EXPECT_THAT(executable_abi_version.proto(), EqualsProto(R"pb(
                platform_name: "CUDA"
                cuda_platform_version {
                  cuda_toolkit_version: "12.0.0"
                  cudnn_version: "9.0.0"
                  cub_version: "2.0.0"
                }
              )pb"));
}

TEST(ExecutableAbiVersionTest, FromDeviceDescriptionRocm) {
  DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      GpuComputeCapability(RocmComputeCapability("gfx942")));

  ASSERT_OK_AND_ASSIGN(
      ExecutableAbiVersion executable_abi_version,
      ExecutableAbiVersion::FromDeviceDescription(device_description));
  EXPECT_THAT(executable_abi_version.platform_name(), "ROCm");
  EXPECT_THAT(executable_abi_version.proto(),
              EqualsProto(R"pb(platform_name: "ROCm")pb"));
}

TEST(ExecutableAbiVersionTest, FromDeviceDescriptionOneAPI) {
  DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      GpuComputeCapability(OneAPIComputeCapability::BMG()));

  ASSERT_OK_AND_ASSIGN(
      ExecutableAbiVersion executable_abi_version,
      ExecutableAbiVersion::FromDeviceDescription(device_description));
  EXPECT_THAT(executable_abi_version.platform_name(), "SYCL");
  EXPECT_THAT(executable_abi_version.proto(),
              EqualsProto(R"pb(platform_name: "SYCL")pb"));
}

}  // namespace

}  // namespace stream_executor
