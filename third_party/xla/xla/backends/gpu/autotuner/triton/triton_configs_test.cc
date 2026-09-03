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

#include "xla/backends/gpu/autotuner/triton/triton_configs.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace xla::gpu {
namespace {

using ::testing::SizeIs;

TEST(TritonConfigsTest, PlatformsReturnNonEmptyConfig) {
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kAmpere),
              SizeIs(30));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwell),
              SizeIs(40));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultCuda),
              SizeIs(25));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultRocm),
              SizeIs(2));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kHopper),
              SizeIs(25));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI300),
              SizeIs(33));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI350),
              SizeIs(58));
}

TEST(TritonConfigsTest, GetDefaultTritonConfigsCuda) {
  se::CudaComputeCapability hopper_cc{se::CudaComputeCapability::kHopper, 0};
  EXPECT_EQ(GetDefaultTritonConfigs(se::GpuComputeCapability{hopper_cc}),
            GetTritonConfigsForPlatform(TritonConfigsPlatform::kHopper));

  se::CudaComputeCapability ampere_cc{se::CudaComputeCapability::kAmpere, 0};
  EXPECT_EQ(GetDefaultTritonConfigs(se::GpuComputeCapability{ampere_cc}),
            GetTritonConfigsForPlatform(TritonConfigsPlatform::kAmpere));

  se::CudaComputeCapability blackwell_cc{se::CudaComputeCapability::kBlackwell,
                                         0};
  EXPECT_EQ(GetDefaultTritonConfigs(se::GpuComputeCapability{blackwell_cc}),
            GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwell));

  se::CudaComputeCapability volta_cc{se::CudaComputeCapability::kVolta, 0};
  EXPECT_EQ(GetDefaultTritonConfigs(se::GpuComputeCapability{volta_cc}),
            GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultCuda));
}

TEST(TritonConfigsTest, GetDefaultTritonConfigsRocm) {
  se::RocmComputeCapability mi300_cc("gfx942");
  EXPECT_EQ(GetDefaultTritonConfigs(se::GpuComputeCapability{mi300_cc}),
            GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI300));

  se::RocmComputeCapability mi350_cc("gfx950");
  EXPECT_EQ(GetDefaultTritonConfigs(se::GpuComputeCapability{mi350_cc}),
            GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI350));

  se::RocmComputeCapability default_rocm_cc("gfx908");
  EXPECT_EQ(GetDefaultTritonConfigs(se::GpuComputeCapability{default_rocm_cc}),
            GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultRocm));
}

}  // namespace
}  // namespace xla::gpu
