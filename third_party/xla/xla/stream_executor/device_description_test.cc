/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/stream_executor/device_description.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace {
using absl_testing::IsOkAndHolds;
using testing::Eq;
using testing::Pointee;

TEST(DeviceDescription, DefaultConstruction) {
  DeviceDescription desc;
  EXPECT_EQ(desc.device_address_bits(), -1);
  EXPECT_EQ(desc.device_memory_size(), -1);
  EXPECT_EQ(desc.clock_rate_ghz(), -1);
  EXPECT_EQ(desc.name(), "<undefined>");
  EXPECT_EQ(desc.platform_version(), "<undefined>");
  constexpr SemanticVersion kZeroVersion = {0, 0, 0};
  EXPECT_EQ(desc.driver_version(), kZeroVersion);
  EXPECT_EQ(desc.runtime_version(), kZeroVersion);
  EXPECT_EQ(desc.pci_bus_id(), "<undefined>");
  EXPECT_EQ(desc.scalar_unit_description(), nullptr);
  EXPECT_EQ(desc.matrix_unit_description(), nullptr);
}

///////////////////////////////////////////////////////////////////////////////
// class RocmComputeCapability tests. To be moved to a separate file once the
// class is refactored out of device_description.h

TEST(RocmComputeCapability, GfxVersion) {
  RocmComputeCapability rcc0;  // default constructed
  auto default_gcn_arch_name = RocmComputeCapability::kInvalidGfx;
  // failure is serious enough to not expect the rest could pass
  ASSERT_EQ(default_gcn_arch_name, rcc0.gfx_version());

  const std::string gfx{"some_string"};
  std::string gcn_arch{gfx};
  ASSERT_EQ(gfx, RocmComputeCapability{gcn_arch}.gfx_version());

  gcn_arch.append(":tail");
  ASSERT_EQ(gfx, RocmComputeCapability{gcn_arch}.gfx_version());

  gcn_arch.append(":even_longer");
  ASSERT_EQ(gfx, RocmComputeCapability{gcn_arch}.gfx_version());
}

TEST(RocmComputeCapability, IsSupportedGfxVersion) {
  ASSERT_TRUE(RocmComputeCapability{"gfx900"}.is_supported_gfx_version());
  ASSERT_TRUE(RocmComputeCapability{"gfx1201"}.is_supported_gfx_version());
  ASSERT_TRUE(RocmComputeCapability{"gfx942"}.is_supported_gfx_version());
  ASSERT_FALSE(RocmComputeCapability{"some_string"}.is_supported_gfx_version());
}

TEST(RocmComputeCapability, Accessors) {
  // there's not much point in testing individual trivial implementations as
  // this require to put here the whole knowledge of RocmComputeCapability.
  // This will make maintanance of the class unnecessary more painful.
  // Testing only the most complicated methods, basically IsThisGfxInAnyList().
  EXPECT_TRUE(RocmComputeCapability{"gfx942"}.gfx9_mi300_series());
  EXPECT_FALSE(RocmComputeCapability{"gfx942x"}.gfx9_mi300_series());
  EXPECT_TRUE(RocmComputeCapability{"gfx950"}.gfx9_mi300_series());
  EXPECT_FALSE(RocmComputeCapability{"gfx951"}.gfx9_mi300_series());

  EXPECT_TRUE(RocmComputeCapability{"gfx942"}.gfx9_mi200_or_later());
  EXPECT_FALSE(RocmComputeCapability{"gfx942x"}.gfx9_mi200_or_later());
  EXPECT_TRUE(RocmComputeCapability{"gfx950"}.gfx9_mi200_or_later());
  EXPECT_FALSE(RocmComputeCapability{"gfx951"}.gfx9_mi200_or_later());
  EXPECT_TRUE(RocmComputeCapability{"gfx90a"}.gfx9_mi200_or_later());
  EXPECT_FALSE(RocmComputeCapability{"gfx90x"}.gfx9_mi200_or_later());

  EXPECT_TRUE(RocmComputeCapability{"gfx942"}.gfx9_mi100_or_later());
  EXPECT_FALSE(RocmComputeCapability{"gfx942x"}.gfx9_mi100_or_later());
  EXPECT_TRUE(RocmComputeCapability{"gfx950"}.gfx9_mi100_or_later());
  EXPECT_FALSE(RocmComputeCapability{"gfx951"}.gfx9_mi100_or_later());
  EXPECT_TRUE(RocmComputeCapability{"gfx90a"}.gfx9_mi100_or_later());
  EXPECT_FALSE(RocmComputeCapability{"gfx90x"}.gfx9_mi100_or_later());
  EXPECT_TRUE(RocmComputeCapability{"gfx908"}.gfx9_mi100_or_later());
  EXPECT_FALSE(RocmComputeCapability{"gfx907"}.gfx9_mi100_or_later());

  EXPECT_TRUE(RocmComputeCapability{"gfx11"}.gfx11());
  EXPECT_FALSE(RocmComputeCapability{"gfx10"}.gfx11());
  EXPECT_FALSE(RocmComputeCapability{"gfx12"}.gfx11());
  EXPECT_TRUE(RocmComputeCapability{"gfx1100"}.gfx11());
  EXPECT_TRUE(RocmComputeCapability{"gfx11xx"}.gfx11());
  EXPECT_TRUE(RocmComputeCapability{"gfx11xxblabla"}.gfx11());

  EXPECT_TRUE(RocmComputeCapability{"gfx12"}.gfx12());
  EXPECT_FALSE(RocmComputeCapability{"gfx11"}.gfx12());
  EXPECT_FALSE(RocmComputeCapability{"gfx13"}.gfx12());
  EXPECT_TRUE(RocmComputeCapability{"gfx1200"}.gfx12());
  EXPECT_TRUE(RocmComputeCapability{"gfx12xx"}.gfx12());
  EXPECT_TRUE(RocmComputeCapability{"gfx12xxblabla"}.gfx12());

  EXPECT_TRUE(RocmComputeCapability{"gfx12"}.fence_before_barrier());
  EXPECT_TRUE(RocmComputeCapability{"anything"}.fence_before_barrier());
  EXPECT_FALSE(RocmComputeCapability{"gfx900"}.fence_before_barrier());
  EXPECT_FALSE(RocmComputeCapability{"gfx906"}.fence_before_barrier());

  EXPECT_FALSE(RocmComputeCapability{"gfx900"}.has_hipblaslt());
  EXPECT_TRUE(RocmComputeCapability{"gfx942"}.has_hipblaslt());
  EXPECT_TRUE(RocmComputeCapability{"gfx90a"}.has_hipblaslt());
  EXPECT_TRUE(RocmComputeCapability{"gfx1200"}.has_hipblaslt());
  EXPECT_TRUE(RocmComputeCapability{"gfx1100"}.has_hipblaslt());
  EXPECT_TRUE(RocmComputeCapability{"gfx1103"}.has_hipblaslt());
}

TEST(GpuComputeCapability, ProtoConversion) {
  EXPECT_THAT(
      GpuComputeCapability::FromProto(
          GpuComputeCapability(CudaComputeCapability::Volta()).ToProto()),
      IsOkAndHolds(GpuComputeCapability(CudaComputeCapability::Volta())));
  EXPECT_THAT(
      GpuComputeCapability::FromProto(
          GpuComputeCapability(RocmComputeCapability("gfx900")).ToProto()),
      IsOkAndHolds(GpuComputeCapability(RocmComputeCapability("gfx900"))));
}

TEST(ExecutionUnitDescription, ProtoConversion) {
  ExecutionUnitDescription desc;
  desc.SetRateInfo(xla::F32, ExecutionUnitDescription::RateInfo{128, 1.5f, 2});
  desc.SetRateInfo(xla::F16, ExecutionUnitDescription::RateInfo{256, 1.5f, 4});

  EXPECT_THAT(ExecutionUnitDescription::FromProto(desc.ToProto()),
              IsOkAndHolds(desc));
}

TEST(DeviceDescription, ProtoConversion) {
  DeviceDescription desc;
  ExecutionUnitDescription scalar_unit_desc;
  scalar_unit_desc.SetRateInfo(xla::F32,
                               ExecutionUnitDescription::RateInfo{64, 1.2f, 1});
  desc.set_scalar_unit_description(scalar_unit_desc);

  ExecutionUnitDescription matrix_unit_desc;
  matrix_unit_desc.SetRateInfo(
      xla::F16, ExecutionUnitDescription::RateInfo{128, 1.2f, 8});
  desc.set_matrix_unit_description(matrix_unit_desc);

  TF_ASSERT_OK_AND_ASSIGN(DeviceDescription from_proto,
                          DeviceDescription::FromProto(desc.ToGpuProto()));

  EXPECT_THAT(from_proto.scalar_unit_description(),
              Pointee(Eq(*desc.scalar_unit_description())));
  EXPECT_THAT(from_proto.matrix_unit_description(),
              Pointee(Eq(*desc.matrix_unit_description())));
}

}  // namespace
}  // namespace stream_executor
