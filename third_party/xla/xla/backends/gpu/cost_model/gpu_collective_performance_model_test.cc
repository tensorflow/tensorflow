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

#include "xla/backends/gpu/cost_model/gpu_collective_performance_model.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/cost_model/gpu_hlo_cost_analysis.h"
#include "xla/backends/gpu/cost_model/hlo_op_profiles.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class RubinCollectivePerformanceModelTest
    : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    if (const char* value = std::getenv("NCCL_MAX_NCHANNELS")) {
      original_max_channels_ = value;
    }
    if (const char* value = std::getenv("NCCL_MIN_NCHANNELS")) {
      original_min_channels_ = value;
    }
    ASSERT_EQ(tsl::setenv("NCCL_MAX_NCHANNELS", "32", /*overwrite=*/1), 0);
    ASSERT_EQ(tsl::setenv("NCCL_MIN_NCHANNELS", "1", /*overwrite=*/1), 0);
  }

  void TearDown() override {
    if (original_max_channels_.has_value()) {
      EXPECT_EQ(tsl::setenv("NCCL_MAX_NCHANNELS",
                            original_max_channels_->c_str(), /*overwrite=*/1),
                0);
    } else {
      EXPECT_EQ(tsl::unsetenv("NCCL_MAX_NCHANNELS"), 0);
    }
    if (original_min_channels_.has_value()) {
      EXPECT_EQ(tsl::setenv("NCCL_MIN_NCHANNELS",
                            original_min_channels_->c_str(), /*overwrite=*/1),
                0);
    } else {
      EXPECT_EQ(tsl::unsetenv("NCCL_MIN_NCHANNELS"), 0);
    }
    HloHardwareIndependentTestBase::TearDown();
  }

  void ExpectAllReduceTime(int active_links,
                           double expected_bus_bandwidth_bytes_per_ns) {
    constexpr int kReplicaCount = 8;
    // Ignore reduction compute to isolate communication bandwidth.
    constexpr int kFlopsPerAdd = 0;
    constexpr absl::Duration kLaunchOverhead = absl::Microseconds(5);
    constexpr double kToleranceNs = 1.0;
    constexpr char kHlo[] = R"(
HloModule test
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT sum = f32[] add(x, y)
}
ENTRY main {
  p = f32[1048576] parameter(0)
  ROOT ar = f32[1048576] all-reduce(p), replica_groups={}, to_apply=add
})";
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnVerifiedModule(kHlo, kReplicaCount));
    const HloOpProfiles::HloOpProfile op_profile = {
        {{HloOpcode::kAdd, F32}, kFlopsPerAdd}};
    GpuHloCostAnalysis analysis(HloCostAnalysis::Options{}, op_profile);
    ASSERT_THAT(module->entry_computation()->Accept(&analysis),
                absl_testing::IsOk());
    const HloInstruction& ar = *module->entry_computation()->root_instruction();
    auto device = TestGpuDeviceInfo::VRNVL72DeviceInfo();
    device.set_device_interconnect_info({.active_links = active_links});

    // Account for reading the f32 input and writing the output. Ring AllReduce
    // has reduce-scatter and all-gather phases, each with ranks - 1 steps.
    const int64_t bytes_accessed = 2 * ShapeUtil::ByteSizeOf(ar.shape());
    constexpr int kRingSteps = 2 * (kReplicaCount - 1);
    constexpr double kRingScalingRatio =
        static_cast<double>(kReplicaCount) / kRingSteps;
    const double expected_communication_ns =
        bytes_accessed /
        (expected_bus_bandwidth_bytes_per_ns * kRingScalingRatio);
    const absl::Duration expected =
        kLaunchOverhead + absl::Nanoseconds(expected_communication_ns);
    EXPECT_NEAR(absl::ToDoubleNanoseconds(
                    GpuPerformanceWithCollectiveModel::ComputeCollectiveTime(
                        ar, &analysis, device)),
                absl::ToDoubleNanoseconds(expected), kToleranceNs);
  }

 private:
  std::optional<std::string> original_max_channels_;
  std::optional<std::string> original_min_channels_;
};

TEST_F(RubinCollectivePerformanceModelTest, WithNvlink) {
  constexpr int kActiveLinks = 18;
  constexpr double kBandwidthPerLinkBytesPerNs = 31.2;
  constexpr double kRingEfficiency = 0.92;
  constexpr double kExpectedBusBandwidthBytesPerNs =
      kActiveLinks * kBandwidthPerLinkBytesPerNs * kRingEfficiency;
  ExpectAllReduceTime(kActiveLinks, kExpectedBusBandwidthBytesPerNs);
}

TEST_F(RubinCollectivePerformanceModelTest, WithUnknownLinks) {
  constexpr int kActiveLinks = 0;
  // Unknown link counts use the LL128 ceiling.
  constexpr int kGraphChannels = 32;
  constexpr int kCtasPerGraphChannel = 2;
  constexpr double kLl128BandwidthPerCtaBytesPerNs = 21.6;
  constexpr double kExpectedBusBandwidthBytesPerNs =
      kGraphChannels * kCtasPerGraphChannel * kLl128BandwidthPerCtaBytesPerNs;
  ExpectAllReduceTime(kActiveLinks, kExpectedBusBandwidthBytesPerNs);
}

}  // namespace
}  // namespace xla::gpu
