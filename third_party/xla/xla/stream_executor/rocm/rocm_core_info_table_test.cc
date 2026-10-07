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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/core_info.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

// Asserts that the table reproduces the peak TFLOPS the AMD white papers quote
void CheckPeakOpsPerNs(const DeviceDescription& device_info, bool is_matrix,
                       xla::PrimitiveType dtype, double expected_tflops) {
  const ExecutionUnitDescription* eu =
      is_matrix ? device_info.matrix_unit_description()
                : device_info.scalar_unit_description();
  ASSERT_NE(eu, nullptr);

  std::optional<ExecutionUnitDescription::RateInfo> rate =
      eu->GetRateInfo(dtype);
  ASSERT_TRUE(rate.has_value())
      << "No rate info for dtype " << xla::PrimitiveType_Name(dtype);

  double flops_per_ns_per_unit =
      rate->clock_rate_ghz * rate->ops_per_clock * 2;  // FMA = 2 ops.
  int64_t n_units = device_info.core_count() * rate->units_per_core;
  double tflops = flops_per_ns_per_unit * n_units / 1000.0;

  // 2% tolerance to absorb boost-clock rounding in the white papers.
  EXPECT_NEAR(tflops, expected_tflops, expected_tflops * 0.02)
      << "dtype: " << xla::PrimitiveType_Name(dtype);
}

// From CDNA 3 white paper
TEST(RocmCoreInfoTableTest, CalculatePeakOpsPerNsMI300X) {
  DeviceDescription d = xla::gpu::TestGpuDeviceInfo::AMDMI300DeviceInfo();
  CheckPeakOpsPerNs(d, /*is_matrix=*/false, xla::F32, 163.4);
  CheckPeakOpsPerNs(d, /*is_matrix=*/false, xla::F64, 81.7);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::F32, 163.4);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::F64, 163.4);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::F16, 1307.4);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::BF16, 1307.4);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::F8E4M3, 2614.9);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::S8, 2614.9);
}

// From CDNA 4 white paper
TEST(RocmCoreInfoTableTest, CalculatePeakOpsPerNsMI350X) {
  DeviceDescription d = xla::gpu::TestGpuDeviceInfo::AMDMI350DeviceInfo();
  CheckPeakOpsPerNs(d, /*is_matrix=*/false, xla::F32, 144.2);
  CheckPeakOpsPerNs(d, /*is_matrix=*/false, xla::F64, 72.1);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::F32, 144.2);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::F64, 72.1);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::F16, 2306.9);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::BF16, 2306.9);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::F8E4M3, 4613.7);
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::S8, 4613.7);
  // MXFP6 / MXFP4 -> 9.2 PF.
  CheckPeakOpsPerNs(d, /*is_matrix=*/true, xla::F4E2M1FN, 9227.5);
}

TEST(RocmCoreInfoTableTest, GetFpusPerCore) {
  auto fpus = [](std::string gfx) {
    return GetFpusPerCore(
        GpuComputeCapability(RocmComputeCapability(std::move(gfx))));
  };
  EXPECT_EQ(fpus("gfx908"), 64);   // CDNA1
  EXPECT_EQ(fpus("gfx90a"), 64);   // CDNA2
  EXPECT_EQ(fpus("gfx942"), 128);  // CDNA3
  EXPECT_EQ(fpus("gfx950"), 128);  // CDNA4
  // Untabulated gfx targets take the default.
  EXPECT_EQ(fpus("gfx1100"), 128);
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
