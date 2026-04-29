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

#include "xla/stream_executor/cuda/cuda_core_info_table.h"

#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

void CheckPeakOpsPerNs(const DeviceDescription& device_info,
                       bool is_matrix_unit, xla::PrimitiveType dtype,
                       double expected_tflops) {
  const ExecutionUnitDescription* eu_descr =
      is_matrix_unit ? device_info.matrix_unit_description()
                     : device_info.scalar_unit_description();

  ASSERT_NE(eu_descr, nullptr);

  std::optional<ExecutionUnitDescription::RateInfo> dtype_rates =
      eu_descr->GetRateInfo(dtype);
  ASSERT_TRUE(dtype_rates.has_value());

  double flops_per_ns_per_unit = dtype_rates->clock_rate_ghz *
                                 dtype_rates->ops_per_clock *
                                 2;  // FMA is 2 ops.
  int64_t n_compute_units =
      device_info.core_count() * dtype_rates->units_per_core;

  float ops_per_ns = flops_per_ns_per_unit * n_compute_units;

  // Allow for 2% error to account for imprecise estimates.
  EXPECT_NEAR(ops_per_ns / 1000.0, expected_tflops, expected_tflops * 0.02)
      << "Failed for dtype: " << xla::PrimitiveType_Name(dtype);
}

TEST(CudaCoreInfoTableTest, CalculatePeakOpsPerNsH100) {
  DeviceDescription h100_device_info =
      xla::gpu::TestGpuDeviceInfo::H100SXMDeviceInfo();
  FillExecutionUnitDesc(h100_device_info.cuda_compute_capability(),
                        h100_device_info.clock_rate_ghz(), h100_device_info);

  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F8E4M3, 1979.0);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::S8, 1979.0);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F16, 989.0);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::BF16, 989.0);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F32, 495.0);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F64, 67.0);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F16, 134.0);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::BF16, 134.0);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F32, 67.0);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::S32, 33.5);
  CheckPeakOpsPerNs(h100_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F64, 33.5);
}

TEST(CudaCoreInfoTableTest, CalculatePeakOpsPerNsB200) {
  DeviceDescription b200_device_info =
      xla::gpu::TestGpuDeviceInfo::B200SXMDeviceInfo();
  FillExecutionUnitDesc(b200_device_info.cuda_compute_capability(),
                        b200_device_info.clock_rate_ghz(), b200_device_info);

  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F4E2M1FN, 9000.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F8E4M3, 4500.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::S8, 4500.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F16, 2200.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::BF16, 2200.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F32, 1100.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F64, 37.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F16, 75.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::BF16, 75.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F32, 75.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::S32, 37.0);
  CheckPeakOpsPerNs(b200_device_info, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F64, 37.0);
}

TEST(CudaCoreInfoTableTest, GetFpusPerCore) {
  EXPECT_EQ(GetFpusPerCore(CudaComputeCapability::Hopper()), 128);
  EXPECT_EQ(GetFpusPerCore(CudaComputeCapability::Ampere()), 64);
  EXPECT_EQ(GetFpusPerCore(CudaComputeCapability::Volta()), 64);
  EXPECT_EQ(GetFpusPerCore(CudaComputeCapability::Pascal()), 64);
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
