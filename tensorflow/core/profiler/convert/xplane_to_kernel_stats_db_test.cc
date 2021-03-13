/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_kernel_stats_db.h"

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(ConvertXplaneToKernelStats, MultiKernels) {
  XSpace space;
  XPlane* device_trace = space.add_planes();
  XPlaneBuilder device_trace_builder(device_trace);

  // Empty default stream
  device_trace_builder.GetOrCreateLine(0);

  XLineBuilder line_builder = device_trace_builder.GetOrCreateLine(0);
  CreateXEvent(&device_trace_builder, &line_builder, "kernel_name_shortest",
               /*offset_ps=*/10000, /*duration_ps=*/1000,
               {{StatType::kTfOp, "mul_786"},
                {StatType::kKernelDetails, R"MULTI(regs:16
static_shared:0
dynamic_shared:0
grid:1,1,1
block:1,1,1
occ_pct:50.0)MULTI"},
                {StatType::kEquation, ""}});

  CreateXEvent(&device_trace_builder, &line_builder, "kernel_name_middle",
               /*offset_ps=*/20000, /*duration_ps=*/2000,
               {{StatType::kTfOp, "Conv2D"},
                {StatType::kKernelDetails, R"MULTI(regs:32
static_shared:0
dynamic_shared:16384
grid:2,1,1
block:32,1,1
occ_pct=13.0)MULTI"},
                {StatType::kEquation, ""}});

  CreateXEvent(&device_trace_builder, &line_builder,
               "volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_tn",
               /*offset_ps=*/30000, /*duration_ps=*/3000,
               {{StatType::kTfOp, "Einsum_80"},
                {StatType::kKernelDetails, R"MULTI(regs:32
static_shared:0
dynamic_shared:16384
grid:3,1,1
block:64,1,1
occ_pct:25.0)MULTI"},
                {StatType::kEquation, ""}});

  KernelReportMap reports;
  ConvertDeviceTraceXPlaneToKernelReports(*device_trace, {}, &reports);
  KernelStatsDb kernel_stats;
  CopyTopKDurationKernelReportsToDb(reports, &kernel_stats);

  EXPECT_EQ(kernel_stats.reports_size(), 3);

  {
    const auto& kernel = kernel_stats.reports().at(2);
    EXPECT_EQ(kernel.name(), "kernel_name_shortest");
    EXPECT_EQ(kernel.registers_per_thread(), 16);
    EXPECT_EQ(kernel.static_shmem_bytes(), 0);
    EXPECT_EQ(kernel.dynamic_shmem_bytes(), 0);
    EXPECT_EQ(kernel.grid_dim().at(0), 1);
    EXPECT_EQ(kernel.grid_dim().at(1), 1);
    EXPECT_EQ(kernel.grid_dim().at(2), 1);
    EXPECT_EQ(kernel.block_dim().at(0), 1);
    EXPECT_EQ(kernel.block_dim().at(1), 1);
    EXPECT_EQ(kernel.block_dim().at(2), 1);
    EXPECT_EQ(kernel.total_duration_ns(), 1);
    EXPECT_FALSE(kernel.is_kernel_using_tensor_core());
    EXPECT_FALSE(kernel.is_op_tensor_core_eligible());
    EXPECT_EQ(kernel.op_name(), "mul_786");
  }

  {
    const auto& kernel = kernel_stats.reports().at(1);
    EXPECT_EQ(kernel.name(), "kernel_name_middle");
    EXPECT_EQ(kernel.registers_per_thread(), 32);
    EXPECT_EQ(kernel.static_shmem_bytes(), 0);
    EXPECT_EQ(kernel.dynamic_shmem_bytes(), 16384);
    EXPECT_EQ(kernel.grid_dim().at(0), 2);
    EXPECT_EQ(kernel.grid_dim().at(1), 1);
    EXPECT_EQ(kernel.grid_dim().at(2), 1);
    EXPECT_EQ(kernel.block_dim().at(0), 32);
    EXPECT_EQ(kernel.block_dim().at(1), 1);
    EXPECT_EQ(kernel.block_dim().at(2), 1);
    EXPECT_EQ(kernel.total_duration_ns(), 2);
    EXPECT_FALSE(kernel.is_kernel_using_tensor_core());
    EXPECT_TRUE(kernel.is_op_tensor_core_eligible());
    EXPECT_EQ(kernel.op_name(), "Conv2D");
  }

  {
    const auto& kernel = kernel_stats.reports().at(0);
    EXPECT_EQ(kernel.name(), "volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_tn");
    EXPECT_EQ(kernel.registers_per_thread(), 32);
    EXPECT_EQ(kernel.static_shmem_bytes(), 0);
    EXPECT_EQ(kernel.dynamic_shmem_bytes(), 16384);
    EXPECT_EQ(kernel.grid_dim().at(0), 3);
    EXPECT_EQ(kernel.grid_dim().at(1), 1);
    EXPECT_EQ(kernel.grid_dim().at(2), 1);
    EXPECT_EQ(kernel.block_dim().at(0), 64);
    EXPECT_EQ(kernel.block_dim().at(1), 1);
    EXPECT_EQ(kernel.block_dim().at(2), 1);
    EXPECT_EQ(kernel.total_duration_ns(), 3);
    EXPECT_TRUE(kernel.is_kernel_using_tensor_core());
    EXPECT_TRUE(kernel.is_op_tensor_core_eligible());
    EXPECT_EQ(kernel.op_name(), "Einsum_80");
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
