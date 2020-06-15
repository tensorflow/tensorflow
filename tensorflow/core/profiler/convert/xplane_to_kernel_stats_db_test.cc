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
  CreateXEvent(&device_trace_builder, &line_builder, "kernel_name_0",
               /*offset_ps=*/10000, /*duration_ps=*/1000,
               {{StatType::kLevel0, "mul_786"},
                {StatType::kKernelDetails, R"MULTI(registers_per_thread:16
static_shared_memory_usage:0
dynamic_shared_memory_usage:0
grid_x:1
grid_y:1
grid_z:1
block_x:1
block_y:1
block_z:1)MULTI"},
                {StatType::kEquation, ""}});

  CreateXEvent(&device_trace_builder, &line_builder, "kernel_name_1",
               /*offset_ps=*/20000, /*duration_ps=*/2000,
               {{StatType::kLevel0, "Conv2D"},
                {StatType::kKernelDetails, R"MULTI(registers_per_thread:32
static_shared_memory_usage:0
dynamic_shared_memory_usage:16384
grid_x:2
grid_y:1
grid_z:1
block_x:32
block_y:1
block_z:1)MULTI"},
                {StatType::kEquation, ""}});

  CreateXEvent(&device_trace_builder, &line_builder,
               "volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_tn",
               /*offset_ps=*/30000, /*duration_ps=*/3000,
               {{StatType::kLevel0, "Einsum_80"},
                {StatType::kKernelDetails, R"MULTI(registers_per_thread:32
static_shared_memory_usage:0
dynamic_shared_memory_usage:16384
grid_x:3
grid_y:1
grid_z:1
block_x:64
block_y:1
block_z:1)MULTI"},
                {StatType::kEquation, ""}});
  KernelStatsDb kernel_stats =
      ConvertDeviceTraceXPlaneToKernelStatsDb(*device_trace, {});

  EXPECT_EQ(kernel_stats.reports_size(), 3);

  const auto& kernel0 = kernel_stats.reports().at(0);
  EXPECT_EQ(kernel0.name(), "kernel_name_0");
  EXPECT_EQ(kernel0.registers_per_thread(), 16);
  EXPECT_EQ(kernel0.static_shmem_bytes(), 0);
  EXPECT_EQ(kernel0.dynamic_shmem_bytes(), 0);
  EXPECT_EQ(kernel0.grid_dim().at(0), 1);
  EXPECT_EQ(kernel0.grid_dim().at(1), 1);
  EXPECT_EQ(kernel0.grid_dim().at(2), 1);
  EXPECT_EQ(kernel0.block_dim().at(0), 1);
  EXPECT_EQ(kernel0.block_dim().at(1), 1);
  EXPECT_EQ(kernel0.block_dim().at(2), 1);
  EXPECT_EQ(kernel0.total_duration_ns(), 1);
  EXPECT_FALSE(kernel0.is_kernel_using_tensor_core());
  EXPECT_FALSE(kernel0.is_op_tensor_core_eligible());
  EXPECT_EQ(kernel0.op_name(), "mul_786");

  const auto& kernel1 = kernel_stats.reports().at(1);
  EXPECT_EQ(kernel1.name(), "kernel_name_1");
  EXPECT_EQ(kernel1.registers_per_thread(), 32);
  EXPECT_EQ(kernel1.static_shmem_bytes(), 0);
  EXPECT_EQ(kernel1.dynamic_shmem_bytes(), 16384);
  EXPECT_EQ(kernel1.grid_dim().at(0), 2);
  EXPECT_EQ(kernel1.grid_dim().at(1), 1);
  EXPECT_EQ(kernel1.grid_dim().at(2), 1);
  EXPECT_EQ(kernel1.block_dim().at(0), 32);
  EXPECT_EQ(kernel1.block_dim().at(1), 1);
  EXPECT_EQ(kernel1.block_dim().at(2), 1);
  EXPECT_EQ(kernel1.total_duration_ns(), 2);
  EXPECT_FALSE(kernel1.is_kernel_using_tensor_core());
  EXPECT_TRUE(kernel1.is_op_tensor_core_eligible());
  EXPECT_EQ(kernel1.op_name(), "Conv2D");

  const auto& kernel2 = kernel_stats.reports().at(2);
  EXPECT_EQ(kernel2.name(), "volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_tn");
  EXPECT_EQ(kernel2.registers_per_thread(), 32);
  EXPECT_EQ(kernel2.static_shmem_bytes(), 0);
  EXPECT_EQ(kernel2.dynamic_shmem_bytes(), 16384);
  EXPECT_EQ(kernel2.grid_dim().at(0), 3);
  EXPECT_EQ(kernel2.grid_dim().at(1), 1);
  EXPECT_EQ(kernel2.grid_dim().at(2), 1);
  EXPECT_EQ(kernel2.block_dim().at(0), 64);
  EXPECT_EQ(kernel2.block_dim().at(1), 1);
  EXPECT_EQ(kernel2.block_dim().at(2), 1);
  EXPECT_EQ(kernel2.total_duration_ns(), 3);
  EXPECT_TRUE(kernel2.is_kernel_using_tensor_core());
  EXPECT_TRUE(kernel2.is_op_tensor_core_eligible());
  EXPECT_EQ(kernel2.op_name(), "Einsum_80");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
