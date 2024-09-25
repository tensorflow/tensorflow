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

#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"

#include <gmock/gmock.h>
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::FieldsAre;

TEST(KernelStatsUtilsTest, TestGroupKernelReportsByOpName) {
  KernelStatsDb kernel_stats_db;
  KernelReport* kernel_report_1 = kernel_stats_db.add_reports();
  kernel_report_1->set_name("op1_kernel1");
  kernel_report_1->set_op_name("op1");
  kernel_report_1->set_total_duration_ns(1000);
  kernel_report_1->set_is_kernel_using_tensor_core(true);
  kernel_report_1->set_is_op_tensor_core_eligible(true);

  KernelReport* kernel_report_2 = kernel_stats_db.add_reports();
  kernel_report_2->set_name("op1_kernel2");
  kernel_report_2->set_op_name("op1");
  kernel_report_2->set_total_duration_ns(1000);
  kernel_report_2->set_is_kernel_using_tensor_core(false);
  kernel_report_2->set_is_op_tensor_core_eligible(true);

  KernelReport* kernel_report_3 = kernel_stats_db.add_reports();
  kernel_report_3->set_name("op2_kernel1");
  kernel_report_3->set_op_name("op2");
  kernel_report_3->set_total_duration_ns(100);
  kernel_report_3->set_is_kernel_using_tensor_core(false);
  kernel_report_3->set_is_op_tensor_core_eligible(false);

  KernelStatsByOpName kernel_stats_by_op_name =
      GroupKernelReportsByOpName(kernel_stats_db);

  // Verifies there are two OpLevelKernelStats
  ASSERT_EQ(kernel_stats_by_op_name.size(), 2);
  auto iter1 = kernel_stats_by_op_name.find("op1");
  auto iter2 = kernel_stats_by_op_name.find("op2");
  ASSERT_NE(iter1, kernel_stats_by_op_name.end());
  ASSERT_NE(iter2, kernel_stats_by_op_name.end());
  const OpLevelKernelStats& op1_stats = iter1->second;
  const OpLevelKernelStats& op2_stats = iter2->second;

  EXPECT_EQ(op1_stats.is_op_tensor_core_eligible, true);
  EXPECT_EQ(op1_stats.total_duration_ns, 2000);
  EXPECT_EQ(op1_stats.tensor_core_duration_ns, 1000);

  EXPECT_EQ(op2_stats.is_op_tensor_core_eligible, false);
  EXPECT_EQ(op2_stats.total_duration_ns, 100);
  EXPECT_EQ(op2_stats.tensor_core_duration_ns, 0);
}

TEST(KernelStatsUtilsTest, KernelDetailsXStatParser) {
  xla::profiler::KernelDetails kernel_info;
  kernel_info.registers_per_thread = 10;
  kernel_info.static_shared_memory_usage = 128;
  kernel_info.dynamic_shared_memory_usage = 256;
  kernel_info.block_x = 32;
  kernel_info.block_y = 8;
  kernel_info.block_z = 4;
  kernel_info.grid_x = 3;
  kernel_info.grid_y = 2;
  kernel_info.grid_z = 1;
  const double occupancy_pct = 50.0;
  std::string xstat_kernel_details = ToXStat(kernel_info, occupancy_pct);
  KernelReport kernel;
  ParseKernelLaunchParams(xstat_kernel_details, &kernel);
  // Verifies that the parser can parse kKernelDetails XStat.
  EXPECT_EQ(kernel.registers_per_thread(), 10);
  EXPECT_EQ(kernel.static_shmem_bytes(), 128);
  EXPECT_EQ(kernel.dynamic_shmem_bytes(), 256);
  EXPECT_EQ(kernel.block_dim()[0], 32);
  EXPECT_EQ(kernel.block_dim()[1], 8);
  EXPECT_EQ(kernel.block_dim()[2], 4);
  EXPECT_EQ(kernel.grid_dim()[0], 3);
  EXPECT_EQ(kernel.grid_dim()[1], 2);
  EXPECT_EQ(kernel.grid_dim()[2], 1);
}

TEST(KernelStatsUtilsTest, KernelDetailsTokenizer) {
  KernelReport kernel;

  // Test odd token count (3): { "odd", "grid", "3,2,1" }
  absl::string_view kernel_details_0 = "odd grid:3,2,1";
  ParseKernelLaunchParams(kernel_details_0, &kernel);
  EXPECT_EQ(kernel.grid_dim()[0], 3);
  EXPECT_EQ(kernel.grid_dim()[1], 2);
  EXPECT_EQ(kernel.grid_dim()[2], 1);

  // Test odd token count (3): { "block", "6,5,4", "odd" }
  absl::string_view kernel_details_1 = "block:6,5,4 odd ";
  ParseKernelLaunchParams(kernel_details_1, &kernel);
  EXPECT_EQ(kernel.block_dim()[0], 6);
  EXPECT_EQ(kernel.block_dim()[1], 5);
  EXPECT_EQ(kernel.block_dim()[2], 4);

  // Test odd token count (3): { "block", "1,2,3", "odd", "grid", "4,5,6" }
  absl::string_view kernel_details_2 = "block:1,2,3 odd grid:4,5,6";
  ParseKernelLaunchParams(kernel_details_2, &kernel);
  EXPECT_EQ(kernel.block_dim()[0], 1);
  EXPECT_EQ(kernel.block_dim()[1], 2);
  EXPECT_EQ(kernel.block_dim()[2], 3);
  EXPECT_EQ(kernel.grid_dim()[0], 4);
  EXPECT_EQ(kernel.grid_dim()[1], 5);
  EXPECT_EQ(kernel.grid_dim()[2], 6);

  // Test even token count (4): { "static_shared", "7", "dynamic_shared", "8" }
  absl::string_view kernel_details_3 = "static_shared:7 dynamic_shared:8";
  ParseKernelLaunchParams(kernel_details_3, &kernel);
  EXPECT_EQ(kernel.static_shmem_bytes(), 7);
  EXPECT_EQ(kernel.dynamic_shmem_bytes(), 8);
}

TEST(KernelStatsUtilsTest, TestInsertOrUpdateKernelReport) {
  KernelReport kr;
  kr.set_name("op1_kernel1");
  kr.set_op_name("op1");
  // Must provide dummy dims since KernelReportMap's comparator assumes array of
  // size 3; values here were suggested by autocomplete
  kr.add_block_dim(32);
  kr.add_block_dim(8);
  kr.add_block_dim(4);
  kr.add_grid_dim(3);
  kr.add_grid_dim(2);
  kr.add_grid_dim(1);

  KernelReportValue krv1;
  krv1.total_duration_ns = 1700;
  krv1.min_duration_ns = 500;
  krv1.max_duration_ns = 1200;
  krv1.occurrences = 2;

  KernelReportValue krv2;
  krv2.total_duration_ns = 900;
  krv2.min_duration_ns = 900;
  krv2.max_duration_ns = 900;
  krv2.occurrences = 1;

  KernelReportMap dst1;
  InsertOrUpdateKernelReport(kr, krv1, &dst1);
  InsertOrUpdateKernelReport(kr, krv2, &dst1);
  EXPECT_THAT(dst1[kr], FieldsAre(2600, 500, 1200, 3));

  KernelReportMap dst2;
  InsertOrUpdateKernelReport(kr, krv2, &dst2);
  InsertOrUpdateKernelReport(kr, krv1, &dst2);
  EXPECT_THAT(dst2[kr], FieldsAre(2600, 500, 1200, 3));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
