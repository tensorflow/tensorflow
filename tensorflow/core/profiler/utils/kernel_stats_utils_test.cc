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

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
