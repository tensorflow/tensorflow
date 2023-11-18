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

#include "tensorflow/core/profiler/convert/op_stats_combiner.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/step_intersection.h"

namespace tensorflow {
namespace profiler {
namespace {

// Tests that the run_environment field of the combined op stats is set
// correctly.
TEST(CombineAllOpStatsTest, CombineRunEnvironment) {
  // Construct OpStatsInfo and all_op_stats_info.
  OpStats dst_op_stats, op_stats_1, op_stats_2;
  op_stats_1.mutable_run_environment()
      ->mutable_host_independent_job_info()
      ->set_profile_duration_ms(100);
  op_stats_2.mutable_run_environment()
      ->mutable_host_independent_job_info()
      ->set_profile_duration_ms(0);
  OpStatsInfo op_stats_info_1(&op_stats_1, TPU, 0),
      op_stats_info_2(&op_stats_2, TPU, 0);
  std::vector<OpStatsInfo> all_op_stats_info = {op_stats_info_1,
                                                op_stats_info_2};

  // Construct dummy step_intersection.
  StepDatabaseResult dummy_step_db_result;
  absl::flat_hash_map<uint32 /*=host_id*/, const StepDatabaseResult*> result;
  result.insert({0, &dummy_step_db_result});
  StepIntersection dummy_step_intersection = StepIntersection(1, result);

  // Combine all op stats.
  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats);

  // Verify that the profile_duration_ms field of the second object is now set.
  EXPECT_EQ(100, dst_op_stats.run_environment()
                     .host_independent_job_info()
                     .profile_duration_ms());
}

TEST(CombineAllOpStatsTest, CombineRunEnvironmentWithUnknownDevice) {
  OpStats dst_op_stats, op_stats_1, op_stats_2;
  op_stats_1.mutable_run_environment()->set_device_type("TPU");
  op_stats_2.mutable_run_environment()->set_device_type("Device");
  OpStatsInfo op_stats_info_1(&op_stats_1, TPU, 0),
      op_stats_info_2(&op_stats_2, TPU, 0);
  std::vector<OpStatsInfo> all_op_stats_info = {op_stats_info_1,
                                                op_stats_info_2};

  // Construct dummy step_intersection.
  StepDatabaseResult dummy_step_db_result;
  absl::flat_hash_map<uint32 /*=host_id*/, const StepDatabaseResult*> result;
  result.insert({0, &dummy_step_db_result});
  StepIntersection dummy_step_intersection = StepIntersection(1, result);

  CombineAllOpStats(all_op_stats_info, dummy_step_intersection, &dst_op_stats);

  EXPECT_EQ("TPU", dst_op_stats.run_environment().device_type());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
