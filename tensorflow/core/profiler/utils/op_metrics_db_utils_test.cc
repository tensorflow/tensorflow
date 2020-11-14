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

#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr double kMaxError = 1E-10;

TEST(OpMetricsDbTest, IdleTimeRatio) {
  OpMetricsDb metrics_db_0;
  metrics_db_0.set_total_time_ps(100000000);
  metrics_db_0.set_total_op_time_ps(60000000);
  EXPECT_NEAR(0.4, IdleTimeRatio(metrics_db_0), kMaxError);

  OpMetricsDb metrics_db_1;
  metrics_db_1.set_total_time_ps(200000000);
  metrics_db_1.set_total_op_time_ps(150000000);
  EXPECT_NEAR(0.25, IdleTimeRatio(metrics_db_1), kMaxError);

  OpMetricsDb metrics_db_2;
  metrics_db_1.set_total_time_ps(0);
  metrics_db_1.set_total_op_time_ps(0);
  EXPECT_NEAR(1.0, IdleTimeRatio(metrics_db_2), kMaxError);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
