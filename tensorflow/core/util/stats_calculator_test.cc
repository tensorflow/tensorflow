/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/stats_calculator.h"

#include <cfloat>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using Detail = StatsCalculator::Detail;

TEST(StatsCalculatorTest, TotalTimeMs) {
  auto options = StatSummarizerOptions();
  StatsCalculator calc(options);

  EXPECT_EQ(0, calc.num_runs());
  calc.UpdateRunTotalUs(1);

  EXPECT_EQ(1, calc.num_runs());
  calc.UpdateRunTotalUs(2);

  EXPECT_EQ(2, calc.num_runs());
  auto run_time_us = calc.run_total_us();
  EXPECT_EQ(1, run_time_us.min());
  EXPECT_FLOAT_EQ(1.5, run_time_us.avg());
}

TEST(StatsCalculatorTest, AddNodeStatsUpdate) {
  auto options = StatSummarizerOptions();
  StatsCalculator calc(options);
  EXPECT_TRUE(calc.GetDetails().empty());

  const int64_t node1_run_order = 1;
  const int64_t run1_start_us = 1;
  const int64_t run1_end_us = 2;
  const int64_t run1_mem_used = 45;
  calc.AddNodeStats("node1", "type_1", node1_run_order, run1_start_us,
                    run1_end_us, run1_mem_used);
  ASSERT_EQ(1, calc.GetDetails().size());
  const Detail& detail = calc.GetDetails().at("node1");
  EXPECT_EQ(1, detail.times_called);
  EXPECT_EQ("node1", detail.name);
  EXPECT_EQ("type_1", detail.type);
  EXPECT_EQ(node1_run_order, detail.run_order);

  const int64_t run2_start_us = 3;
  const int64_t run2_end_us = 5;
  const int64_t run2_mem_used = 145;
  calc.AddNodeStats("node1", "type_1", node1_run_order, run2_start_us,
                    run2_end_us, run2_mem_used);
  EXPECT_EQ(1, calc.GetDetails().size());

  EXPECT_EQ(2, detail.times_called);
  EXPECT_EQ("node1", detail.name);
  EXPECT_EQ("type_1", detail.type);
  EXPECT_EQ(node1_run_order, detail.run_order);

  EXPECT_EQ(run1_start_us + run2_start_us, detail.start_us.sum());
  EXPECT_EQ(run1_end_us + run2_end_us, detail.rel_end_us.sum());
  EXPECT_EQ(run1_mem_used + run2_mem_used, detail.mem_used.sum());
}

TEST(StatsCalculatorTest, UpdateStat) {
  Stat<double> stat;
  EXPECT_TRUE(stat.empty());
  EXPECT_TRUE(stat.all_same());
  stat.UpdateStat(1);
  EXPECT_TRUE(stat.all_same());
  stat.UpdateStat(-1.0);
  EXPECT_FALSE(stat.all_same());
  stat.UpdateStat(100);
  stat.UpdateStat(0);
  EXPECT_EQ(4, stat.count());
  EXPECT_EQ(-1, stat.min());
  EXPECT_EQ(100, stat.max());
  EXPECT_EQ(25, stat.avg());
  EXPECT_EQ(1, stat.first());
  EXPECT_EQ(0, stat.newest());
  EXPECT_EQ(10002, stat.squared_sum());
  EXPECT_EQ(625, stat.avg() * stat.avg());
  // Sample variance
  EXPECT_EQ(7502.0 / 3, stat.sample_variance());
  // Sample standard deviation, from WolframAlpha
  EXPECT_NEAR(50.00666622228147160678152, std::sqrt(stat.sample_variance()),
              FLT_EPSILON);
  // Population variance
  EXPECT_NEAR(7502.0 / 4, stat.variance(), FLT_EPSILON);
  // Population standard deviation, from WolframAlpha
  EXPECT_NEAR(43.30704330706496060826769, stat.std_deviation(), FLT_EPSILON);
}

}  // namespace
}  // namespace tensorflow
