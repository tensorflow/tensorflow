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

#include "xla/tools/compute_xspace_stats.h"

#include <gtest/gtest.h>
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

TEST(ComputeXSpaceStatsTest, IsMemcpyTest) {
  tensorflow::profiler::XEvent event;
  event.add_stats()->set_metadata_id(1);
  EXPECT_TRUE(xla::gpu::IsMemcpy(event, 1));
  EXPECT_FALSE(xla::gpu::IsMemcpy(event, 2));
}

TEST(ComputeXSpaceStatsTest, ProcessLineEventsTest) {
  tensorflow::profiler::XLine line;
  tensorflow::profiler::XEvent* event = line.add_events();
  event->set_duration_ps(1000);
  event->add_stats()->set_metadata_id(1);
  event = line.add_events();
  event->set_duration_ps(2000);
  event->add_stats()->set_metadata_id(2);
  event = line.add_events();
  event->set_duration_ps(3000);
  event->add_stats()->set_metadata_id(1);
  TF_ASSERT_OK_AND_ASSIGN(LineStats stats,
                          xla::gpu::ProcessLineEvents(line, 1));
  EXPECT_EQ(stats.total_time_ps, 6000);
  EXPECT_EQ(stats.memcpy_time_ps, 4000);
}

TEST(ComputeXSpaceStatsTest, CalculateDeviceTimeAndMemcpyTest) {
  tensorflow::profiler::XSpace xspace;
  tensorflow::profiler::XPlane* plane = xspace.add_planes();
  plane->set_name("/device:GPU:0");
  tensorflow::profiler::XStatMetadata* stat_metadata =
      &(*plane->mutable_stat_metadata())[1];
  stat_metadata->set_id(1);
  stat_metadata->set_name("memcpy_details");
  tensorflow::profiler::XLine* line = plane->add_lines();
  tensorflow::profiler::XEvent* event = line->add_events();
  event->set_duration_ps(1000);
  event->add_stats()->set_metadata_id(1);
  event = line->add_events();
  event->set_duration_ps(2000);
  event->add_stats()->set_metadata_id(2);
  event = line->add_events();
  event->set_duration_ps(3000);
  event->add_stats()->set_metadata_id(1);
  TF_ASSERT_OK_AND_ASSIGN(GpuDeviceStats stats,
                          xla::gpu::CalculateGpuDeviceStats(xspace));
  EXPECT_EQ(stats.device_time_us, 0.006);
  EXPECT_EQ(stats.device_memcpy_time_us, 0.004);
  EXPECT_TRUE(xla::gpu::IsMemcpy(*event, 1));
  EXPECT_FALSE(xla::gpu::IsMemcpy(*event, 2));
}

TEST(ComputeXSpaceStatsTest, CalculateCpuTimeTest) {
  tensorflow::profiler::XSpace xspace;
  tensorflow::profiler::XPlane* plane = xspace.add_planes();
  plane->set_name("/host:CPU");
  tensorflow::profiler::XLine* line = plane->add_lines();
  tensorflow::profiler::XEvent* event = line->add_events();
  event->set_duration_ps(1000);
  event = line->add_events();
  event->set_duration_ps(2000);

  tensorflow::profiler::XPlane* task_environment_plane = xspace.add_planes();
  task_environment_plane->set_name("Task Environment");
  tensorflow::profiler::XStatMetadata* stat_metadata =
      &(*task_environment_plane->mutable_stat_metadata())[1];
  stat_metadata->set_id(1);
  stat_metadata->set_name("profile_start_time");
  tensorflow::profiler::XStat* stat = task_environment_plane->add_stats();
  stat->set_metadata_id(1);
  stat->set_uint64_value(1000);
  stat_metadata = &(*task_environment_plane->mutable_stat_metadata())[2];
  stat_metadata->set_id(2);
  stat_metadata->set_name("profile_stop_time");
  stat = task_environment_plane->add_stats();
  stat->set_metadata_id(2);
  stat->set_uint64_value(5000);

  TF_ASSERT_OK_AND_ASSIGN(CpuStats stats, xla::gpu::CalculateCpuStats(xspace));
  EXPECT_EQ(stats.cpu_time_us, 0.003);
  EXPECT_EQ(stats.wall_time_us, 4);
}

}  // namespace
}  // namespace xla::gpu
