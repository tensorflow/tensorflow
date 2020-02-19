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

#include "tensorflow/core/profiler/convert/xplane_to_op_metrics_db.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

void AddTensorFlowOpEvent(absl::string_view tf_op_fullname,
                          int64 start_timestamp_ns, int64 duration_ns,
                          bool on_device, XPlaneBuilder* plane,
                          XLineBuilder* line) {
  XEventBuilder event =
      line->AddEvent(*plane->GetOrCreateEventMetadata(tf_op_fullname));
  event.SetTimestampNs(start_timestamp_ns);
  event.SetDurationNs(duration_ns);
  if (!on_device) return;
  event.ParseAndAddStatValue(*plane->GetOrCreateStatMetadata("level 0"),
                             tf_op_fullname);
}

void SetXPlaneNameAndId(absl::string_view name, int64 id,
                        XPlaneBuilder* plane) {
  plane->SetName(name);
  plane->SetId(id);
}

TEST(ConvertXPlaneToOpMetricsDb, HostOpMetricsDb) {
  static constexpr char TfOp1[] = "TfOp1";
  static constexpr char TfOp2[] = "TfOp2";
  constexpr int64 kTfOp1StartNs = 100000;
  constexpr int64 kTfOp1DurationNs = 8000;
  constexpr int64 kTfOp2StartNs = 110000;
  constexpr int64 kTfOp2DurationNs = 10000;

  XPlane xplane;
  XPlaneBuilder host_plane(&xplane);
  SetXPlaneNameAndId(kHostThreads, /*id=*/0, &host_plane);
  XLineBuilder thread1 = host_plane.GetOrCreateLine(/*line_id=*/10);
  AddTensorFlowOpEvent(absl::StrCat(TfOp1, ":", TfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/false, &host_plane,
                       &thread1);
  XLineBuilder thread2 = host_plane.GetOrCreateLine(/*line_id=*/20);
  AddTensorFlowOpEvent(absl::StrCat(TfOp1, ":", TfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/false, &host_plane,
                       &thread2);
  AddTensorFlowOpEvent(absl::StrCat(TfOp2, ":", TfOp2), kTfOp2StartNs,
                       kTfOp2DurationNs, /*on_device=*/false, &host_plane,
                       &thread2);

  OpMetricsDb op_metrics = ConvertHostThreadsXPlaneToOpMetricsDb(xplane);
  // Op1, Op2, Idle.
  EXPECT_EQ(3, op_metrics.metrics_db_size());
  uint64 total_op_duration =
      NanosToPicos(kTfOp1DurationNs * 2 + kTfOp2DurationNs);
  EXPECT_EQ(total_op_duration, op_metrics.total_op_time_ps());
  uint64 total_duration = NanosToPicos(kTfOp2StartNs - kTfOp1StartNs +
                                       kTfOp2DurationNs + kTfOp1DurationNs);
  EXPECT_EQ(total_duration, op_metrics.total_time_ps());

  // Verifies OpMetricsDb is built correctly.
  const OpMetrics& op_1 = op_metrics.metrics_db().at(0);
  EXPECT_EQ(TfOp1, op_1.name());
  EXPECT_EQ(TfOp1, op_1.category());
  EXPECT_EQ(2, op_1.occurrences());
  EXPECT_EQ(NanosToPicos(kTfOp1DurationNs) * 2, op_1.time_ps());

  const OpMetrics& idle = op_metrics.metrics_db().at(1);
  EXPECT_EQ("IDLE", idle.name());
  // Idle time is the gap between Op2 start and the end of Op1, which is 2000ns.
  EXPECT_EQ(NanosToPicos(2000), idle.time_ps());

  const OpMetrics& op_2 = op_metrics.metrics_db().at(2);
  EXPECT_EQ(TfOp2, op_2.name());
  EXPECT_EQ(TfOp2, op_2.category());
  EXPECT_EQ(1, op_2.occurrences());
  EXPECT_EQ(NanosToPicos(kTfOp2DurationNs), op_2.time_ps());
}

TEST(ConvertXPlaneToOpMetricsDb, DeviceOpMetricsDb) {
  static constexpr char TfOp1[] = "TfOp1";
  static constexpr char TfOp2[] = "TfOp2";
  constexpr int64 kTfOp1StartNs = 100000;
  constexpr int64 kTfOp1DurationNs = 8000;
  constexpr int64 kTfOp2StartNs = 110000;
  constexpr int64 kTfOp2DurationNs = 10000;

  XPlane xplane;
  XPlaneBuilder device_plane(&xplane);
  SetXPlaneNameAndId(absl::StrCat(kGpuPlanePrefix, ":0"), /*id=*/1,
                     &device_plane);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(/*line_id=*/10);
  AddTensorFlowOpEvent(absl::StrCat(TfOp1, ":", TfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/true, &device_plane,
                       &stream1);
  XLineBuilder stream2 = device_plane.GetOrCreateLine(/*line_id=*/20);
  AddTensorFlowOpEvent(absl::StrCat(TfOp1, ":", TfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/true, &device_plane,
                       &stream2);
  AddTensorFlowOpEvent(absl::StrCat(TfOp2, ":", TfOp2), kTfOp2StartNs,
                       kTfOp2DurationNs, /*on_device=*/true, &device_plane,
                       &stream2);

  OpMetricsDb op_metrics = ConvertDeviceTraceXPlaneToOpMetricsDb(
      xplane, /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_giga_bytes_per_second=*/0);

  // Op1, Op2, Idle.
  EXPECT_EQ(3, op_metrics.metrics_db_size());
  uint64 total_op_duration =
      NanosToPicos(kTfOp1DurationNs * 2 + kTfOp2DurationNs);
  EXPECT_EQ(total_op_duration, op_metrics.total_op_time_ps());
  // For device, the total_duration for each device is the total duration merged
  // from all GPU streams, which is from 100000 to 120000.
  uint64 total_duration =
      NanosToPicos(kTfOp2StartNs + kTfOp2DurationNs - kTfOp1StartNs);
  EXPECT_EQ(total_duration, op_metrics.total_time_ps());

  // Verifies OpMetricsDb is built correctly.
  const OpMetrics& op_1 = op_metrics.metrics_db().at(0);
  EXPECT_EQ(TfOp1, op_1.name());
  EXPECT_EQ(TfOp1, op_1.category());
  EXPECT_EQ(2, op_1.occurrences());
  EXPECT_EQ(NanosToPicos(kTfOp1DurationNs) * 2, op_1.time_ps());

  const OpMetrics& op_2 = op_metrics.metrics_db().at(1);
  EXPECT_EQ(TfOp2, op_2.name());
  EXPECT_EQ(TfOp2, op_2.category());
  EXPECT_EQ(1, op_2.occurrences());
  EXPECT_EQ(NanosToPicos(kTfOp2DurationNs), op_2.time_ps());

  const OpMetrics& idle = op_metrics.metrics_db().at(2);
  EXPECT_EQ("IDLE", idle.name());
  // GPU is always busy in this example.
  EXPECT_EQ(NanosToPicos(0), idle.time_ps());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
