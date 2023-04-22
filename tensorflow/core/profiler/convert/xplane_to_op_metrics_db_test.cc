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

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

void AddTensorFlowOpEvent(std::string&& tf_op_fullname,
                          int64 start_timestamp_ns, int64 duration_ns,
                          bool on_device, absl::string_view kernel_name,
                          XPlaneBuilder* plane, XLineBuilder* line) {
  absl::string_view name = on_device ? kernel_name : tf_op_fullname;
  XEventBuilder event = line->AddEvent(*plane->GetOrCreateEventMetadata(name));
  event.SetTimestampNs(start_timestamp_ns);
  event.SetDurationNs(duration_ns);
  if (!on_device) return;
  event.AddStatValue(
      *plane->GetOrCreateStatMetadata("level 0"),
      *plane->GetOrCreateStatMetadata(std::move(tf_op_fullname)));
}

TEST(ConvertXPlaneToOpMetricsDb, HostOpMetricsDb) {
  static constexpr char kTfOp1[] = "TfOp1";
  static constexpr char kTfOp2[] = "TfOp2";
  constexpr int64 kTfOp1StartNs = 100000;
  constexpr int64 kTfOp1DurationNs = 8000;
  constexpr int64 kTfOp2StartNs = 110000;
  constexpr int64 kTfOp2DurationNs = 10000;

  XSpace xspace;
  XPlane* xplane = GetOrCreateHostXPlane(&xspace);
  XPlaneBuilder host_plane(xplane);
  XLineBuilder thread1 = host_plane.GetOrCreateLine(/*line_id=*/10);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/false,
                       /*kernel_name=*/"", &host_plane, &thread1);
  XLineBuilder thread2 = host_plane.GetOrCreateLine(/*line_id=*/20);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/false,
                       /*kernel_name=*/"", &host_plane, &thread2);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp2, ":", kTfOp2), kTfOp2StartNs,
                       kTfOp2DurationNs, /*on_device=*/false,
                       /*kernel_name=*/"", &host_plane, &thread2);

  OpMetricsDb op_metrics = ConvertHostThreadsXPlaneToOpMetricsDb(*xplane);
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
  EXPECT_EQ(kTfOp1, op_1.name());
  EXPECT_EQ(kTfOp1, op_1.category());
  EXPECT_EQ(2, op_1.occurrences());
  EXPECT_EQ(NanosToPicos(kTfOp1DurationNs) * 2, op_1.time_ps());

  const OpMetrics& idle = op_metrics.metrics_db().at(1);
  EXPECT_EQ(kIdle, idle.name());
  // Idle time is the gap between Op2 start and the end of Op1, which is 2000ns.
  EXPECT_EQ(NanosToPicos(2000), idle.time_ps());

  const OpMetrics& op_2 = op_metrics.metrics_db().at(2);
  EXPECT_EQ(kTfOp2, op_2.name());
  EXPECT_EQ(kTfOp2, op_2.category());
  EXPECT_EQ(1, op_2.occurrences());
  EXPECT_EQ(NanosToPicos(kTfOp2DurationNs), op_2.time_ps());
}

TEST(ConvertXPlaneToOpMetricsDb, DeviceOpMetricsDb) {
  // TfOp1 has kernel1 and kernel2; TfOp2 has kernel3.
  static constexpr char kTfOp1[] = "TfOp1";
  static constexpr char kTfOp2[] = "TfOp2";
  static constexpr char kKernel1[] = "kernel1";
  static constexpr char kKernel2[] = "kernel2";
  static constexpr char kKernel3[] = "kernel3";
  constexpr int64 kKernel1StartNs = 100000;
  constexpr int64 kKernel1DurationNs = 8000;
  constexpr int64 kKernel2StartNs = 110000;
  constexpr int64 kKernel2DurationNs = 10000;
  constexpr int64 kKernel3StartNs = 120000;
  constexpr int64 kKernel3DurationNs = 10000;

  XSpace xspace;
  XPlane* xplane = GetOrCreateGpuXPlane(&xspace, /*device_ordinal=*/0);
  XPlaneBuilder device_plane(xplane);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(/*line_id=*/10);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel1StartNs,
                       kKernel1DurationNs, /*on_device=*/true, kKernel1,
                       &device_plane, &stream1);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel2StartNs,
                       kKernel2DurationNs, /*on_device=*/true, kKernel2,
                       &device_plane, &stream1);
  XLineBuilder stream2 = device_plane.GetOrCreateLine(/*line_id=*/20);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel1StartNs,
                       kKernel1DurationNs, /*on_device=*/true, kKernel1,
                       &device_plane, &stream2);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel2StartNs,
                       kKernel2DurationNs, /*on_device=*/true, kKernel2,
                       &device_plane, &stream2);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp2, ":", kTfOp2), kKernel3StartNs,
                       kKernel3DurationNs, /*on_device=*/true, kKernel3,
                       &device_plane, &stream2);

  OpMetricsDb op_metrics = ConvertDeviceTraceXPlaneToOpMetricsDb(*xplane);

  // kernel1, kernel2, kernel3, Idle.
  EXPECT_EQ(4, op_metrics.metrics_db_size());
  uint64 total_op_duration = NanosToPicos(
      kKernel1DurationNs * 2 + kKernel2DurationNs * 2 + kKernel3DurationNs);
  EXPECT_EQ(total_op_duration, op_metrics.total_op_time_ps());
  // For device, the total_duration for each device is the total duration merged
  // from all GPU streams, which is from 100000 to 130000.
  uint64 total_duration =
      NanosToPicos(kKernel3StartNs + kKernel3DurationNs - kKernel1StartNs);
  EXPECT_EQ(total_duration, op_metrics.total_time_ps());

  // Verifies OpMetricsDb is built correctly.
  const OpMetrics& op_1 = op_metrics.metrics_db().at(0);
  EXPECT_EQ(absl::StrCat(kTfOp1, "/", kKernel1), op_1.name());
  EXPECT_EQ(kTfOp1, op_1.category());
  EXPECT_EQ(2, op_1.occurrences());
  EXPECT_EQ(NanosToPicos(kKernel1DurationNs) * 2, op_1.time_ps());

  const OpMetrics& op_2 = op_metrics.metrics_db().at(1);
  EXPECT_EQ(absl::StrCat(kTfOp1, "/", kKernel2), op_2.name());
  EXPECT_EQ(kTfOp1, op_2.category());
  EXPECT_EQ(2, op_2.occurrences());
  EXPECT_EQ(NanosToPicos(kKernel2DurationNs) * 2, op_2.time_ps());

  const OpMetrics& op_3 = op_metrics.metrics_db().at(2);
  EXPECT_EQ(absl::StrCat(kTfOp2, "/", kKernel3), op_3.name());
  EXPECT_EQ(kTfOp2, op_3.category());
  EXPECT_EQ(1, op_3.occurrences());
  EXPECT_EQ(NanosToPicos(kKernel3DurationNs), op_3.time_ps());

  const OpMetrics& idle = op_metrics.metrics_db().at(3);
  EXPECT_EQ(kIdle, idle.name());
  // GPU is always busy in this example.
  EXPECT_EQ(NanosToPicos(0), idle.time_ps());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
