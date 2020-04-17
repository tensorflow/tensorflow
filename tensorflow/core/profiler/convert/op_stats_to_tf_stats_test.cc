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

#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

void AddTensorFlowOpEvent(absl::string_view tf_op_fullname,
                          int64 start_timestamp_ns, int64 duration_ns,
                          bool on_device, absl::string_view kernel_name,
                          XPlaneBuilder* plane, XLineBuilder* line) {
  absl::string_view name = on_device ? kernel_name : tf_op_fullname;
  XEventBuilder event = line->AddEvent(*plane->GetOrCreateEventMetadata(name));
  event.SetTimestampNs(start_timestamp_ns);
  event.SetDurationNs(duration_ns);
  if (!on_device) return;
  event.ParseAndAddStatValue(*plane->GetOrCreateStatMetadata("level 0"),
                             tf_op_fullname);
}

TEST(OpStatsToTfStats, GpuTfStats) {
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

  XSpace space;
  XPlaneBuilder device_plane(space.add_planes());
  device_plane.SetName(absl::StrCat(kGpuPlanePrefix, ":0"));
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

  const OpStats& op_stats = ConvertXSpaceToOpStats(space);
  const TfStatsDatabase& tf_stats = ConvertOpStatsToTfStats(op_stats);

  // TfOp1, TfOp2, Idle
  EXPECT_EQ(3, tf_stats.with_idle().tf_stats_record_size());

  const TfStatsRecord& record_0 = tf_stats.with_idle().tf_stats_record(0);
  EXPECT_EQ(kTfOp1, record_0.op_name());
  EXPECT_EQ(kTfOp1, record_0.op_type());
  EXPECT_EQ(2, record_0.occurrences());
  EXPECT_EQ(NanosToMicros(kKernel1DurationNs) * 2 +
                NanosToMicros(kKernel2DurationNs) * 2,
            record_0.total_self_time_in_us());

  const TfStatsRecord& record_1 = tf_stats.with_idle().tf_stats_record(1);
  EXPECT_EQ(kTfOp2, record_1.op_name());
  EXPECT_EQ(kTfOp2, record_1.op_type());
  EXPECT_EQ(1, record_1.occurrences());
  EXPECT_EQ(NanosToMicros(kKernel3DurationNs),
            record_1.total_self_time_in_us());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
