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

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

XEventBuilder AddTensorFlowOpEvent(std::string&& tf_op_fullname,
                                   int64_t start_timestamp_ns,
                                   int64_t duration_ns, bool on_device,
                                   absl::string_view kernel_name,
                                   XPlaneBuilder* plane, XLineBuilder* line) {
  absl::string_view name = on_device ? kernel_name : tf_op_fullname;
  XEventBuilder event = line->AddEvent(*plane->GetOrCreateEventMetadata(name));
  event.SetTimestampNs(start_timestamp_ns);
  event.SetDurationNs(duration_ns);
  if (!on_device) return event;
  event.AddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      *plane->GetOrCreateStatMetadata(std::move(tf_op_fullname)));
  return event;
}

void AddTensorFlowOpEventWithKernelDetails(std::string&& tf_op_fullname,
                                           int64_t start_timestamp_ns,
                                           int64_t duration_ns, bool on_device,
                                           absl::string_view kernel_name,
                                           absl::string_view kernel_details,
                                           XPlaneBuilder* plane,
                                           XLineBuilder* line) {
  XEventBuilder event =
      AddTensorFlowOpEvent(std::move(tf_op_fullname), start_timestamp_ns,
                           duration_ns, on_device, kernel_name, plane, line);
  if (!on_device) return;
  event.ParseAndAddStatValue(*plane->GetOrCreateStatMetadata("kernel_details"),
                             kernel_details);
}

TEST(OpStatsToTfStats, GpuTfStats) {
  // TfOp1 has kernel1 and kernel2; TfOp2 has kernel3;
  // TfOp3 has kernel4 and kernel5 and is TensorCore eligible.
  static constexpr char kTfOp1[] = "TfOp1";
  static constexpr char kTfOp2[] = "TfOp2";
  static constexpr char kTfOp3[] = "Conv2D";
  static constexpr char kKernel1[] = "kernel1";
  static constexpr char kKernel2[] = "kernel2";
  static constexpr char kKernel3[] = "kernel3";
  // Kernel4 is a kernel using TensorCore
  static constexpr char kKernel4[] = "volta_fp16_s884gemm";
  static constexpr char kKernel5[] = "kernel5";
  constexpr int64_t kKernel1StartNs = 100000;
  constexpr int64_t kKernel1DurationNs = 8000;
  constexpr int64_t kKernel2StartNs = 110000;
  constexpr int64_t kKernel2DurationNs = 10000;
  constexpr int64_t kKernel3StartNs = 120000;
  constexpr int64_t kKernel3DurationNs = 10000;
  constexpr int64_t kKernel4StartNs = 130000;
  constexpr int64_t kKernel4DurationNs = 10000;
  constexpr int64_t kKernel5StartNs = 150000;
  constexpr int64_t kKernel5DurationNs = 10000;

  // Mock kernel details for both kernel4 and kernel5.
  const std::string kKernelDetails = R"MULTI(regs:32
static_shared:0
dynamic_shared:16384
grid:2,1,1
block:32,1,1
occ_pct:100)MULTI";

  XSpace space;
  XPlaneBuilder device_plane(
      GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0));
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
  AddTensorFlowOpEventWithKernelDetails(
      absl::StrCat(kTfOp3, ":", kTfOp3), kKernel4StartNs, kKernel4DurationNs,
      /*on_device=*/true, kKernel4, kKernelDetails, &device_plane, &stream2);
  AddTensorFlowOpEventWithKernelDetails(
      absl::StrCat(kTfOp3, ":", kTfOp3), kKernel5StartNs, kKernel5DurationNs,
      /*on_device=*/true, kKernel5, kKernelDetails, &device_plane, &stream2);

  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  const OpStats op_stats = ConvertXSpaceToOpStats(space, options);
  const TfStatsDatabase tf_stats = ConvertOpStatsToTfStats(op_stats);

  EXPECT_EQ(tf_stats.device_type(), op_stats.run_environment().device_type());

  // TfOp1, TfOp3, TfOp2, Idle
  EXPECT_EQ(4, tf_stats.with_idle().tf_stats_record_size());

  const TfStatsRecord& record_0 = tf_stats.with_idle().tf_stats_record(0);
  EXPECT_EQ(kTfOp1, record_0.op_name());
  EXPECT_EQ(kTfOp1, record_0.op_type());
  EXPECT_EQ(2, record_0.occurrences());
  EXPECT_EQ(
      NanoToMicro(kKernel1DurationNs) * 2 + NanoToMicro(kKernel2DurationNs) * 2,
      record_0.total_self_time_in_us());

  const TfStatsRecord& record_1 = tf_stats.with_idle().tf_stats_record(1);
  EXPECT_EQ(kTfOp3, record_1.op_name());
  EXPECT_EQ(kTfOp3, record_1.op_type());
  EXPECT_EQ(1, record_1.occurrences());
  EXPECT_EQ(NanoToMicro(kKernel4DurationNs) + NanoToMicro(kKernel5DurationNs),
            record_1.total_self_time_in_us());
  // GPU TensorCore utilization is 0.5 because kernel4 is using TensorCore and
  // kernel5 is not using TensorCore, and they have the same duration.
  EXPECT_DOUBLE_EQ(0.5, record_1.gpu_tensorcore_utilization());

  const TfStatsRecord& record_2 = tf_stats.with_idle().tf_stats_record(2);
  EXPECT_EQ(kTfOp2, record_2.op_name());
  EXPECT_EQ(kTfOp2, record_2.op_type());
  EXPECT_EQ(1, record_2.occurrences());
  EXPECT_EQ(NanoToMicro(kKernel3DurationNs), record_2.total_self_time_in_us());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
