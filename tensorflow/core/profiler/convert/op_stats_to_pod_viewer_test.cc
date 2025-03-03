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

#include "tensorflow/core/profiler/convert/op_stats_to_pod_viewer.h"

#include "google/protobuf/any.pb.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/diagnostics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/pod_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/diagnostics.h"
#include "tensorflow/core/profiler/utils/event_span.h"

namespace tensorflow {
namespace profiler {
namespace {

const double kMaxError = 1e-6;
constexpr int kStepNum = 2;
constexpr int kCoreId = 1001;
constexpr int kStepTimePs = 1000;
constexpr int kHostComputePs = 50;
constexpr int kHostCompilePs = 50;
constexpr int kHostToHostPs = 50;
constexpr int kHostToDevicePs = 50;
constexpr int kHostPreparePs = 50;
constexpr int kDeviceCollectivePs = 350;
constexpr int kHostWaitInputPs = 50;
constexpr int kDeviceToDevicePs = 50;
constexpr int kDeviceToHostPs = 50;
constexpr int kDeviceCompute32Ps = 50;
constexpr int kDeviceCompute16Ps = 50;
constexpr int kDeviceWaitDevicePs = 50;
constexpr int kDeviceWaitHostPs = 50;
constexpr int kUnknownTimePs = 50;
static constexpr char kHostname[] = "host:123";

void CreateOpStats(OpStats* op_stats) {
  PerCoreStepInfo* info = op_stats->mutable_step_db()->add_step_sequence();
  info->set_step_num(kStepNum);
  StepInfoResult& step_info = (*info->mutable_step_info_per_core())[kCoreId];
  step_info.set_step_num(kStepNum);
  step_info.set_duration_ps(kStepTimePs);
  GenericStepBreakdown breakdown;
  auto& type_ps = *breakdown.mutable_type_ps();
  type_ps[HOST_COMPUTE] = kHostComputePs;
  type_ps[HOST_COMPILE] = kHostCompilePs;
  type_ps[HOST_TO_HOST] = kHostToHostPs;
  type_ps[HOST_TO_DEVICE] = kHostToDevicePs;
  type_ps[HOST_PREPARE] = kHostPreparePs;
  type_ps[DEVICE_COLLECTIVES] = kDeviceCollectivePs;
  type_ps[HOST_WAIT_INPUT] = kHostWaitInputPs;
  type_ps[DEVICE_TO_DEVICE] = kDeviceToDevicePs;
  type_ps[DEVICE_TO_HOST] = kDeviceToHostPs;
  type_ps[DEVICE_COMPUTE_32] = kDeviceCompute32Ps;
  type_ps[DEVICE_COMPUTE_16] = kDeviceCompute16Ps;
  type_ps[DEVICE_WAIT_DEVICE] = kDeviceWaitDevicePs;
  type_ps[DEVICE_WAIT_HOST] = kDeviceWaitHostPs;
  type_ps[UNKNOWN_TIME] = kUnknownTimePs;
  step_info.mutable_step_breakdown()->PackFrom(breakdown);
  CoreDetails& details = (*op_stats->mutable_core_id_to_details())[kCoreId];
  details.set_hostname(kHostname);
}

TEST(OpStatsToPodViewer, GpuPodViewer) {
  OpStats op_stats;
  CreateOpStats(&op_stats);
  PodViewerDatabase pod_viewer_db = ConvertOpStatsToPodViewer(op_stats);
  EXPECT_EQ(1, pod_viewer_db.pod_stats_sequence().pod_stats_map_size());
  const PodStatsMap& pod_stats_map =
      pod_viewer_db.pod_stats_sequence().pod_stats_map(0);
  EXPECT_EQ(kStepNum, pod_stats_map.step_num());
  const PodStatsRecord& record = pod_stats_map.pod_stats_per_core().at(kCoreId);
  EXPECT_EQ(kStepNum, record.step_num());
  EXPECT_EQ(kHostname, record.host_name());
  EXPECT_NEAR(tsl::profiler::PicoToMicro(kStepTimePs),
              record.total_duration_us(), kMaxError);
  const auto& breakdown = record.step_breakdown_us();
  EXPECT_NEAR(
      tsl::profiler::PicoToMicro(kDeviceCompute32Ps + kDeviceCompute16Ps),
      breakdown.at(kDeviceCompute), kMaxError);
  EXPECT_NEAR(
      tsl::profiler::PicoToMicro(kDeviceToDevicePs + kDeviceWaitDevicePs),
      breakdown.at(kDeviceToDevice), kMaxError);
  EXPECT_NEAR(tsl::profiler::PicoToMicro(kDeviceCollectivePs),
              breakdown.at(kDeviceCollectives), kMaxError);
  EXPECT_NEAR(tsl::profiler::PicoToMicro(kHostComputePs),
              breakdown.at(kHostCompute), kMaxError);
  EXPECT_NEAR(tsl::profiler::PicoToMicro(kHostPreparePs),
              breakdown.at(kHostPrepare), kMaxError);
  EXPECT_NEAR(tsl::profiler::PicoToMicro(kHostWaitInputPs + kHostToDevicePs +
                                         kDeviceWaitHostPs),
              breakdown.at(kInput), kMaxError);
  EXPECT_NEAR(tsl::profiler::PicoToMicro(kDeviceToHostPs),
              breakdown.at(kOutput), kMaxError);
  EXPECT_NEAR(tsl::profiler::PicoToMicro(kHostCompilePs),
              breakdown.at(kCompile), kMaxError);
  EXPECT_NEAR(tsl::profiler::PicoToMicro(kUnknownTimePs),
              breakdown.at(kAllOthers), kMaxError);

  EXPECT_EQ(GetGenericEventTypeStr(kDeviceCollectives), record.bottleneck());
}

TEST(OpStatsToPodViewer, Diagnostics) {
  OpStats op_stats;
  op_stats.mutable_step_db()->set_use_incomplete_step(true);
  PodViewerDatabase pod_viewer_db = ConvertOpStatsToPodViewer(op_stats);
  EXPECT_EQ(1, pod_viewer_db.diagnostics().warnings_size());
  EXPECT_EQ(kErrorIncompleteStep, pod_viewer_db.diagnostics().warnings(0));
}

TEST(OpStatsToPodViewer, DeviceType) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_device_type("GPU");
  PodViewerDatabase pod_viewer_db = ConvertOpStatsToPodViewer(op_stats);
  EXPECT_EQ("GPU", pod_viewer_db.device_type());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
