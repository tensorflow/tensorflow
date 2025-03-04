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

#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/duty_cycle_tracker.h"
#include "tensorflow/core/profiler/convert/multi_xplanes_to_op_stats.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/step_events_to_steps_db.h"
#include "tensorflow/core/profiler/protobuf/diagnostics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_function.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Property;
using ::testing::UnorderedElementsAre;

TEST(ConvertXPlaneToOpStats, GpuPerfEnv) {
  auto space = std::make_unique<XSpace>();
  constexpr double kMaxError = 0.01;
  constexpr int kClockRateKHz = 1530000;
  constexpr int kCoreCount = 80;
  constexpr uint64 kMemoryBandwidthBytesPerSecond =
      uint64{900} * 1000 * 1000 * 1000;
  // Volta.
  constexpr int kComputeCapMajor = 7;
  constexpr int kComputeCapMinor = 0;

  XPlaneBuilder device_plane(
      GetOrCreateGpuXPlane(space.get(), /*device_ordinal=*/0));
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kDevVendor)),
                            kDeviceVendorNvidia);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata("clock_rate"),
                            kClockRateKHz);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata("core_count"),
                            kCoreCount);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("memory_bandwidth"),
      kMemoryBandwidthBytesPerSecond);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_major"),
      kComputeCapMajor);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_minor"),
      kComputeCapMinor);

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  OpStats op_stats;
  TF_CHECK_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                   options, &op_stats));
  const PerfEnv& perf_env = op_stats.perf_env();
  // Change to lower flops number that we do not use sum of the tensor core peak
  // flops and the cuda core peak flops together as peak flops. Only use the
  // tensor core peak flops as all those white papers are using.
  EXPECT_NEAR(125.34, perf_env.peak_tera_flops_per_second(), kMaxError);
  EXPECT_NEAR(
      900,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_HBM_RW),
      kMaxError);
  // Ridge point changed accordingly from above peak flops change.
  EXPECT_NEAR(139.26, perf_env.ridge_point(), kMaxError);
}

TEST(ConvertXPlaneToOpStats, GpuRunEnvironment) {
  auto space = std::make_unique<XSpace>();
  XPlaneBuilder device_plane1(
      GetOrCreateGpuXPlane(space.get(), /*device_ordinal=*/0));
  device_plane1.AddStatValue(*device_plane1.GetOrCreateStatMetadata(
                                 GetStatTypeStr(StatType::kDevVendor)),
                             kDeviceVendorNvidia);
  XPlaneBuilder device_plane2(
      GetOrCreateGpuXPlane(space.get(), /*device_ordinal=*/1));
  device_plane2.AddStatValue(*device_plane2.GetOrCreateStatMetadata(
                                 GetStatTypeStr(StatType::kDevVendor)),
                             kDeviceVendorNvidia);

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  OpStats op_stats;
  TF_CHECK_OK(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot_or.value(), OpStatsOptions(), &op_stats));
  const RunEnvironment& run_env = op_stats.run_environment();

  EXPECT_EQ("Nvidia GPU", run_env.device_type());
  EXPECT_EQ(1, run_env.host_count());
  EXPECT_EQ(1, run_env.task_count());
  EXPECT_EQ(2, run_env.device_core_count());
}

TEST(ConvertXPlaneToOpStats, CpuOnlyStepDbTest) {
  constexpr int64_t kStepNum = 123;
  constexpr int64_t kStepId = 0;

  auto space = std::make_unique<XSpace>();
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(space.get()));
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, kStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kStepId}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, kStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 70);

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  OpStats op_stats;
  TF_CHECK_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                   options, &op_stats));
  const StepDatabaseResult& step_db = op_stats.step_db();

  EXPECT_EQ(step_db.step_sequence_size(), 1);
}

TEST(ConvertXPlaneToOpStats, GpuStepDbTest) {
  constexpr int64_t kStepNum = 123;
  constexpr int64_t kStepId = 0;
  constexpr int64_t kCorrelationId = 100;

  auto space = std::make_unique<XSpace>();
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(space.get()));
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, kStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kStepId}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 20,
               {{StatType::kStepId, kStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 10,
               {{StatType::kCorrelationId, kCorrelationId}});

  XPlaneBuilder device_plane_builder(
      GetOrCreateGpuXPlane(space.get(), /*device_ordinal=*/0));
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 50, 40,
               {{StatType::kCorrelationId, kCorrelationId}});

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  OpStats op_stats;
  TF_CHECK_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                   options, &op_stats));
  const StepDatabaseResult& step_db = op_stats.step_db();

  EXPECT_EQ(step_db.step_sequence_size(), 1);

  PrecisionStats precision_stats =
      op_stats.device_op_metrics_db().precision_stats();
  EXPECT_EQ(precision_stats.compute_16bit_ps(), 0);
  EXPECT_EQ(precision_stats.compute_32bit_ps(), 40);
}

TEST(ConvertXPlaneToOpStats, PropagateAndDedupErrors) {
  XSpace space;
  static constexpr char kError[] = "host: error";
  *space.add_errors() = kError;
  *space.add_errors() = kError;

  OpStats op_stats = ConvertXSpaceToOpStats(space, OpStatsOptions());

  EXPECT_EQ(1, op_stats.diagnostics().errors_size());
  EXPECT_EQ(kError, op_stats.diagnostics().errors(/*index=*/0));
}

TEST(ConvertXPlaneToOpStats, Hostnames) {
  XSpace space;
  static constexpr char kHost[] = "host1";
  *space.add_hostnames() = kHost;

  OpStats op_stats = ConvertXSpaceToOpStats(space, OpStatsOptions());
  EXPECT_EQ(
      kHost,
      op_stats.core_id_to_details().at(kDefaultGpuLocalCoreId).hostname());
}

void BuildXSpaceForTest(XSpace& xspace, absl::string_view hostname) {
  constexpr int64_t kStepNum = 123;
  constexpr int64_t kStepId = 456;
  // Create a host only XSpace for test.
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&xspace));
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, kStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kStepId}});

  auto executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, kStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kStepId}});
  // Create a TensorFlow op that runs for 70 ps.
  CreateXEvent(&host_plane_builder, &executor_thread, "aaa:bbb", 30, 70);
  xspace.add_hostnames(std::string(hostname));
}

TEST(ConvertXPlaneToOpStats, TestConvertMultiXSpacesToCombinedOpStats) {
  static constexpr char kHost1[] = "host1";
  static constexpr char kHost2[] = "host2";

  auto xspace1 = std::make_unique<XSpace>();
  auto xspace2 = std::make_unique<XSpace>();

  BuildXSpaceForTest(*xspace1, kHost1);
  BuildXSpaceForTest(*xspace2, kHost2);

  std::vector<std::string> xspace_paths;
  xspace_paths.push_back("host1.pb");
  xspace_paths.push_back("host2.pb");

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(xspace1));
  xspaces.push_back(std::move(xspace2));

  auto session_snapshot_or =
      SessionSnapshot::Create(std::move(xspace_paths), std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;

  TF_CHECK_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                   options, &combined_op_stats))
      << "Failed to convert multi XSpace to OpStats";

  // Result OpStats has 2 Host Ops, "IDLE" and "aaa:bbb".
  ASSERT_EQ(combined_op_stats.host_op_metrics_db().metrics_db_size(), 2);
  const auto& metric = combined_op_stats.host_op_metrics_db().metrics_db(1);
  EXPECT_EQ(metric.name(), "aaa");
  EXPECT_EQ(metric.category(), "bbb");
  // Each host has the HostOp "aaa:bbb" running for 70 ps, so the combined
  // OpStats has "aaa:bbb" running for 140 ps in total.
  EXPECT_EQ(metric.self_time_ps(), 140);

  // Result OpStats has 1 step, 2 cores.
  ASSERT_EQ(combined_op_stats.step_db().step_sequence_size(), 1);
  ASSERT_EQ(
      combined_op_stats.step_db().step_sequence(0).step_info_per_core_size(),
      2);
  const auto& step_info_per_core =
      combined_op_stats.step_db().step_sequence(0).step_info_per_core();
  // global_core_id is computed using: 1000 * host_id + local_core_id.
  EXPECT_TRUE(step_info_per_core.contains(kDefaultGpuLocalCoreId));
  EXPECT_TRUE(step_info_per_core.contains(1000 + kDefaultGpuLocalCoreId));

  const auto& core_details_map = combined_op_stats.core_id_to_details();
  EXPECT_EQ(kHost1, core_details_map.at(kDefaultGpuLocalCoreId).hostname());
  EXPECT_EQ(kHost2,
            core_details_map.at(1000 + kDefaultGpuLocalCoreId).hostname());
}

TEST(ConvertXPlaneToOpStats, RunEnvironmentExtractedFromTpuPlane) {
  XSpace xspace;
  for (int i : {0, 1, 2, 3}) {
    GetOrCreateTpuXPlane(&xspace, i, "TPU V4", 0, 0);
  }

  OpStats op_stats = ConvertXSpaceToOpStats(xspace, OpStatsOptions());

  EXPECT_EQ(op_stats.run_environment().device_type(), "TPU V4");
  EXPECT_EQ(op_stats.run_environment().device_core_count(), 4);
}

TEST(ConvertXPlaneToOpStats, TpuPerfEnv) {
  auto space = std::make_unique<XSpace>();
  constexpr double kMaxError = 0.01;
  constexpr int kClockRateKHz = 1530000;
  constexpr int kCoreCount = 80;
  constexpr uint64 kMemoryBandwidthBytesPerSecond =
      uint64{900} * 1000 * 1000 * 1000;
  // Volta.
  constexpr int kComputeCapMajor = 7;
  constexpr int kComputeCapMinor = 0;
  constexpr double kDevCapPeakTeraflopsPerSecond = 141.0;
  constexpr double kDevCapPeakHbmBwGigabytesPerSecond = 900.0;
  constexpr double kDevCapPeakSramRdBwGigabytesPerSecond = 101.0;
  constexpr double kDevCapPeakSramWrBwGigabytesPerSecond = 102.0;
  constexpr double kDevCapPeakCmemRdBwGigabytesPerSecond = 101.0;
  constexpr double kDevCapPeakCmemWrBwGigabytesPerSecond = 102.0;
  constexpr double kDevCapPeakVmemRdBwGigabytesPerSecond = 201.0;
  constexpr double kDevCapPeakVmemWrBwGigabytesPerSecond = 202.0;

  XPlaneBuilder device_plane(GetOrCreateTpuXPlane(
      space.get(), /*device_ordinal=*/0, "TPU V4",
      kDevCapPeakTeraflopsPerSecond, kDevCapPeakHbmBwGigabytesPerSecond));
  /*device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kDevVendor)),
                        kDeviceVendorNvidia); // "Google, Inc.");*/
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata("clock_rate"),
                            kClockRateKHz);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata("core_count"),
                            kCoreCount);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("memory_bandwidth"),
      kMemoryBandwidthBytesPerSecond);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_major"),
      kComputeCapMajor);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_minor"),
      kComputeCapMinor);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_sram_rd_bw_gigabytes_per_second"),
                            kDevCapPeakSramRdBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_sram_wr_bw_gigabytes_per_second"),
                            kDevCapPeakSramWrBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_cmem_rd_bw_gigabytes_per_second"),
                            kDevCapPeakCmemRdBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_cmem_wr_bw_gigabytes_per_second"),
                            kDevCapPeakCmemWrBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_vmem_rd_bw_gigabytes_per_second"),
                            kDevCapPeakVmemRdBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_vmem_wr_bw_gigabytes_per_second"),
                            kDevCapPeakVmemWrBwGigabytesPerSecond);

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  OpStats op_stats;
  TF_CHECK_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                   options, &op_stats));
  const PerfEnv& perf_env = op_stats.perf_env();
  EXPECT_NEAR(kDevCapPeakTeraflopsPerSecond,
              perf_env.peak_tera_flops_per_second(), kMaxError);
  EXPECT_NEAR(
      kDevCapPeakHbmBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_HBM_RW),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakSramRdBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_RD),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakSramWrBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_WR),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakCmemRdBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_CMEM_RD),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakCmemWrBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_CMEM_WR),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakVmemRdBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_VMEM_RD),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakVmemWrBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_VMEM_WR),
      kMaxError);
  EXPECT_NEAR(156.67, perf_env.ridge_point(), kMaxError);
}

TEST(ConvertXPlaneToOpStats, TpuRunEnvironment) {
  auto space = std::make_unique<XSpace>();
  XPlaneBuilder device_plane1(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/0, "TPU V4", 0, 0));
  XPlaneBuilder device_plane2(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/1, "TPU V4", 0, 0));

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  OpStats op_stats;
  TF_CHECK_OK(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot_or.value(), OpStatsOptions(), &op_stats));
  const RunEnvironment& run_env = op_stats.run_environment();

  EXPECT_EQ("TPU V4", run_env.device_type());
  EXPECT_EQ(1, run_env.host_count());
  EXPECT_EQ(1, run_env.task_count());
  EXPECT_EQ(2, run_env.device_core_count());
}

TEST(ConvertXPlaneToOpStats, TpuDeviceTraceToStepDb) {
  auto space = std::make_unique<XSpace>();
  constexpr double kDevCapPeakTeraflopsPerSecond = 141.0;
  constexpr double kDevCapPeakHbmBwGigabytesPerSecond = 1000.0;
  XPlaneBuilder xplane_builder(GetOrCreateTpuXPlane(
      space.get(), /*device_ordinal=*/0, "TPU V4",
      kDevCapPeakTeraflopsPerSecond, kDevCapPeakHbmBwGigabytesPerSecond));

  XEventMetadata* event_metadata = xplane_builder.GetOrCreateEventMetadata(1);
  event_metadata->set_name("op_name");
  XStatsBuilder<XEventMetadata> stats(event_metadata, &xplane_builder);

  stats.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kProgramId)),
                     1);
  stats.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kSymbolId)),
                     1);
  stats.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kSelfDurationPs)),
                     10);
  stats.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      "tf_op_name");
  stats.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kHloCategory)),
                     "category");
  XLineBuilder line = xplane_builder.GetOrCreateLine(1);
  line.SetName(kTensorFlowOpLineName);
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetNs(0);
  event.SetDurationNs(10);

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  OpStats op_stats;
  TF_CHECK_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                   options, &op_stats));
  EXPECT_THAT(op_stats.device_op_metrics_db().metrics_db(),
              UnorderedElementsAre(Property(&OpMetrics::name, "op_name"),
                                   Property(&OpMetrics::name, "IDLE")));
}

// Verifies that the step db is generated correctly by intersecting for
// multi-device TPU.
TEST(ConvertXPlaneToOpStats, TpuMultiDeviceStepDbTest) {
  auto space = std::make_unique<XSpace>();

  XPlaneBuilder device_plane_builder1(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/0, "TPU V4", 0, 0));
  XPlaneBuilder device_plane_builder2(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/1, "TPU V4", 0, 0));
  device_plane_builder1.ReserveLines(1);
  device_plane_builder2.ReserveLines(1);

  // Create 1 step in xplane in TPU ordinal 0.
  XStatMetadata* kGroupId1 = device_plane_builder1.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kGroupId));
  XLineBuilder line = device_plane_builder1.GetOrCreateLine(1);
  line.SetName(kXlaOpLineName);
  // Step 1
  XEventMetadata* event_metadata =
      device_plane_builder1.GetOrCreateEventMetadata(1);
  event_metadata->set_name("Step 1");
  XEventBuilder event_builder = line.AddEvent(*event_metadata);
  event_builder.AddStatValue(*kGroupId1, 1);  // step num
  event_builder.SetDurationNs(100);
  event_builder.SetOffsetNs(100);

  // Create 2 steps in xplane in TPU ordinal 1.
  line = device_plane_builder2.GetOrCreateLine(1);
  line.SetName(kXlaOpLineName);
  // Step 1
  XStatMetadata* kGroupId2 = device_plane_builder2.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kGroupId));
  XEventMetadata* event_metadata2 =
      device_plane_builder2.GetOrCreateEventMetadata(2);
  event_metadata2->set_name("Step 1");
  XEventBuilder event_builder2 = line.AddEvent(*event_metadata2);
  event_builder2.AddStatValue(*kGroupId2, 1);  // step num
  event_builder2.SetDurationNs(100);
  event_builder2.SetOffsetNs(300);
  // Step 2
  XStatMetadata* kGroupId3 = device_plane_builder2.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kGroupId));
  XEventMetadata* event_metadata3 =
      device_plane_builder2.GetOrCreateEventMetadata(2);
  event_metadata3->set_name("Step 2");
  XEventBuilder event_builder3 = line.AddEvent(*event_metadata3);
  event_builder3.AddStatValue(*kGroupId3, 2);  // step num
  event_builder3.SetDurationNs(100);
  event_builder3.SetOffsetNs(300);

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats op_stats = ConvertXSpaceToOpStats(*space, options);
  const StepDatabaseResult& step_db = op_stats.step_db();
  // For TPU step events, we intersect the step events by step num across
  // different TPU devices.
  EXPECT_EQ(step_db.step_sequence_size(), 1);
}

TEST(ConvertXPlaneToOpStats, ConstructDutyCycleTrackerFromXlaOps) {
  XSpace space;
  XPlane* device_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  XLineBuilder op_line = device_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kXlaOpLineName);
  CreateXEvent(&device_plane_builder, &op_line, "op.1", /*offset_ps=*/10,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloInfeed}});
  CreateXEvent(&device_plane_builder, &op_line, "op.2", /*offset_ps=*/20,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloCall}});
  CreateXEvent(&device_plane_builder, &op_line, "op.3", /*offset_ps=*/30,
               /*duration_ps=*/10);
  CreateXEvent(&device_plane_builder, &op_line, "op.4", /*offset_ps=*/40,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloOutfeed}});

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  DutyCycleTracker tracker = ConstructDutyCycleTracker(visitor);
  EXPECT_EQ(tracker.GetActiveTimePs(), 20);
  EXPECT_EQ(tracker.GetIdleTimePs(), 20);
}

TEST(ConvertXPlaneToOpStats, ConstructDutyCycleTrackerFromSparseCore) {
  XSpace space;
  XPlane* sc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder sc_plane_builder(sc_plane);
  XLineBuilder op_line = sc_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(&sc_plane_builder, &op_line, "op.1", /*offset_ps=*/10,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &op_line, "op.2", /*offset_ps=*/20,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &op_line, "op.3", /*offset_ps=*/30,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &op_line, "op.4", /*offset_ps=*/40,
               /*duration_ps=*/10);
  XLineBuilder module_line = sc_plane_builder.GetOrCreateLine(1);
  module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &module_line, "module.1", /*offset_ps=*/5,
               /*duration_ps=*/50);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(sc_plane);
  DutyCycleTracker tracker = ConstructDutyCycleTracker(visitor);
  EXPECT_EQ(tracker.GetActiveTimePs(), 40);
  EXPECT_EQ(tracker.GetIdleTimePs(), 10);
}

TEST(ConvertXPlaneToOpStats, MultiCoreChipBusyAndIdleTimeTest) {
  XSpace space;
  CoreDetails tc_core_details;
  tc_core_details.set_local_chip_id(0);
  CoreDetails sc_core_details;
  sc_core_details.set_local_chip_id(0);
  XPlane* tc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder tc_plane_builder(tc_plane);
  tc_plane_builder.AddStatValue(*tc_plane_builder.GetOrCreateStatMetadata(
                                    GetStatTypeStr(StatType::kCoreDetails)),
                                tc_core_details);
  XLineBuilder xla_op_line = tc_plane_builder.GetOrCreateLine(0);
  xla_op_line.SetName(kXlaOpLineName);
  CreateXEvent(&tc_plane_builder, &xla_op_line, "op.1", /*offset_ps=*/10,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloInfeed}});
  CreateXEvent(&tc_plane_builder, &xla_op_line, "op.2", /*offset_ps=*/20,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloCall}});
  CreateXEvent(&tc_plane_builder, &xla_op_line, "op.3", /*offset_ps=*/30,
               /*duration_ps=*/10);
  CreateXEvent(&tc_plane_builder, &xla_op_line, "op.4", /*offset_ps=*/40,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloOutfeed}});

  XPlane* sc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/1, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder sc_plane_builder(sc_plane);
  sc_plane_builder.AddStatValue(*sc_plane_builder.GetOrCreateStatMetadata(
                                    GetStatTypeStr(StatType::kCoreDetails)),
                                sc_core_details);
  XLineBuilder sc_op_line = sc_plane_builder.GetOrCreateLine(0);
  sc_op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "op.1", /*offset_ps=*/10,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "op.2", /*offset_ps=*/20,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "op.3", /*offset_ps=*/30,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "op.4", /*offset_ps=*/40,
               /*duration_ps=*/10);
  XLineBuilder sc_module_line = sc_plane_builder.GetOrCreateLine(1);
  sc_module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.1", /*offset_ps=*/5,
               /*duration_ps=*/50);

  OpStats op_stats = ConvertXSpaceToOpStats(space, OpStatsOptions());
  EXPECT_EQ(op_stats.device_op_metrics_db().idle_time_ps(), 10);
  EXPECT_EQ(op_stats.device_op_metrics_db().busy_time_ps(), 40);
}

TEST(ConvertXPlaneToOpStats, HandleSparseCoreBusyOpMetrics) {
  XSpace space;
  XPlane* tc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder tc_plane_builder(tc_plane);
  tc_plane_builder.SetId(0);
  XLineBuilder tc_step_line = tc_plane_builder.GetOrCreateLine(0);
  tc_step_line.SetName(tsl::profiler::kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "step.1", /*offset_ps=*/10,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&tc_plane_builder, &tc_step_line, "step.2", /*offset_ps=*/20,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&tc_plane_builder, &tc_step_line, "step.3", /*offset_ps=*/30,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&tc_plane_builder, &tc_step_line, "step.4", /*offset_ps=*/40,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{4}}});
  XLineBuilder tc_module_line = tc_plane_builder.GetOrCreateLine(1);
  tc_module_line.SetName(tsl::profiler::kXlaModuleLineName);
  CreateXEvent(&tc_plane_builder, &tc_module_line, "module.1", /*offset_ps=*/10,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&tc_plane_builder, &tc_module_line, "module.2", /*offset_ps=*/20,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&tc_plane_builder, &tc_module_line, "module.3", /*offset_ps=*/30,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&tc_plane_builder, &tc_module_line, "module.4", /*offset_ps=*/40,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{4}}});
  XLineBuilder tc_op_line = tc_plane_builder.GetOrCreateLine(2);
  tc_op_line.SetName(kXlaOpLineName);
  auto& program_id_stat = *tc_plane_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kProgramId));
  auto& symbol_id_stat = *tc_plane_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kSymbolId));
  XStatsBuilder<XEventMetadata> op1_stats(
      tc_plane_builder.GetOrCreateEventMetadata("op.1"), &tc_plane_builder);
  op1_stats.AddStatValue(program_id_stat, 1);
  op1_stats.AddStatValue(symbol_id_stat, 1);
  XStatsBuilder<XEventMetadata> op2_stats(
      tc_plane_builder.GetOrCreateEventMetadata("op.2"), &tc_plane_builder);
  op2_stats.AddStatValue(program_id_stat, 1);
  op2_stats.AddStatValue(symbol_id_stat, 2);
  XStatsBuilder<XEventMetadata> op3_stats(
      tc_plane_builder.GetOrCreateEventMetadata("op.3"), &tc_plane_builder);
  op3_stats.AddStatValue(program_id_stat, 1);
  op3_stats.AddStatValue(symbol_id_stat, 3);
  XStatsBuilder<XEventMetadata> op4_stats(
      tc_plane_builder.GetOrCreateEventMetadata("op.4"), &tc_plane_builder);
  op4_stats.AddStatValue(program_id_stat, 1);
  op4_stats.AddStatValue(symbol_id_stat, 4);
  CreateXEvent(&tc_plane_builder, &tc_op_line, "op.1", /*offset_ps=*/15,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "op.2", /*offset_ps=*/25,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "op.3", /*offset_ps=*/35,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "op.4", /*offset_ps=*/45,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{4}}});
  XPlane* sc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/1, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder sc_plane_builder(sc_plane);
  sc_plane_builder.SetId(1);
  sc_plane_builder.SetName(
      absl::StrCat(sc_plane->name(), " SparseCore ", sc_plane->id()));
  XLineBuilder sc_step_line = sc_plane_builder.GetOrCreateLine(0);
  sc_step_line.SetName(tsl::profiler::kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "step.1", /*offset_ps=*/10,
               /*duration_ps=*/10,
               {{StatType::kStepIdleTimePs, int64_t{5}},
                {StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "step.2", /*offset_ps=*/20,
               /*duration_ps=*/10,
               {{StatType::kStepIdleTimePs, int64_t{5}},
                {StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "step.3", /*offset_ps=*/30,
               /*duration_ps=*/10,
               {{StatType::kStepIdleTimePs, int64_t{5}},
                {StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "step.4", /*offset_ps=*/40,
               /*duration_ps=*/10,
               {{StatType::kStepIdleTimePs, int64_t{5}},
                {StatType::kGroupId, int64_t{4}}});
  XLineBuilder sc_module_line = sc_plane_builder.GetOrCreateLine(1);
  sc_module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.1", /*offset_ps=*/10,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.2", /*offset_ps=*/20,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.3", /*offset_ps=*/30,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.4", /*offset_ps=*/40,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{4}}});
  XLineBuilder sc_op_line = sc_plane_builder.GetOrCreateLine(2);
  sc_op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "scs op.1", /*offset_ps=*/15,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "scs op.2", /*offset_ps=*/25,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "scs op.3", /*offset_ps=*/35,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "scs op.4", /*offset_ps=*/45,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{4}}});
  OpStats op_stats = ConvertXSpaceToOpStats(
      space,
      OpStatsOptions{.generate_op_metrics_db = true, .generate_step_db = true});
  EXPECT_EQ(op_stats.device_op_metrics_db().total_time_ps(), 40);
  EXPECT_EQ(op_stats.device_op_metrics_db().total_op_time_ps(), 20);
  EXPECT_EQ(op_stats.step_db().step_sequence_size(), 4);
  EXPECT_EQ(op_stats.hlo_metrics_db_complete_steps_only().total_time_ps(), 40);
  EXPECT_EQ(op_stats.hlo_metrics_db_complete_steps_only().total_op_time_ps(),
            20);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
