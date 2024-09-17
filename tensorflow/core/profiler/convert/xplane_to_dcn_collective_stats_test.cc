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

#include "tensorflow/core/profiler/convert/xplane_to_dcn_collective_stats.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tsl/platform/status.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

DcnSlackAnalysis CreateDcnSlackAnalysisProto() {
  DcnSlackAnalysis dcn_slack_analysis;
  DcnSlackSummary* dcn_slack_summary =
      dcn_slack_analysis.add_dcn_slack_summary();
  dcn_slack_summary->set_rendezvous("collective");
  dcn_slack_summary->set_recv_op_name("recv-done");
  dcn_slack_summary->set_send_op_name("send");
  dcn_slack_summary->set_slack_us(2);
  dcn_slack_summary->set_observed_duration_us(12);
  dcn_slack_summary->set_stall_duration_us(5);
  dcn_slack_summary->set_occurrences(4);
  dcn_slack_summary->set_bytes_transmitted_over_network(819200);
  return dcn_slack_analysis;
}

SessionSnapshot CreateSessionSnapshot(bool create_cache_file,
                                      bool has_dcn_collective_stats) {
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  std::string path = absl::StrCat("ram://", test_name, "/");
  std::unique_ptr<WritableFile> xplane_file;
  std::vector<std::string> paths = {absl::StrCat(path, "hostname.xplane.pb")};

  auto xspace = std::make_unique<XSpace>();
  XPlane* xplane = FindOrAddMutablePlaneWithName(xspace.get(), "/host:CPU");
  if (has_dcn_collective_stats) {
    XPlaneBuilder xplane_builder(xplane);
    xplane_builder.GetOrCreateEventMetadata("MegaScale:");
  }

  if (create_cache_file) {
    if (has_dcn_collective_stats) {
      tensorflow::Env::Default()
          ->NewAppendableFile(
              absl::StrCat(path, "hostname.dcn_collective_stats.pb"),
              &xplane_file)
          .IgnoreError();
      tensorflow::Env::Default()
          ->NewAppendableFile(
              absl::StrCat(path, "ALL_HOSTS.dcn_collective_stats.pb"),
              &xplane_file)
          .IgnoreError();
    } else {
      tensorflow::Env::Default()
          ->NewAppendableFile(
              absl::StrCat(path, "NO_HOST.dcn_collective_stats.pb"),
              &xplane_file)
          .IgnoreError();
    }
  }

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(xspace));

  absl::StatusOr<SessionSnapshot> session_snapshot_status =
      SessionSnapshot::Create(paths, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_status.status());
  SessionSnapshot session_snapshot = std::move(session_snapshot_status.value());
  if (has_dcn_collective_stats) {
    DcnSlackAnalysis dcn_slack_analysis = CreateDcnSlackAnalysisProto();
    TF_CHECK_OK(session_snapshot.WriteBinaryProto(
        DCN_COLLECTIVE_STATS, "hostname", dcn_slack_analysis));
    TF_CHECK_OK(session_snapshot.WriteBinaryProto(
        DCN_COLLECTIVE_STATS, kAllHostsIdentifier, dcn_slack_analysis));
  }
  return session_snapshot;
}

TEST(ConvertXplaneToDcnCollectiveStats,
     HasAllHostsDcnCollectiveStatsCacheFile) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(true, true);

  absl::StatusOr<bool> status =
      HasDcnCollectiveStatsInMultiXSpace(session_snapshot);
  EXPECT_EQ(status.value(), true);
}

TEST(ConvertXplaneToDcnCollectiveStats, HasNoHostDcnCollectiveStatsCacheFile) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(true, false);

  absl::StatusOr<bool> status =
      HasDcnCollectiveStatsInMultiXSpace(session_snapshot);
  EXPECT_EQ(status.value(), false);
}

TEST(ConvertXplaneToDcnCollectiveStats,
     NoCacheFileButTraceHasDcnCollectiveStats) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, true);

  absl::StatusOr<bool> status =
      HasDcnCollectiveStatsInMultiXSpace(session_snapshot);
  EXPECT_EQ(status.value(), true);
}

TEST(ConvertXplaneToDcnCollectiveStats,
     NoCacheFileNoDcnCollectiveStatsPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, false);

  absl::StatusOr<bool> status =
      HasDcnCollectiveStatsInMultiXSpace(session_snapshot);
  EXPECT_EQ(status.value(), false);
}

TEST(ConvertXplaneToDcnCollectiveStats,
     ConvertXSpaceToDcnCollectiveStatsWhenStatsPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, true);

  absl::StatusOr<bool> status =
      ConvertMultiXSpaceToDcnCollectiveStats(session_snapshot);
  absl::StatusOr<std::optional<std::string>> all_hosts_filepath =
      session_snapshot.GetHostDataFilePath(StoredDataType::DCN_COLLECTIVE_STATS,
                                           kAllHostsIdentifier);
  absl::StatusOr<std::optional<std::string>> host_filepath =
      session_snapshot.GetHostDataFilePath(StoredDataType::DCN_COLLECTIVE_STATS,
                                           "hostname");

  EXPECT_EQ(status.value(), true);
  TF_EXPECT_OK(all_hosts_filepath.status());
  EXPECT_TRUE(all_hosts_filepath.value().has_value());
  EXPECT_FALSE(all_hosts_filepath.value().value().empty());
  TF_EXPECT_OK(host_filepath.status());
  EXPECT_TRUE(host_filepath.value().has_value());
  EXPECT_FALSE(host_filepath.value().value().empty());
}

TEST(ConvertXplaneToDcnCollectiveStats,
     ConvertXSpaceToDcnCollectiveStatsWhenStatsNotPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, false);

  absl::StatusOr<bool> status =
      ConvertMultiXSpaceToDcnCollectiveStats(session_snapshot);
  absl::StatusOr<std::optional<std::string>> filepath =
      session_snapshot.GetHostDataFilePath(StoredDataType::DCN_COLLECTIVE_STATS,
                                           kNoHostIdentifier);

  EXPECT_EQ(status.value(), false);
  TF_EXPECT_OK(filepath.status());
  EXPECT_TRUE(filepath.value().has_value());
  EXPECT_FALSE(filepath.value().value().empty());
}

TEST(ConvertXplaneToDcnCollectiveStats,
     GetHostDcnSlackAnalysisWhenStatsNotPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, false);

  absl::StatusOr<DcnSlackAnalysis> host_dcn_slack_analysis =
      GetDcnSlackAnalysisByHostName(session_snapshot, "hostname");

  TF_EXPECT_OK(host_dcn_slack_analysis.status());
  EXPECT_EQ(host_dcn_slack_analysis.value().dcn_slack_summary_size(), 0);
}

TEST(ConvertXplaneToDcnCollectiveStats,
     GetHostDcnSlackAnalysisWhenStatsPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(true, true);

  absl::StatusOr<DcnSlackAnalysis> host_dcn_slack_analysis =
      GetDcnSlackAnalysisByHostName(session_snapshot, "hostname");

  TF_EXPECT_OK(host_dcn_slack_analysis.status());
  EXPECT_EQ(host_dcn_slack_analysis.value().dcn_slack_summary_size(), 1);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
