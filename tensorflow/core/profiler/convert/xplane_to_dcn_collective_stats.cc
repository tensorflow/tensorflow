/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tensorflow/core/profiler/convert/dcn_slack_analysis_combiner.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/xspace_to_dcn_slack_analysis.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/tensorboard_plugin_profile/protobuf/dcn_slack_analysis.pb.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {

namespace {

bool HasDcnCollectiveStatsInXSpace(const XSpace& xspace) {
  if (const tsl::profiler::XPlane* xplane =
          FindPlaneWithName(xspace, tsl::profiler::kHostThreadsPlaneName);
      xplane != nullptr) {
    for (const auto& [_, metadata] : xplane->event_metadata()) {
      if (absl::StartsWith(metadata.name(), "MegaScale:")) {
        return true;
      }
    }
  }
  return false;
}

absl::StatusOr<bool> GetDcnCollectiveStatsFromMultiXSpaceAndSaveToFile(
    const SessionSnapshot& session_snapshot) {
  DcnSlackAnalysisCombiner combiner;
  for (int idx = 0; idx < session_snapshot.XSpaceSize(); idx++) {
    std::string hostname = session_snapshot.GetHostname(idx);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                        session_snapshot.GetXSpace(idx));

    // The profile does not have dcn collective stats.
    if (!HasDcnCollectiveStatsInXSpace(*xspace)) {
      DcnSlackAnalysis dcnSlackAnalysis;
      TF_RETURN_IF_ERROR(WriteBinaryProto(session_snapshot,
                                          StoredDataType::DCN_COLLECTIVE_STATS,
                                          kNoHostIdentifier, dcnSlackAnalysis));
      return false;
    }

    DcnSlackAnalysis dcnSlackAnalysis =
        ConvertXSpaceToDcnSlackAnalysis(*xspace, nullptr, nullptr);

    TF_RETURN_IF_ERROR(WriteBinaryProto(session_snapshot,
                                        StoredDataType::DCN_COLLECTIVE_STATS,
                                        hostname, dcnSlackAnalysis));

    combiner.Combine(dcnSlackAnalysis);
  }

  DcnSlackAnalysis dcnSlackAnalysis = combiner.Finalize();
  TF_RETURN_IF_ERROR(WriteBinaryProto(session_snapshot,
                                      StoredDataType::DCN_COLLECTIVE_STATS,
                                      kAllHostsIdentifier, dcnSlackAnalysis));

  // The profile has dcn collective stats.
  return true;
}

}  // namespace

absl::StatusOr<bool> HasDcnCollectiveStatsInMultiXSpace(
    const SessionSnapshot& session_snapshot) {
  std::pair<bool, std::string> hasCacheFile;
  TF_ASSIGN_OR_RETURN(hasCacheFile, session_snapshot.HasCacheFile(
                                        StoredDataType::DCN_COLLECTIVE_STATS));

  // Cache file not present, check if trace contains dcn collective stats.
  if (!hasCacheFile.first) {
    for (int idx = 0; idx < session_snapshot.XSpaceSize(); idx++) {
      std::string hostname = session_snapshot.GetHostname(idx);
      TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                          session_snapshot.GetXSpace(idx));

      if (HasDcnCollectiveStatsInXSpace(*xspace)) {
        return true;
      }
    }
    return false;
  }

  if (hasCacheFile.second.empty()) {
    // If the profiler finds a file NO_HOST.dcn_collective_stats.pb, this means
    // dcn collective stats are not present in the profile.
    return false;
  } else {
    // If the profiler finds a file ALL_HOSTS.dcn_collective_stats.pb, this
    // means dcn collective stats are present in the profile.
    return true;
  }
}

absl::StatusOr<bool> ConvertMultiXSpaceToDcnCollectiveStats(
    const SessionSnapshot& session_snapshot) {
  std::pair<bool, std::string> hasCacheFile;
  TF_ASSIGN_OR_RETURN(hasCacheFile, session_snapshot.HasCacheFile(
                                        StoredDataType::DCN_COLLECTIVE_STATS));

  // Cache file not present, generate dcn collective stats.
  if (!hasCacheFile.first) {
    return GetDcnCollectiveStatsFromMultiXSpaceAndSaveToFile(session_snapshot);
  }

  if (hasCacheFile.second.empty()) {
    // If the profiler finds a file NO_HOST.dcn_collective_stats.pb, this means
    // dcn collective stats are not present in the profile.
    return false;
  } else {
    // If the profiler finds a file ALL_HOSTS.dcn_collective_stats.pb, this
    // means dcn collective stats are present in the profile.
    return true;
  }
}

absl::StatusOr<DcnSlackAnalysis> GetDcnSlackAnalysisByHostName(
    const SessionSnapshot& session_snapshot, const std::string hostname) {
  TF_ASSIGN_OR_RETURN(bool hasDcnCollectiveStats,
                      ConvertMultiXSpaceToDcnCollectiveStats(session_snapshot));

  DcnSlackAnalysis dcnSlackAnalysis;
  if (hasDcnCollectiveStats) {
    TF_RETURN_IF_ERROR(ReadBinaryProto(session_snapshot,
                                       StoredDataType::DCN_COLLECTIVE_STATS,
                                       hostname, &dcnSlackAnalysis));
  }

  return dcnSlackAnalysis;
}

}  // namespace profiler
}  // namespace tensorflow
