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

#include "tensorflow/core/profiler/convert/xplane_to_hlo.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/utils/hlo_proto_map.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

namespace {

using tsl::profiler::ProfilerJoinPath;

constexpr char kNoModuleIdentifier[] = "NO_MODULE";
constexpr char kHloProtoSuffix[] = ".hlo_proto.pb";

// Extracts and deduplicates the HLO protos from all the XSpaces.
// Stores the HLO protos as files in the same directory as the xspace files.
absl::StatusOr<bool> GetHloProtoFromMultiXSpaceAndSaveToFile(
    const SessionSnapshot& session_snapshot) {
  // Get all HLO protos from XSpaces and deduplicate.
  HloProtoMap hlo_proto_map;
  for (int i = 0; i < session_snapshot.XSpaceSize(); i++) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                        session_snapshot.GetXSpace(i));
    hlo_proto_map.AddHloProtosFromXSpace(*xspace);
  }

  std::vector<absl::string_view> module_list = hlo_proto_map.GetModuleList();
  // Write an empty identifier if there is no HLO module.
  if (module_list.empty()) {
    std::string file_name =
        ProfilerJoinPath(session_snapshot.GetSessionRunDir(),
                         absl::StrCat(kNoModuleIdentifier, kHloProtoSuffix));
    xla::HloProto empty_hlo;
    TF_RETURN_IF_ERROR(tensorflow::WriteBinaryProto(tensorflow::Env::Default(),
                                                    file_name, empty_hlo));
    // The profile does not have HLO proto.
    return false;
  }

  // Save HLO protos to session run directory.
  for (const absl::string_view module_name : module_list) {
    auto hlo_proto_or = hlo_proto_map.GetHloProtoByModuleName(module_name);
    if (!hlo_proto_or.ok()) {
      return errors::Internal(hlo_proto_or.status().message());
    }
    std::string file_name =
        ProfilerJoinPath(session_snapshot.GetSessionRunDir(),
                         absl::StrCat(module_name, kHloProtoSuffix));
    TF_RETURN_IF_ERROR(tensorflow::WriteBinaryProto(
        tensorflow::Env::Default(), file_name, *hlo_proto_or.value()));
  }

  // The profile has HLO proto.
  return true;
}

}  // namespace

absl::StatusOr<xla::HloProto> GetHloProtoByModuleName(
    const SessionSnapshot& session_snapshot,
    const absl::string_view module_name) {
  std::string file_name =
      ProfilerJoinPath(session_snapshot.GetSessionRunDir(),
                       absl::StrCat(module_name, kHloProtoSuffix));
  xla::HloProto hlo_proto;
  TF_RETURN_IF_ERROR(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                                 file_name, &hlo_proto));
  return hlo_proto;
}

absl::StatusOr<bool> ConvertMultiXSpaceToHloProto(
    const SessionSnapshot& session_snapshot) {
  // Gets all the files in session run directory.
  // TODO(profiler): Move this glob to SessionSnapshot and build a map from file
  // type to file paths.
  std::vector<std::string> results;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->GetChildren(
      std::string(session_snapshot.GetSessionRunDir()), &results));

  // If the profiler finds a filename with hlo proto suffix, this means HLO
  // proto was already generated previously.
  for (const std::string& path : results) {
    if (absl::EndsWith(path, kHloProtoSuffix)) {
      if (absl::EndsWith(path,
                         absl::StrCat(kNoModuleIdentifier, kHloProtoSuffix))) {
        return false;
      } else {
        return true;
      }
    }
  }

  // Generate HLO proto.
  // TODO(jiesun): Maybe generate a tag file at profile collection time, so
  // don't need to read XSpace files for checking whether HLO proto exists or
  // not.
  return GetHloProtoFromMultiXSpaceAndSaveToFile(session_snapshot);
}

}  // namespace profiler
}  // namespace tensorflow
