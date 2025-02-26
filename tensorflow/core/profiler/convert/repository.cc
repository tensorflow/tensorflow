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

#include "tensorflow/core/profiler/convert/repository.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {
std::string GetHostnameByPath(absl::string_view xspace_path) {
  std::string_view file_name = tensorflow::io::Basename(xspace_path);
  // Remove suffix from file_name, preserving entire prefix.
  absl::ConsumeSuffix(&file_name, ".xplane.pb");
  return std::string(file_name);
}
}  // namespace

absl::StatusOr<SessionSnapshot> SessionSnapshot::Create(
    std::vector<std::string> xspace_paths,
    std::optional<std::vector<std::unique_ptr<XSpace>>> xspaces) {
  if (xspace_paths.empty()) {
    return errors::InvalidArgument("Can not find XSpace path.");
  }

  if (xspaces.has_value()) {
    if (xspaces->size() != xspace_paths.size()) {
      return errors::InvalidArgument(
          "The size of the XSpace paths: ", xspace_paths.size(),
          " is not equal ",
          "to the size of the XSpace proto: ", xspaces->size());
    }
    for (size_t i = 0; i < xspace_paths.size(); ++i) {
      auto host_name = GetHostnameByPath(xspace_paths.at(i));
      if (xspaces->at(i)->hostnames_size() > 0 && !host_name.empty()) {
        if (!absl::StrContains(host_name, xspaces->at(i)->hostnames(0))) {
          return errors::InvalidArgument(
              "The hostname of xspace path and preloaded xpace don't match at "
              "index: ",
              i, ". \nThe host name of xpace path is ", host_name,
              " but the host name of preloaded xpace is ",
              xspaces->at(i)->hostnames(0), ".");
        }
      }
    }
  }

  return SessionSnapshot(std::move(xspace_paths), std::move(xspaces));
}

absl::StatusOr<std::unique_ptr<XSpace>> SessionSnapshot::GetXSpace(
    size_t index) const {
  if (index >= xspace_paths_.size()) {
    return errors::InvalidArgument("Can not get the ", index,
                                   "th XSpace. The total number of XSpace is ",
                                   xspace_paths_.size());
  }

  // Return the pre-loaded XSpace proto.
  if (xspaces_.has_value()) {
    if (xspaces_->at(index) == nullptr) {
      return errors::Internal("");
    }
    return std::move(xspaces_->at(index));
  }

  // Return the XSpace proto from file.
  auto xspace_from_file = std::make_unique<XSpace>();
  TF_RETURN_IF_ERROR(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                                 xspace_paths_.at(index),
                                                 xspace_from_file.get()));
  return xspace_from_file;
}

absl::StatusOr<std::unique_ptr<XSpace>> SessionSnapshot::GetXSpaceByName(
    absl::string_view name) const {
  if (auto it = hostname_map_.find(name); it != hostname_map_.end()) {
    return GetXSpace(it->second);
  }

  return errors::InvalidArgument("Can not find the XSpace by name: ", name,
                                 ". The total number of XSpace is ",
                                 xspace_paths_.size());
}

std::string SessionSnapshot::GetHostname(size_t index) const {
  return GetHostnameByPath(xspace_paths_.at(index));
}

std::optional<std::string> SessionSnapshot::GetFilePath(
    absl::string_view toolname, absl::string_view hostname) const {
  if (!has_accessible_run_dir_) return std::nullopt;
  std::string file_name = "";
  if (toolname == "trace_viewer@")
    file_name = absl::StrCat(hostname, ".", "SSTABLE");
  if (!file_name.empty())
    return tensorflow::io::JoinPath(session_run_dir_, file_name);
  return std::nullopt;
}

absl::StatusOr<std::string> SessionSnapshot::GetHostDataFileName(
    const StoredDataType data_type, const std::string host) const {
  for (const auto& format : *kHostDataSuffixes) {
    if (data_type == format.first) return absl::StrCat(host, format.second);
  }
  return absl::InternalError(&"Unknown StoredDataType: "[data_type]);
}

absl::StatusOr<std::optional<std::string>> SessionSnapshot::GetHostDataFilePath(
    const StoredDataType data_type, const std::string host) const {
  // Gets all the files in session run directory.
  std::vector<std::string> results;
  TF_RETURN_IF_ERROR(::tsl::Env::Default()->GetChildren(
      std::string(GetSessionRunDir()), &results));

  TF_ASSIGN_OR_RETURN(std::string filename,
                      GetHostDataFileName(data_type, host));

  for (const std::string& path : results) {
    if (absl::EndsWith(path, filename)) {
      return ::tsl::profiler::ProfilerJoinPath(GetSessionRunDir(), filename);
    }
  }

  return std::nullopt;
}

absl::StatusOr<std::pair<bool, std::string>> SessionSnapshot::HasCacheFile(
    const StoredDataType data_type) const {
  std::optional<std::string> filepath;
  TF_ASSIGN_OR_RETURN(filepath,
                      GetHostDataFilePath(data_type, kNoHostIdentifier));
  if (filepath) {
    // cache file is present but file contains no data_type events
    return std::pair<bool, std::string>(true, std::string());
  }

  TF_ASSIGN_OR_RETURN(filepath,
                      GetHostDataFilePath(data_type, kAllHostsIdentifier));
  if (filepath) {
    // cache file is present and file contains data_type events
    return std::pair<bool, std::string>(true, filepath.value());
  }

  // no cache file present
  return std::pair<bool, std::string>(false, std::string());
}

}  // namespace profiler
}  // namespace tensorflow
