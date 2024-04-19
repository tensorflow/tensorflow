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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_REPOSITORY_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_REPOSITORY_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/statusor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/utils/file_system_utils.h"

namespace tensorflow {
namespace profiler {

constexpr char kAllHostsIdentifier[] = "ALL_HOSTS";
constexpr char kNoHostIdentifier[] = "NO_HOST";

enum StoredDataType {
  DCN_COLLECTIVE_STATS,
};

static auto* kHostDataSuffixes =
    new std::vector<std::pair<StoredDataType, const char*>>(
        {{StoredDataType::DCN_COLLECTIVE_STATS, ".dcn_collective_stats.pb"}});

// File system directory snapshot of a profile session.
class SessionSnapshot {
 public:
  // Performs validation and creates SessionSnapshot.
  // <xspace_paths> are the file paths to XSpace protos.
  // Optionally, <xspaces> can contain the XSpace protos pre-loaded by the
  // profiler plugin.
  static absl::StatusOr<SessionSnapshot> Create(
      std::vector<std::string> xspace_paths,
      std::optional<std::vector<std::unique_ptr<XSpace>>> xspaces);

  // Returns the number of XSpaces in the profile session.
  size_t XSpaceSize() const { return xspace_paths_.size(); }

  // Gets XSpace proto.
  // The caller of this function will take ownership of the XSpace.
  absl::StatusOr<std::unique_ptr<XSpace>> GetXSpace(size_t index) const;

  // Gets XSpace proto.
  // The caller of this function will take ownership of the XSpace.
  absl::StatusOr<std::unique_ptr<XSpace>> GetXSpaceByName(
      absl::string_view name) const;

  // Gets host name.
  std::string GetHostname(size_t index) const;

  // Gets the run directory of the profile session.
  absl::string_view GetSessionRunDir() const { return session_run_dir_; }

  // Gets whether the session has an accessible run dir. If false, any
  // path-based file read will be disabled in this mode.
  bool HasAccessibleRunDir() const { return has_accessible_run_dir_; }

  // Gets the path of the fast file for a given tool.
  std::optional<std::string> GetFilePath(absl::string_view toolname,
                                         absl::string_view host) const;

  // Gets the name of the host data file.
  absl::StatusOr<std::string> GetHostDataFileName(StoredDataType data_type,
                                                  std::string host) const;

  // Gets the path of the host data file.
  absl::StatusOr<std::optional<std::string>> GetHostDataFilePath(
      StoredDataType data_type, std::string host) const;

  /* Gets whether the cache file is present in run dir. First value indicates
  whether cache file is present or not. Second value indicates the path of cache
  file. Possible cases are:
      1. <false, "">: If no cache file is present
      2. <true, "">: If cache file is present but file contains no data_type
     events
      3. <true, filepath>: If cache file is present and file contains data_type
     events
  */
  absl::StatusOr<std::pair<bool, std::string>> HasCacheFile(
      StoredDataType data_type) const;

  template <typename T>
  absl::Status WriteBinaryProto(const StoredDataType data_type,
                                const std::string host, T& proto) const {
    // Gets name for host data file.
    TF_ASSIGN_OR_RETURN(std::string filename,
                        GetHostDataFileName(data_type, host));

    std::string filepath =
        tsl::profiler::ProfilerJoinPath(GetSessionRunDir(), filename);

    return tensorflow::WriteBinaryProto(tsl::Env::Default(), filepath, proto);
  }

  template <typename T>
  absl::Status ReadBinaryProto(const StoredDataType data_type,
                               const std::string host, T* proto) const {
    // Gets file path for host data.
    TF_ASSIGN_OR_RETURN(std::optional<std::string> filepath,
                        GetHostDataFilePath(data_type, host));
    if (filepath) {
      return tensorflow::ReadBinaryProto(tsl::Env::Default(), filepath.value(),
                                         proto);
    }

    return absl::NotFoundError(
        absl::StrCat("No binary proto found for ", host, " and ", data_type));
  }

 private:
  SessionSnapshot(std::vector<std::string> xspace_paths,
                  std::optional<std::vector<std::unique_ptr<XSpace>>> xspaces)
      : xspace_paths_(std::move(xspace_paths)),
        // If the snapshot was initialized by xspaces, the file path and run dir
        // is a path tensorflow can't read from or write to so any file IO
        // encapsulated in this class will be disabled in this mode.
        has_accessible_run_dir_(!xspaces.has_value()),
        xspaces_(std::move(xspaces)) {
    session_run_dir_ = tensorflow::io::Dirname(xspace_paths_.at(0));
    for (size_t i = 0; i < xspace_paths_.size(); ++i) {
      std::string host_name = GetHostname(i);
      hostname_map_[host_name] = i;
    }
  }

  // File paths to XSpace protos.
  std::vector<std::string> xspace_paths_;
  // The run directory of the profile session.
  absl::string_view session_run_dir_;

  absl::flat_hash_map<std::string /*host_name*/, size_t /*index*/>
      hostname_map_;

  const bool has_accessible_run_dir_;

  // XSpace protos pre-loaded by the profiler plugin.
  // TODO(profiler): Use blobstore paths to initialize SessionSnapshot instead
  // of using pre-loaded XSpaces.
  mutable std::optional<std::vector<std::unique_ptr<XSpace>>> xspaces_;
};

// Writes binary proto format T for a host and data_type to a session.
template <typename T>
absl::Status WriteBinaryProto(const SessionSnapshot& session_snapshot,
                              const StoredDataType data_type,
                              const std::string& host, T& proto) {
  return session_snapshot.WriteBinaryProto(data_type, host, proto);
}

// Reads binary proto format T for a host and data_type to a session.
template <typename T>
absl::Status ReadBinaryProto(const SessionSnapshot& session_snapshot,
                             const StoredDataType data_type,
                             const std::string& host, T* proto) {
  return session_snapshot.ReadBinaryProto(data_type, host, proto);
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_REPOSITORY_H_
