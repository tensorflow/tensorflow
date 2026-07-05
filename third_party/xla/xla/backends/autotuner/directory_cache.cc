/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/autotuner/directory_cache.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/time/clock.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {

namespace {

std::string GetCacheRelativePath(
    const autotuner::AutotuneTargetKey& target_key) {
  if (target_key.explicit_version().empty()) {
    return tsl::io::JoinPath(target_key.device(),
                             absl::StrCat(target_key.hlo_fingerprint(), ".pb"));
  }
  return tsl::io::JoinPath(target_key.device(), target_key.explicit_version(),
                           absl::StrCat(target_key.hlo_fingerprint(), ".pb"));
}

}  // namespace

absl::StatusOr<std::vector<autotuner::AutotuneEntry>> DirectoryCache::Read(
    const autotuner::AutotuneTargetKey& target_key) {
  std::string relative_path = GetCacheRelativePath(target_key);
  std::string file_path = tsl::io::JoinPath(directory_path_, relative_path);

  tsl::Env* env = tsl::Env::Default();
  if (!env->FileExists(file_path).ok()) {
    return std::vector<autotuner::AutotuneEntry>{};
  }

  std::string serialized_entry;
  RETURN_IF_ERROR(tsl::ReadFileToString(env, file_path, &serialized_entry));

  autotuner::AutotuneEntry entry;
  if (!entry.ParseFromString(serialized_entry)) {
    return absl::InternalError(
        absl::StrCat("Failed to parse cache entry from file: ", file_path));
  }

  return std::vector<autotuner::AutotuneEntry>{entry};
}

absl::Status DirectoryCache::Write(const autotuner::AutotuneEntry& entry) {
  std::string relative_path = GetCacheRelativePath(entry.key().target());
  std::string file_path = tsl::io::JoinPath(directory_path_, relative_path);

  tsl::Env* env = tsl::Env::Default();

  // Ensure temp directory exists.
  std::string tmp_dir = tsl::io::JoinPath(directory_path_, "tmp");
  if (!env->IsDirectory(tmp_dir).ok()) {
    RETURN_IF_ERROR(env->RecursivelyCreateDir(tmp_dir));
  }

  // Ensure the parent directory of the final destination file exists.
  std::string parent_dir = std::string(tsl::io::Dirname(file_path));
  if (!env->IsDirectory(parent_dir).ok()) {
    RETURN_IF_ERROR(env->RecursivelyCreateDir(parent_dir));
  }

  std::string serialized_entry;
  if (!entry.SerializeToString(&serialized_entry)) {
    return absl::InternalError("Failed to serialize cache entry.");
  }

  // Rename trick: Write to a temporary file, then rename it to the final file
  // to avoid mingled files when multiple threads are writing to the same file.
  // Also avoids reading incomplete files. (This may not work on all file
  // systems.)
  int64_t time_stamp = absl::GetCurrentTimeNanos();
  std::string safe_tmp_filename =
      absl::StrReplaceAll(relative_path, {{"/", "_"}});
  std::string temp_file_path = tsl::io::JoinPath(
      tmp_dir, absl::StrCat("tmp_", safe_tmp_filename, "_", time_stamp));

  RETURN_IF_ERROR(
      tsl::WriteStringToFile(env, temp_file_path, serialized_entry));
  return env->RenameFile(temp_file_path, file_path);
}

}  // namespace xla
