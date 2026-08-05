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

#include "xla/backends/autotuner/directory_store.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {

DirectoryStore::DirectoryStore(std::string directory_path, CacheMode mode)
    : directory_path_(std::move(directory_path)), mode_(mode) {}

std::string DirectoryStore::GetFilePath(
    const autotuner::AutotuneTargetKey& target_key) const {
  std::string file_name = absl::StrCat(target_key.hlo_fingerprint(), ".pb");
  if (target_key.explicit_version().empty()) {
    return tsl::io::JoinPath(directory_path_, target_key.device(), file_name);
  }
  return tsl::io::JoinPath(directory_path_, target_key.device(),
                           target_key.explicit_version(), file_name);
}

absl::StatusOr<std::vector<autotuner::AutotuneEntry>> DirectoryStore::Read(
    const autotuner::AutotuneTargetKey& target_key) {
  std::string path = GetFilePath(target_key);
  tsl::Env* env = tsl::Env::Default();
  if (!env->FileExists(path).ok()) {
    return std::vector<autotuner::AutotuneEntry>{};
  }

  std::string content;
  RETURN_IF_ERROR(tsl::ReadFileToString(env, path, &content));

  autotuner::AutotuneEntry entry;
  if (!entry.ParseFromString(content)) {
    return absl::InternalError(
        absl::StrCat("Failed to parse cache entry from file: ", path));
  }

  return std::vector<autotuner::AutotuneEntry>{entry};
}

absl::Status DirectoryStore::Write(const autotuner::AutotuneEntry& entry) {
  if (mode_ == CacheMode::kReadOnly) {
    return absl::OkStatus();
  }

  std::string path = GetFilePath(entry.key().target());
  tsl::Env* env = tsl::Env::Default();

  std::string dir(tsl::io::Dirname(path));
  RETURN_IF_ERROR(env->RecursivelyCreateDir(dir));

  std::string content;
  if (!entry.SerializeToString(&content)) {
    return absl::InternalError("Failed to serialize autotune entry.");
  }

  // Rename trick: Write to a temporary file, then rename it to the final file
  // to avoid mingled files when multiple threads are writing to the same file.
  std::string tmp_dir = tsl::io::JoinPath(directory_path_, "tmp");
  RETURN_IF_ERROR(env->RecursivelyCreateDir(tmp_dir));
  std::string tmp_path = tsl::io::JoinPath(
      tmp_dir, absl::StrCat("tmp_", absl::GetCurrentTimeNanos()));

  RETURN_IF_ERROR(tsl::WriteStringToFile(env, tmp_path, content));
  return env->RenameFile(tmp_path, path);
}

absl::StatusOr<std::vector<autotuner::AutotuneEntry>>
DirectoryStore::ReadAll() {
  return absl::UnimplementedError("Not implemented.");
}

}  // namespace xla
