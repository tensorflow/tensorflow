/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/checkpoint_callback_manager.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace checkpoint {

const absl::string_view kCheckpointCallbackManagerResourceName =
    "checkpoint_callback_manager";

namespace {

const absl::string_view kCheckpointFileRegex = "^part-[0-9]*-of-[0-9]*$";
const absl::string_view kCheckpointTempDirRegex = "-[0-9]*_temp$";
const absl::string_view kCheckpointDirRegex = "-[0-9]*$";
const absl::string_view kCheckpointTempDirSuffix = "_temp";

}  // namespace

//  Examples:
//    "/foo/bar/checkpoint-1_temp/part-00000-of-00001" -->
//        ("checkpoint-1", "/foo/bar");
//    "/foo/bar/checkpoint-2/part-00000-of-00001" -->
//        ("checkpoint-2", "/foo/bar");
//    "/foo/bar/checkpoint-3" --> ("checkpoint-3", "/foo/bar");
//    "/foo/bar"              --> NotFound error
StatusOr<std::pair<std::string, std::string>>
CheckpointCallbackManager::GetCheckpointIdAndPathFromPrefix(
    absl::string_view prefix) {
  for (absl::string_view path = prefix;; path = io::Dirname(path)) {
    absl::string_view basename = io::Basename(path);

    // Failed to find checkpoint_id
    if (basename.empty()) break;

    // Skip known checkpoint file: e.g., part-00000-of-00001
    if (RE2::PartialMatch(basename, kCheckpointFileRegex)) continue;

    // With _temp suffix: e.g., checkpoint-1_temp
    if (RE2::PartialMatch(basename, kCheckpointTempDirRegex)) {
      // Trim suffix, "_temp".
      return std::make_pair(
          std::string(basename.substr(
              0, basename.length() - kCheckpointTempDirSuffix.length())),
          std::string(io::Dirname(path)));
    }

    // Without _temp suffix: e.g., checkpoint-1
    if (RE2::PartialMatch(basename, kCheckpointDirRegex)) {
      return std::make_pair(std::string(basename),
                            std::string(io::Dirname(path)));
    }
  }
  return errors::NotFound(
      absl::StrCat("Failed to find a checkpoint id. prefix = ", prefix));
}

Status CheckpointCallbackManager::RegisterSaveCallback(
    absl::string_view file_extension, SaveCallback callback) {
  return save_callbacks_.try_emplace(file_extension, std::move(callback)).second
             ? Status::OK()
             : errors::AlreadyExists("A callback already exists.");
}

bool CheckpointCallbackManager::DoesSaveCallbackExist(
    absl::string_view file_extension) const {
  return save_callbacks_.contains(file_extension);
}

Status CheckpointCallbackManager::RegisterRestoreCallback(
    absl::string_view file_extension, RestoreCallback callback) {
  return restore_callbacks_.try_emplace(file_extension, std::move(callback))
                 .second
             ? Status::OK()
             : errors::AlreadyExists("A callback already exists.");
}

bool CheckpointCallbackManager::DoesRestoreCallbackExist(
    absl::string_view file_extension) const {
  return restore_callbacks_.contains(file_extension);
}

void CheckpointCallbackManager::Save(absl::string_view prefix) {
  StatusOr<std::pair<std::string, std::string>> id_and_dir =
      GetCheckpointIdAndPathFromPrefix(prefix);
  if (!id_and_dir.ok()) {
    LOG(WARNING) << id_and_dir.status();
    return;
  }

  for (const auto& name_and_callback : save_callbacks_) {
    const std::string file_path = io::JoinPath(
        id_and_dir->second,
        absl::StrCat(id_and_dir->first, ".", name_and_callback.first));

    // If the file already exists, we are done.
    if (Env::Default()->FileExists(file_path).ok()) {
      continue;
    }

    LOG(INFO) << "Calling a save callback: file_extension = "
              << name_and_callback.first
              << ", checkpoint_id = " << id_and_dir->first;
    // The callback should return a string to store.
    StatusOr<std::string> save_content =
        name_and_callback.second(id_and_dir->first);
    if (!save_content.ok()) {
      LOG(WARNING) << save_content.status();
      continue;
    }

    Status write_status =
        WriteStringToFile(Env::Default(), file_path, *save_content);
    if (!write_status.ok()) {
      LOG(WARNING) << write_status;
    } else {
      LOG(INFO) << "A CheckpointCallbackManager has been written to "
                << file_path;
    }
  }
}

void CheckpointCallbackManager::Restore(absl::string_view prefix) {
  StatusOr<std::pair<std::string, std::string>> id_and_dir =
      GetCheckpointIdAndPathFromPrefix(prefix);
  if (!id_and_dir.ok()) {
    LOG(WARNING) << id_and_dir.status();
    return;
  }

  for (const auto& name_and_callback : restore_callbacks_) {
    const std::string file_path = io::JoinPath(
        id_and_dir->second,
        absl::StrCat(id_and_dir->first, ".", name_and_callback.first));
    if (!Env::Default()->FileExists(file_path).ok()) {
      continue;
    }
    std::string payload;
    Status read_status = ReadFileToString(Env::Default(), file_path, &payload);
    if (!read_status.ok()) {
      LOG(WARNING) << "Failed to read: " << read_status;
      continue;
    }

    LOG(INFO) << "Calling a restore callback: file_extension = "
              << name_and_callback.first
              << ", checkpoint_id = " << id_and_dir->first;
    Status callback_status =
        name_and_callback.second(id_and_dir->first, payload);
    if (!callback_status.ok()) {
      LOG(WARNING) << callback_status;
    }
  }
}

}  // namespace checkpoint
}  // namespace tensorflow
