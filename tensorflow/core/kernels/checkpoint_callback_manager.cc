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
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tsl/platform/regexp.h"

namespace tensorflow {
namespace checkpoint {

const absl::string_view kCheckpointCallbackManagerResourceName =
    "checkpoint_callback_manager";

namespace {

const absl::string_view kCheckpointFileRegex = "^part-[0-9]*-of-[0-9]*";
const absl::string_view kCheckpointTempDirRegex = "-[0-9]*_temp$";
const absl::string_view kCheckpointDirRegex = "-[0-9]*$";
const absl::string_view kCheckpointTempDirSuffix = "_temp";

void TriggerSaveCallbackIfFileNotExist(absl::string_view checkpoint_id,
                                       absl::string_view checkpoint_dir,
                                       absl::string_view file_extension,
                                       SaveCallback callback) {
  const std::string file_path = io::JoinPath(
      checkpoint_dir, absl::StrCat(checkpoint_id, ".", file_extension));

  // If the file already exists, we are done.
  if (Env::Default()->FileExists(file_path).ok()) {
    return;
  }
  LOG(INFO) << "Calling a save callback: file_extension = " << file_extension
            << ", checkpoint_id = " << checkpoint_id;
  // The callback should return a string to store.
  absl::StatusOr<std::string> save_content = callback(checkpoint_id);
  if (!save_content.ok()) {
    LOG(WARNING) << save_content.status();
    return;
  }

  // An empty string means nothing to be saved.
  if (save_content->empty()) {
    return;
  }

  absl::Status write_status =
      WriteStringToFile(Env::Default(), file_path, *save_content);
  if (!write_status.ok()) {
    LOG(WARNING) << write_status;
  } else {
    LOG(INFO) << "A CheckpointCallbackManager has been written to "
              << file_path;
  }
}

void TriggerRestoreCallbackIfFileExists(absl::string_view checkpoint_id,
                                        absl::string_view checkpoint_dir,
                                        absl::string_view file_extension,
                                        RestoreCallback callback) {
  const std::string file_path = io::JoinPath(
      checkpoint_dir, absl::StrCat(checkpoint_id, ".", file_extension));
  if (!Env::Default()->FileExists(file_path).ok()) {
    return;
  }
  std::string payload;
  absl::Status read_status =
      ReadFileToString(Env::Default(), file_path, &payload);
  if (!read_status.ok()) {
    LOG(WARNING) << "Failed to read: " << read_status;
    return;
  }

  LOG(INFO) << "Calling a restore callback: file_extension = " << file_extension
            << ", checkpoint_id = " << checkpoint_id;
  absl::Status callback_status = callback(checkpoint_id, payload);
  if (!callback_status.ok()) {
    LOG(WARNING) << callback_status;
  }
}

}  // namespace

//  Examples:
//    "/foo/bar/checkpoint-1_temp/part-00000-of-00001" -->
//        ("checkpoint-1", "/foo/bar");
//    "/foo/bar/checkpoint-2/part-00000-of-00001" -->
//        ("checkpoint-2", "/foo/bar");
//    "/foo/bar/checkpoint-3" --> ("checkpoint-3", "/foo/bar");
//    "/foo/bar"              --> NotFound error
absl::StatusOr<std::pair<std::string, std::string>>
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

absl::Status CheckpointCallbackManager::RegisterSaveCallback(
    absl::string_view file_extension, SaveCallback callback) {
  SaveCallback lazy_callback = nullptr;
  std::string checkpoint_id;
  std::string checkpoint_dir;
  {
    mutex_lock l(mu_);
    if (!save_callbacks_.try_emplace(file_extension, std::move(callback))
             .second) {
      return errors::AlreadyExists("A callback already exists.");
    }

    // If last_saved_checkpoint_id_and_dir_ is not empty,
    // tries to trigger save callback lazily.
    if (!last_saved_checkpoint_id_and_dir_.first.empty()) {
      lazy_callback = save_callbacks_[file_extension];
      checkpoint_id = last_saved_checkpoint_id_and_dir_.first;
      checkpoint_dir = last_saved_checkpoint_id_and_dir_.second;
    }
  }

  if (lazy_callback != nullptr) {
    TriggerSaveCallbackIfFileNotExist(checkpoint_id, checkpoint_dir,
                                      file_extension, lazy_callback);
  }
  return absl::OkStatus();
}

bool CheckpointCallbackManager::DoesSaveCallbackExist(
    absl::string_view file_extension) {
  tf_shared_lock l(mu_);
  return save_callbacks_.contains(file_extension);
}

absl::Status CheckpointCallbackManager::RegisterRestoreCallback(
    absl::string_view file_extension, RestoreCallback callback) {
  RestoreCallback lazy_callback = nullptr;
  std::string checkpoint_id;
  std::string checkpoint_dir;
  {
    mutex_lock l(mu_);
    if (!restore_callbacks_.try_emplace(file_extension, std::move(callback))
             .second) {
      return errors::AlreadyExists("A callback already exists.");
    }

    // If last_restored_checkpoint_id_and_dir_ is not empty,
    // tries to trigger restore callback lazily.
    if (!last_restored_checkpoint_id_and_dir_.first.empty()) {
      lazy_callback = restore_callbacks_[file_extension];
      checkpoint_id = last_restored_checkpoint_id_and_dir_.first;
      checkpoint_dir = last_restored_checkpoint_id_and_dir_.second;
    }
  }

  if (lazy_callback != nullptr) {
    TriggerRestoreCallbackIfFileExists(checkpoint_id, checkpoint_dir,
                                       file_extension, lazy_callback);
  }
  return absl::OkStatus();
}

bool CheckpointCallbackManager::DoesRestoreCallbackExist(
    absl::string_view file_extension) {
  tf_shared_lock l(mu_);
  return restore_callbacks_.contains(file_extension);
}

void CheckpointCallbackManager::Save(absl::string_view prefix) {
  absl::StatusOr<std::pair<std::string, std::string>> id_and_dir =
      GetCheckpointIdAndPathFromPrefix(prefix);
  if (!id_and_dir.ok()) {
    return;
  }

  // Create a copy to avoid holding lock while calling a callback.
  absl::flat_hash_map<std::string, SaveCallback> copy_of_save_callbacks;
  {
    mutex_lock l(mu_);
    last_saved_checkpoint_id_and_dir_ = *id_and_dir;
    copy_of_save_callbacks = save_callbacks_;
  }

  for (const auto& name_and_callback : copy_of_save_callbacks) {
    TriggerSaveCallbackIfFileNotExist(id_and_dir->first, id_and_dir->second,
                                      name_and_callback.first,
                                      name_and_callback.second);
  }
}

void CheckpointCallbackManager::Restore(absl::string_view prefix) {
  absl::StatusOr<std::pair<std::string, std::string>> id_and_dir =
      GetCheckpointIdAndPathFromPrefix(prefix);
  if (!id_and_dir.ok()) {
    return;
  }

  // Create a copy to avoid holding lock while calling a callback.
  absl::flat_hash_map<std::string, RestoreCallback> copy_of_restore_callbacks;
  {
    mutex_lock l(mu_);
    if (*id_and_dir == last_restored_checkpoint_id_and_dir_) {
      // We don't want to trigger restore callback function multiple times.
      return;
    }
    last_restored_checkpoint_id_and_dir_ = *id_and_dir;
    copy_of_restore_callbacks = restore_callbacks_;
  }

  for (const auto& name_and_callback : copy_of_restore_callbacks) {
    TriggerRestoreCallbackIfFileExists(id_and_dir->first, id_and_dir->second,
                                       name_and_callback.first,
                                       name_and_callback.second);
  }
}

}  // namespace checkpoint
}  // namespace tensorflow
