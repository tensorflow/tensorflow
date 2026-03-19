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
#include "tensorflow/core/data/service/snapshot/snapshot_split_provider.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/path.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNextSplitIndex[] = "next_split_index";
constexpr char kRepetitionIndex[] = "repetition_index";

absl::StatusOr<int64_t> GetRepetitionIndex(const std::string& split_file) {
  absl::string_view repetition_dir_path = tsl::io::Dirname(split_file);
  absl::string_view repetition_dir_name =
      tsl::io::Basename(repetition_dir_path);
  return ParseRepetitionDirectoryName(repetition_dir_name);
}
}  // namespace

SnapshotSplitProvider::SnapshotSplitProvider(
    const std::string& worker_address, const SnapshotTaskDef& snapshot_task,
    int64_t source_index, absl::Duration timeout,
    std::unique_ptr<DataServiceDispatcherClient> dispatcher, Env* env)
    : worker_address_(worker_address),
      snapshot_task_(snapshot_task),
      source_index_(source_index),
      timeout_(timeout),
      env_(env) {
  mutex_lock l(mu_);
  dispatcher_ = std::move(dispatcher);
}

absl::Status SnapshotSplitProvider::GetNext(Tensor* split, bool* end_of_splits)
    TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(GetAndValidateSplit(split, end_of_splits));
  if (!*end_of_splits) {
    ++next_split_index_;
  }
  return absl::OkStatus();
}

absl::Status SnapshotSplitProvider::GetAndValidateSplit(Tensor* split,
                                                        bool* end_of_splits)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (split_to_file_map_.contains(next_split_index_)) {
    return GetSplitFromFile(split_to_file_map_[next_split_index_], split,
                            end_of_splits);
  }

  TF_ASSIGN_OR_RETURN(int64_t dispatcher_split_index,
                      GetSplitFromDispatcher(split, end_of_splits));
  if (dispatcher_split_index == next_split_index_) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(split_to_file_map_, GetSplitsFiles(next_split_index_));
  TF_RETURN_IF_ERROR(ValidateSplitFiles(split_to_file_map_, next_split_index_,
                                        dispatcher_split_index,
                                        *end_of_splits));
  return GetSplitFromFile(split_to_file_map_[next_split_index_], split,
                          end_of_splits);
}

absl::Status SnapshotSplitProvider::GetSplitFromFile(
    const std::string& split_file, Tensor* split, bool* end_of_splits)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(3) << "Getting the next split from file: " << split_file;
  TF_ASSIGN_OR_RETURN(int64_t repetition_index, GetRepetitionIndex(split_file));
  if (repetition_index_ < repetition_index) {
    *end_of_splits = true;
    return absl::OkStatus();
  }
  snapshot_util::TFRecordReaderImpl reader(split_file,
                                           tsl::io::compression::kNone);
  TF_RETURN_IF_ERROR(reader.Initialize(env_));
  TF_ASSIGN_OR_RETURN(std::vector<Tensor> tensors, reader.GetTensors());
  if (tensors.size() != 1) {
    return absl::InternalError(absl::StrCat(
        "A snapshot split file is expected to contain 1 tensor. Got ",
        tensors.size(), " tensors from ", split_file, "."));
  }
  *split = std::move(tensors[0]);
  *end_of_splits = false;
  return absl::OkStatus();
}

absl::StatusOr<int64_t> SnapshotSplitProvider::GetSplitFromDispatcher(
    Tensor* split, bool* end_of_splits) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t local_split_index = 0;
  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [this, split, &local_split_index, end_of_splits]()
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            return dispatcher_->GetSnapshotSplit(
                worker_address_, snapshot_task_.base_path(),
                snapshot_task_.stream_index(), source_index_, repetition_index_,
                *split, local_split_index, *end_of_splits);
          },
      "Get next split for snapshot",
      /*deadline_micros=*/env_->NowMicros() +
          absl::ToInt64Microseconds(timeout_)));
  return local_split_index;
}

absl::StatusOr<absl::btree_map<int64_t, std::string>>
SnapshotSplitProvider::GetSplitsFiles(int64_t start_index) const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  absl::btree_map<int64_t, std::string> split_to_file_map;
  std::string splits_directory = SourceDirectory(
      snapshot_task_.base_path(), snapshot_task_.stream_index(), source_index_);
  TF_ASSIGN_OR_RETURN(std::vector<std::string> repetition_directories,
                      GetChildren(splits_directory, env_));

  for (const std::string& repetition : repetition_directories) {
    std::string repetition_dir = io::JoinPath(splits_directory, repetition);
    TF_ASSIGN_OR_RETURN(std::vector<std::string> split_files,
                        GetChildren(repetition_dir, env_));
    for (const std::string& split_file : split_files) {
      TF_ASSIGN_OR_RETURN(auto split_index, ParseSplitFilename(split_file));
      auto [local_split_index, global_split_index] = split_index;
      if (local_split_index >= start_index) {
        split_to_file_map[local_split_index] =
            tsl::io::JoinPath(repetition_dir, split_file);
      }
    }
  }
  TF_RETURN_IF_ERROR(ValidateSplitFiles(split_to_file_map, start_index));
  return split_to_file_map;
}

absl::Status SnapshotSplitProvider::ValidateSplitFiles(
    const absl::btree_map<int64_t, std::string>& split_files,
    int64_t start_index) const {
  if (split_files.empty()) {
    return absl::OkStatus();
  }

  if (split_files.cbegin()->first != start_index) {
    return absl::InternalError(absl::StrCat("Failed to get split ", start_index,
                                            " for snapshot ",
                                            snapshot_task_.DebugString()));
  }

  int64_t end_index = split_files.rbegin()->first;
  if (end_index - start_index + 1 != split_files.size()) {
    return absl::InternalError(absl::StrCat(
        "Failed to get split ", start_index, ". Some splits between [",
        start_index, ", ", end_index, "] are missing for snapshot ",
        snapshot_task_.DebugString()));
  }
  return absl::OkStatus();
}

absl::Status SnapshotSplitProvider::ValidateSplitFiles(
    const absl::btree_map<int64_t, std::string>& split_files,
    int64_t start_index, int64_t end_index, bool end_of_splits) const {
  TF_RETURN_IF_ERROR(ValidateSplitFiles(split_files, start_index));
  if (end_index < start_index) {
    return absl::InternalError(absl::StrCat(
        "The tf.data service worker is expected to read split ", start_index,
        ", but the dispatcher returns split ", end_index, " for snapshot ",
        snapshot_task_.DebugString()));
  }

  if (end_of_splits) {
    // When `end_of_splits` is true, the dispatcher returns the index past the
    // the last split index. The actual `end_index` is the one before it.
    end_index = end_index - 1;
  }

  if (split_files.empty() || split_files.cbegin()->first != start_index ||
      split_files.rbegin()->first < end_index) {
    return absl::InternalError(absl::StrCat(
        "The tf.data service dispatcher has written split ", end_index,
        ". However, not all splits between [", start_index, ", ", end_index,
        "] are found for snapshot ", snapshot_task_.DebugString()));
  }
  return absl::OkStatus();
}

absl::Status SnapshotSplitProvider::Reset() {
  mutex_lock l(mu_);
  ++repetition_index_;
  LOG(INFO) << "Reset tf.data snapshot split provider for snapshot "
            << snapshot_task_.ShortDebugString() << ", repetition "
            << repetition_index_ << ".";
  return absl::OkStatus();
}

absl::Status SnapshotSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(full_name(kNextSplitIndex), next_split_index_));
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(full_name(kRepetitionIndex), repetition_index_));
  return absl::OkStatus();
}

absl::Status SnapshotSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) TF_LOCKS_EXCLUDED(mu_) {
  int64_t next_split_index = 0;
  int64_t repetition_index = 0;
  TF_RETURN_IF_ERROR(
      reader->ReadScalar(full_name(kNextSplitIndex), &next_split_index));
  TF_RETURN_IF_ERROR(
      reader->ReadScalar(full_name(kRepetitionIndex), &repetition_index));
  mutex_lock l(mu_);
  next_split_index_ = next_split_index;
  repetition_index_ = repetition_index;
  TF_ASSIGN_OR_RETURN(split_to_file_map_, GetSplitsFiles(next_split_index_));
  LOG(INFO) << "Restored snapshot split provider for snapshot "
            << snapshot_task_.ShortDebugString() << ", next split "
            << next_split_index_ << ", repetition " << repetition_index_ << ".";
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
