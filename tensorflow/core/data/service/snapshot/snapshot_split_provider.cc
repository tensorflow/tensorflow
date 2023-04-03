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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNextSplitIndex[] = "next_split_index";

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

Status SnapshotSplitProvider::GetNext(Tensor* split, bool* end_of_splits)
    TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(GetAndValidateSplit(split, end_of_splits));
  if (!*end_of_splits) {
    ++next_split_index_;
  }
  return OkStatus();
}

Status SnapshotSplitProvider::GetAndValidateSplit(Tensor* split,
                                                  bool* end_of_splits)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (split_to_file_map_.contains(next_split_index_)) {
    return GetSplitFromFile(split_to_file_map_[next_split_index_], split,
                            end_of_splits);
  }

  TF_ASSIGN_OR_RETURN(int64_t dispatcher_split_index,
                      GetSplitFromDispatcher(split, end_of_splits));
  if (dispatcher_split_index == next_split_index_) {
    return OkStatus();
  }

  TF_ASSIGN_OR_RETURN(split_to_file_map_, GetSplitsFiles(next_split_index_));
  TF_RETURN_IF_ERROR(ValidateSplitFiles(split_to_file_map_, next_split_index_,
                                        dispatcher_split_index,
                                        *end_of_splits));
  return GetSplitFromFile(split_to_file_map_[next_split_index_], split,
                          end_of_splits);
}

Status SnapshotSplitProvider::GetSplitFromFile(const std::string& split_file,
                                               Tensor* split,
                                               bool* end_of_splits)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(3) << "Getting the next split from file: " << split_file;
  snapshot_util::TFRecordReaderImpl reader(split_file,
                                           tsl::io::compression::kNone);
  TF_RETURN_IF_ERROR(reader.Initialize(env_));
  TF_ASSIGN_OR_RETURN(std::vector<Tensor> tensors, reader.GetTensors());
  if (tensors.size() != 1) {
    return errors::Internal(
        "A snapshot split file is expected to contain 1 tensor. Got ",
        tensors.size(), " tensors from ", split_file, ".");
  }
  *split = std::move(tensors[0]);
  *end_of_splits = false;
  return OkStatus();
}

StatusOr<int64_t> SnapshotSplitProvider::GetSplitFromDispatcher(
    Tensor* split, bool* end_of_splits) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t local_split_index = 0;
  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [this, split, &local_split_index, end_of_splits]()
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            return dispatcher_->GetSnapshotSplit(
                worker_address_, snapshot_task_.base_path(),
                snapshot_task_.stream_index(), source_index_, *split,
                local_split_index, *end_of_splits);
          },
      "Get next split for snapshot",
      /*deadline_micros=*/env_->NowMicros() +
          absl::ToInt64Microseconds(timeout_)));
  return local_split_index;
}

StatusOr<absl::btree_map<int64_t, std::string>>
SnapshotSplitProvider::GetSplitsFiles(int64_t start_index) const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::string splits_directory = SourceDirectory(
      snapshot_task_.base_path(), snapshot_task_.stream_index(), source_index_);
  absl::btree_map<int64_t, std::string> splits;

  TF_ASSIGN_OR_RETURN(std::vector<std::string> split_files,
                      GetChildren(splits_directory, env_));
  for (const std::string& split_file : split_files) {
    TF_ASSIGN_OR_RETURN(auto split_indices, SplitIndices(split_file));
    auto [local_split_index, global_split_index] = split_indices;
    if (local_split_index >= next_split_index_) {
      splits[local_split_index] =
          tsl::io::JoinPath(splits_directory, split_file);
    }
  }
  TF_RETURN_IF_ERROR(ValidateSplitFiles(splits, start_index));
  return splits;
}

Status SnapshotSplitProvider::ValidateSplitFiles(
    const absl::btree_map<int64_t, std::string>& split_files,
    int64_t start_index) const {
  if (split_files.empty()) {
    return OkStatus();
  }

  if (split_files.cbegin()->first != start_index) {
    return errors::Internal("Failed to get split ", start_index,
                            " for snapshot ", snapshot_task_.DebugString());
  }

  int64_t end_index = split_files.rbegin()->first;
  if (end_index - start_index + 1 != split_files.size()) {
    return errors::Internal("Failed to get split ", start_index,
                            ". Some splits between [", start_index, ", ",
                            end_index, "] are missing for snapshot ",
                            snapshot_task_.DebugString());
  }
  return OkStatus();
}

Status SnapshotSplitProvider::ValidateSplitFiles(
    const absl::btree_map<int64_t, std::string>& split_files,
    int64_t start_index, int64_t end_index, bool end_of_splits) const {
  TF_RETURN_IF_ERROR(ValidateSplitFiles(split_files, start_index));
  if (end_index < start_index) {
    return errors::Internal(
        "The tf.data service worker is expected to read split ", start_index,
        ", but the dispatcher returns split ", end_index, " for snapshot ",
        snapshot_task_.DebugString());
  }

  if (end_of_splits) {
    // When `end_of_splits` is true, the dispatcher returns the index past the
    // the last split index. The actual `end_index` is the one before it.
    end_index = end_index - 1;
  }

  if (split_files.empty() || split_files.cbegin()->first != start_index ||
      split_files.rbegin()->first < end_index) {
    return errors::Internal(
        "The tf.data service dispatcher has written split ", end_index,
        ". However, not all splits between [", start_index, ", ", end_index,
        "] are found for snapshot ", snapshot_task_.DebugString());
  }
  return OkStatus();
}

Status SnapshotSplitProvider::Reset() { return OkStatus(); }

Status SnapshotSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(full_name(kNextSplitIndex), next_split_index_));
  return OkStatus();
}

Status SnapshotSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) TF_LOCKS_EXCLUDED(mu_) {
  int64_t next_split_index = 0;
  TF_RETURN_IF_ERROR(
      reader->ReadScalar(full_name(kNextSplitIndex), &next_split_index));
  mutex_lock l(mu_);
  next_split_index_ = next_split_index;
  TF_ASSIGN_OR_RETURN(split_to_file_map_, GetSplitsFiles(next_split_index_));
  return OkStatus();
}

}  // namespace data
}  // namespace tensorflow
