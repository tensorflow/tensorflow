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

#include "absl/time/time.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
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
    const std::string& dispatcher_address,
    const std::string& dispatcher_protocol, const std::string& worker_address,
    const SnapshotTaskDef& snapshot_task, int64_t source_index,
    absl::Duration timeout, Env* env)
    : dispatcher_address_(dispatcher_address),
      dispatcher_protocol_(dispatcher_protocol),
      worker_address_(worker_address),
      snapshot_task_(snapshot_task),
      source_index_(source_index),
      timeout_(timeout),
      env_(env) {}

Status SnapshotSplitProvider::GetNext(Tensor* split, bool* end_of_splits)
    TF_LOCKS_EXCLUDED(mu_) {
  // TODO(b/258691097): Do not need to read from file multiple times.
  StatusOr<std::string> split_file = GetSplitFilename();
  if (split_file.ok()) {
    return GetSplitFromFile(*split_file, split, end_of_splits);
  }
  if (!errors::IsNotFound(split_file.status())) {
    return split_file.status();
  }
  return GetSplitFromDispatcher(split, end_of_splits);
}

StatusOr<std::string> SnapshotSplitProvider::GetSplitFilename() const
    TF_LOCKS_EXCLUDED(mu_) {
  std::string splits_directory = SourceDirectory(
      snapshot_task_.base_path(), snapshot_task_.stream_index(), source_index_);
  std::vector<std::string> split_filenames;
  TF_RETURN_IF_ERROR(env_->GetChildren(splits_directory, &split_filenames));

  mutex_lock l(mu_);
  for (const std::string& split_filename : split_filenames) {
    TF_ASSIGN_OR_RETURN(auto split_index, SplitIndex(split_filename));
    auto [local_split_index, global_split_index] = split_index;
    if (local_split_index == next_split_index_) {
      return tsl::io::JoinPath(splits_directory, split_filename);
    }
  }
  return errors::NotFound("No split file found for split ", next_split_index_,
                          " in directory ", splits_directory);
}

Status SnapshotSplitProvider::GetSplitFromFile(const std::string& split_file,
                                               Tensor* split,
                                               bool* end_of_splits)
    TF_LOCKS_EXCLUDED(mu_) {
  VLOG(3) << "Getting the next split from file: " << split_file;
  snapshot_util::TFRecordReader reader(split_file, tsl::io::compression::kNone,
                                       DataTypeVector{1, DT_VARIANT});
  std::vector<Tensor> tensors;
  TF_RETURN_IF_ERROR(reader.Initialize(env_));
  TF_RETURN_IF_ERROR(reader.ReadTensors(&tensors));
  if (tensors.size() != 1) {
    return errors::Internal(
        "A snapshot split file is expected to contain 1 tensor. Got ",
        tensors.size(), " tensors from ", split_file, ".");
  }
  *split = std::move(tensors[0]);
  *end_of_splits = false;
  mutex_lock l(mu_);
  ++next_split_index_;
  return OkStatus();
}

Status SnapshotSplitProvider::GetSplitFromDispatcher(Tensor* split,
                                                     bool* end_of_splits)
    TF_LOCKS_EXCLUDED(mu_) {
  VLOG(3) << "Getting the next split from dispatcher at "
          << dispatcher_address_;
  mutex_lock l(mu_);
  if (!dispatcher_) {
    dispatcher_ = std::make_unique<DataServiceDispatcherClient>(
        dispatcher_address_, dispatcher_protocol_);
  }
  // TODO(b/258691097): Checks the local_split_index and read splits from disk
  // if the local_split_index is unexpected.
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
  ++next_split_index_;
  return OkStatus();
}

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
  return OkStatus();
}

Status SnapshotSplitProvider::Reset() {
  return errors::FailedPrecondition(
      "tf.data SnapshotSplitProvider does not support `Reset`.");
}

}  // namespace data
}  // namespace tensorflow
