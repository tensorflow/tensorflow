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
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/utils.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/regexp.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace data {
namespace {

StatusOr<int64_t> GetCheckpointIndex(const std::string& checkpoint_name) {
  // The checkpoint names end in "checkpoint_<chunk_index>".
  static constexpr LazyRE2 kFilenameRe = {R"(checkpoint_(\d+)$)"};
  int64_t chunk_index = 0;
  if (!RE2::PartialMatch(checkpoint_name, *kFilenameRe, &chunk_index)) {
    return errors::Internal("Invalid checkpoint name: ", checkpoint_name);
  }
  return chunk_index;
}

}  // namespace

constexpr int64_t SnapshotWriterParams::kDefaultMaxChunkSizeBytes;

SnapshotStreamWriter::SnapshotStreamWriter(
    const SnapshotWriterParams& params, std::unique_ptr<TaskIterator> iterator)
    : params_(params),
      iterator_(std::move(iterator)),
      snapshot_thread_(RunSnapshotThread()) {
  DCHECK_NE(iterator_, nullptr);
}

Status SnapshotStreamWriter::Wait() TF_LOCKS_EXCLUDED(mu_) {
  snapshot_thread_.reset();
  mutex_lock l(mu_);
  return status_;
}

std::unique_ptr<Thread> SnapshotStreamWriter::RunSnapshotThread() {
  auto snapshot_fn = [this]() TF_LOCKS_EXCLUDED(mu_) {
    Status status = WriteSnapshotFn();
    if (!status.ok()) {
      mutex_lock l(mu_);
      status_ = std::move(status);
    }
  };
  return absl::WrapUnique(params_.env->StartThread(
      /*thread_options=*/{}, /*name=*/"tf_data_service_snapshot_thread",
      std::move(snapshot_fn)));
}

Status SnapshotStreamWriter::WriteSnapshotFn() TF_LOCKS_EXCLUDED(mu_) {
  TF_RETURN_IF_ERROR(InitializeDirectories());
  TF_RETURN_IF_ERROR(Restore());
  while (ShouldWriteChunk()) {
    TF_RETURN_IF_ERROR(WriteChunk());
  }
  mutex_lock l(mu_);
  return status_;
}

Status SnapshotStreamWriter::InitializeDirectories() {
  TF_RETURN_IF_ERROR(params_.env->RecursivelyCreateDir(
      UncommittedChunksDirectory(params_.snapshot_path, params_.stream_id)));
  TF_RETURN_IF_ERROR(params_.env->RecursivelyCreateDir(
      CheckpointsDirectory(params_.snapshot_path, params_.stream_id)));
  return OkStatus();
}

bool SnapshotStreamWriter::ShouldWriteChunk() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return !end_of_sequence_ && status_.ok();
}

Status SnapshotStreamWriter::WriteChunk() {
  std::string chunk_file_path = GetChunkFilePath();
  snapshot_util::TFRecordWriter writer(chunk_file_path, params_.compression);
  TF_RETURN_IF_ERROR(writer.Initialize(params_.env));
  while (ShouldWriteRecord()) {
    TF_RETURN_IF_ERROR(WriteRecord(writer));
  }
  TF_RETURN_IF_ERROR(writer.Close());
  return CommitChunk(chunk_file_path);
}

std::string SnapshotStreamWriter::GetChunkFilePath() const {
  return tsl::io::JoinPath(
      UncommittedChunksDirectory(params_.snapshot_path, params_.stream_id),
      absl::StrCat("chunk_", chunk_index_));
}

Status SnapshotStreamWriter::CommitChunk(const std::string& chunk_file_path) {
  std::string chunk_basename(tsl::io::Basename(chunk_file_path));
  std::string committed_chunk_filename = tsl::io::JoinPath(
      CommittedChunksDirectory(params_.snapshot_path), chunk_basename);
  // Writes the checkpoint before committing the chunk. If the worker fails in
  // between, the restarted worker will synchronize the checkpoint with the
  // committed chunks.
  if (ShouldSave()) {
    TF_RETURN_IF_ERROR(Save());
  }
  TF_RETURN_IF_ERROR(
      params_.env->RenameFile(chunk_file_path, committed_chunk_filename));
  ++chunk_index_;
  chunk_size_bytes_ = 0;
  return OkStatus();
}

bool SnapshotStreamWriter::ShouldWriteRecord() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return chunk_size_bytes_ < params_.max_chunk_size_bytes &&
         !end_of_sequence_ && status_.ok();
}

Status SnapshotStreamWriter::WriteRecord(
    snapshot_util::TFRecordWriter& writer) {
  std::vector<Tensor> element;
  TF_RETURN_IF_ERROR(iterator_->GetNext(element, end_of_sequence_));
  if (end_of_sequence_) {
    return writer.Close();
  }
  TF_RETURN_IF_ERROR(writer.WriteTensors(element));
  chunk_size_bytes_ += EstimatedSizeBytes(element);
  return OkStatus();
}

void SnapshotStreamWriter::Cancel() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  status_ = errors::Cancelled(
      "The tf.data service snapshot writer has been cancelled.");
}

bool SnapshotStreamWriter::ShouldSave() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return !end_of_sequence_ && status_.ok();
}

Status SnapshotStreamWriter::Save() {
  // TODO(b/258691097): Delete previous checkpoints.
  std::string uncommitted_checkpoint_path;
  if (!params_.env->LocalTempFilename(&uncommitted_checkpoint_path)) {
    return errors::Internal(
        "Failed to create temp files for distributed snapshot checkpoints.");
  }
  std::string committed_checkpoint_path = CheckpointPath(chunk_index_);

  snapshot_util::TFRecordWriter writer(uncommitted_checkpoint_path,
                                       params_.compression);
  TF_RETURN_IF_ERROR(writer.Initialize(params_.env));
  TF_ASSIGN_OR_RETURN(Tensor serialized, iterator_->Save());
  TF_RETURN_IF_ERROR(writer.WriteTensors({serialized}));
  TF_RETURN_IF_ERROR(writer.Close());
  return params_.env->RenameFile(uncommitted_checkpoint_path,
                                 committed_checkpoint_path);
}

Status SnapshotStreamWriter::Restore() {
  StatusOr<int64_t> checkpoint_index = LastCheckpointIndex();
  if (errors::IsNotFound(checkpoint_index.status())) {
    // No checkpoint has been written. Does not restore anything.
    return OkStatus();
  }
  TF_RETURN_IF_ERROR(checkpoint_index.status());

  std::string checkpoint_path = CheckpointPath(*checkpoint_index);
  snapshot_util::TFRecordReader reader(checkpoint_path, params_.compression,
                                       DataTypeVector{1, DT_VARIANT});
  TF_RETURN_IF_ERROR(reader.Initialize(params_.env));
  std::vector<Tensor> serialized_tensors;
  TF_RETURN_IF_ERROR(reader.ReadTensors(&serialized_tensors));
  if (serialized_tensors.size() != 1) {
    return errors::Internal(
        "A snapshot checkpoint file is expected to contain 1 Tensor. Got ",
        serialized_tensors.size(),
        " tensors from checkpoint file: ", checkpoint_path);
  }
  TF_RETURN_IF_ERROR(iterator_->Restore(serialized_tensors[0]));
  TF_RETURN_IF_ERROR(SyncCheckpointWithChunks());
  chunk_index_ = *checkpoint_index + 1;
  return OkStatus();
}

StatusOr<int64_t> SnapshotStreamWriter::LastCheckpointIndex() const {
  std::string checkpoints_directory =
      CheckpointsDirectory(params_.snapshot_path, params_.stream_id);
  std::vector<std::string> checkpoint_names;
  TF_RETURN_IF_ERROR(
      params_.env->GetChildren(checkpoints_directory, &checkpoint_names));
  if (checkpoint_names.empty()) {
    return errors::NotFound("No checkpoint has been written in directory ",
                            checkpoints_directory);
  }

  int64_t last_index = 0;
  for (const std::string& checkpoint_name : checkpoint_names) {
    TF_ASSIGN_OR_RETURN(int64_t checkpoint_index,
                        GetCheckpointIndex(checkpoint_name));
    last_index = std::max(last_index, checkpoint_index);
  }
  return last_index;
}

Status SnapshotStreamWriter::SyncCheckpointWithChunks() {
  // TODO(b/258691097): Commit uncommitted chunk files written before the
  // checkpoint and delete chunk files written after the checkpoint.
  return OkStatus();
}

std::string SnapshotStreamWriter::CheckpointPath(int64_t chunk_index) const {
  return tsl::io::JoinPath(
      CheckpointsDirectory(params_.snapshot_path, params_.stream_id),
      absl::StrCat("checkpoint_", chunk_index));
}
}  // namespace data
}  // namespace tensorflow
