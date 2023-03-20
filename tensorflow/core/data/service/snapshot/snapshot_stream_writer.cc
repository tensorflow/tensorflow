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
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/utils.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/regexp.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace {

constexpr int64_t kTFRecordReaderOutputBufferSize = 512 << 20;  // 512MB

// Extracts the index from `filename`. If `filename` is `prefix_<index>`, this
// returns <index>. If `filename` does not start with `prefix`, returns an
// internal error.
StatusOr<int64_t> GetFileIndex(const std::string& filename,
                               const std::string& prefix) {
  RE2 kFilenameRe(absl::StrCat(prefix, R"(_(\d+)$)"));
  int64_t index = 0;
  if (!RE2::PartialMatch(filename, kFilenameRe, &index)) {
    return errors::Internal("Failed to extract the index for file `", filename,
                            "` with prefix `", prefix, "`.");
  }
  return index;
}

}  // namespace

constexpr int64_t SnapshotWriterParams::kDefaultMaxChunkSizeBytes;

SnapshotStreamWriter::SnapshotStreamWriter(
    const SnapshotWriterParams& params, std::unique_ptr<TaskIterator> iterator)
    : params_(params), iterator_(std::move(iterator)) {
  DCHECK_NE(iterator_.get(), nullptr);
  snapshot_thread_ = absl::WrapUnique(params_.env->StartThread(
      /*thread_options=*/{}, /*name=*/"tf_data_service_snapshot_thread",
      [this]() { WriteSnapshotAndLog(); }));
}

void SnapshotStreamWriter::WriteSnapshotAndLog() TF_LOCKS_EXCLUDED(mu_) {
  if (StreamAlreadyCompleted()) {
    LOG(INFO) << "Distributed tf.data snapshot stream has already been "
              << "completed for " << params_.DebugString();
    mutex_lock l(mu_);
    completed_ = true;
    return;
  }

  LOG(INFO) << "Writing distributed tf.data snapshot stream: "
            << params_.DebugString();
  Status status = WriteSnapshot();
  if (errors::IsFailedPrecondition(status)) {
    LOG(INFO) << "Stopped writing distributed tf.data snapshot stream due to a "
                 "transient error: "
              << params_.DebugString()
              << ". It will be retried. Status: " << status;
    return;
  }
  status = FinalizeStream(status);
  mutex_lock l(mu_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to write distributed tf.data snapshot stream: "
               << params_.DebugString() << ". Status: " << status;
    completed_ = std::move(status);
    return;
  }
  LOG(INFO) << "Finished writing distributed tf.data snapshot stream: "
            << params_.DebugString();
  completed_ = true;
}

Status SnapshotStreamWriter::WriteSnapshot() TF_LOCKS_EXCLUDED(mu_) {
  // TODO(b/258691097): Write the "LEASE" file periodically.
  TF_RETURN_IF_ERROR(InitializeDirectories());
  TF_RETURN_IF_ERROR(Restore());
  while (ShouldWriteChunk()) {
    TF_RETURN_IF_ERROR(WriteChunk());
  }
  mutex_lock l(mu_);
  return completed_.status();
}

bool SnapshotStreamWriter::StreamAlreadyCompleted() const {
  std::string done_file_path =
      StreamDoneFilePath(params_.snapshot_path, params_.stream_index);
  return params_.env->FileExists(done_file_path).ok();
}

Status SnapshotStreamWriter::InitializeDirectories() {
  TF_RETURN_IF_ERROR(
      params_.env->RecursivelyCreateDir(params_.UncommittedChunksDirectory()));
  TF_RETURN_IF_ERROR(
      params_.env->RecursivelyCreateDir(params_.CheckpointsDirectory()));
  return OkStatus();
}

bool SnapshotStreamWriter::ShouldWriteChunk() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return !end_of_sequence_ && completed_.ok();
}

Status SnapshotStreamWriter::WriteChunk() {
  LOG(INFO) << "Writing distributed tf.data snapshot stream "
            << params_.stream_index << ", chunk " << chunk_index_ << ".";
  std::string chunk_file_path = GetChunkFilePath();
  snapshot_util::TFRecordWriter writer(chunk_file_path, params_.compression);
  TF_RETURN_IF_ERROR(writer.Initialize(params_.env));
  while (ShouldWriteRecord()) {
    TF_RETURN_IF_ERROR(WriteRecord(writer));
  }
  TF_RETURN_IF_ERROR(writer.Close());
  return CommitChunk();
}

Status SnapshotStreamWriter::CommitChunk() {
  // Writes the checkpoint before committing the chunk. If the worker fails in
  // between, the restarted worker will synchronize the checkpoint with the
  // committed chunks.
  if (ShouldSave()) {
    TF_RETURN_IF_ERROR(Save());
  }
  TF_RETURN_IF_ERROR(
      params_.env->RenameFile(GetChunkFilePath(), GetCommittedChunkFilePath()));
  ++chunk_index_;
  metrics::RecordTFDataServiceSnapshotBytesCommitted(chunk_size_bytes_);
  chunk_size_bytes_ = 0;
  return OkStatus();
}

std::string SnapshotStreamWriter::GetChunkFilePath() const {
  return tsl::io::JoinPath(params_.UncommittedChunksDirectory(),
                           absl::StrCat("chunk_", chunk_index_));
}

std::string SnapshotStreamWriter::GetCommittedChunkFilePath() const {
  return tsl::io::JoinPath(
      params_.CommittedChunksDirectory(),
      absl::StrCat("chunk_", params_.stream_index, "_", chunk_index_));
}

bool SnapshotStreamWriter::ShouldWriteRecord() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return chunk_size_bytes_ < params_.max_chunk_size_bytes &&
         !end_of_sequence_ && completed_.ok();
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

Status SnapshotStreamWriter::FinalizeStream(Status status) {
  if (status.ok()) {
    status = WriteDoneFile();
  }
  if (!status.ok()) {
    // If writing snapshot fails and writing the error file also fails, returns
    // the former status.
    WriteErrorFile(status).IgnoreError();
  }
  Status s = DeleteCheckpoints();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to clean up checkpoints at "
               << params_.CheckpointsDirectory() << ": " << s;
  }
  return status;
}

Status SnapshotStreamWriter::WriteDoneFile() {
  std::string done_file_path =
      StreamDoneFilePath(params_.snapshot_path, params_.stream_index);
  return AtomicallyWriteStringToFile(done_file_path, "", params_.env);
}

Status SnapshotStreamWriter::WriteErrorFile(const Status& status) {
  std::string error_file_path =
      tsl::io::JoinPath(params_.StreamDirectory(), "ERROR");
  return AtomicallyWriteStringToFile(error_file_path, status.ToString(),
                                     params_.env);
}

StatusOr<bool> SnapshotStreamWriter::Completed() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return completed_;
}

StatusOr<bool> SnapshotStreamWriter::Wait() TF_LOCKS_EXCLUDED(mu_) {
  snapshot_thread_.reset();
  mutex_lock l(mu_);
  return completed_;
}

void SnapshotStreamWriter::Cancel() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  completed_ = errors::Cancelled(
      "The tf.data service snapshot writer has been cancelled.");
}

bool SnapshotStreamWriter::ShouldSave() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  if (end_of_sequence_) {
    // If this is the last chunk, we only write checkpoints when there are more
    // than one chunk. For example, if there are 3 chunks, the files will be:
    // 1. Write checkpoint 1
    // 2. Commit chunk 1
    // 3. Write checkpoint 2
    // 4. Commit chunk 2
    // 5. Write checkpoint 3
    // 6. Commit chunk 3
    // 7. Write DONE file
    // If there is only one chunk, we do not need to write a checkpoint.
    return chunk_index_ > 0 && chunk_size_bytes_ > 0;
  }
  return completed_.ok();
}

Status SnapshotStreamWriter::Save() {
  LOG(INFO) << "Checkpointing distributed tf.data snapshot writer. Stream "
            << params_.stream_index << ", chunk " << chunk_index_
            << ", chunk size in bytes: " << chunk_size_bytes_ << ".";
  std::string checkpoint_path = CheckpointPath(chunk_index_);
  TF_ASSIGN_OR_RETURN(std::vector<Tensor> serialized_iterator,
                      iterator_->Save());
  TF_RETURN_IF_ERROR(AtomicallyWriteTFRecords(
      checkpoint_path, serialized_iterator, params_.compression, params_.env));
  return DeleteOutdatedCheckpoints();
}

Status SnapshotStreamWriter::DeleteOutdatedCheckpoints() {
  if (params_.test_only_keep_temp_files) {
    return OkStatus();
  }

  std::vector<std::string> checkpoint_filenames;
  TF_RETURN_IF_ERROR(params_.env->GetChildren(params_.CheckpointsDirectory(),
                                              &checkpoint_filenames));
  for (const std::string& checkpoint_filename : checkpoint_filenames) {
    std::string checkpoint_filepath =
        tsl::io::JoinPath(params_.CheckpointsDirectory(), checkpoint_filename);
    if (IsTemporaryFile(checkpoint_filename)) {
      TF_RETURN_IF_ERROR(params_.env->DeleteFile(checkpoint_filepath));
      continue;
    }

    TF_ASSIGN_OR_RETURN(int64_t checkpoint_index,
                        GetFileIndex(checkpoint_filename, "checkpoint"));
    if (checkpoint_index < chunk_index_) {
      TF_RETURN_IF_ERROR(params_.env->DeleteFile(checkpoint_filepath));
    }
  }
  return OkStatus();
}

Status SnapshotStreamWriter::DeleteCheckpoints() {
  if (params_.test_only_keep_temp_files) {
    return OkStatus();
  }
  if (params_.env->FileExists(params_.CheckpointsDirectory()).ok()) {
    int64_t undeleted_files, undeleted_dirs;
    return params_.env->DeleteRecursively(params_.CheckpointsDirectory(),
                                          &undeleted_files, &undeleted_dirs);
  }
  return OkStatus();
}

Status SnapshotStreamWriter::Restore() {
  StatusOr<int64_t> checkpoint_index = LastCheckpointIndex();
  if (errors::IsNotFound(checkpoint_index.status())) {
    // No checkpoint has been written. Does not restore anything.
    return OkStatus();
  }
  TF_RETURN_IF_ERROR(checkpoint_index.status());

  std::string checkpoint_path = CheckpointPath(*checkpoint_index);
  snapshot_util::TFRecordReaderImpl reader(checkpoint_path, params_.compression,
                                           kTFRecordReaderOutputBufferSize);
  TF_RETURN_IF_ERROR(reader.Initialize(params_.env));
  TF_ASSIGN_OR_RETURN(std::vector<Tensor> serialized_tensors,
                      reader.GetTensors());
  TF_RETURN_IF_ERROR(iterator_->Restore(serialized_tensors));
  TF_RETURN_IF_ERROR(SyncCheckpointWithChunks(*checkpoint_index));
  chunk_index_ = *checkpoint_index + 1;
  LOG(INFO) << "Restored distributed tf.data snapshot writer. Stream "
            << params_.stream_index << ", chunk " << *checkpoint_index << ".";
  return OkStatus();
}

StatusOr<int64_t> SnapshotStreamWriter::LastCheckpointIndex() const {
  TF_ASSIGN_OR_RETURN(std::vector<std::string> checkpoint_names,
                      GetChildren(params_.CheckpointsDirectory(), params_.env));
  if (checkpoint_names.empty()) {
    return errors::NotFound("No checkpoint has been written in directory ",
                            params_.CheckpointsDirectory());
  }

  int64_t last_index = 0;
  for (const std::string& checkpoint_name : checkpoint_names) {
    TF_ASSIGN_OR_RETURN(int64_t checkpoint_index,
                        GetFileIndex(checkpoint_name, "checkpoint"));
    last_index = std::max(last_index, checkpoint_index);
  }
  return last_index;
}

Status SnapshotStreamWriter::SyncCheckpointWithChunks(
    int64_t checkpoint_index) {
  // In case the worker fails after writing the checkpoint but before committing
  // a chunk file, this will synchronize the checkpoint with the chunks. It will
  // commit uncommitted chunk files written before the checkpoint and delete
  // chunk files written after the checkpoint.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::string> uncommitted_chunks,
      GetChildren(params_.UncommittedChunksDirectory(), params_.env));
  for (const std::string& uncommitted_chunk : uncommitted_chunks) {
    std::string uncommitted_chunk_filename = tsl::io::JoinPath(
        params_.UncommittedChunksDirectory(), uncommitted_chunk);
    TF_ASSIGN_OR_RETURN(int64_t chunk_index,
                        GetFileIndex(uncommitted_chunk, "chunk"));
    std::string committed_chunk_filename = tsl::io::JoinPath(
        params_.CommittedChunksDirectory(),
        absl::StrCat("chunk_", params_.stream_index, "_", chunk_index));
    if (chunk_index <= checkpoint_index) {
      TF_RETURN_IF_ERROR(params_.env->RenameFile(uncommitted_chunk_filename,
                                                 committed_chunk_filename));
    } else {
      TF_RETURN_IF_ERROR(params_.env->DeleteFile(uncommitted_chunk_filename));
    }
  }
  return OkStatus();
}

std::string SnapshotStreamWriter::CheckpointPath(int64_t chunk_index) const {
  return tsl::io::JoinPath(params_.CheckpointsDirectory(),
                           absl::StrCat("checkpoint_", chunk_index));
}
}  // namespace data
}  // namespace tensorflow
