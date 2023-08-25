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

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/common.h"
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
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {
namespace {

constexpr int64_t kTFRecordReaderOutputBufferSize = 512 << 20;  // 512MB
constexpr int64_t kUnknownNumElements = -1;

// Extracts the index from the `filename` of an uncommitted chunk. The chunk
// file name is expected to be chunk_<chunk_index>.
absl::StatusOr<int64_t> GetUncommittedChunkIndex(const std::string& filename) {
  std::vector<std::string> tokens = absl::StrSplit(filename, '_');
  int64_t chunk_index = 0;
  if (tokens.size() != 2 || tokens[0] != "chunk" ||
      !absl::SimpleAtoi(tokens[1], &chunk_index) || chunk_index < 0) {
    return absl::InternalError(
        absl::StrCat("Invalid chunk file name: ", filename,
                     ". Expected chunk_<chunk_index>."));
  }
  return chunk_index;
}

}  // namespace

SnapshotStreamWriter::SnapshotStreamWriter(
    const SnapshotWriterParams& params, std::unique_ptr<TaskIterator> iterator)
    : params_(params), iterator_(std::move(iterator)) {
  DCHECK_NE(iterator_.get(), nullptr);
  last_commit_time_ = absl::FromUnixMicros(params_.env->NowMicros());
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
  if (IsPreemptedError(status)) {
    LOG(INFO) << "tf.data service snapshot writer is cancelled: " << status;
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
  iterator_ = nullptr;  // Reclaims iterator resources.
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
  LOG(INFO) << "Writing distributed tf.data snapshot " << params_.snapshot_path
            << ", stream " << params_.stream_index << ", chunk " << chunk_index_
            << ".";

  std::string uncommitted_chunk_file_path =
      tsl::io::JoinPath(params_.UncommittedChunksDirectory(),
                        absl::StrCat("chunk_", chunk_index_));
  snapshot_util::TFRecordWriter writer(uncommitted_chunk_file_path,
                                       params_.compression);
  TF_RETURN_IF_ERROR(writer.Initialize(params_.env));
  while (ShouldWriteRecord()) {
    TF_RETURN_IF_ERROR(WriteRecord(writer));
  }
  TF_RETURN_IF_ERROR(writer.Close());
  chunk_file_to_num_elements_[absl::StrCat("chunk_", chunk_index_)] =
      chunk_num_elements_;
  if (ShouldCommit()) {
    TF_RETURN_IF_ERROR(Commit());
  }
  metrics::RecordTFDataServiceSnapshotBytesCommitted(chunk_size_bytes_);
  ++chunk_index_;
  chunk_size_bytes_ = 0;
  chunk_num_elements_ = 0;
  return OkStatus();
}

bool SnapshotStreamWriter::ShouldCommit() const {
  {
    mutex_lock l(mu_);
    if (!completed_.ok()) {
      return false;
    }
  }
  const absl::Time now = absl::FromUnixMicros(params_.env->NowMicros());
  return end_of_sequence_ ||
         now > last_commit_time_ + params_.checkpoint_interval;
}

Status SnapshotStreamWriter::Commit() {
  // Writes the checkpoint before committing the chunks. If the worker fails in
  // between, the restarted worker will commit the uncommitted chunks.
  TF_RETURN_IF_ERROR(Save());
  TF_ASSIGN_OR_RETURN(
      std::vector<std::string> uncommitted_chunks,
      GetChildren(params_.UncommittedChunksDirectory(), params_.env));
  if (uncommitted_chunks.size() != chunk_file_to_num_elements_.size()) {
    return absl::InternalError(absl::StrCat(
        "Failed to write tf.data snapshot: Expected ",
        chunk_file_to_num_elements_.size(), " uncommitted chunks, but got ",
        uncommitted_chunks.size(), "."));
  }
  // Commits all chunks since the last commit.
  for (int64_t i = 0; i < uncommitted_chunks.size(); ++i) {
    const std::string& uncommitted_chunk = uncommitted_chunks[i];
    TF_ASSIGN_OR_RETURN(int64_t chunk_index,
                        GetUncommittedChunkIndex(uncommitted_chunk));
    if (chunk_index <= chunk_index_) {
      std::string uncommitted_chunk_path = tsl::io::JoinPath(
          params_.UncommittedChunksDirectory(), uncommitted_chunk);
      std::string committed_chunk_path = tsl::io::JoinPath(
          params_.CommittedChunksDirectory(),
          absl::StrCat("chunk_", params_.stream_index, "_", chunk_index, "_",
                       chunk_file_to_num_elements_[uncommitted_chunk]));
      TF_RETURN_IF_ERROR(params_.env->RenameFile(uncommitted_chunk_path,
                                                 committed_chunk_path));
    }
  }
  last_committed_chunk_ = chunk_index_;
  last_commit_time_ = absl::FromUnixMicros(params_.env->NowMicros());
  chunk_file_to_num_elements_.clear();
  return OkStatus();
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
  tsl::profiler::TraceMe activity("SnapshotWriteRecord",
                                  tsl::profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(writer.WriteTensors(element));
  chunk_size_bytes_ += EstimatedSizeBytes(element);
  ++chunk_num_elements_;
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

Status SnapshotStreamWriter::Save() {
  LOG(INFO) << "Checkpointing distributed tf.data snapshot writer for snapshot "
            << params_.DebugString() << ". Stream " << params_.stream_index
            << ", chunk " << chunk_index_
            << ", chunk size in bytes: " << chunk_size_bytes_
            << ", number of elements in chunk: " << chunk_num_elements_ << ".";
  tsl::profiler::TraceMe activity("SnapshotCheckpoint",
                                  tsl::profiler::TraceMeLevel::kInfo);
  absl::Time start_time = absl::FromUnixMicros(params_.env->NowMicros());
  std::string checkpoint_path =
      CheckpointPath(chunk_index_, chunk_num_elements_);
  TF_ASSIGN_OR_RETURN(std::vector<Tensor> serialized_iterator,
                      iterator_->Save());
  TF_RETURN_IF_ERROR(AtomicallyWriteTFRecords(
      checkpoint_path, serialized_iterator, params_.compression, params_.env));
  absl::Time end_time = absl::FromUnixMicros(params_.env->NowMicros());
  LOG(INFO) << "Wrote checkpoint file " << checkpoint_path << ". "
            << "Checkpointing distributed tf.data snapshot writer took "
            << (end_time - start_time);
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

    TF_ASSIGN_OR_RETURN(auto checkpoint_filename_tokens,
                        ParseCheckpointFilename(checkpoint_filename));
    auto [checkpoint_index, unused] = checkpoint_filename_tokens;
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
  LOG(INFO) << "Deleting tf.data snapshot checkpoints directory: "
            << params_.CheckpointsDirectory();
  if (params_.env->FileExists(params_.CheckpointsDirectory()).ok()) {
    int64_t undeleted_files, undeleted_dirs;
    return params_.env->DeleteRecursively(params_.CheckpointsDirectory(),
                                          &undeleted_files, &undeleted_dirs);
  }
  return OkStatus();
}

Status SnapshotStreamWriter::Restore() {
  StatusOr<std::string> checkpoint_name = LastCheckpointName();
  if (errors::IsNotFound(checkpoint_name.status())) {
    // No checkpoint has been written. Deletes any uncommitted chunks.
    // Otherwise, it may attempt to write an existing file.
    return SyncCheckpointWithChunks(/*checkpoint_index=*/std::nullopt,
                                    kUnknownNumElements);
  }
  TF_RETURN_IF_ERROR(checkpoint_name.status());
  snapshot_util::TFRecordReaderImpl reader(CheckpointPath(*checkpoint_name),
                                           params_.compression,
                                           kTFRecordReaderOutputBufferSize);
  TF_RETURN_IF_ERROR(reader.Initialize(params_.env));
  TF_ASSIGN_OR_RETURN(std::vector<Tensor> serialized_tensors,
                      reader.GetTensors());
  TF_RETURN_IF_ERROR(iterator_->Restore(serialized_tensors));
  TF_ASSIGN_OR_RETURN(auto checkpoint_name_tokens,
                      ParseCheckpointFilename(*checkpoint_name));
  auto [checkpoint_index, checkpoint_num_elements] = checkpoint_name_tokens;
  TF_RETURN_IF_ERROR(
      SyncCheckpointWithChunks(checkpoint_index, checkpoint_num_elements));
  last_committed_chunk_ = chunk_index_ = checkpoint_index + 1;
  LOG(INFO) << "Restored distributed tf.data snapshot writer. Snapshot "
            << params_.snapshot_path << ", stream " << params_.stream_index
            << ", chunk " << checkpoint_index << ".";
  return OkStatus();
}

StatusOr<std::string> SnapshotStreamWriter::LastCheckpointName() const {
  TF_ASSIGN_OR_RETURN(std::vector<std::string> checkpoint_names,
                      GetChildren(params_.CheckpointsDirectory(), params_.env));
  if (checkpoint_names.empty()) {
    return errors::NotFound("No checkpoint has been written in directory ",
                            params_.CheckpointsDirectory());
  }

  int64_t last_index = -1;
  std::string last_checkpoint_name = "";
  for (const std::string& checkpoint_name : checkpoint_names) {
    TF_ASSIGN_OR_RETURN(auto checkpoint_name_tokens,
                        ParseCheckpointFilename(checkpoint_name));
    auto [checkpoint_index, unused] = checkpoint_name_tokens;
    if (checkpoint_index > last_index) {
      last_index = checkpoint_index;
      last_checkpoint_name = checkpoint_name;
    }
  }
  return last_checkpoint_name;
}

Status SnapshotStreamWriter::SyncCheckpointWithChunks(
    std::optional<int64_t> checkpoint_index, int64_t checkpoint_num_elements) {
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
                        GetUncommittedChunkIndex(uncommitted_chunk));
    if (checkpoint_index.has_value() && chunk_index <= *checkpoint_index) {
      int64_t chunk_num_elements = chunk_index == *checkpoint_index
                                       ? checkpoint_num_elements
                                       : kUnknownNumElements;
      std::string committed_chunk_filename =
          tsl::io::JoinPath(params_.CommittedChunksDirectory(),
                            absl::StrCat("chunk_", params_.stream_index, "_",
                                         chunk_index, "_", chunk_num_elements));
      TF_RETURN_IF_ERROR(params_.env->RenameFile(uncommitted_chunk_filename,
                                                 committed_chunk_filename));
    } else {
      TF_RETURN_IF_ERROR(params_.env->DeleteFile(uncommitted_chunk_filename));
    }
  }
  return OkStatus();
}

std::string SnapshotStreamWriter::CheckpointPath(
    int64_t chunk_index, int64_t chunk_num_elements) const {
  return tsl::io::JoinPath(
      params_.CheckpointsDirectory(),
      absl::StrCat("checkpoint_", chunk_index, "_", chunk_num_elements));
}

std::string SnapshotStreamWriter::CheckpointPath(
    const std::string& checkpoint_name) const {
  return tsl::io::JoinPath(params_.CheckpointsDirectory(), checkpoint_name);
}

}  // namespace data
}  // namespace tensorflow
