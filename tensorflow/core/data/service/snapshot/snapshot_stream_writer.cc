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
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/byte_size.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/parallel_tfrecord_writer.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/utils.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {
namespace {

constexpr ByteSize kTFRecordReaderOutputBufferSize = ByteSize::GB(1);
constexpr int64_t kUnknownNumElements = -1;

constexpr const char kFileShardDelimiter[] = "_CHUNK_SHARDS_";

// Extracts the index from the `filename` of an uncommitted chunk. The file name
// is expected to be chunk_<chunk_index>_CHUNK_SHARDS_<unique_file_id>.
absl::StatusOr<int64_t> GetUncommittedChunkIndex(const std::string& filename) {
  std::vector<std::string> tokens =
      absl::StrSplit(filename, kFileShardDelimiter);
  if (tokens.size() != 2) {
    return absl::InternalError(
        absl::StrCat("Invalid tf.data snapshot chunk file: ", filename,
                     ". Expected sharded chunk files."));
  }

  tokens = absl::StrSplit(tokens[0], '_');
  int64_t chunk_index = 0;
  if (tokens.size() != 2 || tokens[0] != "chunk" ||
      !absl::SimpleAtoi(tokens[1], &chunk_index) || chunk_index < 0) {
    return absl::InternalError(
        absl::StrCat("Invalid tf.data snapshot chunk file: ", filename,
                     ". Expected chunk_<chunk_index>."));
  }
  return chunk_index;
}

size_t TotalNumElements(
    const ParallelTFRecordWriter::FileToStatsMap& file_stats) {
  size_t num_elements = 0;
  for (const auto& [file, stats] : file_stats) {
    num_elements += stats.num_records;
  }
  return num_elements;
}

ByteSize TotalBytes(const ParallelTFRecordWriter::FileToStatsMap& file_stats) {
  ByteSize bytes;
  for (const auto& [file, stats] : file_stats) {
    bytes += stats.estimated_size;
  }
  return bytes;
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
  absl::Status status = WriteSnapshot();
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

absl::Status SnapshotStreamWriter::WriteSnapshot() TF_LOCKS_EXCLUDED(mu_) {
  // TODO(b/258691097): Write the "LEASE" file periodically.
  TF_RETURN_IF_ERROR(InitializeDirectories());
  TF_RETURN_IF_ERROR(Restore());
  while (ShouldWriteChunks()) {
    TF_RETURN_IF_ERROR(WriteChunks());
  }
  mutex_lock l(mu_);
  return completed_.status();
}

bool SnapshotStreamWriter::StreamAlreadyCompleted() const {
  std::string done_file_path =
      StreamDoneFilePath(params_.snapshot_path, params_.stream_index);
  return params_.env->FileExists(done_file_path).ok();
}

absl::Status SnapshotStreamWriter::InitializeDirectories() {
  TF_RETURN_IF_ERROR(
      params_.env->RecursivelyCreateDir(params_.UncommittedChunksDirectory()));
  TF_RETURN_IF_ERROR(
      params_.env->RecursivelyCreateDir(params_.CheckpointsDirectory()));
  return absl::OkStatus();
}

bool SnapshotStreamWriter::ShouldWriteChunks() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return !end_of_sequence_ && completed_.ok();
}

absl::Status SnapshotStreamWriter::WriteChunks() {
  LOG(INFO) << "Writing distributed tf.data snapshot " << params_.snapshot_path
            << ", stream " << params_.stream_index << ", chunk " << chunk_index_
            << ".";

  std::string chunks_prefix = tsl::io::JoinPath(
      params_.UncommittedChunksDirectory(),
      absl::StrCat("chunk_", chunk_index_, kFileShardDelimiter));
  ParallelTFRecordWriter writer(TranslateFileName(chunks_prefix),
                                params_.compression, params_.env,
                                params_.max_chunk_size);
  do {
    TF_RETURN_IF_ERROR(WriteRecord(writer));
  } while (ShouldWriteRecord());
  TF_ASSIGN_OR_RETURN(const ParallelTFRecordWriter::FileToStatsMap file_stats,
                      writer.Finalize());
  TF_RETURN_IF_ERROR(Completed().status());
  TF_RETURN_IF_ERROR(Commit(file_stats));
  metrics::RecordTFDataServiceSnapshotBytesCommitted(
      TotalBytes(file_stats).ToUnsignedBytes());
  return absl::OkStatus();
}

bool SnapshotStreamWriter::ShouldWriteRecord() const {
  mutex_lock l(mu_);
  if (!completed_.ok() || end_of_sequence_) {
    return false;
  }
  const absl::Time now = absl::FromUnixMicros(params_.env->NowMicros());
  // Adjusts the checkpoint interval to speed up initial commits during startup.
  // It will grow gradually from 5 min to the configured checkpoint interval.
  const absl::Duration adjusted_checkpoint_interval = std::min(
      params_.checkpoint_interval, absl::Minutes(0.5 * chunk_index_ + 5));
  return now < last_commit_time_ + adjusted_checkpoint_interval;
}

absl::Status SnapshotStreamWriter::WriteRecord(ParallelTFRecordWriter& writer) {
  std::vector<Tensor> element;
  TF_RETURN_IF_ERROR(iterator_->GetNext(element, end_of_sequence_));
  if (end_of_sequence_) {
    return absl::OkStatus();
  }
  return writer.Write(std::move(element));
}

absl::Status SnapshotStreamWriter::Commit(
    const ParallelTFRecordWriter::FileToStatsMap& file_stats) {
  // Writes the checkpoint before committing the chunks. Once the checkpoint is
  // written, the chunks before the checkpoint are considered done. If the
  // worker restarts before committing the files in `file_stats`, the restarted
  // worker should commit the uncommitted chunks (see SyncCheckpointWithChunks).
  TF_RETURN_IF_ERROR(Save(file_stats));

  // Commits all chunks since the last commit.
  for (const auto& [file, stats] : file_stats) {
    std::string committed_chunk_path =
        tsl::io::JoinPath(params_.CommittedChunksDirectory(),
                          absl::StrCat("chunk_", params_.stream_index, "_",
                                       chunk_index_++, "_", stats.num_records));
    TF_RETURN_IF_ERROR(params_.env->RenameFile(file, committed_chunk_path));
  }
  last_commit_time_ = absl::FromUnixMicros(params_.env->NowMicros());
  return absl::OkStatus();
}

absl::Status SnapshotStreamWriter::FinalizeStream(absl::Status status) {
  if (status.ok()) {
    status = WriteDoneFile();
  }
  if (!status.ok()) {
    // If writing snapshot fails and writing the error file also fails, returns
    // the former status.
    WriteErrorFile(status).IgnoreError();
  }
  absl::Status s = DeleteCheckpoints();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to clean up checkpoints at "
               << params_.CheckpointsDirectory() << ": " << s;
  }
  return status;
}

absl::Status SnapshotStreamWriter::WriteDoneFile() {
  std::string done_file_path =
      StreamDoneFilePath(params_.snapshot_path, params_.stream_index);
  return AtomicallyWriteStringToFile(done_file_path, "", params_.env);
}

absl::Status SnapshotStreamWriter::WriteErrorFile(const absl::Status& status) {
  std::string error_file_path =
      tsl::io::JoinPath(params_.StreamDirectory(), "ERROR");
  return AtomicallyWriteStringToFile(error_file_path, status.ToString(),
                                     params_.env);
}

absl::StatusOr<bool> SnapshotStreamWriter::Completed() const
    TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return completed_;
}

absl::StatusOr<bool> SnapshotStreamWriter::Wait() TF_LOCKS_EXCLUDED(mu_) {
  snapshot_thread_.reset();
  mutex_lock l(mu_);
  return completed_;
}

void SnapshotStreamWriter::Cancel() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  completed_ = absl::CancelledError(
      "The tf.data service snapshot writer has been cancelled.");
}

absl::Status SnapshotStreamWriter::Save(
    const ParallelTFRecordWriter::FileToStatsMap& file_stats) {
  const size_t num_elements = TotalNumElements(file_stats);
  const ByteSize byte_size = TotalBytes(file_stats);
  LOG(INFO) << "Checkpointing distributed tf.data snapshot writer for snapshot "
            << params_.DebugString() << ". Stream " << params_.stream_index
            << ", chunk " << chunk_index_
            << ", number of elements in chunk: " << num_elements
            << ", chunk size: " << byte_size << ".";
  tsl::profiler::TraceMe activity("SnapshotCheckpoint",
                                  tsl::profiler::TraceMeLevel::kInfo);
  absl::Time start_time = absl::FromUnixMicros(params_.env->NowMicros());
  // The checkpoint index identifies the first chunk index after the checkpoint:
  // When a worker restarts, all the files before `checkpoint_index` should be
  // committed; all the files at/after `checkpoint_index` should be discarded.
  int64_t checkpoint_index = chunk_index_ + file_stats.size();
  std::string checkpoint_path = CheckpointPath(checkpoint_index, num_elements);
  TF_ASSIGN_OR_RETURN(std::vector<Tensor> serialized_iterator,
                      iterator_->Save());
  TF_RETURN_IF_ERROR(AtomicallyWriteTFRecords(
      checkpoint_path, serialized_iterator, params_.compression, params_.env));
  absl::Time end_time = absl::FromUnixMicros(params_.env->NowMicros());
  LOG(INFO) << "Wrote checkpoint file " << checkpoint_path << ". "
            << "Checkpointing distributed tf.data snapshot writer took "
            << (end_time - start_time);
  return DeleteOutdatedCheckpoints(checkpoint_index);
}

absl::Status SnapshotStreamWriter::DeleteOutdatedCheckpoints(
    int64_t checkpoint_index) {
  if (params_.test_only_keep_temp_files) {
    return absl::OkStatus();
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
    auto [checkpoint_file_index, _] = checkpoint_filename_tokens;
    if (checkpoint_file_index < checkpoint_index) {
      TF_RETURN_IF_ERROR(params_.env->DeleteFile(checkpoint_filepath));
    }
  }
  return absl::OkStatus();
}

absl::Status SnapshotStreamWriter::DeleteCheckpoints() {
  if (params_.test_only_keep_temp_files) {
    return absl::OkStatus();
  }
  LOG(INFO) << "Deleting tf.data snapshot checkpoints directory: "
            << params_.CheckpointsDirectory();
  if (params_.env->FileExists(params_.CheckpointsDirectory()).ok()) {
    int64_t undeleted_files, undeleted_dirs;
    return params_.env->DeleteRecursively(params_.CheckpointsDirectory(),
                                          &undeleted_files, &undeleted_dirs);
  }
  return absl::OkStatus();
}

absl::Status SnapshotStreamWriter::Restore() {
  absl::StatusOr<std::string> checkpoint_name = LastCheckpointName();
  if (absl::IsNotFound(checkpoint_name.status())) {
    // No checkpoint has been written. Deletes any uncommitted chunks.
    // Otherwise, it may attempt to write an existing file.
    return SyncCheckpointWithChunks(/*checkpoint_index=*/std::nullopt,
                                    kUnknownNumElements);
  }
  TF_RETURN_IF_ERROR(checkpoint_name.status());
  snapshot_util::TFRecordReaderImpl reader(
      CheckpointPath(*checkpoint_name), params_.compression,
      kTFRecordReaderOutputBufferSize.ToUnsignedBytes());
  TF_RETURN_IF_ERROR(reader.Initialize(params_.env));
  TF_ASSIGN_OR_RETURN(std::vector<Tensor> serialized_tensors,
                      reader.GetTensors());
  TF_RETURN_IF_ERROR(iterator_->Restore(serialized_tensors));
  TF_ASSIGN_OR_RETURN(auto checkpoint_name_tokens,
                      ParseCheckpointFilename(*checkpoint_name));
  auto [checkpoint_index, checkpoint_num_elements] = checkpoint_name_tokens;
  TF_RETURN_IF_ERROR(
      SyncCheckpointWithChunks(checkpoint_index, checkpoint_num_elements));
  chunk_index_ = checkpoint_index;
  LOG(INFO) << "Restored distributed tf.data snapshot writer. Snapshot "
            << params_.snapshot_path << ", stream " << params_.stream_index
            << ", chunk " << checkpoint_index << ".";
  return absl::OkStatus();
}

absl::StatusOr<std::string> SnapshotStreamWriter::LastCheckpointName() const {
  TF_ASSIGN_OR_RETURN(std::vector<std::string> checkpoint_names,
                      GetChildren(params_.CheckpointsDirectory(), params_.env));
  if (checkpoint_names.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No checkpoint has been written in directory ",
                     params_.CheckpointsDirectory()));
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

absl::Status SnapshotStreamWriter::SyncCheckpointWithChunks(
    std::optional<int64_t> checkpoint_index, int64_t checkpoint_num_elements) {
  // In case the worker fails after writing the checkpoint but before committing
  // a chunk file, this will synchronize the checkpoint with the chunks. It will
  // commit uncommitted chunk files written before the checkpoint and delete
  // chunk files written after the checkpoint.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::string> uncommitted_chunks,
      GetChildren(params_.UncommittedChunksDirectory(), params_.env));

  TF_ASSIGN_OR_RETURN(int64_t last_committed_chunk_index,
                      LastCommittedChunkIndex());
  int64_t next_chunk_index = last_committed_chunk_index + 1;
  for (const std::string& uncommitted_chunk : uncommitted_chunks) {
    std::string uncommitted_chunk_filename = tsl::io::JoinPath(
        params_.UncommittedChunksDirectory(), uncommitted_chunk);
    TF_ASSIGN_OR_RETURN(int64_t uncommitted_chunk_index,
                        GetUncommittedChunkIndex(uncommitted_chunk));
    if (checkpoint_index.has_value() &&
        uncommitted_chunk_index < *checkpoint_index) {
      int64_t chunk_num_elements = (next_chunk_index == *checkpoint_index - 1)
                                       ? checkpoint_num_elements
                                       : kUnknownNumElements;
      std::string committed_chunk_filename = tsl::io::JoinPath(
          params_.CommittedChunksDirectory(),
          absl::StrCat("chunk_", params_.stream_index, "_", next_chunk_index,
                       "_", chunk_num_elements));
      TF_RETURN_IF_ERROR(params_.env->RenameFile(uncommitted_chunk_filename,
                                                 committed_chunk_filename));
      ++next_chunk_index;
    } else {
      TF_RETURN_IF_ERROR(params_.env->DeleteFile(uncommitted_chunk_filename));
    }
  }
  if (checkpoint_index.has_value() && next_chunk_index != *checkpoint_index) {
    return absl::InternalError(absl::StrCat(
        "Failed to recover tf.data snapshot writer: Unable to find chunks [",
        next_chunk_index, ", ", *checkpoint_index, ")."));
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> SnapshotStreamWriter::LastCommittedChunkIndex() {
  std::string committed_chunks_directory = params_.CommittedChunksDirectory();
  TF_ASSIGN_OR_RETURN(
      std::vector<std::string> committed_chunks,
      GetChildren(params_.CommittedChunksDirectory(), params_.env));

  int64_t last_committed_chunk_index = -1;
  for (const std::string& committed_chunk : committed_chunks) {
    TF_ASSIGN_OR_RETURN(auto chunk_filename_tokens,
                        ParseChunkFilename(committed_chunk));
    const auto [stream_index, chunk_index, _] = chunk_filename_tokens;
    if (stream_index != params_.stream_index) {
      continue;
    }
    if (chunk_index > last_committed_chunk_index) {
      last_committed_chunk_index = chunk_index;
    }
  }
  return last_committed_chunk_index;
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
