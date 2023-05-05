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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_STREAM_WRITER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_STREAM_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

constexpr int64_t kDefaultMaxChunkSizeBytes = 2 * (size_t{1} << 30);  // 2GB

struct SnapshotWriterParams {
  // The directory path of the snapshot. See the comment on SnapshotStreamWriter
  // for how the directory is structured.
  std::string snapshot_path;

  // The index of the snapshot stream. A stream is one shard of the snapshot
  // processed by a worker.
  int64_t stream_index = 0;

  // Compression method as defined in tsl/lib/io/compression.h.
  std::string compression;

  // The Tensorflow environment.
  Env* env = nullptr;

  // The maximum number of bytes in each chunk.
  int64_t max_chunk_size_bytes = kDefaultMaxChunkSizeBytes;

  // If true, keep temporary files (e.g., checkpoints) after completing the
  // snapshot. Used only for unit testing.
  bool test_only_keep_temp_files = false;

  std::string StreamDirectory() const {
    return tensorflow::data::StreamDirectory(snapshot_path, stream_index);
  }

  std::string CommittedChunksDirectory() const {
    return tensorflow::data::CommittedChunksDirectory(snapshot_path);
  }

  std::string UncommittedChunksDirectory() const {
    return tensorflow::data::UncommittedChunksDirectory(snapshot_path,
                                                        stream_index);
  }

  std::string CheckpointsDirectory() const {
    return tensorflow::data::CheckpointsDirectory(snapshot_path, stream_index);
  }

  std::string DebugString() const {
    return absl::Substitute(
        "SnapshotWriterParams { base_path: $0, stream: $1, compression: $2 }",
        snapshot_path, stream_index, compression);
  }
};

// Responsible for writing one snapshot stream, which is organized as following:
//
// - snapshot
//   - DONE
//   - ERROR
//   - snapshot.metadata
//   - dataset_def.proto
//   - chunks
//     - chunk_<stream_index>_<chunk_index>_<num_elements>
//   - streams
//     - stream_0
//       - DONE
//       - ERROR
//       - splits
//         - split_<local_split_index>_<global_split_index>
//       - uncommitted chunks
//         - chunk_<chunk_index>
//       - checkpoints
//         - checkpoint_<chunk_index>_<num_elements>
//
// This class is thread-safe.
class SnapshotStreamWriter {
 public:
  // Creates a SnapshotStreamWriter. Once created, it will start writing the
  // snapshot stream. Users can call `Wait` to wait for it to finish.
  explicit SnapshotStreamWriter(const SnapshotWriterParams& params,
                                std::unique_ptr<TaskIterator> iterator);
  virtual ~SnapshotStreamWriter() = default;
  SnapshotStreamWriter(const SnapshotStreamWriter&) = delete;
  SnapshotStreamWriter& operator=(const SnapshotStreamWriter&) = delete;

  // Returns true if the snapshot stream has completed. A snapshot stream is
  // completed if the dataset has reached the end of sequence and a DONE file is
  // written. Returns an error if the snapshot has failed. This does not block
  // the caller.
  StatusOr<bool> Completed() const;

  // Waits for the writer to finish writing the snapshot stream and returns the
  // final status.
  StatusOr<bool> Wait();

  // Cancels the writer. If cancelled, `Wait` will return a Cancelled error.
  void Cancel();

 private:
  // Writes the snapshot and any debugging log when necessary.
  void WriteSnapshotAndLog();

  // Writes the snapshot. Returns an error if writing fails or the task has been
  // cancelled.
  Status WriteSnapshot();

  // Returns true if the stream is already completed and there is no additional
  // work to perform.
  bool StreamAlreadyCompleted() const;

  // Creates directories to store uncommitted chunks and checkpoints.
  Status InitializeDirectories();

  // Returns true until the snapshot stream writer is finished, which may be due
  // to reaching the end of its iterator, encountering an error, or being
  // cancelled.
  bool ShouldWriteChunk() const;

  // Writes the next chunk.
  Status WriteChunk();

  // Commits the current chunk.
  Status CommitChunk();

  // Returns the path of the current chunk.
  std::string GetChunkFilePath() const;
  std::string GetCommittedChunkFilePath() const;

  // Returns true if the writer should write the next record to the current
  // chunk.
  bool ShouldWriteRecord() const;

  // Writes the next record to the current chunk.
  Status WriteRecord(snapshot_util::TFRecordWriter& writer);

  // Writes a DONE file when the stream is finished. Writes an ERROR file if it
  // failed.
  Status FinalizeStream(Status status);
  Status WriteDoneFile();
  Status WriteErrorFile(const Status& status);

  // Returns true if the writer should write an iterator checkpoint.
  bool ShouldSave() const;

  // Saves an iterator checkpoint.
  Status Save();

  // After committing a checkpoint, deletes the previous checkpoints.
  Status DeleteOutdatedCheckpoints();

  // Deletes all checkpoints.
  Status DeleteCheckpoints();

  // Restores from the last checkpoint.
  Status Restore();

  // Returns the filename of the most recent checkpoint.
  StatusOr<std::string> LastCheckpointName() const;

  // Synchronizes the checkpoint with the committed chunks. This is called when
  // the worker restores the snapshot in case the worker fails after writing the
  // checkpoint but before committing a chunk file. If no checkpoint has been
  // written, `checkpoint_index` is nullopt.
  Status SyncCheckpointWithChunks(std::optional<int64_t> checkpoint_index,
                                  int64_t checkpoint_num_elements);

  // Returns the path of the checkpoint for `chunk_index` with
  // `chunk_num_elements`.
  std::string CheckpointPath(int64_t chunk_index,
                             int64_t chunk_num_elements) const;

  // Returns the path of the checkpoint for `checkpoint_name`.
  std::string CheckpointPath(const std::string& checkpoint_name) const;

  const SnapshotWriterParams params_;

  // The dataset iterator that produces the dataset elements.
  std::unique_ptr<TaskIterator> iterator_;

  // Index of the current chunk.
  int64_t chunk_index_ = 0;
  // Size of the current chunk.
  int64_t chunk_size_bytes_ = 0;
  // Number of elements in current chunk.
  int64_t chunk_num_elements_ = 0;

  // True if the dataset is exhausted.
  bool end_of_sequence_ = false;

  mutable mutex mu_;

  // Whether the writer is completed:
  // - If the snapshot is successful, this is true.
  // - If any error happens during the snapshot write, it is the error status.
  // - If the snapshot has not finished, this is false.
  StatusOr<bool> completed_ TF_GUARDED_BY(mu_) = false;

  std::unique_ptr<Thread> snapshot_thread_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_STREAM_WRITER_H_
