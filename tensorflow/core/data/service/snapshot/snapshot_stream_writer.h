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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

struct SnapshotWriterParams {
  // The directory path of the snapshot. See the comment on SnapshotStreamWriter
  // for how the directory is structured.
  std::string snapshot_path;

  // The ID of the stream. A stream is one shard of the snapshot processed by a
  // worker.
  int64_t stream_id = 0;

  // Compression method as defined in tsl/lib/io/compression.h.
  std::string compression;

  // The Tensorflow environment.
  Env* env = nullptr;

  // The maximum number of bytes in each chunk.
  int64_t max_chunk_size_bytes = kDefaultMaxChunkSizeBytes;

 private:
  static constexpr int64_t kDefaultMaxChunkSizeBytes =
      10 * (size_t{1} << 30);  // 10GB
};

// Responsible for writing one snapshot stream, which is organized as following:
//
// - snapshot
//   - LEASE
//   - DONE
//   - snapshot.metadata
//   - dataset_def.proto
//   - committed chunks
//     - chunk_<stream_index>_<chunk_index>
//   - streams
//     - stream_0
//       - LEASE
//       - DONE
//       - splits
//         - split_<local_split_index>_<global_split_index>
//       - uncommitted chunks
//         - chunk_<chunk_index>
//       - checkpoints
//         - checkpoint_<local_split_index>_<chunk_index>
//
// This class is thread-safe.
// TODO(b/258691666): Support chunking, checkpointing, and fault tolerance.
class SnapshotStreamWriter {
 public:
  // Creates a SnapshotStreamWriter. Once created, it will start writing the
  // snapshot stream. Users can call `Wait` to wait for it to finish.
  // TODO(b/258691666): Create a new `TaskIterator` that persists splits.
  explicit SnapshotStreamWriter(const SnapshotWriterParams& params,
                                std::unique_ptr<TaskIterator> iterator);

  // Waits for the writer to finish writing the snapshot stream.
  Status Wait();

  // Cancels the writer. If cancelled, `Wait` will return a Cancelled error.
  void Cancel();

 private:
  // Runs `WriteSnapshotFn` on a dedicated thread.
  std::unique_ptr<Thread> RunSnapshotThread();

  // Function to write the snapshot. Returns an error if writing fails or the
  // task has been cancelled.
  Status WriteSnapshotFn();

  // Creates directories to store uncommitted chunks and checkpoints.
  Status InitializeDirectories();

  // Returns true until the snapshot stream writer is finished, which may be due
  // to reaching the end of its iterator, encountering an error, or being
  // cancelled.
  bool ShouldWriteChunk() const;

  // Writes the next chunk.
  Status WriteChunk();

  // Returns the path of the current chunk.
  std::string GetChunkFilePath() const;

  // Commits the current chunk.
  Status CommitChunk(const std::string& chunk_file_path);

  // Returns true if the writer should write the next record to the current
  // chunk.
  bool ShouldWriteRecord() const;

  // Writes the next record to the current chunk.
  Status WriteRecord(snapshot_util::TFRecordWriter& writer);

  // Returns true if the writer should write an iterator checkpoint.
  bool ShouldSave() const;

  // Saves an iterator checkpoint.
  Status Save();

  // Restores from the last checkpoint.
  Status Restore();

  // Returns the index of the last checkpointed chunk.
  StatusOr<int64_t> LastCheckpointIndex() const;

  // Synchronizes the checkpoint with the committed chunks. This will commit
  // uncommitted chunk files written before the checkpoint and delete chunk
  // files written after the checkpoint.
  Status SyncCheckpointWithChunks();

  // Returns the path of the checkpoint for `chunk_index`.
  std::string CheckpointPath(int64_t chunk_index) const;

  const SnapshotWriterParams params_;

  // The dataset iterator that produces the dataset elements.
  std::unique_ptr<TaskIterator> iterator_;

  // Index of the current chunk.
  int64_t chunk_index_ = 0;
  // Size of the current chunk.
  int64_t chunk_size_bytes_ = 0;

  // True if the dataset is exhausted.
  bool end_of_sequence_ = false;

  mutable mutex mu_;

  // Status of the writer:
  // - If the snapshotting is successful, it is an OK status.
  // - If any error happens during the snapshot write, it is the error status.
  // - If the writer is cancelled, it is a Cancelled status.
  Status status_ TF_GUARDED_BY(mu_);

  std::unique_ptr<Thread> snapshot_thread_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_STREAM_WRITER_H_
