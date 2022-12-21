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
// TODO(b/258691666): Support chunking, checkpointing, and fault tolerance.
class SnapshotStreamWriter {
 public:
  // Creates a SnapshotStreamWriter. Once created, it will start writing the
  // snapshot stream. Users can call `Wait` to wait for it to finish.
  // TODO(b/258691666): Create a new `TaskIterator` that persists splits.
  // TODO(b/258691666): Create a structure for the input params.
  explicit SnapshotStreamWriter(
      std::unique_ptr<TaskIterator> iterator, const std::string& snapshot_path,
      int64_t stream_id, const std::string& compression, Env* env,
      std::optional<int64_t> max_chunk_size_bytes = std::nullopt);

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

  // Creates a directory to store uncommitted chunks.
  Status CreateChunksDirectory();

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

  // Returns the status of the writer:
  // - If the snapshotting is successful, returns an OK status.
  // - If any error happens during the snapshot write, returns the error status.
  // - If the writer is cancelled, returns a Cancelled status.
  Status status() const;

  Env* const env_;
  const std::string snapshot_path_;
  const int64_t stream_id_;
  const std::string compression_;
  const int64_t max_chunk_size_bytes_;

  mutable mutex mu_;
  std::unique_ptr<TaskIterator> iterator_ TF_GUARDED_BY(mu_);

  // Index of the current chunk.
  int64_t chunk_index_ TF_GUARDED_BY(mu_) = 0;
  // Size of the current chunk.
  int64_t chunk_size_bytes_ TF_GUARDED_BY(mu_) = 0;

  // True if the dataset is exhausted.
  bool end_of_sequence_ TF_GUARDED_BY(mu_) = false;
  // Status of the writer. See the comment of `status()` for a detailed
  // description.
  Status status_ TF_GUARDED_BY(mu_);

  std::unique_ptr<Thread> snapshot_thread_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_STREAM_WRITER_H_
