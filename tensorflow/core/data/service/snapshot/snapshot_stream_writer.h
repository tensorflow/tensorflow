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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "tensorflow/core/data/service/byte_size.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/snapshot/parallel_tfrecord_writer.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

constexpr ByteSize kDefaultMaxChunkSize = ByteSize::GB(6);
constexpr absl::Duration kDefaultCheckpointInterval = absl::Minutes(30);

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
  ByteSize max_chunk_size = kDefaultMaxChunkSize;

  // How often should checkpoints be written at the steady state. We write
  // checkpoints (and committing chunks) more frequently at the startup time to
  // avoid starving training jobs during startup.
  absl::Duration checkpoint_interval = kDefaultCheckpointInterval;

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
  absl::StatusOr<bool> Completed() const;

  // Waits for the writer to finish writing the snapshot stream and returns the
  // final status.
  absl::StatusOr<bool> Wait();

  // Cancels the writer. If cancelled, `Wait` will return a Cancelled error.
  void Cancel();

 private:
  // Writes the snapshot and any debugging log when necessary.
  void WriteSnapshotAndLog();

  // Writes the snapshot. Returns an error if writing fails or the task has been
  // cancelled.
  absl::Status WriteSnapshot();

  // Returns true if the stream is already completed and there is no additional
  // work to perform.
  bool StreamAlreadyCompleted() const;

  // Creates directories to store uncommitted chunks and checkpoints.
  absl::Status InitializeDirectories();

  // Returns true until the snapshot stream writer is finished, which may be due
  // to reaching the end of its iterator, encountering an error, or being
  // cancelled.
  bool ShouldWriteChunks() const;

  // Writes the chunk files.
  absl::Status WriteChunks();

  // Returns true if it should write more records to the current chunks. Returns
  // false if it should checkpoint and commit the current chunks, there are no
  // more records to write, or there is an error.
  bool ShouldWriteRecord() const;

  // Writes the next record to the current chunks.
  absl::Status WriteRecord(ParallelTFRecordWriter& writer);

  // Commits the chunks since the last commit.
  absl::Status Commit(const ParallelTFRecordWriter::FileToStatsMap& file_stats);

  // Writes a DONE file when the stream is finished. Writes an ERROR file if it
  // failed.
  absl::Status FinalizeStream(absl::Status status);
  absl::Status WriteDoneFile();
  absl::Status WriteErrorFile(const absl::Status& status);

  // Saves an iterator checkpoint.
  absl::Status Save(const ParallelTFRecordWriter::FileToStatsMap& file_stats);

  // After committing a checkpoint, deletes the previous checkpoints.
  absl::Status DeleteOutdatedCheckpoints(int64_t checkpoint_index);

  // Deletes all checkpoints.
  absl::Status DeleteCheckpoints();

  // Restores from the last checkpoint.
  absl::Status Restore();

  // Returns the filename of the most recent checkpoint.
  absl::StatusOr<std::string> LastCheckpointName() const;

  // Synchronizes the checkpoint with the committed chunks. This is called when
  // the worker restores the snapshot in case the worker fails after writing the
  // checkpoint but before committing a chunk file. If no checkpoint has been
  // written, `checkpoint_index` is nullopt.
  absl::Status SyncCheckpointWithChunks(std::optional<int64_t> checkpoint_index,
                                        int64_t checkpoint_num_elements);

  // Index of the last committed chunk.
  absl::StatusOr<int64_t> LastCommittedChunkIndex();

  // Returns the path of the checkpoint for `chunk_index` with
  // `chunk_num_elements`.
  std::string CheckpointPath(int64_t chunk_index,
                             int64_t chunk_num_elements) const;

  // Returns the path of the checkpoint for `checkpoint_name`.
  std::string CheckpointPath(const std::string& checkpoint_name) const;

  const SnapshotWriterParams params_;

  // The dataset iterator that produces the dataset elements.
  std::unique_ptr<TaskIterator> iterator_;

  // Index of the next chunk to write.
  int64_t chunk_index_ = 0;
  // Timestamp when the last chunks are committed.
  absl::Time last_commit_time_ = absl::Now();

  // True if the dataset is exhausted.
  bool end_of_sequence_ = false;

  mutable mutex mu_;

  // Whether the writer is completed:
  // - If the snapshot is successful, this is true.
  // - If any error happens during the snapshot write, it is the error status.
  // - If the snapshot has not finished, this is false.
  absl::StatusOr<bool> completed_ TF_GUARDED_BY(mu_) = false;

  std::unique_ptr<Thread> snapshot_thread_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_STREAM_WRITER_H_
