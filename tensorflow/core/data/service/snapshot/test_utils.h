/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_TEST_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_TEST_UTILS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace testing {

// Reads the records from a distributed tf.data snapshot written at `base_path`.
template <class T>
tsl::StatusOr<std::vector<T>> ReadSnapshot(const std::string& base_path,
                                           const std::string& compression) {
  std::vector<T> result;
  std::string chunks_directory = CommittedChunksDirectory(base_path);
  TF_ASSIGN_OR_RETURN(std::vector<string> chunk_files,
                      GetChildren(chunks_directory, Env::Default()));
  for (const std::string& chunk_file : chunk_files) {
    std::string chunk_file_path =
        tsl::io::JoinPath(chunks_directory, chunk_file);
    snapshot_util::TFRecordReader tfrecord_reader(chunk_file_path, compression,
                                                  DataTypeVector{DT_INT64});
    TF_RETURN_IF_ERROR(tfrecord_reader.Initialize(Env::Default()));

    while (true) {
      std::vector<Tensor> tensors;
      Status status = tfrecord_reader.ReadTensors(&tensors);
      if (absl::IsOutOfRange(status)) {
        break;
      }
      TF_RETURN_IF_ERROR(status);
      result.push_back(tensors[0].unaligned_flat<T>().data()[0]);
    }
  }
  return result;
}

// Writes a partial snapshot to test checkpointing and recovering. It can be
// used to write the specified committed chunks, uncommitted chunks, and
// checkpoints.
class PartialSnapshotWriter {
 public:
  static tsl::StatusOr<PartialSnapshotWriter> Create(
      const DatasetDef& dataset, const std::string& snapshot_path,
      int64_t stream_index, const std::string& compression,
      int64_t max_chunk_size_bytes = 1);
  virtual ~PartialSnapshotWriter() = default;
  PartialSnapshotWriter(const PartialSnapshotWriter&) = delete;
  PartialSnapshotWriter& operator=(const PartialSnapshotWriter&) = delete;
  PartialSnapshotWriter(PartialSnapshotWriter&&) = default;
  PartialSnapshotWriter& operator=(PartialSnapshotWriter&&) = delete;

  // Writes the specified chunks.
  tsl::Status WriteCommittedChunks(
      const absl::flat_hash_set<int64_t>& committed_chunk_indexes) const;

  // Writes the specified uncommitted chunks.
  tsl::Status WriteUncommittedChunks(
      const absl::flat_hash_set<int64_t>& uncommitted_chunk_indexes) const;

  // Writes the specified checkpoints.
  tsl::Status WriteCheckpoints(
      const absl::flat_hash_set<int64_t>& checkpoint_indexes) const;

 private:
  PartialSnapshotWriter(const DatasetDef& dataset,
                        const std::string& snapshot_path, int64_t stream_index,
                        const std::string& compression,
                        int64_t max_chunk_size_bytes);

  tsl::Status Initialize();

  const DatasetDef dataset_;
  const std::string snapshot_path_;
  const int64_t stream_index_;
  const std::string compression_;
  const int64_t max_chunk_size_bytes_;

  std::string tmp_snapshot_path_;
};

// Creates a test iterator for the input dataset. The iterator will generate all
// elements of the dataset.
tsl::StatusOr<std::unique_ptr<StandaloneTaskIterator>> TestIterator(
    const DatasetDef& dataset_def);

}  // namespace testing
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_TEST_UTILS_H_
