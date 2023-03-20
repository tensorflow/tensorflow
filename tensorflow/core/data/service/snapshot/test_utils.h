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
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/snapshot/snapshot_reader.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace testing {

// Reads the records from a distributed tf.data snapshot written at `base_path`.
template <class T>
tsl::StatusOr<std::vector<T>> ReadSnapshot(const std::string& base_path,
                                           const std::string& compression) {
  experimental::DistributedSnapshotMetadata metadata;
  metadata.set_compression(compression);
  SnapshotReaderParams params{base_path, metadata, DataTypeVector{DT_INT64},
                              /*shapes=*/{}, Env::Default()};
  SnapshotReader reader(params);
  std::vector<T> result;
  while (true) {
    TF_ASSIGN_OR_RETURN(GetNextResult next, reader.GetNext());
    if (next.end_of_sequence) {
      return result;
    }
    result.push_back(next.tensors[0].unaligned_flat<T>().data()[0]);
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
