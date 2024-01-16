/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PARALLEL_TFRECORD_WRITER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PARALLEL_TFRECORD_WRITER_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace data {

// Uses multiple threads to write TFRecords in parallel. Users add data without
// waiting for the file writes, and it writes one shard of file per thread.
// Returns the file names when writes are finished. This class is thread-safe.
//
// Usage example:
//
// ParallelTFRecordWriter writer(
//     "/path/to/directory", tsl::io::compression::kSnappy, Env::Default());
//
// std::vector<Tensor> record;
// bool end_of_sequence = false;
// TF_RETURN_IF_ERROR(iterator.GetNext(record, end_of_sequence));
// while (!end_of_sequence) {
//   TF_RETURN_IF_ERROR(writer.Write(record));
//   TF_RETURN_IF_ERROR(iterator.GetNext(record, end_of_sequence));
// }
// TF_ASSIGN_OR_RETURN(absl::flat_hash_set<std::string> files,
//                     writer.Finalize());
class ParallelTFRecordWriter {
 public:
  explicit ParallelTFRecordWriter(const std::string& directory,
                                  const std::string& compression, tsl::Env* env,
                                  int64_t num_write_threads = 10,
                                  int64_t buffer_size_per_thread = 1);
  virtual ~ParallelTFRecordWriter();
  ParallelTFRecordWriter(const ParallelTFRecordWriter&) = delete;
  ParallelTFRecordWriter& operator=(const ParallelTFRecordWriter&) = delete;

  // Writes `record`. If there is sufficient buffer space, it returns without
  // waiting for the record to be written to the file. If the buffer is full,
  // blocks until there is enough space to buffer the record.
  absl::Status Write(std::vector<Tensor> record);

  // Flushes the writer and finalizes the files. Returns the absolute paths.
  // When the writer is finalized, `Write` will return FailedPreconditionErrors.
  // The caller should make sure all `Write` calls have finished before calling
  // `Finalize`. Will block until the writer is closed or an error occurs.
  absl::StatusOr<absl::flat_hash_set<std::string>> Finalize();

 private:
  // Run by a thread to write buffered records to a unique file.
  void WriteFile();

  // Whether there are more records to be written.
  bool ShouldWriteRecord() const;

  // Writes one record.
  absl::Status WriteRecord(snapshot_util::TFRecordWriter& writer);

  // Generates a unique file name in the requested directory.
  absl::StatusOr<std::string> GetUniqueFile() const;

  // Updates the status of the writer and notifies waiters.
  void UpdateStatus(absl::Status status);

  tsl::Env* const env_;
  const std::string directory_;
  const std::string compression_;
  const int64_t buffer_size_;

  mutable absl::Mutex mu_;
  mutable absl::CondVar ready_to_push_;
  mutable absl::CondVar ready_to_pop_;

  bool finalized_ ABSL_GUARDED_BY(mu_) = false;
  absl::Status status_ ABSL_GUARDED_BY(mu_);

  // Set of written files.
  absl::flat_hash_set<std::string> files_ ABSL_GUARDED_BY(mu_);

  // Buffer to hold the records to be written. The size should be bounded by
  // `buffer_size_`.
  std::deque<std::vector<Tensor>> buffer_ ABSL_GUARDED_BY(mu_);

  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PARALLEL_TFRECORD_WRITER_H_
