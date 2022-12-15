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

#include <memory>
#include <string>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// TODO(b/258691666): Support chunking, checkpointing, and fault tolerance.
class SnapshotStreamWriter {
 public:
  // Creates a SnapshotStreamWriter. Once created, it will start writing the
  // snapshot stream. Users can call `Wait` to wait for it to finish.
  // TODO(b/258691666): Create a new `TaskIterator` that persists splits.
  explicit SnapshotStreamWriter(std::unique_ptr<TaskIterator> iterator,
                                const std::string& snapshot_stream_path,
                                Env* env);

  // Waits for the task runner to finish writing the snapshot shard.
  Status Wait();

  // Cancels the task runner. If cancelled, `Wait` will return a Cancelled
  // error.
  void Cancel();

 private:
  // Runs `WriteSnapshotFn` on a dedicated thread.
  std::unique_ptr<Thread> RunSnapshotThread();

  // Function to write the snapshot. Returns an error if writing fails or the
  // task has been cancelled.
  Status WriteSnapshotFn();

  // Gets the next element from the input iterator.
  StatusOr<GetElementResult> GetNext();

  // Returns true if the task runner has been cancelled.
  bool IsCancelled() const;

  const std::string snapshot_stream_path_;
  Env* const env_;

  mutable mutex mu_;
  std::unique_ptr<TaskIterator> iterator_ TF_GUARDED_BY(mu_);
  Status status_ TF_GUARDED_BY(mu_);
  bool cancelled_ TF_GUARDED_BY(mu_) = false;

  std::unique_ptr<Thread> snapshot_thread_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_STREAM_WRITER_H_
