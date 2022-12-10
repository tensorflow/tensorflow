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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace data {

SnapshotStreamWriter::SnapshotStreamWriter(
    std::unique_ptr<TaskIterator> iterator,
    const std::string& snapshot_stream_path, Env* env)
    : snapshot_stream_path_(snapshot_stream_path),
      env_(env),
      iterator_(std::move(iterator)),
      snapshot_thread_(RunSnapshotThread()) {}

std::unique_ptr<Thread> SnapshotStreamWriter::RunSnapshotThread() {
  auto snapshot_fn = [this]() TF_LOCKS_EXCLUDED(mu_) {
    Status status = WriteSnapshotFn();
    {
      mutex_lock l(mu_);
      status_ = std::move(status);
    }
  };
  return absl::WrapUnique(env_->StartThread(
      /*thread_options=*/{}, /*name=*/"tf_data_service_snapshot_thread",
      std::move(snapshot_fn)));
}

Status SnapshotStreamWriter::WriteSnapshotFn() {
  // TODO(b/258691666): Support compression.
  snapshot_util::TFRecordWriter writer(snapshot_stream_path_,
                                       tsl::io::compression::kNone);
  TF_RETURN_IF_ERROR(writer.Initialize(env_));
  auto cleanup = gtl::MakeCleanup([&writer] { writer.Close().IgnoreError(); });
  while (!IsCancelled()) {
    std::vector<Tensor> element;
    bool end_of_sequence = false;
    {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(iterator_->GetNext(element, end_of_sequence));
    }
    if (end_of_sequence) {
      return writer.Close();
    }
    TF_RETURN_IF_ERROR(writer.WriteTensors(element));
  }
  return errors::Cancelled(
      "The tf.data service snapshot writer has been cancelled.");
}

Status SnapshotStreamWriter::Wait() TF_LOCKS_EXCLUDED(mu_) {
  snapshot_thread_.reset();
  mutex_lock l(mu_);
  return status_;
}

void SnapshotStreamWriter::Cancel() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  cancelled_ = true;
}

bool SnapshotStreamWriter::IsCancelled() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return cancelled_;
}

}  // namespace data
}  // namespace tensorflow
