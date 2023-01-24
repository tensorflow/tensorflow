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
#include "tensorflow/core/data/service/snapshot/snapshot_split_provider.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "absl/time/time.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

SnapshotSplitProvider::SnapshotSplitProvider(
    const std::string& address, const std::string& protocol,
    const SnapshotTaskDef& snapshot_task, int64_t source_index,
    absl::Duration timeout)
    : address_(address),
      protocol_(protocol),
      snapshot_task_(snapshot_task),
      source_index_(source_index),
      timeout_(timeout) {}

Status SnapshotSplitProvider::GetNext(Tensor* split, bool* end_of_splits)
    TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  if (!dispatcher_) {
    dispatcher_ =
        std::make_unique<DataServiceDispatcherClient>(address_, protocol_);
  }
  return grpc_util::Retry(
      [this, split, end_of_splits] {
        return dispatcher_->GetSnapshotSplit(
            snapshot_task_.base_path(), snapshot_task_.stream_index(),
            source_index_, *split, *end_of_splits);
      },
      "Get next split for snapshot",
      /*deadline_micros=*/Env::Default()->NowMicros() +
          absl::ToInt64Microseconds(timeout_));
}

Status SnapshotSplitProvider::Reset() {
  return errors::FailedPrecondition(
      "tf.data SnapshotSplitProvider does not support `Reset`.");
}

Status SnapshotSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) {
  return errors::Unimplemented(
      "Save is not implemented for tf.data SnapshotSplitProvider.");
}

Status SnapshotSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) {
  return errors::Unimplemented(
      "Restore is not implemented for tf.data SnapshotSplitProvider.");
}

}  // namespace data
}  // namespace tensorflow
