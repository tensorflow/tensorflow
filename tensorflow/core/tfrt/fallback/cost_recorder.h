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

// This file defines a recorder for op cost measurement.

#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_COST_RECORDER_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_COST_RECORDER_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/tfrt/fallback/op_cost_map.pb.h"
#include "tfrt/host_context/shared_context.h"  // from @tf_runtime

namespace tfrt {
class HostContext;
}  // namespace tfrt

namespace tensorflow {
namespace tfrt_stub {
class CostRecorder : public tfrt::SharedContext {
 public:
  explicit CostRecorder(tfrt::HostContext* host) {}

  void RecordCost(int64_t op_key, const uint64_t execution_time);

  size_t size();
  Status WriteToFile();

 private:
  tensorflow::mutex op_cost_map_mutex_;
  // Map op key to {sum of op execution time in microseconds, number of op}.
  absl::flat_hash_map<int64_t, std::pair<uint64_t, uint64_t>> op_cost_map_
      TF_GUARDED_BY(op_cost_map_mutex_);
};
}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_COST_RECORDER_H_
