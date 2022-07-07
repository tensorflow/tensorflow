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

// This file defines a recorder for op cost measurement

#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_COST_RECORDER_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_COST_RECORDER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tfrt/host_context/shared_context.h"  // from @tf_runtime

namespace tfrt {
class HostContext;
}  // namespace tfrt

namespace tensorflow {
namespace tfrt_stub {
class CostRecorder : public tfrt::SharedContext {
 public:
  explicit CostRecorder(tfrt::HostContext* host) {}

  // TODO(xiangll): This is used for cost measurement only. Clean up after the
  // measurement is done.
  void RecordCost(const absl::string_view op_name,
                  const uint64_t run_duration) {
    cost_per_op_map_[op_name] = run_duration;
  }

 private:
  // Map op name to op run duration in terms of microseconds.
  absl::flat_hash_map<absl::string_view, uint64_t> cost_per_op_map_;
};
}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_COST_RECORDER_H_
