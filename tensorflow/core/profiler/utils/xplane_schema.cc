/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {

const int kNumStatTypes = static_cast<int>(StatType::kHloModule) + 1;

static constexpr absl::string_view kStatTypeStrMap[kNumStatTypes] = {
    "unknown",         "id",
    "parent_step_id",  "function_step_id",
    "device_ordinal",  "chip_ordinal",
    "node_ordinal",    "model_id",
    "queue_addr",      "request_id",
    "run_id",          "correlation_id",
    "graph_type",      "step_num",
    "iter_num",        "index_on_host",
    "bytes_reserved",  "bytes_allocated",
    "bytes_available", "fragmentation",
    "kernel_details",  "group_id",
    "step_name",       "level 0",
    "tf_op",           "hlo_op",
    "hlo_module",
};

absl::Span<const absl::string_view> GetStatTypeStrMap() {
  return absl::MakeConstSpan(kStatTypeStrMap, kNumStatTypes);
}

}  // namespace profiler
}  // namespace tensorflow
