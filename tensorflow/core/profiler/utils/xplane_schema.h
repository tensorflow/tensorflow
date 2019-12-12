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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {

constexpr int kNumStatTypes = 27;

enum StatType {
  kUnknown = 0,
  // TraceMe arguments.
  kStepId,
  kParentStepId,
  kFunctionStepId,
  kDeviceOrdinal,
  kChipOrdinal,
  kNodeOrdinal,
  kModelId,
  kQueueAddr,
  kRequestId,
  kRunId,
  kCorrelationId,
  kGraphType,
  kStepNum,
  kIterNum,
  kIndexOnHost,
  kBytesReserved,
  kBytesAllocated,
  kBytesAvailable,
  kFragmentation,
  kKernelDetails,
  // Stats added when processing traces.
  kGroupId,
  kStepName,
  kLevel0,
  kTfOp,
  kHloOp,
  kHloModule,
};

constexpr absl::string_view kStatTypeStrMap[] = {
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

inline absl::string_view GetStatTypeStr(StatType stat_type) {
  return kStatTypeStrMap[stat_type];
}

inline bool IsStatType(StatType stat_type, absl::string_view stat_name) {
  return kStatTypeStrMap[stat_type] == stat_name;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
