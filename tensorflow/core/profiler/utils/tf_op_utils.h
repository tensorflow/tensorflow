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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_

#include <utility>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

namespace tensorflow {
namespace profiler {

// Special op types.
constexpr absl::string_view kUnknownOp = "";  // op types are non-empty strings
constexpr absl::string_view kDatasetOp = "Dataset";
constexpr absl::string_view kIterator = "Iterator";
constexpr absl::string_view kSeparator = "::";

// Breaks a TensorFlow op fullname into name and type.
struct TfOp {
  absl::string_view name;
  absl::string_view type;
};

TfOp ParseTfOpFullname(absl::string_view tf_op_fullname);

// Trace event name for TF ops is the op type so they have the same color in
// trace viewer.
std::string TfOpEventName(absl::string_view tf_op_fullname);

// Returns true if the given name is not a TensorFlow op.
inline bool IsUnknownOp(absl::string_view tf_op_type) {
  return tf_op_type == kUnknownOp;
}

// Returns true if the given name is a TensorFlow Dataset Op.
inline bool IsDatasetOp(absl::string_view tf_op_type) {
  return tf_op_type == kDatasetOp;
}

constexpr size_t kNumStatType = 27;

enum class StatType {
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

constexpr std::array<absl::string_view, kNumStatType> kStatTypeStrMap({
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
});

inline absl::string_view GetStatTypeStr(StatType stat_type) {
  return kStatTypeStrMap.at(static_cast<std::size_t>(stat_type));
}

inline bool IsStatType(StatType stat_type, absl::string_view stat_name) {
  return kStatTypeStrMap.at(static_cast<std::size_t>(stat_type)) == stat_name;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
