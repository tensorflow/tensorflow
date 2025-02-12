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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_GPU_EVENT_STATS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_GPU_EVENT_STATS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// Stats from a GPU stream XEvent.
struct GpuEventStats {
  explicit GpuEventStats(const XEventVisitor* event);

  bool IsKernel() const { return !kernel_details.empty(); }
  bool IsMemCpy() const { return !memcpy_details.empty(); }
  bool IsCudaGraphExecution() const { return cuda_graph_exec_id.has_value(); }

  bool IsXlaOp() const { return !hlo_op_names.empty(); }
  bool IsTfOp() const { return !tf_op_fullname.empty(); }

  // Stats from TensorFlow.
  absl::string_view tf_op_fullname;
  absl::string_view equation;
  absl::string_view tensor_shapes;

  // Stats from XLA.
  std::vector<absl::string_view> hlo_op_names;
  absl::string_view hlo_module_name;
  std::optional<uint64_t> program_id;

  // Stats from CUPTI.
  absl::string_view kernel_details;
  absl::string_view memcpy_details;
  std::optional<int64_t> correlation_id;
  std::optional<int64_t> scope_range_id;

  // Stats derived by grouping.
  std::optional<int64_t> group_id;
  bool is_eager = false;
  std::optional<uint64_t> cuda_graph_exec_id;
  std::optional<uint64_t> cuda_graph_id_for_inner_node;
};

// Stats for a host-side GPU launch XEvent.
struct LaunchEventStats {
  explicit LaunchEventStats(const XEventVisitor* event);

  bool IsLaunch() const {
    return device_id.has_value() && correlation_id.has_value();
  }

  // Stats from CUPTI.
  std::optional<int64_t> device_id;
  std::optional<int64_t> correlation_id;

  // Stat derived by grouping.
  std::optional<int64_t> group_id;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_GPU_EVENT_STATS_H_
