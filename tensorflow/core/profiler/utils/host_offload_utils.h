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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_HOST_OFFLOAD_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_HOST_OFFLOAD_UTILS_H_

#include <cstdint>
#include <optional>
#include <queue>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/layout.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

class HostOffloadEventProcessor {
 public:
  HostOffloadEventProcessor(XPlaneBuilder* plane_builder,
                            XLineBuilder* host_offload_op_line_builder)
      : plane_builder_(plane_builder),
        host_offload_op_line_builder_(host_offload_op_line_builder) {}
  ~HostOffloadEventProcessor() = default;

  void ProcessHostOffloadOpEvent(const XEventVisitor& event,
                                 std::optional<int64_t> group_id);

  bool IsHostOffloadOpName(const XEventVisitor& event) const;

 private:
  std::string GetOffloadInstructionID(absl::string_view op_name) const;
  std::string GetOffloadInstructionName(absl::string_view op_name) const;

  absl::flat_hash_map<std::string, std::queue<const XEventVisitor*>>
      seen_events_;
  std::string host_memory_label_ =
      absl::StrCat("S(", xla::Layout::kHostMemorySpace, ")");

  XPlaneBuilder* plane_builder_;
  XLineBuilder* host_offload_op_line_builder_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HOST_OFFLOAD_UTILS_H_
