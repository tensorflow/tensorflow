/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_RENDER_OP_METRIC_GETTER_H_
#define XLA_HLO_TOOLS_HLO_DIFF_RENDER_OP_METRIC_GETTER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace hlo_diff {

// An interface for getting op metrics.
class OpMetricGetter {
 public:
  virtual ~OpMetricGetter() = default;

  // Returns the op time in picoseconds.
  virtual absl::StatusOr<uint64_t> GetOpTimePs(
      absl::string_view op_name) const = 0;
};

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_OP_METRIC_GETTER_H_
