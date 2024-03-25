/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_HOST_OFFLOAD_LEGALIZE_H_
#define XLA_SERVICE_HOST_OFFLOAD_LEGALIZE_H_

#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

class HloCostAnalysis;

// This pass legalizes the graph for the "host memory offloading" pass to
// correctly identified buffers that are meant to be move on the host. Any
// legalization that could block that is welcome into this pass.
class HostOffloadLegalize : public HloModulePass {
 public:
  explicit HostOffloadLegalize(int64_t host_memory_space_color,
                               bool after_layout)
      : kHostMemorySpaceColor(host_memory_space_color),
        after_layout_(after_layout) {}
  ~HostOffloadLegalize() override = default;

  absl::string_view name() const override { return "host-offload-legalize"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const int64_t kHostMemorySpaceColor;
  const bool after_layout_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HOST_OFFLOAD_LEGALIZE_H_
