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
#ifndef XLA_SERVICE_HOST_OFFLOADER_H_
#define XLA_SERVICE_HOST_OFFLOADER_H_

#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

class HloCostAnalysis;

// This pass does "host memory offloading". If a tensor is annotated to be moved
// to or from the host, this pass will remove the annotations and update each
// tensor's layout with host memory spaces and insert copies* if necessary. This
// pass checks to make sure that no compute is done on the tensors annotated for
// host memory offload; if there is compute, it is considered a user error and
// an error will be returned.
//
// * TODO(b/319293918): Inserting of copies is not yet implemented.
class HostOffloader : public HloModulePass {
 public:
  static constexpr absl::string_view kPipelineForwardTarget = "PipelineForward";
  static constexpr absl::string_view kPipelineBackwardTarget =
      "PipelineBackward";

  explicit HostOffloader(int64_t host_memory_space_color)
      : kHostMemorySpaceColor(host_memory_space_color) {}
  ~HostOffloader() override = default;

  absl::string_view name() const override { return "host-offloader"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const int64_t kHostMemorySpaceColor;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  absl::flat_hash_set<HloInstruction*> custom_calls_to_remove_;
  absl::flat_hash_set<HloInstruction*> broadcasts_to_replace_;
  absl::flat_hash_set<const HloBuffer*> buffers_to_move_to_host_memory_;

  Status HandlePipelineForwardCustomCall(HloInstruction* custom_call);
  void HandlePipelineBackwardCustomCall(HloInstruction* custom_call);
  Status DynamifySlice(HloInstruction* slice);
};

}  // namespace xla

#endif  // XLA_SERVICE_HOST_OFFLOADER_H_
