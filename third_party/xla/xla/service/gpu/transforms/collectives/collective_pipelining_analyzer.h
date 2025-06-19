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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_PIPELINING_ANALYZER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_PIPELINING_ANALYZER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Traverses the HLO and adds metadata to collectives' frontend attributes if it
// finds that a collective is trivially pipeline-able. That is:
// * No heavy dependencies will be pipelined along with it.
// * Loop tuple size will not increase.
// The output of this pass is supposed to be consumed in subsequent
// `CollectivePipeliner` passes.
//
// We define a collective to be trivially pipeline-eable if it is
// followed/preceded by dynamic-update-slice/dynamic-slice which operates only
// on induction variable, and no heavy (anything not no-op and simple converts,
// reshapes, transposes, etc.) ops follow/preceed it.
// The pass supports detecting such cases for AllReduce, AllGather, and
// ReduceScatter.
//
// Advantage of having this as a pass is that in case heuristic is not good
// enough, users will have an escape hatch to disable it via
// `xla_disable_hlo_passes` flag.
class CollectivePipeliningAnalyzer : public HloModulePass {
 public:
  CollectivePipeliningAnalyzer() = default;

  absl::string_view name() const override {
    return "collective-pipelining-analyzer";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_PIPELINING_ANALYZER_H_
