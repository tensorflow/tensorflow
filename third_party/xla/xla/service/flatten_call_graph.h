/* Copyright 2017 The OpenXLA Authors.

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

// Flatten the call graph for an HLO module into a tree.

#ifndef XLA_SERVICE_FLATTEN_CALL_GRAPH_H_
#define XLA_SERVICE_FLATTEN_CALL_GRAPH_H_

#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

// Flattening associates each call site with a unique computation (for
// sequential calling contexts) This simplifies buffer assignment and
// points-to analysis (see b/36865746 for details).
class FlattenCallGraph : public HloModulePass {
 public:
  absl::string_view name() const override { return "flatten-call-graph"; }

  // Duplicates computations called from multiple call- or while-nodes to
  // flatten the call graph.
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_FLATTEN_CALL_GRAPH_H_
