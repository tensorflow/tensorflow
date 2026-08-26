/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/disable_functional_ops_lowering_for_xla_pass.h"

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

namespace {

// Returns the effective global JIT level for `options`. Unlike
// `GetGlobalJitLevelForGraph`, this also works when `options.session_options`
// is null, which can happen when auto-clustering is enabled purely via the
// TF_XLA_FLAGS=--tf_xla_auto_jit=N environment variable and the execution
// path never populates a SessionOptions. In that case we combine just the
// flag-provided auto-jit level with whether the graph is a single-GPU graph,
// mirroring what `GetGlobalJitLevelForGraph` does for a DEFAULT ConfigProto
// setting.
OptimizerOptions::GlobalJitLevel GetEffectiveGlobalJitLevel(
    const GraphOptimizationPassOptions& options, const Graph& graph) {
  if (options.session_options != nullptr) {
    return GetGlobalJitLevelForGraph(options);
  }

  const XlaAutoJitFlag& auto_jit_flag =
      GetMarkForCompilationPassFlags()->xla_auto_jit_flag;
  auto level_or_off = [](int32_t level) {
    return level == OptimizerOptions::DEFAULT
               ? OptimizerOptions::OFF
               : static_cast<OptimizerOptions::GlobalJitLevel>(level);
  };
  return IsSingleGpuGraph(graph)
             ? level_or_off(auto_jit_flag.optimization_level_single_gpu)
             : level_or_off(auto_jit_flag.optimization_level_general);
}

}  // namespace

absl::Status DisableFunctionalOpsLoweringForXlaPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr || options.graph->get() == nullptr) {
    return absl::OkStatus();
  }
  Graph* graph = options.graph->get();
  if (GetEffectiveGlobalJitLevel(options, *graph) < OptimizerOptions::ON_1) {
    return absl::OkStatus();
  }
  for (Node* n : graph->op_nodes()) {
    if (!n->IsIfNode() && !n->IsCaseNode() && !n->IsWhileNode()) continue;
    bool lower = false;
    if (TryGetNodeAttr(n->attrs(), "_lower_using_switch_merge", &lower) &&
        lower) {
      n->ClearAttr("_lower_using_switch_merge");
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
