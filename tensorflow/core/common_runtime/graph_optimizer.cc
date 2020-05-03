/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/graph_optimizer.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"

namespace tensorflow {

GraphOptimizer::GraphOptimizer(const OptimizerOptions& opts) : opts_(opts) {
  if (opts_.opt_level() >= OptimizerOptions::L1) {
    opts_.set_do_common_subexpression_elimination(true);
    opts_.set_do_constant_folding(true);
  }
}

GraphOptimizer::~GraphOptimizer() {}

void GraphOptimizer::Optimize(
    FunctionLibraryRuntime* runtime, Env* env, const Device* device,
    std::unique_ptr<Graph>* graph,
    const std::unordered_map<string, std::vector<PartialTensorShape>>*
        shape_map,
    const NodePredicate& cse_consider_fn, const NodePredicate& cf_consider_fn,
    bool inline_multi_device_functions,
    bool inline_impl_selection_group_functions,
    bool inline_with_single_device_body_placer) {
  Graph* g = graph->get();
  DumpGraph("Initial", g);

  bool changed = true;
  const int kMaxRounds = 10;
  for (int rounds = 0; rounds < kMaxRounds; ++rounds) {
    changed = false;
    if (RemoveListArrayConverter(g)) {
      DumpGraph("RemoveListArrayConverter", g);
      changed = true;
    }
    if (opts_.do_function_inlining() && RemoveDeadNodes(g)) {
      DumpGraph("RemoveDeadNodes", g);
      changed = true;
    }
    if (opts_.do_function_inlining() && RemoveIdentityNodes(g)) {
      DumpGraph("RemoveIdentityNodes", g);
      changed = true;
    }

    if (opts_.do_constant_folding()) {
      ConstantFoldingOptions cf_opts;
      cf_opts.shape_map = shape_map;
      cf_opts.consider = cf_consider_fn;
      if (opts_.max_folded_constant_in_bytes() > 0) {
        cf_opts.max_constant_size_in_bytes =
            opts_.max_folded_constant_in_bytes();
      }
      bool was_mutated;
      ConstantFold(cf_opts, runtime, env, device, g, &was_mutated)
          .IgnoreError();
      if (was_mutated) {
        RemoveDeadNodes(g);
        DumpGraph("ConstFolding", g);
        changed = true;
      }
    }

    if (opts_.do_function_inlining() && FixupSourceAndSinkEdges(g)) {
      DumpGraph("FixupSourceAndSinkEdges", g);
      changed = true;
    }
    if (opts_.do_common_subexpression_elimination() &&
        OptimizeCSE(g, cse_consider_fn)) {
      DumpGraph("OptimizeCSE", g);
      changed = true;
    }
    if (opts_.do_function_inlining()) {
      ExpandInlineFunctionsOptions expand_inline_opts;
      expand_inline_opts.native_options.inlined_function_body_placer =
          InlinedFunctionBodyPlacer::SingleDevice();

      // Force single device placement strategy for multi-device function body.
      if (inline_with_single_device_body_placer) {
        expand_inline_opts.multi_device_options.inlined_function_body_placer =
            InlinedFunctionBodyPlacer::SingleDevice();
      }

      if (!inline_multi_device_functions) {
        // GraphOptimizer is running:
        //   (1) After partitioning when executing with a Session API.
        //   (2) For a single device function body after instantiation.
        // We can't inline multi-device functions in these cases, because it
        // might lead to multiple device assignments.
        expand_inline_opts.multi_device_options.disable_inlining = true;
      }
      if (inline_impl_selection_group_functions) {
        expand_inline_opts.native_options
            .inline_impl_selection_group_functions = true;
        expand_inline_opts.multi_device_options
            .inline_impl_selection_group_functions = true;
      }

      bool was_mutated = ExpandInlineFunctions(runtime, g, expand_inline_opts);
      if (was_mutated) {
        DumpGraph("ExpandInlineFunctions", g);
        changed = true;
      }
    }
    if (!changed) break;
  }

  // Note that we use the Graph constructor that copies the input
  // FunctionLibraryDefinition, since the original lib def will go out of scope.
  std::unique_ptr<Graph> copy(new Graph(g->flib_def()));
  CopyGraph(*g, copy.get());
  graph->swap(copy);

  DumpGraph("ReCopy", graph->get());
}

void GraphOptimizer::Optimize(FunctionLibraryRuntime* runtime, Env* env,
                              const Device* device,
                              std::unique_ptr<Graph>* graph,
                              const Options& options) {
  Optimize(runtime, env, device, graph, options.shape_map,
           options.cse_consider_fn, options.cf_consider_fn,
           options.inline_multi_device_functions,
           options.inline_impl_selection_group_functions,
           options.inline_with_single_device_body_placer);
}

}  // end namespace tensorflow
