/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/force_xla_constants_on_host_pass.h"

#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

Status ForceXlaConstantsOnHostPass::Run(
    const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();

  OptimizerOptions opts;
  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, options.session_options->env, /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, options.flib_def, opts);
  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  for (Node* node : graph->nodes()) {
    if (CanCreateXlaKernel(node->def())) {
      const FunctionBody* fbody = nullptr;
      std::vector<int> constant_arg_indices;
      std::vector<int> resource_arg_indices;

      NameAttrList function;
      TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(node->def(), &function));

      // Force all constants to be on the host memory.
      TF_RETURN_IF_ERROR(GetBodyAndConstantsAndResources(
          flr, function, &fbody, &constant_arg_indices, &resource_arg_indices));
      VLOG(3) << "Found constant arg indices: "
              << absl::StrJoin(constant_arg_indices, ", ");

      node->AddAttr("_input_hostmem", constant_arg_indices);
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
