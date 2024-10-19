/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/lower_function_call_op.h"

#include <utility>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/lower_function_call_inline_policy.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"

namespace tensorflow {

using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;
using OutputControlSrc = InlineFunctionBodyOptions::OutputControlSource;

absl::Status RewriteFunctionCallNode(Node* n, Graph* g,
                                     const FunctionLibraryDefinition& flib_def,
                                     bool keep_caller_fetchable) {
  VLOG(2) << "Lower function call node: " << SummarizeNode(*n);

  // We support lowering of two types of functions that could be invoked by the
  // node `n`: 1) native functions and 2) multi-device functions.
  // NOTE(ezhulenev): We explicitly choose not to deal with SymbolicGradient,
  // because it has been deprecated for a long time.
  InlineFunctionBodyOptions inline_options;
  inline_options.keep_caller_node = keep_caller_fetchable
                                        ? KeepCallerNode::kFetchable
                                        : KeepCallerNode::kTargetable;

  FunctionCallInlinePolicy policy = GetFunctionCallInlinePolicy(n);
  if (policy == FunctionCallInlinePolicy::kMultiDevicePlacer) {
    // Multi-device function calls (PartitionedCall or StatefulPartitionedCall
    // ops) can execute on multiple devices and accept DT_RESOURCE inputs that
    // belong to different devices. This type of functions was added in
    // Tensorflow 2.0 Eager mode, and it has control outputs to represent
    // side-effects that must always execute (see `control_ret` in FunctionDef).
    inline_options.output_control_src = OutputControlSrc::kControlOutputs;
    inline_options.inlined_function_body_placer =
        InlinedFunctionBodyPlacer::MultiDevice();
  } else if (policy == FunctionCallInlinePolicy::kSingleDevicePlacer) {
    // Native function call (node.type_string() is the function name). These
    // functions are always executed on a single-device, which is the device of
    // the function call node.
    inline_options.output_control_src = OutputControlSrc::kDataOutputs;
    inline_options.inlined_function_body_placer =
        InlinedFunctionBodyPlacer::SingleDevice();
  } else {
    return errors::InvalidArgument("Unsupported function inlining policy");
  }

  core::RefCountPtr<FunctionRecord> fdef;
  if (n->IsPartitionedCall()) {
    NameAttrList func;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "f", &func));
    fdef = flib_def.FindRecord(func.name());
  } else if (n->type_string() == FunctionLibraryDefinition::kGradientOp) {
    VLOG(2) << "Skip SymbolicGradient lowering";
    return absl::OkStatus();
  } else {
    fdef = flib_def.FindRecord(n->type_string());
  }

  if (fdef == nullptr) {
    return errors::Internal("Can't find a function: node=", SummarizeNode(*n));
  }

  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(std::move(fdef), n->attrs(), &flib_def, &fbody));

  if (flags::Global().enable_function_pruning_before_inlining.value()) {
    // TODO(b/341325107): Enable this path by default and remove the flag.
    VLOG(2) << "Pruning enabled before inlining";
    // NOTE(mrry): We pass `fbody->arg_nodes` as an additional set of roots,
    // because otherwise the `FunctionBody` state will become inconsistent.
    // The unused `Identity` nodes will be colocated with the arguments, and
    // pruned in a subsequent pass.
    PruneFunctionBody(
        fbody->record->fdef(), fbody->graph,
        absl::Span<Node*>(fbody->arg_nodes.data(), fbody->arg_nodes.size()));
  } else {
    VLOG(2) << "Pruning disabled before inlining";
  }

  absl::Status can_inline_function_call =
      ValidateInlining(n, fbody.get(), inline_options);
  if (can_inline_function_call.ok()) {
    TF_RETURN_IF_ERROR(
        InlineFunctionBody(flib_def, g, n, fbody.get(), inline_options));
  } else {
    VLOG(2) << "Failed to inline function call node: "
            << can_inline_function_call.message();
  }

  return absl::OkStatus();
}

}  // namespace tensorflow
