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

#include "tensorflow/compiler/tf2xla/kernels/if_while_utils.h"

#include "tensorflow/compiler/tf2xla/const_analysis.h"

namespace tensorflow {

const char kPropagateCompileTimeConsts[] = "_xla_propagate_compile_time_consts";

absl::InlinedVector<int, 5> ConvertCompileTimeConstArgumentsToConst(
    XlaOpKernelContext* ctx, std::vector<XlaCompiler::Argument>* args,
    int xla_expression_offset,
    std::function<bool(int arg_idx)> should_resolve_constant) {
  absl::InlinedVector<int, 5> resolved_constant_idxs;
  for (int i = 0; i < args->size(); i++) {
    XlaCompiler::Argument* arg = &(*args)[i];
    const XlaExpression& expression =
        ctx->InputExpression(i + xla_expression_offset);
    // If the input tensor is a compile time constant build a kConstant type
    // argument.
    if (should_resolve_constant(i)) {
      // NOTE: We can not simply check that this is Kind::kConstant because
      // this could be the output of a MetadataOnly op e.g. Size.
      xla::StatusOr<absl::optional<Tensor>> maybe_constant =
          expression.ResolveConstant(ctx->compiler()->client());
      if (maybe_constant.ok() && maybe_constant.ValueOrDie().has_value()) {
        arg->kind = XlaCompiler::Argument::kConstant;
        arg->type = expression.dtype();
        arg->constant_value = std::move(maybe_constant.ValueOrDie().value());
        arg->shape = expression.GetShape().ValueOrDie();
        resolved_constant_idxs.push_back(i);
      }
    }
  }
  return resolved_constant_idxs;
}

Status FindMustBeConstNodes(XlaOpKernelContext* ctx,
                            const NameAttrList& func_name,
                            std::vector<bool>* must_be_const_nodes,
                            const FunctionBody** body) {
  TF_RETURN_IF_ERROR(ctx->compiler()->FindFunctionBody(func_name, body));
  must_be_const_nodes->resize((*body)->graph->num_node_ids(), false);
  return BackwardsConstAnalysis(*((*body)->graph),
                                /*compile_time_const_arg_indices=*/nullptr,
                                must_be_const_nodes, ctx->function_library());
}

}  // namespace tensorflow
