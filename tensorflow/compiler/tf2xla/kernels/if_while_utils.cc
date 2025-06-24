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

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/hlo/builder/value_inference.h"
#include "xla/literal.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"

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
      VLOG(1) << "Trying to resolve constant " << i;
      // NOTE: We can not simply check that this is Kind::kConstant because
      // this could be the output of a MetadataOnly op e.g. Size.

      // If we can infer the constant values of an inner computation's argument,
      // replace them with constants. If that fails, we fallback to infer the
      // bounds of the argument.
      absl::StatusOr<std::optional<Tensor>> maybe_constant =
          expression.ResolveConstant(ctx->compiler()->client());
      absl::StatusOr<std::optional<Tensor>> bounds =
          expression.ResolveConstant(ctx->compiler()->client(), false,
                                     xla::ValueInferenceMode::kUpperBound);
      if ((maybe_constant.ok() && maybe_constant->has_value()) ||
          (bounds.ok() && bounds->has_value())) {
        absl::StatusOr<Tensor> values_are_dynamic =
            expression.ResolveDynamism();
        bool all_values_are_static = false;
        if (values_are_dynamic.ok()) {
          xla::Literal literal =
              HostTensorToLiteral(values_are_dynamic.value()).value();
          all_values_are_static = literal.IsAll(0);
        }

        if (all_values_are_static) {
          arg->kind = XlaCompiler::Argument::kConstant;
          arg->type = expression.dtype();
          arg->constant_value = std::move(maybe_constant.value().value());
          arg->shape = expression.GetShape().value();
          resolved_constant_idxs.push_back(i);
        } else {
          arg->value_bound.emplace(std::move(bounds.value().value()));
          arg->value_dynamism.emplace(std::move(values_are_dynamic.value()));
        }
      }
    }
  }
  return resolved_constant_idxs;
}

absl::Status FindMustBeConstNodes(XlaOpKernelContext* ctx,
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
