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

#include "tensorflow/compiler/tf2xla/kernels/case_op.h"

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/kernels/if_while_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/dynamic_shaped_ops.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

XlaCaseOp::XlaCaseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("branches", &unpruned_branches_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types_));
  if (!ctx->GetAttr(kXlaTokenInputNodesAttrName, &token_input_nodes_).ok()) {
    has_token_input_output_ = false;
  } else {
    has_token_input_output_ = !token_input_nodes_.empty();
  }
  if (ctx->HasAttr(kPropagateCompileTimeConsts)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kPropagateCompileTimeConsts,
                                     &propagate_compile_time_consts_));
  }
  if (!ctx->GetAttr(kXlaOriginalOutsideCompilationNodeName,
                    &original_node_name_)
           .ok())
    original_node_name_ = name();
}

std::pair<std::vector<NameAttrList>, xla::XlaOp>
XlaCaseOp::GetPrunedBranchesAndIndex(XlaOpKernelContext* ctx) {
  xla::Literal branch_index_literal;
  bool branch_index_is_constant =
      ctx->ConstantInput(0, &branch_index_literal).ok();

  if (!branch_index_is_constant) {
    return {unpruned_branches_, ctx->Input(0)};
  }

  int32_t branch_index = branch_index_literal.Get<int32>({});
  if (branch_index < 0 || branch_index >= unpruned_branches_.size()) {
    branch_index = unpruned_branches_.size() - 1;
  }

  std::vector<NameAttrList> pruned_branch = {unpruned_branches_[branch_index]};
  return {pruned_branch, xla::ZerosLike(ctx->Input(0))};
}

// TODO(b/35949885): There is duplication here with the handling of the
// while_op/if_op. Refactor the common code out/rework.
void XlaCaseOp::Compile(XlaOpKernelContext* ctx) {
  OP_REQUIRES(ctx, !unpruned_branches_.empty(),
              errors::InvalidArgument("Must provide at least one case branch"));
  OP_REQUIRES(ctx, input_type(0) == DT_INT32,
              errors::InvalidArgument(
                  "branch_index argument must be a int32 for XLA compilation"));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->InputShape(0)),
              errors::InvalidArgument(
                  "branch_index argument must be scalar for XLA compilation"));

  xla::XlaBuilder* b = ctx->builder();

  // We opportunistically prune out branches if the branch index is a
  // compile-time constant.  This is important in the context of the DeviceIndex
  // ops (and other such ops that may come later) since we may have a Case with
  // trivially unselected branches that cannot be compiled into HLO.
  std::vector<NameAttrList> branches;
  xla::XlaOp branch_index;
  std::tie(branches, branch_index) = GetPrunedBranchesAndIndex(ctx);

  int num_branches = branches.size();

  VLOG(1) << "Building Case: " << input_types_.size() << " inputs";

  std::vector<XlaCompiler::Argument> arguments(input_types_.size());
  int num_resource_args = 0;
  for (int i = 0; i < input_types_.size(); ++i) {
    XlaCompiler::Argument& arg = arguments[i];
    DataType type = ctx->input_type(i + 1);

    if (type == DT_RESOURCE) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(i + 1, &resource));
      XlaCompiler::PopulateArgumentFromResource(*resource, &arg);
      OP_REQUIRES(ctx, arg.initialized,
                  errors::Unimplemented("Uninitialized arguments: ", arg.name));
      VLOG(2) << "Resource " << resource->name()
              << " type: " << DataTypeString(arg.type)
              << " shape: " << arg.HumanString()
              << " initialized: " << arg.initialized;

      num_resource_args++;
    } else {
      arg.kind = XlaCompiler::Argument::kParameter;
      arg.type = input_types_[i];
      // Use the xla::Shape for the input instead of ctx->InputShape. This is
      // necessary for forwarding shapes of DT_VARIANTs, e.g. TensorLists.
      auto shape_or = ctx->builder()->GetShape(ctx->Input(i + 1));
      OP_REQUIRES_OK(ctx, shape_or.status());
      arg.shape = shape_or.value();
      VLOG(2) << "Arg type: " << DataTypeString(arg.type)
              << " shape: " << arg.HumanString();
    }
  }

  if (propagate_compile_time_consts_) {
    std::vector<std::vector<bool>> case_branch_must_be_const_nodes(
        num_branches);
    std::vector<const FunctionBody*> case_bodies(num_branches);
    for (int branch_idx = 0; branch_idx < num_branches; branch_idx++) {
      OP_REQUIRES_OK(ctx, FindMustBeConstNodes(
                              ctx, branches[branch_idx],
                              &case_branch_must_be_const_nodes[branch_idx],
                              &case_bodies[branch_idx]));
    }

    // Replaces `kParameter` type args in `arguments` with `kConstant` if
    // the op input corresponding to that arg is a compile-time const. This
    // is necessary to propagate compile time consts to ops in the branch
    // functions.
    auto arg_is_parameter = [&](int arg_idx) {
      if (arguments[arg_idx].kind != XlaCompiler::Argument::kParameter) {
        return false;
      }
      return true;
    };
    ConvertCompileTimeConstArgumentsToConst(ctx, &arguments,
                                            /*xla_expression_offset=*/1,
                                            arg_is_parameter);
  }

  // Compile each branch of the conditional.
  XlaCompiler::CompileOptions options;
  options.use_tuple_arg = true;
  options.return_updated_values_for_all_resources = true;
  options.is_entry_computation = false;
  options.add_token_input_output = has_token_input_output_;
  XlaCompiler* compiler = ctx->compiler();

  std::vector<XlaCompiler::CompilationResult> branch_results(num_branches);
  for (int j = 0; j < num_branches; ++j) {
    OP_REQUIRES_OK(ctx,
                   compiler->CompileFunction(options, branches[j], arguments,
                                             &branch_results[j]));
    OP_REQUIRES_OK(
        ctx,
        ctx->xla_context()->RecordCollectiveInfoFromNestedCompilationResult(
            branch_results[j]));
  }

  bool has_tensor_array_gradients = false;
  for (XlaCompiler::CompilationResult& result : branch_results) {
    for (const XlaCompiler::ResourceUpdate& update : result.resource_updates) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx,
                     ctx->GetResourceInput(update.input_index + 1, &resource));
      XlaCompiler::Argument& arg = arguments[update.input_index];

      // Add any TensorArray gradients touched by the then/else computation to
      // the enclosing graph.
      for (const string& grad_source : update.tensor_array_gradients_accessed) {
        VLOG(5) << "TensorArray " << resource->name() << " accessed gradient "
                << grad_source;
        XlaResource* gradient;
        OP_REQUIRES_OK(ctx, resource->GetOrCreateTensorArrayGradient(
                                grad_source, b, &gradient));
      }
      // Add all of the TensorArray gradients to the argument. For simplicity,
      // we always pass all known gradients.
      for (const auto& gradient : resource->tensor_array_gradients()) {
        arg.tensor_array_gradients.insert(gradient.first);
      }
      if (!resource->tensor_array_gradients().empty()) {
        has_tensor_array_gradients = true;
      }
    }
  }

  // Recompile the functions to update the argument shapes for tensor arrays.
  if (has_tensor_array_gradients) {
    for (int j = 0; j < num_branches; ++j) {
      branch_results[j] = {};
      OP_REQUIRES_OK(ctx,
                     compiler->CompileFunction(options, branches[j], arguments,
                                               &branch_results[j]));
    }
  }

  xla::Shape branch0_input_shape;
  std::vector<const xla::XlaComputation*> result_computations(num_branches);
  for (int j = 0; j < num_branches; ++j) {
    // Check that all branches have identical input shapes.
    OP_REQUIRES(ctx, branch_results[j].xla_input_shapes.size() == 1,
                errors::FailedPrecondition("Expected one input shape"));
    xla::Shape branch_input_shape = branch_results[j].xla_input_shapes[0];
    if (j == 0) {
      branch0_input_shape = branch_input_shape;
    }
    OP_REQUIRES(ctx, branch_input_shape.IsTuple(),
                errors::FailedPrecondition("Expected tuple shape"));
    OP_REQUIRES(
        ctx,
        xla::ShapeUtil::Compatible(branch0_input_shape, branch_input_shape),
        errors::InvalidArgument(
            "Input shapes of 0 and ", j, " branches do not match: ",
            xla::ShapeUtil::HumanString(branch0_input_shape), " vs. ",
            xla::ShapeUtil::HumanString(branch_input_shape)));

    if (j == 0) {
      VLOG(2) << "Input shape: "
              << xla::ShapeUtil::HumanString(branch0_input_shape);
      VLOG(2) << "Output shape: "
              << xla::ShapeUtil::HumanString(
                     branch_results[0].xla_output_shape);
    }

    // Check that all branches have same TensorList output indices.
    for (int output_index = 0; output_index < branch_results[0].outputs.size();
         output_index++) {
      bool is_tensor_list_in_branch_0 =
          branch_results[0].outputs[output_index].is_tensor_list;
      bool is_tensor_list_in_branch_j =
          branch_results[j].outputs[output_index].is_tensor_list;
      OP_REQUIRES(
          ctx, is_tensor_list_in_branch_0 == is_tensor_list_in_branch_j,
          errors::FailedPrecondition("Output #", output_index, " is ",
                                     (is_tensor_list_in_branch_0 ? "" : "not"),
                                     " a TensorList in branch 0, but is ",
                                     (is_tensor_list_in_branch_j ? "" : "not"),
                                     " a TensorList in branch ", j));
    }

    // We set return_updated_values_for_all_resources=true and we pass the same
    // arguments to both computations, so the resource update count must match.
    OP_REQUIRES(ctx,
                branch_results[0].resource_updates.size() ==
                    branch_results[j].resource_updates.size(),
                errors::FailedPrecondition(
                    "Different number of resources in 0 and ", j, " branch"));
    for (int i = 0; i < branch_results[0].resource_updates.size(); ++i) {
      const auto& lhs = branch_results[0].resource_updates[i];
      const auto& rhs = branch_results[j].resource_updates[i];
      bool equal = lhs.input_index == rhs.input_index &&
                   lhs.shape == rhs.shape &&
                   lhs.tensor_array_gradients_accessed ==
                       rhs.tensor_array_gradients_accessed;
      OP_REQUIRES(ctx, equal,
                  errors::FailedPrecondition("Mismatch in resource of 0 and ",
                                             j, " branch for resource ", i));
    }
    result_computations[j] = branch_results[j].computation.get();
  }

  // Prepare the input arg Tuple.
  int num_inputs = branch_results[0].input_mapping.size();
  std::vector<xla::XlaOp> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    int input_num = branch_results[0].input_mapping[i] + 1;
    if (has_token_input_output_ && i == num_inputs - 1) {
      // Set token input for this "case" op.
      std::vector<xla::XlaOp> token_inputs;
      token_inputs.reserve(token_input_nodes_.size());
      for (const string& node_name : token_input_nodes_) {
        auto token_or = compiler->GetNodeToken(node_name);
        OP_REQUIRES_OK(ctx, token_or.status());
        token_inputs.push_back(token_or.value());
      }
      inputs[i] = xla::AfterAll(b, token_inputs);
    } else if (ctx->input_type(input_num) == DT_RESOURCE) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(input_num, &resource));
      OP_REQUIRES_OK(ctx, resource->Pack(&inputs[i], b));
    } else {
      inputs[i] = ctx->Input(input_num);
    }
  }
  auto input_tuple = xla::Tuple(b, inputs);
  xla::XlaOp outputs = xla::DynamicConditional(
      ctx->builder(), branch_index, absl::MakeSpan(result_computations),
      std::vector<xla::XlaOp>(num_branches, input_tuple));
  // Sets non-variable outputs.
  for (int i = 0; i < output_types_.size(); ++i) {
    xla::XlaOp output_handle = xla::GetTupleElement(outputs, i);
    if (VLOG_IS_ON(2)) {
      LOG(INFO) << "Setting output " << i;
      auto shape_or = b->GetShape(output_handle);
      if (shape_or.ok()) {
        LOG(INFO) << "Shape for output " << i << ": "
                  << xla::ShapeUtil::HumanString(shape_or.value());
      } else {
        LOG(INFO) << "Shape unknown for output " << i;
      }
    }
    // We have checked that all branches have same TensorList output indices.
    if (branch_results[0].outputs[i].is_tensor_list) {
      ctx->SetTensorListOutput(i, output_handle);
    } else {
      ctx->SetOutput(i, output_handle);
    }
  }
  if (has_token_input_output_) {
    // Set token output for this "Case" op. Token output is the last output of
    // XLA computation, which comes after all "normal" TF outputs and resource
    // updates. For "Case" node, num of resource updates equals to number of
    // resource args because we set `return_updated_values_for_all_resources`
    // to true in XlaCompiler option.
    xla::XlaOp token_output =
        xla::GetTupleElement(outputs, output_types_.size() + num_resource_args);
    auto shape_or = b->GetShape(token_output);
    OP_REQUIRES_OK(ctx, shape_or.status());
    OP_REQUIRES(ctx, shape_or.value().IsToken(),
                errors::FailedPrecondition(
                    "Token output is not token type: ",
                    xla::ShapeUtil::HumanString(shape_or.value())));
    OP_REQUIRES_OK(ctx,
                   compiler->SetNodeToken(original_node_name_, token_output));
  }

  // Updates the values of any resource variables modified by the conditional
  // bodies.
  for (const XlaCompiler::CompilationResult& result : branch_results) {
    for (int i = 0; i < result.resource_updates.size(); ++i) {
      const XlaCompiler::ResourceUpdate& update = result.resource_updates[i];
      XlaResource* resource;
      OP_REQUIRES_OK(ctx,
                     ctx->GetResourceInput(update.input_index + 1, &resource));
      if (update.modified) {
        int pos = static_cast<int>(result.outputs.size()) + i;
        OP_REQUIRES_OK(ctx,
                       resource->SetFromPack(
                           arguments[update.input_index].tensor_array_gradients,
                           xla::GetTupleElement(outputs, pos), b));
      }
      VLOG(2) << "Case variable: pos: " << update.input_index
              << " name: " << resource->name()
              << " modified: " << update.modified
              << " type: " << DataTypeString(update.type)
              << " shape: " << update.shape.DebugString();
    }
  }
  VLOG(1) << "Done building Case";
}

REGISTER_XLA_OP(Name("Case").AllowResourceTypes().AllowVariantTypes(),
                XlaCaseOp);
REGISTER_XLA_OP(Name("StatelessCase").AllowResourceTypes().AllowVariantTypes(),
                XlaCaseOp);

}  // namespace tensorflow
