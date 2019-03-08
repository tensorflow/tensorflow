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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

XlaCaseOp::XlaCaseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("branches", &branches_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types_));
  if (!ctx->GetAttr(kXlaTokenInputNodesAttrName, &token_input_nodes_).ok()) {
    has_token_input_output_ = false;
  } else {
    has_token_input_output_ = !token_input_nodes_.empty();
  }
}

// TODO(b/35949885): There is duplication here with the handling of the
// while_op. Refactor the common code out/rework.
void XlaCaseOp::Compile(XlaOpKernelContext* ctx) {
  xla::XlaBuilder* b = ctx->builder();
  int num_branches = branches_.size();
  OP_REQUIRES(ctx, num_branches >= 1,
              errors::InvalidArgument("Must provide at least one case branch"));
  OP_REQUIRES(ctx, input_type(0) == DT_INT32,
              errors::InvalidArgument(
                  "branch_index argument must be a int32 for XLA compilation"));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->InputShape(0)),
              errors::InvalidArgument(
                  "branch_index argument must be scalar for XLA compilation"));

  VLOG(1) << "Building Case: " << input_types_.size() << " inputs";

  std::vector<XlaCompiler::Argument> arguments(input_types_.size());
  int num_resource_args = 0;
  for (int i = 0; i < input_types_.size(); ++i) {
    XlaCompiler::Argument& arg = arguments[i];
    DataType type = ctx->input_type(i + 1);

    if (type == DT_RESOURCE) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(i + 1, &resource));

      arg.initialized = resource->initialized();
      arg.kind = XlaCompiler::Argument::kResource;
      arg.resource_kind = resource->kind();

      arg.type = resource->type();
      arg.shape = resource->shape();
      OP_REQUIRES(ctx, arg.initialized,
                  errors::Unimplemented("Uninitialized arguments: ", arg.name));
      arg.max_array_size = resource->max_array_size();
      for (const auto& gradient : resource->tensor_array_gradients()) {
        arg.tensor_array_gradients.insert(gradient.first);
      }
      arg.name = resource->name();
      VLOG(2) << "Resource " << resource->name()
              << " type: " << DataTypeString(arg.type)
              << " shape: " << arg.HumanString()
              << " initialized: " << arg.initialized;

      num_resource_args++;
    } else {
      arg.kind = XlaCompiler::Argument::kParameter;
      arg.type = input_types_[i];
      arg.shape = ctx->InputShape(i + 1);
      VLOG(2) << "Arg type: " << DataTypeString(arg.type)
              << " shape: " << arg.HumanString();
    }
  }

  // Compile each branch of the conditional.
  XlaCompiler::CompileOptions options;
  options.use_tuple_arg = true;
  options.resolve_compile_time_constants = false;
  options.return_updated_values_for_all_resources = true;
  options.is_entry_computation = false;
  options.add_token_input_output = has_token_input_output_;
  XlaCompiler* compiler = ctx->compiler();

  std::vector<XlaCompiler::CompilationResult> branch_results(num_branches);
  std::vector<XlaCompiler::CompilationResult*> branch_results_p(num_branches);
  for (int j = 0; j < num_branches; ++j) {
    OP_REQUIRES_OK(ctx,
                   compiler->CompileFunction(options, branches_[j], arguments,
                                             &branch_results[j]));
    branch_results_p[j] = &branch_results[j];
  }

  bool has_tensor_array_gradients = false;
  for (XlaCompiler::CompilationResult* result : branch_results_p) {
    for (const XlaCompiler::ResourceUpdate& update : result->resource_updates) {
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
                     compiler->CompileFunction(options, branches_[j], arguments,
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
    OP_REQUIRES(ctx, branch_results[j].xla_input_shapes.size() == 1,
                errors::FailedPrecondition("Expected one input shape"));
    OP_REQUIRES(
        ctx,
        xla::ShapeUtil::Compatible(branch0_input_shape, branch_input_shape),
        errors::InvalidArgument(
            "Input shapes of 0 and ", j, " branches do not match: ",
            xla::ShapeUtil::HumanString(branch0_input_shape), " vs. ",
            xla::ShapeUtil::HumanString(branch_input_shape)));

    // Check that all branches have identical output shapes.
    OP_REQUIRES(
        ctx,
        xla::ShapeUtil::Compatible(branch_results[0].xla_output_shape,
                                   branch_results[j].xla_output_shape),
        errors::InvalidArgument(
            "Output shapes of 0 and ", j, " branches do not match: ",
            xla::ShapeUtil::HumanString(branch_results[0].xla_output_shape),
            " vs. ",
            xla::ShapeUtil::HumanString(branch_results[j].xla_output_shape)));

    if (j == 0) {
      VLOG(2) << "Input shape: "
              << xla::ShapeUtil::HumanString(branch0_input_shape);
      VLOG(2) << "Output shape: "
              << xla::ShapeUtil::HumanString(
                     branch_results[0].xla_output_shape);
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
      for (const string& node_name : token_input_nodes_) {
        auto token_or = compiler->GetNodeToken(node_name);
        OP_REQUIRES_OK(ctx, token_or.status());
        token_inputs.push_back(token_or.ValueOrDie());
      }
      inputs[i] = xla::AfterAll(b, token_inputs);
    } else if (ctx->input_type(input_num) == DT_RESOURCE) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(input_num, &resource));
      OP_REQUIRES_OK(ctx, resource->Pack(&inputs[i], b));
    } else {
      inputs[i] = ctx->Input(i + 1);
    }
  }
  auto input_tuple = xla::Tuple(b, inputs);

  xla::XlaOp outputs =
      xla::Conditional(ctx->Input(0), absl::MakeSpan(result_computations),
                       std::vector<xla::XlaOp>(num_branches, input_tuple));
  // Sets non-variable outputs.
  for (int i = 0; i < output_types_.size(); ++i) {
    xla::XlaOp output_handle = xla::GetTupleElement(outputs, i);
    if (VLOG_IS_ON(2)) {
      LOG(INFO) << "Setting output " << i;
      auto shape_or = b->GetShape(output_handle);
      if (shape_or.ok()) {
        LOG(INFO) << "Shape for output " << i << ": "
                  << xla::ShapeUtil::HumanString(shape_or.ValueOrDie());
      } else {
        LOG(INFO) << "Shape unknown for output " << i;
      }
    }
    ctx->SetOutput(i, output_handle);
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
    OP_REQUIRES(ctx, shape_or.ValueOrDie().IsToken(),
                errors::FailedPrecondition(
                    "Token output is not token type: ",
                    xla::ShapeUtil::HumanString(shape_or.ValueOrDie())));
    OP_REQUIRES_OK(ctx, compiler->SetNodeToken(name(), token_output));
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

REGISTER_XLA_OP(Name("Case").AllowResourceTypes(), XlaCaseOp);

}  // namespace tensorflow
