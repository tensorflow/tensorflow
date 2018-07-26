/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/if_op.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace tensorflow {

XlaIfOp::XlaIfOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  const NameAttrList* name_attr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("then_branch", &name_attr));
  then_branch_ = *name_attr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("else_branch", &name_attr));
  else_branch_ = *name_attr;

  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tcond", &cond_type_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types_));
}

// TODO(b/35949885): There is duplication here with the handling of the
// while_op. Refactor the common code out/rework.
void XlaIfOp::Compile(XlaOpKernelContext* ctx) {
  xla::XlaBuilder* b = ctx->builder();

  OP_REQUIRES(ctx, cond_type_ == DT_BOOL,
              errors::InvalidArgument(
                  "Condition argument must be a boolean for XLA compilation"));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->InputShape(0)),
              errors::InvalidArgument(
                  "Condition argument must be a scalar for XLA compilation"));

  VLOG(1) << "Building If: " << input_types_.size() << " inputs";

  std::vector<XlaCompiler::Argument> arguments(input_types_.size());
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
      arg.tensor_array_size = resource->tensor_array_size();
      for (const auto& gradient : resource->tensor_array_gradients()) {
        arg.tensor_array_gradients.insert(gradient.first);
      }
      arg.name = resource->name();
      VLOG(2) << "Resource " << resource->name()
              << " type: " << DataTypeString(arg.type)
              << " shape: " << arg.shape.DebugString()
              << " initialized: " << arg.initialized;
    } else {
      arg.kind = XlaCompiler::Argument::kParameter;
      arg.type = input_types_[i];
      arg.shape = ctx->InputShape(i + 1);
      VLOG(2) << "Arg type: " << DataTypeString(arg.type)
              << " shape: " << arg.shape.DebugString();
    }
  }

  // Compile both branches of the conditional.
  XlaCompiler::CompileOptions options;
  options.use_tuple_arg = true;
  options.resolve_compile_time_constants = false;
  options.return_updated_values_for_all_resources = true;
  options.is_entry_computation = false;
  XlaCompiler* compiler = ctx->compiler();

  XlaCompiler::CompilationResult then_result;
  OP_REQUIRES_OK(ctx, compiler->CompileFunction(options, then_branch_,
                                                arguments, &then_result));
  XlaCompiler::CompilationResult else_result;
  OP_REQUIRES_OK(ctx, compiler->CompileFunction(options, else_branch_,
                                                arguments, &else_result));

  bool has_tensor_array_gradients = false;
  for (XlaCompiler::CompilationResult* result : {&then_result, &else_result}) {
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
      if (!resource->tensor_array_gradients().empty())
        has_tensor_array_gradients = true;
    }
  }

  // Recompile the functions to update the argument shapes for tensor arrays.
  if (has_tensor_array_gradients) {
    then_result = {};
    OP_REQUIRES_OK(ctx, compiler->CompileFunction(options, then_branch_,
                                                  arguments, &then_result));
    else_result = {};
    OP_REQUIRES_OK(ctx, compiler->CompileFunction(options, else_branch_,
                                                  arguments, &else_result));
  }

  // Check that both branches have identical input shapes.
  OP_REQUIRES(ctx, then_result.xla_input_shapes.size() == 1,
              errors::FailedPrecondition("Expected one input shape"));
  xla::Shape then_input_shape = then_result.xla_input_shapes[0];
  OP_REQUIRES(ctx, xla::ShapeUtil::IsTuple(then_input_shape),
              errors::FailedPrecondition("Expected tuple shape"));
  OP_REQUIRES(ctx, else_result.xla_input_shapes.size() == 1,
              errors::FailedPrecondition("Expected one input shape"));
  xla::Shape else_input_shape = else_result.xla_input_shapes[0];
  OP_REQUIRES(ctx, xla::ShapeUtil::IsTuple(else_input_shape),
              errors::FailedPrecondition("Expected tuple shape"));
  OP_REQUIRES(ctx,
              xla::ShapeUtil::Compatible(then_input_shape, else_input_shape),
              errors::InvalidArgument(
                  "Input shapes of then and else branches do not match: ",
                  xla::ShapeUtil::HumanString(then_input_shape), " vs. ",
                  xla::ShapeUtil::HumanString(else_input_shape)));

  // Check that both branches have identical output shapes.
  OP_REQUIRES(
      ctx,
      xla::ShapeUtil::Compatible(then_result.xla_output_shape,
                                 else_result.xla_output_shape),
      errors::InvalidArgument(
          "Output shapes of then and else branches do not match: ",
          xla::ShapeUtil::HumanString(then_result.xla_output_shape), " vs. ",
          xla::ShapeUtil::HumanString(else_result.xla_output_shape)));

  VLOG(2) << "Input shape: " << xla::ShapeUtil::HumanString(then_input_shape);
  VLOG(2) << "Output shape: "
          << xla::ShapeUtil::HumanString(then_result.xla_output_shape);

  // We set return_updated_values_for_all_resources=true and we pass the same
  // arguments to both computations, so the resource update count must match.
  OP_REQUIRES(ctx,
              then_result.resource_updates.size() ==
                  else_result.resource_updates.size(),
              errors::FailedPrecondition(
                  "Different number of resources in then and else branch"));
  for (int i = 0; i < then_result.resource_updates.size(); ++i) {
    const auto& lhs = then_result.resource_updates[i];
    const auto& rhs = else_result.resource_updates[i];
    bool equal = lhs.input_index == rhs.input_index && lhs.shape == rhs.shape &&
                 lhs.tensor_array_gradients_accessed ==
                     rhs.tensor_array_gradients_accessed;
    OP_REQUIRES(
        ctx, equal,
        errors::FailedPrecondition(
            "Mismatch in resource of then and else branch for resource ", i));
  }

  int num_inputs = then_result.input_mapping.size();
  std::vector<xla::XlaOp> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    int input_num = then_result.input_mapping[i] + 1;
    if (ctx->input_type(input_num) == DT_RESOURCE) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(input_num, &resource));
      OP_REQUIRES_OK(ctx, resource->Pack(&inputs[i], b));
    } else {
      inputs[i] = ctx->Input(i + 1);
    }
  }

  xla::XlaOp outputs = xla::Conditional(
      ctx->Input(0), xla::Tuple(b, inputs), *then_result.computation,
      xla::Tuple(b, inputs), *else_result.computation);
  // Sets non-variable outputs.
  for (int i = 0; i < output_types_.size(); ++i) {
    if (ctx->input_type(i) != DT_RESOURCE) {
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
  }

  // Updates the values of any resource variables modified by the conditional
  // bodies.
  for (XlaCompiler::CompilationResult* result : {&then_result, &else_result}) {
    for (int i = 0; i < result->resource_updates.size(); ++i) {
      const XlaCompiler::ResourceUpdate& update = result->resource_updates[i];
      XlaResource* resource;
      OP_REQUIRES_OK(ctx,
                     ctx->GetResourceInput(update.input_index + 1, &resource));
      if (update.modified) {
        int pos = result->outputs.size() + i;
        OP_REQUIRES_OK(ctx,
                       resource->SetFromPack(
                           arguments[update.input_index].tensor_array_gradients,
                           xla::GetTupleElement(outputs, pos), b));
      }
      VLOG(2) << "If variable: pos: " << update.input_index
              << " name: " << resource->name()
              << " modified: " << update.modified
              << " type: " << DataTypeString(update.type)
              << " shape: " << update.shape.DebugString();
    }
  }
  VLOG(1) << "Done building If";
}

REGISTER_XLA_OP(Name("If").AllowResourceTypes(), XlaIfOp);
REGISTER_XLA_OP(Name("XlaIf").AllowResourceTypes(), XlaIfOp);

}  // namespace tensorflow
