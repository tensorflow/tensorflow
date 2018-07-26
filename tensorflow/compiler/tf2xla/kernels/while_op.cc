/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/while_op.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace {

// Builds XlaCompiler argument descriptions `args` from `ctx`.
Status MakeXlaCompilerArgumentsFromInputs(
    XlaOpKernelContext* ctx, std::vector<XlaCompiler::Argument>* args,
    bool* has_uninitialized_vars, bool* has_tensor_arrays) {
  VLOG(2) << "Num inputs " << ctx->num_inputs();
  args->resize(ctx->num_inputs());
  *has_uninitialized_vars = false;
  *has_tensor_arrays = false;
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    VLOG(2) << " Input " << i
            << " type: " << DataTypeString(ctx->input_type(i))
            << " shape: " << ctx->InputShape(i).DebugString();
    XlaCompiler::Argument& arg = (*args)[i];
    DataType type = ctx->input_type(i);
    // When reading a resource input, use the type and shape of the resource's
    // current value.
    if (type == DT_RESOURCE) {
      XlaResource* resource;
      TF_RETURN_IF_ERROR(ctx->GetResourceInput(i, &resource));

      arg.initialized = resource->initialized();
      arg.kind = XlaCompiler::Argument::kResource;
      arg.resource_kind = resource->kind();
      if (arg.resource_kind == XlaResource::kTensorArray) {
        *has_tensor_arrays = true;
      }

      arg.type = resource->type();
      arg.shape = resource->shape();
      if (!arg.initialized) {
        *has_uninitialized_vars = true;
      }
      arg.tensor_array_size = resource->tensor_array_size();
      for (const auto& gradient : resource->tensor_array_gradients()) {
        arg.tensor_array_gradients.insert(gradient.first);
      }
      arg.name = resource->name();
      VLOG(2) << "    resource " << resource->name()
              << " type: " << DataTypeString(arg.type)
              << " shape: " << arg.shape.DebugString()
              << " initialized: " << arg.initialized;

    } else {
      arg.kind = XlaCompiler::Argument::kParameter;
      arg.type = ctx->input_type(i);
      arg.shape = ctx->InputShape(i);
    }
  }
  return Status::OK();
}

}  // anonymous namespace

XlaWhileOp::XlaWhileOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  const NameAttrList* name_attr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("cond", &name_attr));
  cond_name_attr_ = *name_attr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("body", &name_attr));
  body_name_attr_ = *name_attr;
}

void XlaWhileOp::Compile(XlaOpKernelContext* ctx) {
  VLOG(1) << "WhileOp::Compile";

  std::vector<XlaCompiler::Argument> arguments;
  bool has_uninitialized_vars;
  bool has_tensor_arrays;
  OP_REQUIRES_OK(
      ctx, MakeXlaCompilerArgumentsFromInputs(
               ctx, &arguments, &has_uninitialized_vars, &has_tensor_arrays));

  xla::XlaBuilder* builder = ctx->builder();
  XlaCompiler* compiler = ctx->compiler();

  VLOG(1) << "Compiling body";

  // All resource that are inputs to the loop's body must also be
  // present as loop body outputs; the signature of the loop's input and
  // output must match. We ensure this by asking the compiler to include the
  // current values of all resources, even if they haven't been updated by the
  // computation. We must also ask the compiler to keep compile-time constant
  // outputs as part of the generated computation, for the same reason.
  // TODO(phawkins): consider adding loop-invariant inputs to XLA's While()
  // operator.
  XlaCompiler::CompileOptions body_options;
  body_options.use_tuple_arg = true;
  body_options.return_updated_values_for_all_resources = true;
  body_options.resolve_compile_time_constants = false;
  body_options.is_entry_computation = false;
  XlaCompiler::CompilationResult body;
  OP_REQUIRES_OK(ctx, compiler->CompileFunction(body_options, body_name_attr_,
                                                arguments, &body));

  // We must use a static shape for parameters to an XLA compilation. However,
  // we may not know the shape of a resource if it is first
  // written inside the loop. Furthermore, we do not know ahead of time which
  // gradient TensorArrays will be created by the TensorArrayGradV3 operator.
  //
  // Ideally we would change TensorFlow to provide static shape always, but
  // but this is not easy to do. So if uninitialized resources or TensorArrays
  // are used by the loop body, we compile the body function twice:
  // 1) once with uninitialized resource inputs and no TensorArray gradient
  //    inputs. We then discard the computation but we assume resource shapes
  //    and the set of gradients read or written will reach a fixpoint after one
  //    iteration.
  //    Hence we can use the output shapes and TensorArray gradients of each
  //    resource as the "true" shapes.
  // 2) again with the "correct" resource information determined by (1).
  if (has_uninitialized_vars || has_tensor_arrays) {
    VLOG(2) << "Recompiling loop body: has_uninitialized_vars: "
            << has_uninitialized_vars
            << " has_tensor_arrays: " << has_tensor_arrays;
    // Initializes any uninitialized resource with zero values of the
    // shape determined by the first compilation.
    for (int i = 0; i < body.resource_updates.size(); ++i) {
      const XlaCompiler::ResourceUpdate& update = body.resource_updates[i];
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));

      XlaCompiler::Argument& arg = arguments[update.input_index];
      if (!arg.initialized) {
        VLOG(2) << "Update shape for argument " << update.input_index << " "
                << update.shape.DebugString();
        arg.initialized = true;

        arg.shape = update.shape;
        OP_REQUIRES_OK(ctx,
                       resource->SetTypeAndShape(update.type, update.shape));

        OP_REQUIRES_OK(ctx, resource->SetZeroValue(builder));
      }

      // Add any TensorArray gradients touched by the body to the enclosing
      // graph.
      for (const string& grad_source : update.tensor_array_gradients_accessed) {
        VLOG(4) << "TensorArray " << resource->name() << " accessed gradient "
                << grad_source;
        XlaResource* gradient;
        OP_REQUIRES_OK(ctx, resource->GetOrCreateTensorArrayGradient(
                                grad_source, builder, &gradient));
      }

      // Add all of the TensorArray gradients to the argument. For simplicity,
      // we always pass all known gradients.
      for (const auto& gradient : resource->tensor_array_gradients()) {
        arg.tensor_array_gradients.insert(gradient.first);
      }
    }
    // Recompile the body with the "correct" resource shapes.
    VLOG(1) << "Recompiling body with corrected resource shapes";
    body = {};
    OP_REQUIRES_OK(ctx, compiler->CompileFunction(body_options, body_name_attr_,
                                                  arguments, &body));
  }

  VLOG(1) << "Compiling condition";

  XlaCompiler::CompileOptions cond_options;
  cond_options.use_tuple_arg = true;
  cond_options.resolve_compile_time_constants = false;
  cond_options.is_entry_computation = false;
  XlaCompiler::CompilationResult cond;
  OP_REQUIRES_OK(ctx, compiler->CompileFunction(cond_options, cond_name_attr_,
                                                arguments, &cond));

  OP_REQUIRES(ctx, body.xla_input_shapes.size() == 1,
              errors::FailedPrecondition("Expected one input shape"));
  xla::Shape body_input_shape = body.xla_input_shapes[0];
  OP_REQUIRES(ctx, xla::ShapeUtil::IsTuple(body_input_shape),
              errors::FailedPrecondition("Expected tuple shape"));
  OP_REQUIRES(ctx, cond.xla_input_shapes.size() == 1,
              errors::FailedPrecondition("Expected one input shape"));
  xla::Shape cond_input_shape = cond.xla_input_shapes[0];
  OP_REQUIRES(ctx, xla::ShapeUtil::IsTuple(cond_input_shape),
              errors::FailedPrecondition("Expected tuple shape"));

  VLOG(2) << "Body shape: " << xla::ShapeUtil::HumanString(body_input_shape)
          << " -> " << xla::ShapeUtil::HumanString(body.xla_output_shape);
  VLOG(2) << "Cond shape: " << xla::ShapeUtil::HumanString(cond_input_shape)
          << " -> " << xla::ShapeUtil::HumanString(cond.xla_output_shape);

  OP_REQUIRES(ctx,
              xla::ShapeUtil::Compatible(body_input_shape, cond_input_shape),
              errors::InvalidArgument(
                  "Input shapes of loop body and condition do not match: ",
                  xla::ShapeUtil::HumanString(body_input_shape), " vs. ",
                  xla::ShapeUtil::HumanString(cond_input_shape)));
  OP_REQUIRES(
      ctx, xla::ShapeUtil::Compatible(body_input_shape, body.xla_output_shape),
      errors::InvalidArgument(
          "Input and output shapes of loop body do not match: ",
          xla::ShapeUtil::HumanString(body_input_shape), " vs. ",
          xla::ShapeUtil::HumanString(body.xla_output_shape)));

  xla::Shape expected_cond_output_shape = xla::ShapeUtil::MakeTupleShape(
      {xla::ShapeUtil::MakeShape(xla::PRED, {})});
  OP_REQUIRES(ctx,
              xla::ShapeUtil::Compatible(cond.xla_output_shape,
                                         expected_cond_output_shape),
              errors::InvalidArgument(
                  "Output shape of loop condition should be (pred[]), got: ",
                  xla::ShapeUtil::HumanString(cond.xla_output_shape)));

  int num_inputs = body.input_mapping.size();
  std::vector<xla::XlaOp> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    int input_num = body.input_mapping[i];
    if (ctx->input_type(input_num) == DT_RESOURCE) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(input_num, &resource));
      OP_REQUIRES_OK(ctx, resource->Pack(&inputs[i], builder));
    } else {
      inputs[i] = ctx->Input(i);
    }
  }

  xla::XlaOp init = xla::Tuple(builder, inputs);

  VLOG(1) << "Building while loop";

  // Wraps the condition in a computation that unpacks the output tuple.
  xla::XlaComputation cond_wrapper;
  {
    std::unique_ptr<xla::XlaBuilder> cb =
        builder->CreateSubBuilder("cond_wrapper");
    auto inputs = xla::Parameter(cb.get(), 0, cond_input_shape, "inputs");
    auto outputs = xla::Call(cb.get(), *cond.computation, {inputs});
    xla::GetTupleElement(outputs, 0);
    xla::StatusOr<xla::XlaComputation> result = cb->Build();
    OP_REQUIRES_OK(ctx, result.status());
    cond_wrapper = std::move(result.ValueOrDie());
  }

  xla::XlaOp while_result = xla::While(cond_wrapper, *body.computation, init);

  // Sets non-variable outputs.
  for (int i = 0; i < ctx->num_outputs(); ++i) {
    if (ctx->input_type(i) != DT_RESOURCE) {
      ctx->SetOutput(body.input_mapping[i],
                     xla::GetTupleElement(while_result, i));
    }
  }

  // Updates the values of any resource variables modified by the loop.
  for (int i = 0; i < body.resource_updates.size(); ++i) {
    const XlaCompiler::ResourceUpdate& update = body.resource_updates[i];
    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));
    if (update.modified) {
      int pos = body.outputs.size() + i;
      OP_REQUIRES_OK(ctx,
                     resource->SetFromPack(
                         arguments[update.input_index].tensor_array_gradients,
                         xla::GetTupleElement(while_result, pos), builder));
    }
    VLOG(2) << "Loop-carried variable: pos: " << update.input_index
            << " name: " << resource->name() << " modified: " << update.modified
            << " type: " << DataTypeString(update.type)
            << " shape: " << update.shape.DebugString();
    // Copies the identity of the resource variable from input to output
    // unchanged, even if the variable was not modified.
    ctx->op_kernel_context()->set_output(
        update.input_index,
        ctx->op_kernel_context()->input(update.input_index));
  }

  VLOG(1) << "Done building while loop";
}

REGISTER_XLA_OP(Name("While").AllowResourceTypes(), XlaWhileOp);
REGISTER_XLA_OP(Name("XlaWhile").AllowResourceTypes(), XlaWhileOp);

}  // namespace tensorflow
