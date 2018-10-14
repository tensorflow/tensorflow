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

#include "tensorflow/compiler/tf2xla/kernels/while_op.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class XlaSelectAndScatterOp : public XlaOpKernel {
 public:
  explicit XlaSelectAndScatterOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("select", &select_computation_));
    OP_REQUIRES_OK(context, context->GetAttr("scatter", &scatter_computation_));
  }

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    const DataType dtype = context->input_type(0);

    std::vector<int64> window_dimensions;
    std::vector<int64> window_strides;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector(
                                "window_dimensions", &window_dimensions));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("window_strides",
                                                              &window_strides));

    const int rank = input_shape.dims();
    OP_REQUIRES(context, rank == window_dimensions.size(),
                errors::InvalidArgument(
                    "The size of window_dimensions must be equal to the input "
                    "rank (",
                    window_dimensions.size(), " vs. ", rank, ")"));
    OP_REQUIRES(context, rank == window_strides.size(),
                errors::InvalidArgument(
                    "The size of window_strides must be equal to the input "
                    "rank (",
                    window_strides.size(), " vs. ", rank, ")"));

    XlaCompiler::CompileOptions compile_options;
    compile_options.use_tuple_arg = false;
    compile_options.resolve_compile_time_constants = false;
    compile_options.is_entry_computation = false;
    compile_options.always_return_tuple = false;

    // Build the select function.
    XlaCompiler::Argument select_arg;
    select_arg.kind = XlaCompiler::Argument::kParameter;
    select_arg.type = dtype;
    select_arg.shape = TensorShape();

    XlaCompiler::CompilationResult select;
    OP_REQUIRES_OK(context, context->compiler()->CompileFunction(
                                compile_options, *select_computation_,
                                {select_arg, select_arg}, &select));

    xla::Shape select_output_shape = xla::ShapeUtil::MakeShape(xla::PRED, {});
    OP_REQUIRES(
        context,
        xla::ShapeUtil::Compatible(select.xla_output_shape,
                                   select_output_shape),
        errors::InvalidArgument(
            "Invalid output shape of XlaSelectAndScatter select. Expected ",
            xla::ShapeUtil::HumanString(select_output_shape), " got ",
            xla::ShapeUtil::HumanString(select.xla_output_shape)));

    // Build the scatter function.
    XlaCompiler::Argument scatter_arg;
    scatter_arg.kind = XlaCompiler::Argument::kParameter;
    scatter_arg.type = dtype;
    scatter_arg.shape = TensorShape();

    XlaCompiler::CompilationResult scatter;
    OP_REQUIRES_OK(context, context->compiler()->CompileFunction(
                                compile_options, *scatter_computation_,
                                {scatter_arg, scatter_arg}, &scatter));

    xla::Shape scalar_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeToXLAShape(dtype, TensorShape(), &scalar_shape));
    OP_REQUIRES(
        context,
        xla::ShapeUtil::Compatible(scatter.xla_output_shape, scalar_shape),
        errors::InvalidArgument(
            "Invalid output shape of scatter. Expected ",
            xla::ShapeUtil::HumanString(scalar_shape), " got ",
            xla::ShapeUtil::HumanString(scatter.xla_output_shape)));

    const TensorShape padding_shape = context->InputShape("padding");
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(padding_shape) &&
                    padding_shape.dim_size(1) == 2,
                errors::InvalidArgument(
                    "padding must be a matrix with minor dimension 2, got ",
                    padding_shape.DebugString()));
    xla::Literal padding_literal;
    OP_REQUIRES_OK(context, context->ConstantInputAsInt64Literal(
                                "padding", &padding_literal));
    std::vector<std::pair<int64, int64>> padding(padding_shape.dim_size(0));
    for (int i = 0; i < padding.size(); ++i) {
      padding[i] = {padding_literal.Get<int64>({i, 0}),
                    padding_literal.Get<int64>({i, 1})};
    }

    xla::XlaOp output = xla::SelectAndScatterWithGeneralPadding(
        context->Input("operand"), *select.computation, window_dimensions,
        window_strides, padding, context->Input("source"),
        context->Input("init_value"), *scatter.computation);
    context->SetOutput(0, output);
  }

 private:
  const NameAttrList* select_computation_;
  const NameAttrList* scatter_computation_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaSelectAndScatterOp);
};

REGISTER_XLA_OP(Name("XlaSelectAndScatter")
                    .CompileTimeConstInput("window_dimensions")
                    .CompileTimeConstInput("window_strides")
                    .CompileTimeConstInput("padding"),
                XlaSelectAndScatterOp);

}  // namespace
}  // namespace tensorflow
