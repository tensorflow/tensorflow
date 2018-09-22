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

class ReduceWindowOp : public XlaOpKernel {
 public:
  explicit ReduceWindowOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("computation", &computation_));
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

    // Build the reducer function.
    XlaCompiler::Argument reducer_arg;
    reducer_arg.kind = XlaCompiler::Argument::kParameter;
    reducer_arg.type = dtype;
    reducer_arg.shape = TensorShape();

    XlaCompiler::CompileOptions compile_options;
    compile_options.use_tuple_arg = false;
    compile_options.resolve_compile_time_constants = false;
    compile_options.is_entry_computation = false;
    compile_options.always_return_tuple = false;
    XlaCompiler::CompilationResult reducer;
    OP_REQUIRES_OK(context, context->compiler()->CompileFunction(
                                compile_options, *computation_,
                                {reducer_arg, reducer_arg}, &reducer));

    xla::Shape scalar_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeToXLAShape(dtype, TensorShape(), &scalar_shape));
    OP_REQUIRES(
        context,
        xla::ShapeUtil::Compatible(reducer.xla_output_shape, scalar_shape),
        errors::InvalidArgument(
            "Invalid output shape of ReduceWindow reducer. Expected ",
            xla::ShapeUtil::HumanString(scalar_shape), " got ",
            xla::ShapeUtil::HumanString(reducer.xla_output_shape)));

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

    xla::XlaOp output = xla::ReduceWindowWithGeneralPadding(
        context->Input(0), context->Input(1), *reducer.computation,
        window_dimensions, window_strides, padding);
    context->SetOutput(0, output);
  }

 private:
  const NameAttrList* computation_;

  TF_DISALLOW_COPY_AND_ASSIGN(ReduceWindowOp);
};

REGISTER_XLA_OP(Name("XlaReduceWindow")
                    .CompileTimeConstInput("window_dimensions")
                    .CompileTimeConstInput("window_strides")
                    .CompileTimeConstInput("padding"),
                ReduceWindowOp);

}  // namespace
}  // namespace tensorflow
