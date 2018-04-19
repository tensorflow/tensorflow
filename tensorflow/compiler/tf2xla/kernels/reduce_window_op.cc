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
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class ReduceWindowOp : public XlaOpKernel {
 public:
  explicit ReduceWindowOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("computation", &computation_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("window_dimensions", &window_dimensions_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("window_strides", &window_strides_));
    OP_REQUIRES_OK(context, context->GetAttr("padding_low", &padding_low_));
    OP_REQUIRES_OK(context, context->GetAttr("padding_high", &padding_high_));
  }

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    const DataType dtype = context->input_type(0);

    const int rank = input_shape.dims();
    OP_REQUIRES(context, rank == window_dimensions_.size(),
                errors::InvalidArgument(
                    "The size of window_dimensions must be equal to the input "
                    "rank (",
                    window_dimensions_.size(), " vs. ", rank, ")"));
    OP_REQUIRES(context, rank == window_strides_.size(),
                errors::InvalidArgument(
                    "The size of window_strides must be equal to the input "
                    "rank (",
                    window_strides_.size(), " vs. ", rank, ")"));
    OP_REQUIRES(context, rank == padding_low_.size(),
                errors::InvalidArgument(
                    "The size of padding_low must be equal to the input "
                    "rank (",
                    padding_low_.size(), " vs. ", rank, ")"));
    OP_REQUIRES(context, rank == padding_high_.size(),
                errors::InvalidArgument(
                    "The size of padding_high must be equal to the input "
                    "rank (",
                    padding_high_.size(), " vs. ", rank, ")"));

    xla::ComputationBuilder* builder = context->builder();

    // Build the reducer function.
    XlaCompiler::Argument reducer_arg;
    reducer_arg.kind = XlaCompiler::Argument::kParameter;
    reducer_arg.type = dtype;
    reducer_arg.shape = TensorShape();

    XlaCompiler::CompileOptions compile_options;
    compile_options.use_tuple_arg = false;
    compile_options.resolve_compile_time_constants = false;
    compile_options.is_entry_computation = false;
    XlaCompiler::CompilationResult reducer;
    OP_REQUIRES_OK(context, context->compiler()->CompileFunction(
                                compile_options, *computation_,
                                {reducer_arg, reducer_arg}, &reducer));

    xla::Shape scalar_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeToXLAShape(dtype, TensorShape(), &scalar_shape));
    OP_REQUIRES(context,
                xla::ShapeUtil::Compatible(
                    reducer.xla_output_shape,
                    xla::ShapeUtil::MakeTupleShape({scalar_shape})),
                errors::InvalidArgument(
                    "Invalid output shape of ReduceWindow reducer. Expected ",
                    xla::ShapeUtil::HumanString(scalar_shape), " got ",
                    xla::ShapeUtil::HumanString(reducer.xla_output_shape)));

    // Wraps the reducer in a computation that unpacks the output tuple.
    xla::Computation wrapper;
    {
      std::unique_ptr<xla::ComputationBuilder> cb =
          builder->CreateSubBuilder("wrapper");
      auto x = cb->Parameter(0, scalar_shape, "x");
      auto y = cb->Parameter(1, scalar_shape, "y");
      auto outputs = cb->Call(*reducer.computation, {x, y});
      cb->GetTupleElement(outputs, 0);
      xla::StatusOr<xla::Computation> result = cb->Build();
      OP_REQUIRES_OK(context, result.status());
      wrapper = std::move(result.ValueOrDie());
    }

    std::vector<std::pair<int64, int64>> padding(rank);
    for (int i = 0; i < rank; ++i) {
      padding[i] = {padding_low_[i], padding_high_[i]};
    }

    xla::ComputationDataHandle output = builder->ReduceWindowWithGeneralPadding(
        context->Input(0), context->Input(1), wrapper, window_dimensions_,
        window_strides_, padding);
    context->SetOutput(0, output);
  }

 private:
  const NameAttrList* computation_;
  std::vector<int64> window_dimensions_;
  std::vector<int64> window_strides_;
  std::vector<int64> padding_low_;
  std::vector<int64> padding_high_;

  TF_DISALLOW_COPY_AND_ASSIGN(ReduceWindowOp);
};

REGISTER_XLA_OP(Name("XlaReduceWindow"), ReduceWindowOp);

}  // namespace
}  // namespace tensorflow
