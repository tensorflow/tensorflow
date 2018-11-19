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

#include "absl/algorithm/container.h"
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

class XlaReduceOp : public XlaOpKernel {
 public:
  explicit XlaReduceOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("reducer", &reducer_));
    OP_REQUIRES_OK(context, context->GetAttr("dimensions_to_reduce",
                                             &dimensions_to_reduce_));
    std::set<int64> dims_set(dimensions_to_reduce_.begin(),
                             dimensions_to_reduce_.end());
    OP_REQUIRES(
        context, dims_set.size() == dimensions_to_reduce_.size(),
        errors::InvalidArgument("Duplicate dimension in dimensions_to_reduce "
                                "argument to XlaReduce"));
  }

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape("input");
    const TensorShape init_value_shape = context->InputShape("init_value");
    const DataType dtype = context->input_type(0);

    const int rank = input_shape.dims();
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(init_value_shape),
                errors::InvalidArgument("init_value must be a scalar"));

    auto dim_in_range = [rank](int64 dim) { return dim >= 0 && dim < rank; };
    OP_REQUIRES(context,
                rank >= dimensions_to_reduce_.size() &&
                    absl::c_all_of(dimensions_to_reduce_, dim_in_range),
                errors::InvalidArgument(
                    "Invalid dimensions_to_reduce argument to XlaReduce"));

    // Build the reducer function.
    XlaCompiler::Argument reducer_arg;
    reducer_arg.kind = XlaCompiler::Argument::kParameter;
    reducer_arg.type = dtype;
    reducer_arg.shape = TensorShape();

    XlaCompiler::CompileOptions compile_options;
    compile_options.use_tuple_arg = false;
    compile_options.always_return_tuple = false;
    compile_options.resolve_compile_time_constants = false;
    compile_options.is_entry_computation = false;
    XlaCompiler::CompilationResult reducer;
    OP_REQUIRES_OK(context, context->compiler()->CompileFunction(
                                compile_options, *reducer_,
                                {reducer_arg, reducer_arg}, &reducer));

    xla::Shape scalar_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeToXLAShape(dtype, TensorShape(), &scalar_shape));
    OP_REQUIRES(
        context,
        xla::ShapeUtil::Compatible(reducer.xla_output_shape, scalar_shape),
        errors::InvalidArgument(
            "Invalid output shape of XlaReduce reducer. Expected ",
            xla::ShapeUtil::HumanString(scalar_shape), " got ",
            xla::ShapeUtil::HumanString(reducer.xla_output_shape)));

    xla::XlaOp output =
        xla::Reduce(context->Input("input"), context->Input("init_value"),
                    *reducer.computation, dimensions_to_reduce_);
    context->SetOutput(0, output);
  }

 private:
  const NameAttrList* reducer_;
  std::vector<int64> dimensions_to_reduce_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaReduceOp);
};

REGISTER_XLA_OP(Name("XlaReduce"), XlaReduceOp);

}  // namespace
}  // namespace tensorflow
