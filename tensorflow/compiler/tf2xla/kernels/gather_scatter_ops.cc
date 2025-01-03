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

#include <cstdint>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class GatherOp : public XlaOpKernel {
 public:
  explicit GatherOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    string dnums_attr;
    OP_REQUIRES_OK(context, context->GetAttr("dimension_numbers", &dnums_attr));
    OP_REQUIRES(
        context, dnums_.ParsePartialFromString(dnums_attr),
        errors::InvalidArgument("Error parsing gather dimension numbers"));
    OP_REQUIRES_OK(
        context, context->GetAttr("indices_are_sorted", &indices_are_sorted_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64_t> slice_sizes;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntVector("slice_sizes", &slice_sizes));
    xla::XlaOp result =
        xla::Gather(ctx->Input("operand"), ctx->Input("start_indices"), dnums_,
                    slice_sizes, indices_are_sorted_);
    ctx->SetOutput(0, result);
  }

 private:
  xla::GatherDimensionNumbers dnums_;
  bool indices_are_sorted_;
};

REGISTER_XLA_OP(Name("XlaGather").CompileTimeConstantInput("slice_sizes"),
                GatherOp);

class ScatterOp : public XlaOpKernel {
 public:
  explicit ScatterOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    OP_REQUIRES_OK(
        context, context->GetAttr("update_computation", &update_computation_));
    string dnums_attr;
    OP_REQUIRES_OK(context, context->GetAttr("dimension_numbers", &dnums_attr));
    OP_REQUIRES(
        context, dnums_.ParsePartialFromString(dnums_attr),
        errors::InvalidArgument("Error parsing scatter dimension numbers"));
    OP_REQUIRES_OK(
        context, context->GetAttr("indices_are_sorted", &indices_are_sorted_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = ctx->input_type(0);

    XlaCompiler::Argument update_computation_arg;
    update_computation_arg.kind = XlaCompiler::Argument::kParameter;
    update_computation_arg.type = dtype;
    update_computation_arg.shape = TensorShape();

    XlaCompiler::CompileOptions compile_options;
    compile_options.use_tuple_arg = false;
    compile_options.always_return_tuple = false;
    compile_options.is_entry_computation = false;
    XlaCompiler::CompilationResult update_computation;
    OP_REQUIRES_OK(ctx, ctx->compiler()->CompileFunction(
                            compile_options, *update_computation_,
                            {update_computation_arg, update_computation_arg},
                            &update_computation));

    xla::XlaOp result =
        xla::Scatter(ctx->Input("operand"), ctx->Input("scatter_indices"),
                     ctx->Input("updates"), *update_computation.computation,
                     dnums_, indices_are_sorted_);
    ctx->SetOutput(0, result);
  }

 private:
  const NameAttrList* update_computation_;
  xla::ScatterDimensionNumbers dnums_;
  bool indices_are_sorted_;
};

REGISTER_XLA_OP(Name("XlaScatter"), ScatterOp);

}  // namespace
}  // namespace tensorflow
