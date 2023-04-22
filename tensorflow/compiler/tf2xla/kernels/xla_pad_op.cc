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
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class XlaPadOp : public XlaOpKernel {
 public:
  explicit XlaPadOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape("input");
    const TensorShape padding_value_shape =
        context->InputShape("padding_value");

    std::vector<int64> padding_low;
    std::vector<int64> padding_high;
    std::vector<int64> padding_interior;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("padding_low",
                                                              &padding_low));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("padding_high",
                                                              &padding_high));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector(
                                "padding_interior", &padding_interior));

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(padding_value_shape),
                errors::InvalidArgument("padding_value must be a scalar"));
    const int rank = input_shape.dims();
    OP_REQUIRES(context, rank == padding_low.size(),
                errors::InvalidArgument(
                    "The size of padding_low must be equal to the input "
                    "rank (",
                    padding_low.size(), " vs. ", rank, ")"));
    OP_REQUIRES(context, rank == padding_high.size(),
                errors::InvalidArgument(
                    "The size of padding_high must be equal to the input "
                    "rank (",
                    padding_high.size(), " vs. ", rank, ")"));
    OP_REQUIRES(context, rank == padding_interior.size(),
                errors::InvalidArgument(
                    "The size of padding_interior must be equal to the input "
                    "rank (",
                    padding_interior.size(), " vs. ", rank, ")"));

    auto non_negative = [](int64 x) { return x >= 0; };
    OP_REQUIRES(
        context, absl::c_all_of(padding_interior, non_negative),
        errors::InvalidArgument("padding_interior must be non-negative, got [",
                                absl::StrJoin(padding_interior, ","), "]"));

    xla::PaddingConfig padding_config;
    for (int i = 0; i < rank; ++i) {
      auto* dim = padding_config.add_dimensions();
      dim->set_edge_padding_low(padding_low[i]);
      dim->set_edge_padding_high(padding_high[i]);
      dim->set_interior_padding(padding_interior[i]);
    }

    xla::XlaOp output =
        xla::Pad(context->Input("input"), context->Input("padding_value"),
                 padding_config);
    context->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaPadOp);
};

REGISTER_XLA_OP(Name("XlaPad")
                    .CompileTimeConstantInput("padding_low")
                    .CompileTimeConstantInput("padding_high")
                    .CompileTimeConstantInput("padding_interior"),
                XlaPadOp);

}  // namespace
}  // namespace tensorflow
