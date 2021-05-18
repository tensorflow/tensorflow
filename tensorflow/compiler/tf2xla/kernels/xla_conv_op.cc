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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class XlaConvOp : public XlaOpKernel {
 public:
  explicit XlaConvOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    string dnums_attr;
    OP_REQUIRES_OK(context, context->GetAttr("dimension_numbers", &dnums_attr));
    OP_REQUIRES(
        context, dnums_.ParsePartialFromString(dnums_attr),
        errors::InvalidArgument("Error parsing convolution dimension numbers"));
    string precision_config_attr;
    OP_REQUIRES_OK(
        context, context->GetAttr("precision_config", &precision_config_attr));
    OP_REQUIRES(context,
                precision_config_.ParsePartialFromString(precision_config_attr),
                errors::InvalidArgument("Error parsing precision config."));
    preferred_element_type_ = absl::nullopt;
  }

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape lhs_shape = context->InputShape(0);
    const TensorShape rhs_shape = context->InputShape(1);
    const TensorShape padding_shape = context->InputShape("padding");
    std::vector<int64> window_strides;
    std::vector<int64> lhs_dilation;
    std::vector<int64> rhs_dilation;
    int64 feature_group_count;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("window_strides",
                                                              &window_strides));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("lhs_dilation",
                                                              &lhs_dilation));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("rhs_dilation",
                                                              &rhs_dilation));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(
                                "feature_group_count", &feature_group_count));

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

    // We do only minimal checking, relying on XLA to check the shape
    // invariants.
    xla::XlaOp output = xla::ConvGeneralDilated(
        context->Input(0), context->Input(1), window_strides, padding,
        lhs_dilation, rhs_dilation, dnums_, feature_group_count,
        /*batch_group_count=*/1, &precision_config_, preferred_element_type_);
    context->SetOutput(0, output);
  }

 protected:
  absl::optional<xla::PrimitiveType> preferred_element_type_;

 private:
  xla::ConvolutionDimensionNumbers dnums_;
  xla::PrecisionConfig precision_config_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaConvOp);
};

REGISTER_XLA_OP(Name("XlaConv")
                    .CompileTimeConstantInput("window_strides")
                    .CompileTimeConstantInput("lhs_dilation")
                    .CompileTimeConstantInput("rhs_dilation")
                    .CompileTimeConstantInput("feature_group_count")
                    .CompileTimeConstantInput("padding"),
                XlaConvOp);

class XlaConvV2Op : public XlaConvOp {
 public:
  explicit XlaConvV2Op(OpKernelConstruction* context) : XlaConvOp(context) {
    DataType preferred_element_dtype;
    OP_REQUIRES_OK(context, context->GetAttr("preferred_element_type",
                                             &preferred_element_dtype));
    xla::PrimitiveType preferred_element_type;
    OP_REQUIRES_OK(context, DataTypeToPrimitiveType(preferred_element_dtype,
                                                    &preferred_element_type));
    preferred_element_type_ = preferred_element_type;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaConvV2Op);
};

REGISTER_XLA_OP(Name("XlaConvV2")
                    .CompileTimeConstantInput("window_strides")
                    .CompileTimeConstantInput("lhs_dilation")
                    .CompileTimeConstantInput("rhs_dilation")
                    .CompileTimeConstantInput("feature_group_count")
                    .CompileTimeConstantInput("padding"),
                XlaConvV2Op);

}  // namespace
}  // namespace tensorflow
