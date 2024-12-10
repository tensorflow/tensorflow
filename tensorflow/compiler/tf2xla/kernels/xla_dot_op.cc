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

#include <optional>

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

class XlaDotOp : public XlaOpKernel {
 public:
  explicit XlaDotOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    string dnums_attr;
    OP_REQUIRES_OK(context, context->GetAttr("dimension_numbers", &dnums_attr));
    OP_REQUIRES(
        context, dnums_.ParsePartialFromString(dnums_attr),
        errors::InvalidArgument("Error parsing convolution dimension numbers"));
    string precision_config_attr;
    OP_REQUIRES_OK(
        context, context->GetAttr("precision_config", &precision_config_attr));
    OP_REQUIRES(
        context,
        precision_config_.ParsePartialFromString(precision_config_attr),
        errors::InvalidArgument("Error parsing convolution dimension numbers"));
    preferred_element_type_ = std::nullopt;
  }

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape lhs_shape = context->InputShape(0);
    const TensorShape rhs_shape = context->InputShape(1);

    // We do only minimal checking, relying on XLA to check the shape
    // invariants.
    xla::XlaOp output =
        xla::DotGeneral(context->Input(0), context->Input(1), dnums_,
                        &precision_config_, preferred_element_type_);
    context->SetOutput(0, output);
  }

 protected:
  std::optional<xla::PrimitiveType> preferred_element_type_;

 private:
  xla::DotDimensionNumbers dnums_;
  xla::PrecisionConfig precision_config_;
  XlaDotOp(const XlaDotOp&) = delete;
  void operator=(const XlaDotOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaDot"), MlirXlaOpKernel);

class XlaDotV2Op : public XlaDotOp {
 public:
  explicit XlaDotV2Op(OpKernelConstruction* context) : XlaDotOp(context) {
    DataType preferred_element_dtype;
    OP_REQUIRES_OK(context, context->GetAttr("preferred_element_type",
                                             &preferred_element_dtype));
    xla::PrimitiveType preferred_element_type;
    OP_REQUIRES_OK(context, DataTypeToPrimitiveType(preferred_element_dtype,
                                                    &preferred_element_type));
    preferred_element_type_ = preferred_element_type;
  }

 private:
  XlaDotV2Op(const XlaDotV2Op&) = delete;
  void operator=(const XlaDotV2Op&) = delete;
};

REGISTER_XLA_OP(Name("XlaDotV2"), MlirXlaOpKernel);

}  // namespace
}  // namespace tensorflow
