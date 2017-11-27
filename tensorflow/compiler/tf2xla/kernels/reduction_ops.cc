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

// XLA-specific reduction Ops.

#include "tensorflow/compiler/tf2xla/kernels/reduction_ops.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

class SumOp : public XlaReductionOp {
 public:
  explicit SumOp(OpKernelConstruction* ctx) : XlaReductionOp(ctx) {}
  void BuildReducer(xla::ComputationBuilder* builder,
                    const xla::ComputationDataHandle& scalar_lhs,
                    const xla::ComputationDataHandle& scalar_rhs) override {
    builder->Add(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Sum"), SumOp);

class ProdOp : public XlaReductionOp {
 public:
  explicit ProdOp(OpKernelConstruction* ctx) : XlaReductionOp(ctx) {}

  xla::ComputationDataHandle InitialValue(
      xla::ComputationBuilder* builder) override {
    return XlaHelpers::One(builder, input_type(0));
  }

  void BuildReducer(xla::ComputationBuilder* builder,
                    const xla::ComputationDataHandle& scalar_lhs,
                    const xla::ComputationDataHandle& scalar_rhs) override {
    builder->Mul(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Prod"), ProdOp);

class MinOp : public XlaReductionOp {
 public:
  explicit MinOp(OpKernelConstruction* ctx) : XlaReductionOp(ctx) {}

  xla::ComputationDataHandle InitialValue(
      xla::ComputationBuilder* builder) override {
    xla::PrimitiveType type;
    TF_CHECK_OK(DataTypeToPrimitiveType(input_type(0), &type));
    return builder->ConstantLiteral(xla::Literal::MaxValue(type));
  }

  void BuildReducer(xla::ComputationBuilder* builder,
                    const xla::ComputationDataHandle& scalar_lhs,
                    const xla::ComputationDataHandle& scalar_rhs) override {
    builder->Min(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Min"), MinOp);

class MaxOp : public XlaReductionOp {
 public:
  explicit MaxOp(OpKernelConstruction* ctx) : XlaReductionOp(ctx) {}

  xla::ComputationDataHandle InitialValue(
      xla::ComputationBuilder* builder) override {
    xla::PrimitiveType type;
    TF_CHECK_OK(DataTypeToPrimitiveType(input_type(0), &type));
    return builder->ConstantLiteral(xla::Literal::MinValue(type));
  }

  void BuildReducer(xla::ComputationBuilder* builder,
                    const xla::ComputationDataHandle& scalar_lhs,
                    const xla::ComputationDataHandle& scalar_rhs) override {
    builder->Max(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Max"), MaxOp);

class MeanOp : public XlaReductionOp {
 public:
  explicit MeanOp(OpKernelConstruction* ctx) : XlaReductionOp(ctx) {}

  void BuildReducer(xla::ComputationBuilder* builder,
                    const xla::ComputationDataHandle& scalar_lhs,
                    const xla::ComputationDataHandle& scalar_rhs) override {
    builder->Add(scalar_lhs, scalar_rhs);
  }

  xla::ComputationDataHandle BuildFinalizer(
      xla::ComputationBuilder* builder,
      const xla::ComputationDataHandle& reduce_output,
      int64 num_elements_reduced) override {
    auto divisor = XlaHelpers::IntegerLiteral(builder, input_type(0),
                                              num_elements_reduced);
    return builder->Div(reduce_output, divisor);
  }
};

REGISTER_XLA_OP(Name("Mean"), MeanOp);

class AllOp : public XlaReductionOp {
 public:
  explicit AllOp(OpKernelConstruction* ctx) : XlaReductionOp(ctx) {}

  xla::ComputationDataHandle InitialValue(
      xla::ComputationBuilder* builder) override {
    return builder->ConstantR0<bool>(true);
  }

  void BuildReducer(xla::ComputationBuilder* builder,
                    const xla::ComputationDataHandle& scalar_lhs,
                    const xla::ComputationDataHandle& scalar_rhs) override {
    builder->And(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("All"), AllOp);

class AnyOp : public XlaReductionOp {
 public:
  explicit AnyOp(OpKernelConstruction* ctx) : XlaReductionOp(ctx) {}

  xla::ComputationDataHandle InitialValue(
      xla::ComputationBuilder* builder) override {
    return builder->ConstantR0<bool>(false);
  }

  void BuildReducer(xla::ComputationBuilder* builder,
                    const xla::ComputationDataHandle& scalar_lhs,
                    const xla::ComputationDataHandle& scalar_rhs) override {
    builder->Or(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Any"), AnyOp);

}  // namespace
}  // namespace tensorflow
