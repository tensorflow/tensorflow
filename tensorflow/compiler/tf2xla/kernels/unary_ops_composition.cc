/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/kernels/elu_op.h"
#include "tensorflow/compiler/tf2xla/kernels/relu_op.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

using XlaUnaryOpGenerator = std::function<xla::XlaOp(xla::XlaOp)>;
using XlaOpGeneratorMap = absl::flat_hash_map<string, XlaUnaryOpGenerator>;

void PopulateXlaOpGeneratorMap(XlaOpGeneratorMap* op_generator_map) {
  auto add_xla_op_generator = [&](std::string name,
                                  XlaUnaryOpGenerator xla_op_generator) {
    CHECK(op_generator_map->insert({name, xla_op_generator}).second);
  };

#define ADD_XLA_OP_GENERATOR(Name) add_xla_op_generator(#Name, xla::Name);

  ADD_XLA_OP_GENERATOR(Abs);
  ADD_XLA_OP_GENERATOR(Acos);
  ADD_XLA_OP_GENERATOR(Acosh);
  ADD_XLA_OP_GENERATOR(Asin);
  ADD_XLA_OP_GENERATOR(Asinh);
  ADD_XLA_OP_GENERATOR(Atan);
  ADD_XLA_OP_GENERATOR(Atanh);
  ADD_XLA_OP_GENERATOR(Ceil);
  ADD_XLA_OP_GENERATOR(Cos);
  ADD_XLA_OP_GENERATOR(Cosh);
  ADD_XLA_OP_GENERATOR(Expm1);
  ADD_XLA_OP_GENERATOR(Exp);
  ADD_XLA_OP_GENERATOR(Floor);
  add_xla_op_generator(
      "Inv", [](xla::XlaOp x) { return xla::ScalarLike(x, 1.0) / x; });
  ADD_XLA_OP_GENERATOR(Log);
  ADD_XLA_OP_GENERATOR(Log1p);
  ADD_XLA_OP_GENERATOR(Neg);
  ADD_XLA_OP_GENERATOR(Reciprocal);
  add_xla_op_generator("Rint", xla::RoundToEven);
  ADD_XLA_OP_GENERATOR(Round);
  ADD_XLA_OP_GENERATOR(Rsqrt);
  add_xla_op_generator("Sigmoid", xla::Logistic);
  ADD_XLA_OP_GENERATOR(Sin);
  ADD_XLA_OP_GENERATOR(Sinh);
  ADD_XLA_OP_GENERATOR(Sqrt);
  ADD_XLA_OP_GENERATOR(Square);
  ADD_XLA_OP_GENERATOR(Tan);
  ADD_XLA_OP_GENERATOR(Tanh);

  ADD_XLA_OP_GENERATOR(Elu);
  ADD_XLA_OP_GENERATOR(Relu);
  ADD_XLA_OP_GENERATOR(Relu6);
  ADD_XLA_OP_GENERATOR(Selu);

#undef ADD_XLA_OP_GENERATOR
}

const XlaOpGeneratorMap& GetXlaOpGeneratorMap() {
  static XlaOpGeneratorMap* result = []() {
    auto* result = new XlaOpGeneratorMap;
    PopulateXlaOpGeneratorMap(result);
    return result;
  }();

  return *result;
}

class UnaryOpsCompositionOp : public XlaOpKernel {
 public:
  explicit UnaryOpsCompositionOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("op_names", &op_names_));

    const XlaOpGeneratorMap& op_generator_map = GetXlaOpGeneratorMap();
    for (absl::string_view op_name : op_names_) {
      OP_REQUIRES(ctx, op_generator_map.contains(op_name),
                  errors::Unimplemented(
                      op_name, " not supported in _UnaryOpsComposition"));
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp x = ctx->Input(0);
    const XlaOpGeneratorMap& op_generator_map = GetXlaOpGeneratorMap();
    for (absl::string_view op_name : op_names_) {
      x = op_generator_map.find(op_name)->second(x);
    }
    ctx->SetOutput(0, x);
  }

 private:
  std::vector<string> op_names_;
};

REGISTER_XLA_OP(Name("_UnaryOpsComposition"), UnaryOpsCompositionOp);

}  // namespace
}  // namespace tensorflow
