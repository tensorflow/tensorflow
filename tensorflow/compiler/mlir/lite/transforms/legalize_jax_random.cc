/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// The full pipline of converting jax random include 2 steps.
// 1. Rename the jax random functions to tflite wrapped functions with the aid
//    of "jax.named_call". For example, in the dumped hlo, the
//    jax.random.uniform will have name "tfl_wrapped_jax_random_uniform".
// 2. Replace the body of "tfl_wrapped_jax_random_uniform" and
//    "tfl_wrapped_jax_random_normal" with tfl.CustomOp("RandomUniform") and
//     tfl.CustomOp("RandomStandardNormal"), respectively.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

struct LegalizeJaxRandomPass
    : public PassWrapper<LegalizeJaxRandomPass, FunctionPass> {
 public:
  StringRef getArgument() const final { return "tfl-legalize-random"; }
  StringRef getDescription() const final {
    return "Replace jax.random.uniform/normal with tfl.custom.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect, mhlo::MhloDialect>();
  }
  void runOnFunction() override;
};

inline OpaqueElementsAttr CustomOption(ImplicitLocOpBuilder *builder,
                                       const std::string &content) {
  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(builder->getContext()->getLoadedDialect("tfl"),
                                 type,
                                 StringRef(content.data(), content.size()));
}

inline bool IsJaxRandomUniform(mlir::FuncOp func) {
  return func.getName().contains("tfl_wrapped_jax_random_uniform");
}

inline bool IsJaxRandomNormal(mlir::FuncOp func) {
  return func.getName().contains("tfl_wrapped_jax_random_normal");
}

void LegalizeJaxRandomPass::runOnFunction() {
  auto func = getFunction();
  if (!IsJaxRandomUniform(func) && !IsJaxRandomNormal(func)) return;
  auto result_tuple_ty =
      func.getType().getResult(0).dyn_cast_or_null<TupleType>();
  if (!result_tuple_ty) return;
  if (result_tuple_ty.size() != 1) return;
  auto result_ty = result_tuple_ty.getType(0).dyn_cast<ShapedType>();

  func.eraseBody();
  func.addEntryBlock();
  ImplicitLocOpBuilder builder(func.getLoc(), func.getBody());
  llvm::SmallVector<int32_t> result_shape_i32;
  auto result_shape = result_ty.getShape();
  for (auto element : result_shape) {
    result_shape_i32.push_back(static_cast<int32_t>(element));
  }
  auto result_shape_attr = builder.getI32TensorAttr(result_shape_i32);
  Value result_shape_tensor = builder.create<mhlo::ConstOp>(result_shape_attr);
  auto custom_code =
      IsJaxRandomUniform(func) ? "RandomUniform" : "RandomStandardNormal";

  llvm::SmallVector<Type> result_ty_vec({result_ty});
  llvm::SmallVector<Value> result_shape_tensor_vec({result_shape_tensor});
  auto attr = CustomOption(&builder, "");
  Value random_result =
      builder
          .create<TFL::CustomOp>(TypeRange(result_ty_vec),
                                 ValueRange(result_shape_tensor_vec),
                                 custom_code, attr)
          .getResult(0);
  Value tulple_result = builder.create<mhlo::TupleOp>(random_result);
  builder.create<mlir::ReturnOp>(tulple_result);
}

static PassRegistration<LegalizeJaxRandomPass> pass;
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeJaxRandomPass() {
  return std::make_unique<LegalizeJaxRandomPass>();
}

}  // namespace TFL
}  // namespace mlir
