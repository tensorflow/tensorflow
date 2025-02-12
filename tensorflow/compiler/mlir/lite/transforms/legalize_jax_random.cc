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

// The full pipeline of converting jax random include 2 steps.
// 1. Rename the jax random functions to tflite wrapped functions with the aid
//    of "jax.named_call". For example, in the dumped hlo, the
//    jax.random.uniform will have name "tfl_wrapped_jax_random_uniform".
// 2. Replace the body of "tfl_wrapped_jax_random_uniform" and
//    "tfl_wrapped_jax_random_normal" with tfl.CustomOp("RandomUniform") and
//     tfl.CustomOp("RandomStandardNormal"), respectively.

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
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
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_LEGALIZEJAXRANDOMPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

struct LegalizeJaxRandomPass
    : public impl::LegalizeJaxRandomPassBase<LegalizeJaxRandomPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeJaxRandomPass)

  void runOnOperation() override;
};

inline ConstBytesAttr CustomOption(ImplicitLocOpBuilder *builder,
                                   const std::string &content) {
  return ConstBytesAttr::get(builder->getContext(),
                             StringRef(content.data(), content.size()));
}

inline bool IsJaxRandomUniform(mlir::func::FuncOp func) {
  return func.getName().contains("tfl_wrapped_jax_random_uniform");
}

inline bool IsJaxRandomNormal(mlir::func::FuncOp func) {
  return func.getName().contains("tfl_wrapped_jax_random_normal");
}

void LegalizeJaxRandomPass::runOnOperation() {
  auto func = getOperation();
  if (!IsJaxRandomUniform(func) && !IsJaxRandomNormal(func)) return;
  auto result_tuple_ty =
      mlir::dyn_cast_or_null<TupleType>(func.getFunctionType().getResult(0));
  if (!result_tuple_ty) return;
  if (result_tuple_ty.size() != 1) return;
  auto result_ty = mlir::dyn_cast<ShapedType>(result_tuple_ty.getType(0));

  func.eraseBody();
  func.addEntryBlock();
  ImplicitLocOpBuilder builder(func.getLoc(), func.getBody());
  llvm::SmallVector<int32_t> result_shape_i32;
  auto result_shape = result_ty.getShape();
  for (auto element : result_shape) {
    result_shape_i32.push_back(static_cast<int32_t>(element));
  }
  auto result_shape_attr = builder.getI32TensorAttr(result_shape_i32);
  Value result_shape_tensor =
      builder.create<stablehlo::ConstantOp>(result_shape_attr);
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
  Value tulple_result = builder.create<stablehlo::TupleOp>(random_result);
  builder.create<mlir::func::ReturnOp>(tulple_result);
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeJaxRandomPass() {
  return std::make_unique<LegalizeJaxRandomPass>();
}

}  // namespace TFL
}  // namespace mlir
