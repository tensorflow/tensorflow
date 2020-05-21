/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/tftext_utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

constexpr char kWhitespaceTokenizer[] = "tftext:WhitespaceTokenizer";
constexpr char kTFAPIImplements[] = "tf.api_implements";

inline OpaqueElementsAttr emptyCustomOption(OpBuilder* builder) {
  std::string content = "";
  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(
      builder->getContext()->getRegisteredDialect("tfl"), type, content);
}

inline RankedTensorType getInputType(mlir::FuncOp func, int idx) {
  return func.getType()
      .getInput(idx)
      .dyn_cast_or_null<mlir::RankedTensorType>();
}

inline RankedTensorType getResultType(mlir::FuncOp func, int idx) {
  return func.getType()
      .getResult(idx)
      .dyn_cast_or_null<mlir::RankedTensorType>();
}

LogicalResult VerifyWhitespaceTokenizer(mlir::FuncOp func) {
  if (func.getNumResults() != 2) {
    return failure();
  }
  if (func.getNumArguments() != 1) {
    return failure();
  }
  auto input_type = getInputType(func, 0);
  if (!input_type || input_type.getRank() != 1 ||
      !input_type.getElementType().isa<mlir::TF::StringType>()) {
    return failure();
  }
  auto value_type = getResultType(func, 0);
  if (!value_type || value_type.getRank() != 1 ||
      !value_type.getElementType().isa<mlir::TF::StringType>()) {
    return failure();
  }
  auto offset_type = getResultType(func, 1);
  if (offset_type.getRank() != 1 ||
      !offset_type.getElementType().isInteger(64)) {
    return failure();
  }
  return success();
}

LogicalResult ConvertWhitespaceTokenizer(mlir::FuncOp func,
                                         llvm::StringRef api) {
  func.eraseBody();
  func.addEntryBlock();
  func.setAttr(kTFAPIImplements, StringAttr::get(api, func.getContext()));

  Value text = func.getArgument(0);
  auto output_type = func.getType().getResult(0);
  auto offset_type = func.getType().getResult(1);
  SmallVector<Type, 2> shape = {output_type, offset_type};
  ArrayRef<Type> output_types(shape);

  OpBuilder builder(func.getBody());

  auto op = builder.create<mlir::TFL::CustomOp>(func.getLoc(), output_types,
                                                ValueRange(text), api,
                                                emptyCustomOption(&builder));

  builder.create<mlir::ReturnOp>(func.getLoc(), op.getResults());
  return success();
}
}  // namespace

LogicalResult ConvertTFTextAPI(mlir::FuncOp func, llvm::StringRef api) {
  if (api.str() == kWhitespaceTokenizer) {
    if (succeeded(VerifyWhitespaceTokenizer(func))) {
      return ConvertWhitespaceTokenizer(func, api);
    }
  }
  return failure();
}

}  // namespace TFL
}  // namespace mlir
