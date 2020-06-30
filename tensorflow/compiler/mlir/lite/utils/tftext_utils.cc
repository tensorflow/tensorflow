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
  // In the case of input tensor with 0 rank.
  // Whitespace tokenizer generates 1 output:
  // * String tensor for tokens.
  //
  // In the case of 1-D input tensor,
  // Whitespace tokenizer generates 2 outputs to make up a ragged tensor:
  // * 1st output is the value of ragged tensor;
  // * 2nd output is the offset.
  //
  // In the case of batched input tesnor,
  // Whitespace tokenizer has 3 outputs to make up a nested ragged tensor:
  // * 1st output is the value of ragged tensor;
  // * 2nd output is the inner offset;
  // * 3rd output is the outer offset.
  auto input_type = getInputType(func, 0);
  if (!input_type || !input_type.getElementType().isa<mlir::TF::StringType>() ||
      !input_type.hasRank()) {
    return func.emitError() << "Input should be a string tensor";
  }

  const std::vector<int> kValidNumOfOutput = {1, 2, 3};
  if (input_type.getRank() >= kValidNumOfOutput.size()) {
    return func.emitError()
           << "Unrecognized input rank: " << input_type.getRank();
  }
  if (func.getNumResults() != kValidNumOfOutput[input_type.getRank()]) {
    return func.emitError()
           << "Expect " << kValidNumOfOutput[input_type.getRank()]
           << "output(s) when input has rank " << input_type.getRank();
  }

  auto value_type = getResultType(func, 0);
  if (!value_type || !value_type.hasRank() || value_type.getRank() != 1 ||
      !value_type.getElementType().isa<mlir::TF::StringType>()) {
    return func.emitError() << "1st output should be string tensor";
  }
  if (func.getNumResults() > 1) {
    auto offset_type = getResultType(func, 1);
    if (!offset_type || !offset_type.hasRank() || offset_type.getRank() != 1 ||
        !offset_type.getElementType().isInteger(64)) {
      return func.emitError() << "2nd output should be int64 tensor";
    }
  }
  if (func.getNumResults() > 2) {
    auto offset_type = getResultType(func, 2);
    if (!offset_type || !offset_type.hasRank() || offset_type.getRank() != 1 ||
        !offset_type.getElementType().isInteger(64)) {
      return func.emitError() << "3rd output should be int64 tensor";
    }
  }

  return success();
}

LogicalResult ConvertWhitespaceTokenizer(mlir::FuncOp func,
                                         llvm::StringRef api) {
  func.eraseBody();
  func.addEntryBlock();
  func.setAttr(kTFAPIImplements, StringAttr::get(api, func.getContext()));
  Value text = func.getArgument(0);
  OpBuilder builder(func.getBody());

  auto op = builder.create<mlir::TFL::CustomOp>(
      func.getLoc(), func.getType().getResults(), ValueRange(text), api,
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
