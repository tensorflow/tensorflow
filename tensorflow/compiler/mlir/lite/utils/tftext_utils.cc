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

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

constexpr char kNgrams[] = "tftext:Ngrams";
constexpr char kWhitespaceTokenizer[] = "tftext:WhitespaceTokenizer";
constexpr char kCustomSgnnProjection[] = "tftext:custom:SgnnProjection";
constexpr char kTFImplements[] = "tf._implements";

using mlir::TF::FuncAttr;
using mlir::TF::StringType;

inline OpaqueElementsAttr CustomOption(OpBuilder* builder,
                                       const std::string& content) {
  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(builder->getContext()->getLoadedDialect("tfl"),
                                 type,
                                 StringRef(content.data(), content.size()));
}

inline TensorType GetInputType(FuncOp func, int idx) {
  return func.getType().getInput(idx).dyn_cast_or_null<TensorType>();
}

inline TensorType GetResultType(FuncOp func, int idx) {
  return func.getType().getResult(idx).dyn_cast_or_null<TensorType>();
}

inline bool RankEquals(const TensorType& type, int rank) {
  return type && type.hasRank() && type.getRank() == rank;
}

LogicalResult VerifyWhitespaceTokenizer(FuncOp func) {
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
  auto input_type = GetInputType(func, 0);
  if (!input_type || !input_type.getElementType().isa<StringType>() ||
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

  auto value_type = GetResultType(func, 0);
  if (!RankEquals(value_type, 1) ||
      !value_type.getElementType().isa<StringType>()) {
    return func.emitError() << "1st output should be string tensor";
  }
  if (func.getNumResults() > 1) {
    auto offset_type = GetResultType(func, 1);
    if (!RankEquals(offset_type, 1) ||
        !offset_type.getElementType().isInteger(64)) {
      return func.emitError() << "2nd output should be int64 tensor";
    }
  }
  if (func.getNumResults() > 2) {
    auto offset_type = GetResultType(func, 2);
    if (!RankEquals(offset_type, 1) ||
        !offset_type.getElementType().isInteger(64)) {
      return func.emitError() << "3rd output should be int64 tensor";
    }
  }

  return success();
}

LogicalResult ConvertWhitespaceTokenizer(FuncOp func, llvm::StringRef api,
                                         FuncAttr attr) {
  func.eraseBody();
  func.addEntryBlock();
  func->setAttr(kTFImplements, attr);
  OpBuilder builder(func.getBody());
  std::string empty_option_buffer;
  auto op = builder.create<CustomOp>(
      func.getLoc(), func.getType().getResults(), func.getArguments(), api,
      CustomOption(&builder, empty_option_buffer));
  builder.create<ReturnOp>(func.getLoc(), op.getResults());
  return success();
}

LogicalResult VerifyNgrams(FuncOp func) {
  // The inputs and outputs should be the same:
  // * A string tensor for tokens/ragged tensor values.
  // * Zero or more row_split tensors.
  constexpr int kValues = 0;
  constexpr int kRowSplits = 1;

  if (func.getType().getInputs().size() != func.getType().getResults().size()) {
    return func.emitError() << "Mismatched number of inputs and outputs.";
  }

  int row_splits = func.getType().getInputs().size() - kRowSplits;
  if (row_splits == 0) {
    auto input_values = GetInputType(func, kValues);
    if (!input_values || !input_values.getElementType().isa<StringType>()) {
      return func.emitError()
             << "Input " << kValues << " should be a string tensor";
    }
    auto output_values = GetResultType(func, kValues);
    if (!output_values || !output_values.getElementType().isa<StringType>()) {
      return func.emitError()
             << "Output " << kValues << " should be a string tensor";
    }

    if (input_values.hasRank() && output_values.hasRank() &&
        input_values.getRank() != output_values.getRank()) {
      return func.emitError() << "Input " << kValues << " and output "
                              << kValues << " should have the same rank";
    }
  } else {
    auto input_values = GetInputType(func, kValues);
    if (!RankEquals(input_values, 1) ||
        !input_values.getElementType().isa<StringType>()) {
      return func.emitError()
             << "Input " << kValues << " should be a 1D string tensor";
    }
    auto output_values = GetResultType(func, kValues);
    if (!RankEquals(output_values, 1) ||
        !output_values.getElementType().isa<StringType>()) {
      return func.emitError()
             << "Output " << kValues << " should be a 1D string tensor";
    }

    for (int i = 0; i < row_splits; ++i) {
      const int row_index = i + kRowSplits;
      auto input_row_splits = GetInputType(func, row_index);
      if (!RankEquals(input_row_splits, 1) ||
          !input_row_splits.getElementType().isInteger(64)) {
        return func.emitError()
               << "Input " << row_index << " should be a 1D int64 tensor";
      }
      auto output_row_splits = GetResultType(func, row_index);
      if (!RankEquals(output_row_splits, 1) ||
          !output_row_splits.getElementType().isInteger(64)) {
        return func.emitError()
               << "Output " << row_index << " should be a 1D int64 tensor";
      }
    }
  }

  return success();
}

LogicalResult CreateNgramsCustomOption(FuncOp func, DictionaryAttr attrs,
                                       std::string& custom_option_buffer) {
  flexbuffers::Builder fbb;
  size_t start_map = fbb.StartMap();

  auto width = attrs.get("width").dyn_cast_or_null<IntegerAttr>();
  if (!width) {
    return func.emitError() << "'width' attribute is not set or not an integer";
  }
  fbb.Int("width", width.getInt());

  auto string_separator =
      attrs.get("string_separator").dyn_cast_or_null<StringAttr>();
  if (!string_separator) {
    return func.emitError()
           << "'string_separator' attribute is not set or not a string";
  }
  // StringAttrs are not guaranteed to be NUL terminated, but flexbuffers
  // strings expect NUL terminated strings.
  std::string string_separator_str(string_separator.getValue().data(),
                                   string_separator.getValue().size());
  fbb.String("string_separator", string_separator_str);

  auto axis = attrs.get("axis").dyn_cast_or_null<IntegerAttr>();
  if (!axis) {
    return func.emitError() << "'axis' attribute is not set or not an integer";
  }
  fbb.Int("axis", axis.getInt());

  auto reduction_type =
      attrs.get("reduction_type").dyn_cast_or_null<StringAttr>();
  if (!reduction_type) {
    return func.emitError()
           << "'reduction_type' attribute is not set or not a string";
  }
  // StringAttrs are not guaranteed to be NUL terminated, but flexbuffers
  // strings expect NUL terminated strings.
  std::string reduction_type_str(reduction_type.getValue().data(),
                                 reduction_type.getValue().size());
  fbb.String("reduction_type", reduction_type_str);

  fbb.EndMap(start_map);
  fbb.Finish();
  custom_option_buffer.assign(fbb.GetBuffer().begin(), fbb.GetBuffer().end());
  return success();
}

LogicalResult ConvertNgrams(FuncOp func, llvm::StringRef api, FuncAttr attr) {
  func.eraseBody();
  func.addEntryBlock();
  func->setAttr(kTFImplements, attr);
  OpBuilder builder(func.getBody());
  std::string custom_option_buffer;
  if (failed(CreateNgramsCustomOption(func, attr.getAttrs(),
                                      custom_option_buffer))) {
    return failure();
  }
  auto op = builder.create<CustomOp>(
      func.getLoc(), func.getType().getResults(), func.getArguments(), api,
      CustomOption(&builder, custom_option_buffer));
  builder.create<ReturnOp>(func.getLoc(), op.getResults());
  return success();
}

LogicalResult VerifySgnnProjection(FuncOp func, FuncAttr attr) {
  if (func.getType().getNumInputs() != 2 ||
      func.getType().getNumResults() != 1) {
    return func.emitError() << "Mismatched number of inputs and outputs.";
  }
  auto values_type = GetInputType(func, 0);
  if (!values_type || !values_type.getElementType().isa<StringType>()) {
    return func.emitError() << "First input should be a string tensor";
  }
  auto row_splits_type = GetInputType(func, 1);
  if (!row_splits_type ||
      !row_splits_type.getElementType().isa<IntegerType>()) {
    return func.emitError() << "Second input should be an integer tensor";
  }

  auto hash_seed =
      attr.getAttrs().get("hash_seed").dyn_cast_or_null<ArrayAttr>();
  if (!hash_seed) {
    return func.emitError()
           << "'hash_seed' attribute is not set or not an array";
  }
  auto output_type = GetResultType(func, 0);
  if (!output_type || !output_type.getElementType().isa<FloatType>() ||
      !RankEquals(output_type, 2)) {
    return func.emitError() << "Output should be a 2D float tensor.";
  }
  if (output_type.getDimSize(1) != hash_seed.size()) {
    return func.emitError()
           << "Output 2nd dimension should be the num of hash seeds.";
  }

  auto buckets = attr.getAttrs().get("buckets").dyn_cast_or_null<IntegerAttr>();
  if (!buckets) {
    return func.emitError() << "'buckets' attribute is not set or not int";
  }

  return success();
}

LogicalResult CreateSgnnProjectionCustomOption(
    FuncOp func, DictionaryAttr attrs, std::string& custom_option_buffer) {
  flexbuffers::Builder fbb;
  size_t start_map = fbb.StartMap();

  auto hash_seed = attrs.get("hash_seed").dyn_cast_or_null<ArrayAttr>();
  auto vector_start = fbb.StartVector("hash_seed");
  for (int i = 0; i < hash_seed.size(); i++) {
    fbb.Add(static_cast<int32_t>(
        (hash_seed.getValue().data() + i)->dyn_cast<IntegerAttr>().getInt()));
  }
  fbb.EndVector(vector_start, /*typed=*/true, /*fixed=*/false);

  auto buckets = attrs.get("buckets").dyn_cast_or_null<IntegerAttr>();
  fbb.Int("buckets", buckets.getInt());

  fbb.EndMap(start_map);
  fbb.Finish();
  custom_option_buffer.assign(fbb.GetBuffer().begin(), fbb.GetBuffer().end());
  return success();
}

LogicalResult ConvertSgnnProjection(FuncOp func, llvm::StringRef api,
                                    FuncAttr attr) {
  // See more details in tensorflow_models/sequence_projection/sgnn/sgnn.py
  func.eraseBody();
  func.addEntryBlock();
  func->setAttr(kTFImplements, attr);
  OpBuilder builder(func.getBody());
  std::string custom_option_buffer;
  if (failed(CreateSgnnProjectionCustomOption(func, attr.getAttrs(),
                                              custom_option_buffer))) {
    return failure();
  }
  auto op = builder.create<CustomOp>(
      func.getLoc(), func.getType().getResults(), func.getArguments(), api,
      CustomOption(&builder, custom_option_buffer));
  builder.create<ReturnOp>(func.getLoc(), op.getResults());
  return success();
}
}  // namespace

LogicalResult ConvertTFTextAPI(FuncOp func, llvm::StringRef api,
                               FuncAttr attr) {
  if (api.str() == kWhitespaceTokenizer) {
    if (succeeded(VerifyWhitespaceTokenizer(func))) {
      return ConvertWhitespaceTokenizer(func, api, attr);
    }
  } else if (api.str() == kNgrams) {
    if (succeeded(VerifyNgrams(func))) {
      return ConvertNgrams(func, api, attr);
    }
  } else if (api.str() == kCustomSgnnProjection) {
    if (succeeded(VerifySgnnProjection(func, attr))) {
      return ConvertSgnnProjection(func, api, attr);
    }
  }
  return failure();
}

bool IsTFTextRegistered(const tensorflow::OpRegistry* op_registery) {
  const std::vector<std::string> kTFTextOps = {
      "WhitespaceTokenizeWithOffsets",
  };
  for (const auto& iter : kTFTextOps) {
    if (op_registery->LookUp(iter)) {
      return true;
    }
  }
  return false;
}

}  // namespace TFL
}  // namespace mlir
