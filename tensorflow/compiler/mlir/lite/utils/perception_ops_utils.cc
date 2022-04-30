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
#include "tensorflow/compiler/mlir/lite/utils/perception_ops_utils.h"

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace mlir {
namespace TFL {

namespace {

constexpr char kTFImplements[] = "tf._implements";
constexpr char kMaxUnpooling[] = "MaxUnpooling2D";
constexpr char kImageWarping[] = "DenseImageWarp";

inline OpaqueElementsAttr CustomOption(OpBuilder* builder,
                                       const std::string& content) {
  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(builder->getContext()->getLoadedDialect("tfl"),
                                 type,
                                 StringRef(content.data(), content.size()));
}

inline LogicalResult HasIntegerArrayWithSize(func::FuncOp* func,
                                             const DictionaryAttr& attrs,
                                             const std::string& attr_name,
                                             int N) {
  ArrayAttr array_attr = attrs.get(attr_name).dyn_cast_or_null<ArrayAttr>();
  if (array_attr == nullptr || array_attr.size() != N) {
    return func->emitWarning()
           << "'" << attr_name << "' attribute for " << kMaxUnpooling
           << " must be set and has size of " << N;
  }
  for (Attribute integer_attr : array_attr.getValue()) {
    IntegerAttr value = integer_attr.dyn_cast<IntegerAttr>();
    if (!value) {
      return func->emitWarning()
             << "'" << attr_name << "' attribute for " << kMaxUnpooling
             << " does not contain integer values";
    }
  }
  return success();
}

inline LogicalResult GetIntegerArraySafe(
    func::FuncOp* func, const DictionaryAttr& attrs,
    const std::string& attr_name, llvm::SmallVectorImpl<int32_t>* results,
    int N) {
  ArrayAttr array_attr = attrs.get(attr_name).dyn_cast_or_null<ArrayAttr>();
  if (array_attr == nullptr || array_attr.size() != N) {
    return func->emitError()
           << "'" << attr_name << "' attribute for " << kMaxUnpooling
           << " must be set and has size of " << N;
  }
  results->reserve(N);

  for (Attribute integer_attr : array_attr.getValue()) {
    IntegerAttr value = integer_attr.dyn_cast<IntegerAttr>();
    if (!value) {
      return func->emitError()
             << "'" << attr_name << "' attribute for " << kMaxUnpooling
             << " does not contain integer values";
    }
    results->push_back(value.getInt());
  }
  return success();
}

}  // namespace

LogicalResult ConvertMaxUnpoolingFunc::RewriteFunc() {
  func_.eraseBody();
  func_.addEntryBlock();
  func_->setAttr(kTFImplements,
                 StringAttr::get(func_.getContext(), kMaxUnpooling));

  OpBuilder builder(func_.getBody());
  std::string custom_option_buffer;
  if (failed(CreateCustomOptions(custom_option_buffer))) {
    return failure();
  }
  auto op = builder.create<CustomOp>(
      func_.getLoc(), func_.getFunctionType().getResults(),
      func_.getArguments(), kMaxUnpooling,
      CustomOption(&builder, custom_option_buffer));
  builder.create<func::ReturnOp>(func_.getLoc(), op.getResults());

  return success();
}

LogicalResult ConvertMaxUnpoolingFunc::VerifySignature() {
  // Verify high-level function signature.
  if (func_.getNumArguments() != 2) {
    return func_.emitWarning()
           << "Invalid number of arguments to " << kMaxUnpooling << ": "
           << func_.getNumArguments();
  }
  if (func_.getFunctionType().getNumResults() != 1) {
    return func_.emitWarning()
           << "Invalid number of results from " << kMaxUnpooling << ": "
           << func_.getFunctionType().getNumResults();
  }

  auto attrs = attr_.getAttrs();

  if (failed(HasIntegerArrayWithSize(&func_, attrs, "pool_size", 2))) {
    return failure();
  }

  if (failed(HasIntegerArrayWithSize(&func_, attrs, "strides", 2))) {
    return failure();
  }

  // Retrieves padding.
  auto padding = attrs.get("padding").dyn_cast_or_null<StringAttr>();
  if (!padding) {
    return func_.emitWarning() << "'padding' attribute for " << kMaxUnpooling
                               << " is not set or not a string";
  }
  if (!padding.getValue().equals("VALID") &&
      !padding.getValue().equals("SAME")) {
    return func_.emitWarning()
           << "Padding for " << kMaxUnpooling << " must be 'SAME' or 'VALID'";
  }
  return success();
}

LogicalResult ConvertMaxUnpoolingFunc::CreateCustomOptions(
    std::string& custom_option_buffer) {
  auto attrs = attr_.getAttrs();
  TfLitePoolParams pool_params;

  llvm::SmallVector<int32_t, 2> pool_size;
  if (failed(GetIntegerArraySafe(&func_, attrs, "pool_size", &pool_size, 2))) {
    return failure();
  }
  pool_params.filter_height = pool_size[0];
  pool_params.filter_width = pool_size[1];

  // Retrieve strides.
  llvm::SmallVector<int32_t, 2> strides;
  if (failed(GetIntegerArraySafe(&func_, attrs, "strides", &strides, 2))) {
    return failure();
  }
  pool_params.stride_height = strides[0];
  pool_params.stride_width = strides[1];

  // Retrieves padding.
  auto padding = attrs.get("padding").dyn_cast_or_null<StringAttr>();
  if (!padding) {
    return func_.emitError() << "'padding' attribute for " << kMaxUnpooling
                             << " is not set or not a string";
  }
  if (padding.getValue().equals("VALID")) {
    pool_params.padding = kTfLitePaddingValid;
  } else if (padding.getValue().equals("SAME")) {
    pool_params.padding = kTfLitePaddingSame;
  } else {
    return func_.emitError()
           << "Padding for " << kMaxUnpooling << " must be 'SAME' or 'VALID'";
  }

  pool_params.activation = kTfLiteActNone;
  pool_params.computed.padding = TfLitePaddingValues{0, 0, 0, 0};

  custom_option_buffer.assign(reinterpret_cast<char*>(&pool_params),
                              sizeof(TfLitePoolParams));
  return success();
}

LogicalResult ConvertDenseImageWarpFunc::RewriteFunc() {
  func_.eraseBody();
  func_.addEntryBlock();
  func_->setAttr(kTFImplements,
                 StringAttr::get(func_.getContext(), kImageWarping));

  OpBuilder builder(func_.getBody());
  auto op = builder.create<CustomOp>(func_.getLoc(),
                                     func_.getFunctionType().getResults(),
                                     func_.getArguments(), kImageWarping,
                                     CustomOption(&builder, /*content=*/""));
  builder.create<func::ReturnOp>(func_.getLoc(), op.getResults());

  return success();
}

LogicalResult ConvertDenseImageWarpFunc::VerifySignature() {
  // Verify high-level function signature.
  if (func_.getNumArguments() != 2) {
    return func_.emitWarning()
           << "Invalid number of arguments to " << kImageWarping << ": "
           << func_.getNumArguments();
  }
  if (func_.getFunctionType().getNumResults() != 1) {
    return func_.emitWarning()
           << "Invalid number of results from " << kImageWarping << ": "
           << func_.getFunctionType().getNumResults();
  }

  // Check types and shapes.
  auto image_type =
      func_.getFunctionType().getInput(0).dyn_cast_or_null<RankedTensorType>();
  if (!image_type || !image_type.getElementType().isF32() ||
      image_type.getRank() != 4) {
    return func_.emitWarning() << "Image should be a 4D float tensor";
  }

  auto flow_type =
      func_.getFunctionType().getInput(1).dyn_cast_or_null<RankedTensorType>();
  if (!flow_type || !flow_type.getElementType().isF32() ||
      flow_type.getRank() != 4) {
    return func_.emitWarning() << "Flow should be a 4D float tensor";
  }

  auto output_type =
      func_.getFunctionType().getResult(0).dyn_cast_or_null<RankedTensorType>();
  if (!output_type || !output_type.getElementType().isF32() ||
      output_type.getRank() != 4) {
    return func_.emitWarning() << "Output should be a 4D float tensor";
  }

  return success();
}

}  // namespace TFL
}  // namespace mlir
