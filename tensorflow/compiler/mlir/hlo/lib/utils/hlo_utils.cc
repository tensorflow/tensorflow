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

#include "mlir-hlo/utils/hlo_utils.h"

#include <numeric>

#include "mlir/IR/Attributes.h"

namespace mlir {
namespace hlo {

DenseIntElementsAttr getBroadcastDimensionsAttr(Builder* b, Value x, Value y,
                                                bool allow_empty) {
  TensorType xType = x.getType().dyn_cast<RankedTensorType>();
  TensorType yType = y.getType().dyn_cast<RankedTensorType>();
  if (!xType || !yType) return {};
  if (allow_empty && xType == yType) return {};

  // If the shapes have the same rank, then there is nothing to do.
  auto xRank = xType.getRank(), yRank = yType.getRank();
  if (allow_empty && xRank == yRank) return {};

  // Otherwise if the ranks of the inputs don't match, TensorFlow automatically
  // reshapes the smaller by padding with dimensions of size 1 as a prefix. In
  // other words to pad a 5-vector to a 3-dimensional tensor it is reshaped to
  // have shape [1,1,5]. XLA's automatic broadcast code is able to broadcast
  // from lower to higher rank, but doesn't assume you want to pad as a prefix
  // of the dimensions, and instead needs to be told which dimensions of the
  // higher rank tensor to match to the lower rank tensor.
  auto maxRank = std::max(xRank, yRank);
  auto minRank = std::min(xRank, yRank);

  // Match the lower rank tensor along the larger-numbered dimensions of the
  // higher rank tensor.
  SmallVector<int64_t, 4> broadcastDimensions(minRank);
  std::iota(broadcastDimensions.begin(), broadcastDimensions.end(),
            maxRank - minRank);

  RankedTensorType type =
      RankedTensorType::get({minRank}, b->getIntegerType(64));
  return DenseIntElementsAttr::get(type, broadcastDimensions);
}

DenseElementsAttr GetScalarOfType(Type ty, int64_t raw_value) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);

  if (auto float_ty = ty.dyn_cast<FloatType>()) {
    APFloat value(float_ty.getFloatSemantics(), raw_value);
    return DenseElementsAttr::get(scalar_ty, value);
  } else if (auto int_ty = ty.dyn_cast<IntegerType>()) {
    APInt value(int_ty.getWidth(), static_cast<int64_t>(raw_value), true);
    return DenseElementsAttr::get(scalar_ty, value);
  } else if (auto complex_ty = ty.dyn_cast<ComplexType>()) {
    Type complex_element_ty = complex_ty.getElementType();
    if (complex_element_ty.isF32()) {
      return DenseElementsAttr::get(
          scalar_ty, static_cast<std::complex<float>>(raw_value));
    } else if (complex_element_ty.isF64()) {
      return DenseElementsAttr::get(
          scalar_ty, static_cast<std::complex<double>>(raw_value));
    }
  }
  llvm_unreachable("unsupported type");
}

static APFloat GetScalarLimitOfFloatType(FloatType float_ty,
                                         ScalarLimit limit) {
  auto& semantics = float_ty.getFloatSemantics();
  switch (limit) {
    case kLowest:
      return APFloat::getLargest(semantics, /*negative=*/true);
    case kInfinityLowest:
      return APFloat::getInf(semantics, /*negative=*/true);
    case kMax:
      return APFloat::getLargest(semantics, /*negative=*/false);
    case kInfinityMax:
      return APFloat::getInf(semantics, /*negative=*/false);
  }
  llvm_unreachable("invalid limit");
}

// Returns a scalar value for the given integer type.
//
// The argument 'scalar' describes which scalar value to return. `integer_value`
// is used to specify the integer value for kInteger. For any other scalar,
// integer_value is ignored.
static APInt GetScalarLimitOfIntegerType(IntegerType integer_ty,
                                         ScalarLimit limit) {
  unsigned width = integer_ty.getWidth();
  bool is_bool = (width == 1);
  switch (limit) {
    case kLowest:
    case kInfinityLowest:
      if (integer_ty.isUnsigned() || is_bool) {
        return APInt::getMinValue(width);
      } else {
        return APInt::getSignedMinValue(width);
      }

    case kMax:
    case kInfinityMax:
      if (integer_ty.isUnsigned() || is_bool) {
        return APInt::getMaxValue(width);
      } else {
        return APInt::getSignedMaxValue(width);
      }
  }
  llvm_unreachable("invalid limit");
}

DenseElementsAttr GetScalarLimitOfType(Type ty, ScalarLimit limit) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);
  if (auto float_ty = ty.dyn_cast<FloatType>()) {
    return DenseElementsAttr::get(scalar_ty,
                                  GetScalarLimitOfFloatType(float_ty, limit));
  } else if (auto integer_ty = ty.dyn_cast<IntegerType>()) {
    return DenseElementsAttr::get(
        scalar_ty, GetScalarLimitOfIntegerType(integer_ty, limit));
  }
  llvm_unreachable("unsupported type");
}

std::string LmhloToMhloOpName(llvm::StringRef op_name,
                              mlir::MLIRContext* context) {
  assert(op_name.startswith("lmhlo.") && "Expected an LMHLO op");

  if (op_name == "lmhlo.dot") {
    return "mhlo.dot_general";
  }

  if (op_name == "lmhlo.dynamic_slice") {
    return "mhlo.dynamic-slice";
  }

  std::string mhlo_op_name(op_name.drop_front(1));
  if (context->isOperationRegistered(mhlo_op_name)) return mhlo_op_name;
  return "";
}

bool IsSequenceStartingWith0(Attribute attr) {
  DenseIntElementsAttr denseAttr = attr.dyn_cast<DenseIntElementsAttr>();
  for (int64_t i = 0, e = denseAttr.getNumElements(); i < e; ++i)
    if (denseAttr.getValues<APInt>()[i].getSExtValue() != i) return false;
  return true;
}

int64_t getArgumentIndex(mlir::FuncOp op, Value value) {
  BlockArgument arg = value.dyn_cast<BlockArgument>();
  if (!arg || arg.getOwner() != &op.front()) return -1;
  return arg.getArgNumber();
}

}  // namespace hlo
}  // namespace mlir
