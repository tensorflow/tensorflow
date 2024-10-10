/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/fusions/triton/xla_triton_ops.h"

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"  // IWYU pragma: keep
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"  // IWYU pragma: keep
#include "mlir/IR/ValueRange.h"
#include "xla/service/gpu/fusions/triton/xla_triton_dialect.cc.inc"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using mlir::Dialect;
using mlir::DictionaryAttr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpaqueProperties;
using mlir::RankedTensorType;
using mlir::RegionRange;
using mlir::SmallVectorImpl;
using mlir::TensorOrMemDesc;
using mlir::Type;
using mlir::ValueRange;
using mlir::triton::DialectInferLayoutInterface;
using mlir::triton::DotOp;

namespace xla {
namespace triton {

void XlaTritonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/service/gpu/fusions/triton/xla_triton_ops.cc.inc"
      >();
}

LogicalResult SparseDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return DotOp::inferReturnTypes(context, location, operands, attributes,
                                 properties, regions, inferredReturnTypes);
}

LogicalResult SparseDotOp::verify() {
  // Implied properties of 2:4 sparse dots.
  constexpr int kContractingFactor = 2;
  constexpr int kMetadataElementsPerPackedValue = 8;
  // Verify operand A.
  auto aTensorTy = llvm::cast<TensorOrMemDesc>(getOperand(0).getType());
  auto aElemTy = aTensorTy.getElementType();
  if (!aElemTy.isF16() && !aElemTy.isBF16())
    return emitError("element type of operand A is not supported");
  auto aShape = aTensorTy.getShape();
  if (aShape.size() != 2) return emitError("shape of operand A is incorrect");

  // Verify operand B.
  auto bTensorTy = llvm::cast<TensorOrMemDesc>(getOperand(1).getType());
  auto bElemTy = bTensorTy.getElementType();
  if (!bElemTy.isF16() && !bElemTy.isBF16())
    return emitError("element type of operand B is not supported");
  auto bShape = bTensorTy.getShape();
  if (bShape.size() != 2) return emitError("shape of operand B is incorrect");

  // Verify operand C.
  auto cTensorTy = llvm::cast<RankedTensorType>(getOperand(2).getType());
  auto cElemTy = cTensorTy.getElementType();
  if (!cElemTy.isF32())
    return emitError("element type of operand C is not supported");
  auto cShape = cTensorTy.getShape();
  if (cShape.size() != 2) return emitError("shape of operand C is incorrect");

  // Check operand dependencies.
  if (aShape[0] != cShape[0] || bShape[1] != cShape[1] ||
      bShape[0] != aShape[1] * kContractingFactor)
    return emitError("operand shape dimensions are incorrect");
  if (aElemTy != bElemTy)
    return emitError("operand element types do not match");

  // Verify sparse metadata.
  auto metaTy = llvm::cast<RankedTensorType>(getOperand(3).getType());
  auto metaShape = metaTy.getShape();
  if (!metaTy.getElementType().isInteger(16) || metaShape.size() != 2)
    return emitError("sparse metadata tensor is invalid");
  if (metaShape[0] != aShape[0] ||
      metaShape[1] * kMetadataElementsPerPackedValue != aShape[1])
    return emitError("sparse metadata shape dimensions are incorrect");

  // Verify tensor encoding.
  auto aEncoding = aTensorTy.getEncoding();
  auto bEncoding = bTensorTy.getEncoding();
  if (!aEncoding && !bEncoding) return mlir::success();
  if (!aEncoding || !bEncoding)
    return emitError("mismatching encoding between A and B operands");

  Dialect &dialect = aEncoding.getDialect();
  auto interface = llvm::cast<DialectInferLayoutInterface>(&dialect);
  return interface->verifyDotOpEncodingCompatibility(getOperation(), aEncoding,
                                                     bEncoding);
}

}  // namespace triton
}  // namespace xla

#define GET_OP_CLASSES
#include "xla/service/gpu/fusions/triton/xla_triton_ops.cc.inc"
