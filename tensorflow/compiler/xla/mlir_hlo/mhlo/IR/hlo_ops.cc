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

// This file defines the operations used in the MHLO dialect.

#include "mhlo/IR/hlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "mhlo/IR/hlo_ops.h.inc"
#include "mhlo/IR/hlo_ops_common.h"
#include "mhlo/IR/mhlo_bytecode.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/TypeInference.h"
#include "utils/convert_op_folder.h"
#include "utils/hlo_utils.h"

namespace mlir {
#include "hlo_patterns.cc.inc"
}  // namespace mlir

#include "mhlo/IR/hlo_ops_enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "mhlo/IR/hlo_ops_attrs.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "mhlo/IR/hlo_ops_typedefs.cc.inc"

namespace mlir {
namespace mhlo {

namespace detail {
/// A type representing a collection of other types.
struct AsyncBundleTypeStorage final
    : public TypeStorage,
      public llvm::TrailingObjects<AsyncBundleTypeStorage, Type> {
  using KeyTy = TypeRange;

  explicit AsyncBundleTypeStorage(unsigned numTypes) : numElements(numTypes) {}

  // Construction.
  static AsyncBundleTypeStorage* construct(TypeStorageAllocator& allocator,
                                           TypeRange key) {
    // Allocate a new storage instance.
    auto byteSize = AsyncBundleTypeStorage::totalSizeToAlloc<Type>(key.size());
    auto* rawMem =
        allocator.allocate(byteSize, alignof(AsyncBundleTypeStorage));
    auto* result = ::new (rawMem) AsyncBundleTypeStorage(key.size());

    // Copy in the element types into the trailing storage.
    std::uninitialized_copy(key.begin(), key.end(),
                            result->getTrailingObjects<Type>());
    return result;
  }

  bool operator==(const KeyTy& key) const { return key == getTypes(); }

  // Return the number of held types.
  unsigned size() const { return numElements; }

  // Return the held types.
  ArrayRef<Type> getTypes() const {
    return {getTrailingObjects<Type>(), size()};
  }

  void getFlattenedTypes(SmallVectorImpl<Type>& types) {
    for (Type type : getTypes()) {
      if (auto nestedTuple = type.dyn_cast<TupleType>())
        nestedTuple.getFlattenedTypes(types);
      else
        types.push_back(type);
    }
  }

  // The number of tuple elements.
  unsigned numElements;
};

}  // namespace detail
/// Return the elements types for this tuple.
ArrayRef<Type> AsyncBundleType::getTypes() const {
  return getImpl()->getTypes();
}
void AsyncBundleType::getFlattenedTypes(SmallVectorImpl<Type>& types) {
  getImpl()->getFlattenedTypes(types);
}

namespace {

static constexpr int64_t kDynamicPrintValue = -1;

void createArgs(ArrayRef<OpAsmParser::UnresolvedOperand> operands,
                ArrayRef<Type> types,
                SmallVector<OpAsmParser::Argument>& args) {
  for (auto argAndType : llvm::zip(operands, types)) {
    auto& arg = args.emplace_back();
    arg.ssaName = std::get<0>(argAndType);
    arg.type = std::get<1>(argAndType);
  }
}

// Checks if the vector `nums` has duplicates.
const auto hasDuplicates = [](const ArrayRef<int64_t> nums) {
  llvm::SmallDenseSet<int64_t> set(nums.begin(), nums.end());
  return set.size() != nums.size();
};

//===----------------------------------------------------------------------===//
// Utilities for the canonicalize patterns
//===----------------------------------------------------------------------===//

// This is an upper limit on how many elements can be folded by an op folder.
// This limit doesn't apply to some special cases like adding a zero,
// multiplying by one, doing many operations with splats.
constexpr int64_t kFoldOpEltLimit = 65536;

// Clamps value to the range [lower, upper].  Requires lower <= upper.
template <typename T>
static T clamp(const T& value, const T& lower, const T& upper) {
  assert(lower <= upper);
  return std::max(lower, std::min(value, upper));
}

// Verifies that dimension attribute for the op correctly indexes in operand or
// result shape.
template <typename OpT>
static LogicalResult verifyDimAttr(OpT op) {
  int64_t rank = -1;
  if (auto ty =
          op.getOperand().getType().template dyn_cast<RankedTensorType>()) {
    rank = ty.getRank();
  } else if (auto ty = op.getType().template dyn_cast<RankedTensorType>()) {
    rank = ty.getRank();
  } else {
    return success();
  }

  int64_t dim = op.getDimension();
  if (dim < 0 || dim >= rank)
    return op.emitOpError() << "requires dimension attribute in range [0, "
                            << rank << "); found (" << dim << ")";
  return success();
}

// Given the start indices and slice sizes for a dynamic-slice that can be
// converted to a static slice, returns the limits for the static slice.
DenseIntElementsAttr buildSliceLimits(DenseIntElementsAttr startIndices,
                                      DenseIntElementsAttr sliceSizes,
                                      Builder* builder) {
  SmallVector<int64_t, 4> sliceLimits;
  for (int64_t i = 0; i < sliceSizes.getNumElements(); ++i) {
    int64_t startIndex = startIndices.getValues<IntegerAttr>()[i].getInt();
    int64_t sliceSize = sliceSizes.getValues<IntegerAttr>()[i].getInt();
    sliceLimits.push_back(startIndex + sliceSize);
  }
  return builder->getI64TensorAttr(sliceLimits);
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter& rewriter, Operation* op,
                                Region& region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-block region");
  Block* block = &region.front();
  Operation* terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

#include "mhlo/IR/mhlo_canonicalize.inc"

// Common shape function helper for RngNormal and RngUniform.
static LogicalResult rngInferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (operands.size() != 3)
    return emitOptionalError(location, "expected 3 operands");

  SmallVector<int64_t> shapeVector;
  Value shapeOperand = operands[2];
  auto shapeOperandType = shapeOperand.getType().cast<ShapedType>();
  Type elementType = getElementTypeOrSelf(operands[1]);

  // Operand `shape` (1D by ODS) may be a constant or not, if `shape` is:
  // 1, not constant and have dynimic dim (tensor<?x>): infer tensor<*x>.
  // 2. not constant nor dynimic (e.g. tensor<3xi64>): infer tensor<?x?x?x>.
  // 3. constant (e.g. dense<[2, 3, 5]>): infer tensor<2x3x5x>.

  // Match to check whether the `shape` operand is a constant.
  DenseIntElementsAttr shape;
  if (!matchPattern(shapeOperand, m_Constant(&shape))) {
    int size = shapeOperandType.getDimSize(0);
    if (hlo::isDynamicDimSize(size)) {
      inferredReturnShapes.emplace_back(elementType);
      return success();
    }
    shapeVector.resize(size, ShapedType::kDynamic);
    inferredReturnShapes.emplace_back(shapeVector, elementType);
    return success();
  }

  // `shape` operand is a constant.
  shapeVector.reserve(shape.size());
  for (const APInt& fp : shape.getValues<APInt>())
    shapeVector.push_back(fp.getSExtValue());
  inferredReturnShapes.emplace_back(shapeVector, elementType);
  return success();
}

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value maybeCastTo(OpBuilder& b, Location loc, Value value, Type type) {
  if (type == value.getType()) return value;
  assert(type.isIndex() || value.getType().isIndex());
  return b.create<arith::IndexCastOp>(loc, type, value);
}

DenseElementsAttr reshape(DenseElementsAttr attr, ShapedType newType) {
  // TODO(b/232866626): DenseElementsAttr::reshape is broken for bool splats.
  // Once that ticket is fixed, we can remove this conditional.
  if (attr.isSplat() && newType.getElementType().isInteger(/*width=*/1)) {
    auto splatValue = attr.getValues<bool>()[0];
    return DenseElementsAttr::get(newType, {splatValue});
  }
  return attr.reshape(newType);
}

//===----------------------------------------------------------------------===//
// Utilities for verifiers
//===----------------------------------------------------------------------===//

// Convert a 1D dense int64 attribute to a list of values.
SmallVector<int64_t> convertDenseIntAttr(
    llvm::Optional<mlir::DenseIntElementsAttr> optionalAttr) {
  if (!optionalAttr.has_value()) return SmallVector<int64_t>{};

  mlir::DenseIntElementsAttr attr = *optionalAttr;
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

// Convert a 1D or Nx2 dense int64 attribute to a list of tuples.
FailureOr<SmallVector<std::pair<int64_t, int64_t>>> convertNx2Attribute(
    llvm::Optional<mlir::DenseIntElementsAttr> optionalAttr, Location loc) {
  if (!optionalAttr.has_value())
    return SmallVector<std::pair<int64_t, int64_t>>{};
  mlir::DenseIntElementsAttr attr = *optionalAttr;

  auto attrType = attr.getType().cast<RankedTensorType>();  // ensured by ODS.
  if (attrType.getRank() > 1) {
    if (attrType.getRank() != 2 || attrType.getShape()[1] != 2)
      return (mlir::emitError(loc) << "expects the shape of padding-attribute "
                                      "to be {N, 2}, but got {"
                                   << attrType.getShape() << "}.",
              failure());
  } else {
    // Padding values can be provided as a 1D vector as well.
    if (attr.getValues<int64_t>().size() % 2 != 0)
      return (mlir::emitError(loc)
                  << "expects the padding-entries to have even number of "
                     "elements, but got "
                  << attr.getValues<int64_t>().size() << " elements.",
              failure());
  }

  auto it = attr.getValues<int64_t>().begin();
  SmallVector<std::pair<int64_t, int64_t>> out(attr.getNumElements() / 2);
  for (auto& item : out) {
    int64_t first = *it;
    ++it;
    int64_t second = *it;
    ++it;
    item = {first, second};
  }
  return out;
}
}  // namespace

//===----------------------------------------------------------------------===//
// Utilities for attributes
//===----------------------------------------------------------------------===//

LogicalResult TypeExtensionsAttr::verifyEncoding(
    llvm::ArrayRef<int64_t> bounds, mlir::Type elementType,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  return hlo::verifyBounds(
      getBounds(), RankedTensorType::get(bounds, elementType), emitError);
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

void CollectivePermuteOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                                Type resultType, Value operand,
                                DenseIntElementsAttr sourceTargetPairs) {
  CollectivePermuteOp::build(odsBuilder, odsState, resultType, operand,
                             sourceTargetPairs, /*channel_handle=*/nullptr);
}

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceScatterOp::verify() {
  if (failed(mlir::hlo::verifyReplicaGroups(*this, /*isUniformSized=*/true)))
    return failure();
  auto operandType = getOperand().getType().cast<TensorType>();
  bool operandTypeRanked = operandType.isa<RankedTensorType>();
  Block& block = getComputation().front();
  if (failed(hlo::verifyReducerShape(
          this->getLoc(), block, {operandType},
          {RankedTensorType::get({}, operandType.getElementType())},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!operandTypeRanked)))
    return failure();

  return mlir::hlo::verifyReduceScatter(
      *this,
      /*operandTypes=*/{getOperand().getType()},
      /*resultTypes=*/{getType()},
      /*scatterDimension=*/getScatterDimension());
}

void ReduceScatterOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                            Type resultType, Value operand,
                            IntegerAttr scatterDimension,
                            DenseIntElementsAttr replicaGroups,
                            ChannelHandleAttr channelHandle) {
  ReduceScatterOp::build(odsBuilder, odsState, resultType, operand,
                         scatterDimension, replicaGroups, channelHandle,
                         /*use_global_device_ids=*/nullptr);
}

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support quantization or sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                        \
  LogicalResult Op::inferReturnTypeComponents(                                \
      MLIRContext* context, Optional<Location> location,                      \
      ValueShapeRange operands, DictionaryAttr attributes,                    \
      RegionRange regions,                                                    \
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {          \
    return inferReturnTypeComponentsFromOperands(context, location, operands, \
                                                 attributes, regions,         \
                                                 inferredReturnShapes);       \
  }

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AddOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AllReduceOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AndOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Atan2Op)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CbrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CeilOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ClzOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CollectivePermuteOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CopyOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CosineOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CrossReplicaSumOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DivOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DomainOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ExpOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Expm1Op)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(FloorOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LogOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Log1pOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LogisticOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MaxOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MulOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NegOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NotOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(OrOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PopulationCountOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PowOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReducePrecisionOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RemOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReverseOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RoundNearestEvenOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RoundOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RsqrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftLeftOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightArithmeticOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightLogicalOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SignOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SineOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SqrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SubtractOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(XorOp)

//===----------------------------------------------------------------------===//
// Async ops
//===----------------------------------------------------------------------===//

Type maybeTupleFromTypes(MLIRContext* ctx, ArrayRef<Type> types) {
  if (types.size() == 1) return types[0];
  return TupleType::get(ctx, TypeRange(types));
}

LogicalResult AsyncStartOp::verify() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  func::FuncOp callee =
      module.lookupSymbol<func::FuncOp>(getCalledComputation());
  if (!callee) {
    return emitOpError() << "can't find function: " << getCalledComputation();
  }
  FunctionType calleeType = callee.getFunctionType();
  auto calleeInputTypes = calleeType.getInputs();
  auto calleeResultTypes = calleeType.getResults();

  auto calleeThreadName = callee->getAttrOfType<StringAttr>("execution_thread");
  if (!calleeThreadName)
    return emitOpError() << "callee must have execution_thread attribute.";
  if (calleeThreadName != getExecutionThread()) {
    return emitOpError()
           << "execution_thread does not match the execution_thread of "
           << getCalledComputation() << ".  Got: \"" << getExecutionThread()
           << "\", but expected " << calleeThreadName << ".";
  }

  if (calleeType.getNumInputs() != getOperands().size()) {
    return emitOpError() << "number of operands doesn't match operands for "
                         << getCalledComputation()
                         << ". Got: " << getOperands().size()
                         << ", but expected: " << calleeType.getNumInputs()
                         << ".";
  }
  for (int i = 0; i < static_cast<int64_t>(getOperands().size()); ++i) {
    if (calleeType.getInput(i) != getOperandTypes()[i]) {
      return emitOpError() << "type mismatch on argument #" << i << " of "
                           << getCalledComputation()
                           << ". Got: " << getOperandTypes()[i]
                           << ", but expected: " << calleeType.getInput(i)
                           << ".";
    }
  }

  auto resultTypes = getResult().getType().cast<AsyncBundleType>().getTypes();
  if (resultTypes.size() < 2)
    return emitOpError() << "result is expected to be a bundle of at least 2 "
                            "components, but got "
                         << resultTypes.size();
  if (resultTypes[0] != maybeTupleFromTypes(getContext(), calleeInputTypes)) {
    return emitOpError()
           << "component #0 of return type doesn't match callee input types";
  }
  if (resultTypes[1] != maybeTupleFromTypes(getContext(), calleeResultTypes)) {
    return emitOpError()
           << "component #1 of return type doesn't match callee result types";
  }

  return success();
}

LogicalResult AsyncUpdateOp::verify() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  func::FuncOp callee =
      module.lookupSymbol<func::FuncOp>(getCalledComputation());
  if (!callee) {
    return emitOpError() << "can't find function: " << getCalledComputation();
  }
  FunctionType calleeType = callee.getFunctionType();
  auto calleeInputTypes = calleeType.getInputs();
  auto calleeResultTypes = calleeType.getResults();
  auto bundleTypes = getBundle().getType().cast<AsyncBundleType>().getTypes();

  auto calleeThreadName = callee->getAttrOfType<StringAttr>("execution_thread");
  if (!calleeThreadName)
    return emitOpError() << "callee must have execution_thread attribute.";
  if (calleeThreadName != getExecutionThread()) {
    return emitOpError() << "execution_thread does not match name of "
                         << getCalledComputation() << ".  Got: \""
                         << getExecutionThread() << "\", but expected "
                         << calleeThreadName << ".";
  }

  if (bundleTypes.size() < 2)
    return emitOpError() << "operand is expected to be a bundle of at least 2 "
                            "components, but got "
                         << bundleTypes.size();
  if (bundleTypes[0] != maybeTupleFromTypes(getContext(), calleeInputTypes)) {
    return emitOpError() << "component #0 of operand bundle type doesn't match "
                            "callee input types";
  }
  if (bundleTypes[1] != maybeTupleFromTypes(getContext(), calleeResultTypes)) {
    return emitOpError() << "component #1 of operand bundle type doesn't match "
                            "callee result types";
  }

  return success();
}

LogicalResult AsyncUpdateOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  AsyncUpdateOp::Adaptor adaptor(operands, attributes, regions);
  auto stateType = adaptor.getBundle().getType().cast<AsyncBundleType>();
  inferredReturnTypes.push_back(stateType);
  return success();
}

LogicalResult AsyncDoneOp::verify() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  func::FuncOp callee =
      module.lookupSymbol<func::FuncOp>(getCalledComputation());
  if (!callee) {
    return emitOpError() << "can't find function: " << getCalledComputation();
  }
  FunctionType calleeType = callee.getFunctionType();
  auto calleeInputTypes = calleeType.getInputs();
  auto calleeResultTypes = calleeType.getResults();
  auto bundleTypes = getBundle().getType().cast<AsyncBundleType>().getTypes();

  auto calleeThreadName = callee->getAttrOfType<StringAttr>("execution_thread");
  if (!calleeThreadName)
    return emitOpError() << "callee must have execution_thread attribute.";
  if (calleeThreadName != getExecutionThread()) {
    return emitOpError() << "execution_thread does not match name of "
                         << getCalledComputation() << ".  Got: \""
                         << getExecutionThread() << "\", but expected "
                         << calleeThreadName << ".";
  }

  if (bundleTypes.size() < 2)
    return emitOpError() << "operand is expected to be a bundle of at least 2 "
                            "components, but got "
                         << bundleTypes.size();
  if (bundleTypes[0] != maybeTupleFromTypes(getContext(), calleeInputTypes)) {
    return emitOpError()
           << "operand type component #0 doesn't match callee input types";
  }
  if (bundleTypes[1] != maybeTupleFromTypes(getContext(), calleeResultTypes)) {
    return emitOpError()
           << "operand type component #1 doesn't match callee result types";
  }

  return success();
}

LogicalResult AsyncDoneOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  AsyncDoneOp::Adaptor adaptor(operands, attributes, regions);
  ModuleOp module =
      adaptor.getBundle().getDefiningOp()->getParentOfType<ModuleOp>();
  auto calledComputation = adaptor.getCalledComputationAttr();
  func::FuncOp callee = module.lookupSymbol<func::FuncOp>(calledComputation);
  if (!callee) {
    return adaptor.getBundle().getDefiningOp()->emitOpError()
           << "can't find function: " << calledComputation;
  }
  llvm::append_range(inferredReturnTypes,
                     callee.getFunctionType().getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

// Builds a constant op with the specified attribute `value`.
void ConstantOp::build(OpBuilder& /*builder*/, OperationState& result,
                       Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<ElementsAttr>()) {
    type = elemAttr.getType();
  } else if (value.isa<BoolAttr, FloatAttr, IntegerAttr>()) {
    // All XLA types must be tensor types. In the build() method, we want to
    // provide more flexibility by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid XLA constants.
    type =
        RankedTensorType::get(/*shape=*/{}, value.cast<TypedAttr>().getType());
    value = DenseElementsAttr::get(type.cast<TensorType>(), value);
  } else if (auto complexAttr = value.dyn_cast<complex::NumberAttr>()) {
    type = RankedTensorType::get(/*shape=*/{},
                                 complexAttr.cast<TypedAttr>().getType());
    value =
        DenseElementsAttr::get(type.cast<TensorType>(), complexAttr.getValue());
  }

  // TODO: support other XLA specific types.
  assert(type && "unsupported attribute type for building mhlo.constant");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes);
  Type type = adaptor.getValue().getType();
  inferredReturnTypes.push_back(type);
  return success();
}

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1) return false;
  auto lhsTy = l.front().cast<TensorType>();
  auto rhsTy = r.front().cast<TensorType>();
  // For comparisons of the uniform quantized element based tensor type, use the
  // storage type since the constant value will be stored through the underlying
  // storage type.
  if (auto rhsElemTy =
          rhsTy.getElementType().dyn_cast<quant::QuantizedType>()) {
    rhsTy = hlo::getSameShapeTensorType(rhsTy, rhsElemTy.getStorageType());
  }
  return lhsTy == rhsTy;
}

ParseResult ConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse the generic form.
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseRParen()) return failure();
    if (parser.parseOptionalAttrDict(result.attributes)) return failure();
    if (parser.parseColon() || parser.parseLParen() || parser.parseRParen() ||
        parser.parseArrow())
      return failure();
    Type resultTy;
    if (parser.parseType(resultTy)) {
      return failure();
    }
    result.addTypes(resultTy);
    return success();
  }

  ElementsAttr valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  if (parser.parseCustomAttributeWithFallback(valueAttr, Type{}, "value",
                                              result.attributes)) {
    return failure();
  }
  result.addTypes(valueAttr.getType());
  return success();
}

/// Print a `constant` op.
///
/// op ::= attr-dict $value
///
/// When the `value` and `output` have different type, it just uses the default
/// operator assembly format as a fallback.
void ConstantOp::print(::mlir::OpAsmPrinter& p) {
  // If not all types are the same, use generic form.
  if (getValue().getType() != getType()) {
    p.printGenericOp(getOperation(), /*printOpName=*/false);
    return;
  }

  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << ' ';
  p.printStrippedAttrOrType(getValueAttr());
}

//===----------------------------------------------------------------------===//
// CustomCallOp
//===----------------------------------------------------------------------===//

void CustomCallOp::build(
    ::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
    ::mlir::TypeRange resultType, ::mlir::ValueRange operands,
    ::mlir::StringAttr callTargetName, ::mlir::BoolAttr hasSideEffect,
    ::mlir::StringAttr backendConfig,
    ::mlir::mhlo::CustomCallApiVersionAttr apiVersion,
    ::mlir::ArrayAttr calledComputations, ::mlir::ArrayAttr operandLayouts,
    ::mlir::ArrayAttr resultLayouts) {
  return CustomCallOp::build(
      odsBuilder, odsState, resultType, operands, callTargetName, hasSideEffect,
      backendConfig, apiVersion, calledComputations,
      CustomCallScheduleAttr::get(odsBuilder.getContext(),
                                  CustomCallSchedule::NONE),
      operandLayouts, resultLayouts, nullptr);
}

LogicalResult CustomCallOp::verify() {
  // If both operand and result layout attributes are not specified then nothing
  // to verify.
  if (getOperandLayouts().has_value() || getResultLayouts().has_value()) {
    // Layout constraints for either both operands & results or none should be
    // specified.
    if (getOperandLayouts().has_value() != getResultLayouts().has_value())
      return emitOpError() << "Layout attributes should be specified for "
                              "either both operands and results or none.";

    // Helper function to verify types and the corresponding layouts.
    auto verifyTypesAndLayouts =
        [this](TypeRange types, mlir::ArrayAttr layouts,
               const std::string& valueName) -> LogicalResult {
      if (types.size() != layouts.size())
        return emitOpError()
               << "Number of " << valueName << "s must match the number of "
               << valueName << " layouts, " << types.size()
               << " != " << layouts.size();

      for (const auto& indexedTypeAndLayout :
           llvm::enumerate(llvm::zip(types, layouts))) {
        // Get index for more descriptive error message.
        auto index = indexedTypeAndLayout.index();

        auto type = std::get<0>(indexedTypeAndLayout.value());
        auto layout = std::get<1>(indexedTypeAndLayout.value())
                          .cast<DenseIntElementsAttr>();

        if (type.isa<TupleType>())
          return emitOpError() << "Tuple types are not fully supported with "
                                  "layout constraints yet";
        auto tensorType = type.dyn_cast<TensorType>();

        // For non-tensor types such as !mhlo.token, the layout should be empty.
        if (!tensorType) {
          if (layout.empty()) continue;
          return emitOpError()
                 << "Only tensor types can have non-empty layout: " << valueName
                 << " #" << index << " of type " << type << " has layout "
                 << layout;
        }

        // For unranked tensors, we cannot verify the compatibility with layout
        // any further.
        if (!tensorType.hasRank()) continue;

        // Layout must be a permutation of [0, N) where N is the rank of the
        // tensor type.
        std::vector<int64_t> range(tensorType.getRank());
        std::iota(range.begin(), range.end(), 0);
        if (tensorType.getRank() != layout.size() ||
            !std::is_permutation(range.begin(), range.end(), layout.begin()))
          return emitOpError()
                 << "incorrect layout " << layout << " for type " << type
                 << ", layout must be a permutation of [0, "
                 << tensorType.getRank() << ")";
      }
      return success();
    };

    // At this point both `operand_layouts` and `result_layouts` are defined.
    ArrayAttr operandLayouts = this->getOperandLayouts().value();
    ArrayAttr resultLayouts = this->getResultLayouts().value();

    // Full support for layouts for arbitrary nesting of tuples is not
    // supported yet.
    //
    // If result does not have any tuples, then i-th element of `result_layouts`
    // specifies the layout constraints on i-th result.
    //
    // For the common case of a single tuple result packing non-tuple values,
    // the i-th element of `result_layouts` specifies layout for i-th element of
    // the result tuple.
    TypeRange resultTypes;
    if (getNumResults() == 1 && getResult(0).getType().isa<TupleType>())
      resultTypes = getResult(0).getType().cast<TupleType>().getTypes();
    else
      resultTypes = getResultTypes();

    // Verify that operands and operand layouts match.
    if (failed(verifyTypesAndLayouts(getOperandTypes(), operandLayouts,
                                     "operand")))
      return failure();

    // Verify that results and result layouts match.
    if (failed(verifyTypesAndLayouts(resultTypes, resultLayouts, "result")))
      return failure();
  }

  // Check output_operand_aliases

  auto aliasArrayAttr = getOutputOperandAliases();
  for (auto attr : aliasArrayAttr) {
    auto alias = attr.cast<OutputOperandAliasAttr>();
    auto outputTupleIndices = alias.getOutputTupleIndices();
    auto operandIndex = alias.getOperandIndex();
    auto operandTupleIndices = alias.getOperandTupleIndices();

    if (operandIndex < 0 ||
        operandIndex >= static_cast<int64_t>(getInputs().size()))
      return emitOpError()
             << "expects operandIndex in the output_operand_alias attribute "
                "to be in range [0, "
             << getInputs().size() << "); got: " << operandIndex << ".";

    Type operandPart = getOperand(operandIndex).getType();
    for (auto i : operandTupleIndices) {
      if (!operandPart.isa<TupleType>() ||
          i >= static_cast<int64_t>(operandPart.cast<TupleType>().size()) ||
          i < 0)
        return emitOpError()
               << "operand_tuple_indices in the output_operand_alias "
                  "attribute out of bounds";
      operandPart = operandPart.cast<TupleType>().getType(i);
    }
    Type outputPart = getNumResults() > 1
                          ? TupleType::get(getContext(), getResultTypes())
                          : getResult(0).getType();
    for (auto i : outputTupleIndices) {
      if (!outputPart.isa<TupleType>() ||
          i >= static_cast<int64_t>(outputPart.cast<TupleType>().size()) ||
          i < 0)
        return emitOpError()
               << "output_tuple_indices in the output_operand_alias "
                  "attribute out of bounds";
      outputPart = outputPart.cast<TupleType>().getType(i);
    }
    if (operandPart != outputPart)
      return emitOpError()
             << "shapes mismatch in the output_operand_alias attribute: "
             << "operand part has type " << operandPart
             << " and output part has type " << outputPart;
  }

  // Check backend_config attribute.
  if (auto backendConfig = getBackendConfig()) {
    if (getApiVersion() == CustomCallApiVersion::API_VERSION_TYPED_FFI) {
      // Typed FFI custom calls require `backend_config` to be a DictionaryAttr.
      if (backendConfig->isa<mlir::StringAttr>())
        return emitOpError()
               << "unsupported user-encoded backend config,"
                  " backend config must be a dictionary attribute.";
    } else {
      // Older API versions require user-encoded `backend_config` string.
      if (backendConfig->isa<mlir::DictionaryAttr>())
        return emitOpError()
               << "unsupported dictionary attribute backend config, backend"
                  " config must be a user-encoded string attribute.";
    }
  }

  return success();
}

void CustomCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // CustomCall has "all possible effects" unless the has_side_effect is present
  // and set to false.
  auto hasSideEffect = (*this)->getAttrOfType<BoolAttr>("has_side_effect");
  if (hasSideEffect && !hasSideEffect.getValue()) return;
  effects.emplace_back(MemoryEffects::Allocate::get());
  effects.emplace_back(MemoryEffects::Free::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get());
}

//===----------------------------------------------------------------------===//
// CholeskyOp
//===----------------------------------------------------------------------===//

// The following properties are already enforced by the ODS:
//   P0. a.element_type is floating or complex
// We intend to verify the following properties
//   P1. The 'a' argument to Cholesky must have rank >= 2, got shape %s
//   P2. The two minor dimensions of 'a' must have equal size, got %s.
LogicalResult CholeskyOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  CholeskyOp::Adaptor adaptor(operands, attributes, regions);
  Type aType = adaptor.getA().getType();
  RankedTensorType aRankedType = aType.dyn_cast<RankedTensorType>();
  if (!aRankedType) {
    inferredReturnShapes.emplace_back(
        aType.cast<TensorType>().getElementType());
    return success();
  }

  ArrayRef<int64_t> aShape = aRankedType.getShape();
  if (aShape.size() < 2) {
    return emitOptionalError(
        location, "argument 'a' must have rank >= 2, got shape ", aShape, ".");
  }

  int64_t lastDim = aShape[aShape.size() - 1];
  int64_t penultimateDim = aShape[aShape.size() - 2];
  if (!hlo::isDynamicDimSize(lastDim) &&
      !hlo::isDynamicDimSize(penultimateDim) && lastDim != penultimateDim) {
    return emitOptionalError(
        location, "minor dimensions of 'a' must have equal size, got shape ",
        aShape, ".");
  }
  inferredReturnShapes.emplace_back(aRankedType.getShape(),
                                    aRankedType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// DotOp
//===----------------------------------------------------------------------===//
namespace {
bool dimCompatible(int64_t a, int64_t b) {
  return hlo::isDynamicDimSize(a) || hlo::isDynamicDimSize(b) || a == b;
}

ShapedType inferDotReturnType(ShapedType lhs, ShapedType rhs) {
  auto elementType = lhs.getElementType();
  if (!lhs.hasRank() || !rhs.hasRank()) {
    return UnrankedTensorType::get(elementType);
  }

  // vector dot vector
  if (1 == lhs.getRank() && 1 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(0), rhs.getDimSize(0))) {
    return RankedTensorType::get({}, elementType);
  }
  // matrix dot vector
  if (2 == lhs.getRank() && 1 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(1), rhs.getDimSize(0))) {
    return RankedTensorType::get({lhs.getDimSize(0)}, elementType);
  }
  // vector dot matrix
  if (1 == lhs.getRank() && 2 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(0), rhs.getDimSize(0))) {
    return RankedTensorType::get({rhs.getDimSize(1)}, elementType);
  }
  // matrix dot matrix
  if (2 == lhs.getRank() && 2 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(1), rhs.getDimSize(0))) {
    int64_t shape[2] = {lhs.getDimSize(0), rhs.getDimSize(1)};
    return RankedTensorType::get(shape, elementType);
  }
  return {};
}
}  // namespace

LogicalResult DotOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  DotOp::Adaptor op(operands);
  auto lhsType = op.getLhs().getType().cast<ShapedType>();
  auto rhsType = op.getRhs().getType().cast<ShapedType>();
  inferredReturnTypes.push_back(inferDotReturnType(lhsType, rhsType));
  return success();
}

//===----------------------------------------------------------------------===//
// DotGeneralOp
//===----------------------------------------------------------------------===//

LogicalResult DotGeneralOp::verify() {
  auto dimNumbers = this->getDotDimensionNumbers();

  ArrayRef<int64_t> lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
  ArrayRef<int64_t> rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
  ArrayRef<int64_t> lhsContractingDims =
      dimNumbers.getLhsContractingDimensions();
  ArrayRef<int64_t> rhsContractingDims =
      dimNumbers.getRhsContractingDimensions();

  if (lhsBatchingDims.size() != rhsBatchingDims.size()) {
    return emitOpError() << "lhs and rhs should have the same number of "
                            "batching dimensions";
  }
  if (lhsContractingDims.size() != rhsContractingDims.size()) {
    return emitOpError() << "lhs and rhs should have the same number of "
                            "contracting dimensions";
  }

  llvm::SmallDenseSet<int64_t> dimSet;

  auto checkDimsDistinct =
      [this](ArrayRef<int64_t> batchingDims, ArrayRef<int64_t> contractingDims,
             llvm::SmallDenseSet<int64_t>& dimSet, llvm::StringRef lhs,
             llvm::StringRef rhs) -> LogicalResult {
    auto dims = llvm::concat<const int64_t>(batchingDims, contractingDims);
    for (auto dim : dims) {
      auto [_, wasInserted] = dimSet.insert(dim);
      if (!wasInserted) {
        return emitOpError() << "has duplicated dimension from " << lhs
                             << " and " << rhs << ": " << dim;
      }
    }
    return success();
  };

  if (failed(checkDimsDistinct(lhsBatchingDims, lhsContractingDims, dimSet,
                               "lhs_batching_dimensions",
                               "lhs_contracting_dimensions"))) {
    return failure();
  }
  dimSet.clear();
  if (failed(checkDimsDistinct(rhsBatchingDims, rhsContractingDims, dimSet,
                               "rhs_batching_dimensions",
                               "rhs_contracting_dimensions"))) {
    return failure();
  }

  auto checkDimsInRange = [this](int64_t rank, ArrayRef<int64_t> dims,
                                 llvm::StringRef dimName) -> LogicalResult {
    auto inRange = [&](int64_t i) -> bool { return 0 <= i && i < rank; };
    const auto* dimsNotInRange =
        std::find_if_not(dims.begin(), dims.end(), inRange);
    if (dimsNotInRange != dims.end()) {
      return emitOpError() << dimName << " value: " << *dimsNotInRange
                           << " is out of range: "
                           << "[0, " << rank << ")";
    }
    return success();
  };

  auto lhsType = this->getLhs().getType().dyn_cast<RankedTensorType>();
  auto rhsType = this->getRhs().getType().dyn_cast<RankedTensorType>();

  if (lhsType) {
    if (failed(checkDimsInRange(lhsType.getRank(), lhsBatchingDims,
                                "lhs_batching_dimensions")) ||
        failed(checkDimsInRange(lhsType.getRank(), lhsContractingDims,
                                "lhs_contracting_dimensions"))) {
      return failure();
    }
  }
  if (rhsType) {
    if (failed(checkDimsInRange(rhsType.getRank(), rhsBatchingDims,
                                "rhs_batching_dimensions")) ||
        failed(checkDimsInRange(rhsType.getRank(), rhsContractingDims,
                                "rhs_contracting_dimensions"))) {
      return failure();
    }
  }

  if (lhsType && rhsType) {
    // Dimension sizes must be compatible for lhs/rhs.
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    for (auto [lhs, rhs] : llvm::zip(lhsBatchingDims, rhsBatchingDims)) {
      if (lhsShape[lhs] != rhsShape[rhs]) {
        return emitOpError() << "batching dimension sizes must match for "
                                "lhs/rhs";
      }
    }
    for (auto [lhs, rhs] : llvm::zip(lhsContractingDims, rhsContractingDims)) {
      if (lhsShape[lhs] != rhsShape[rhs]) {
        return emitOpError() << "contracting dimension sizes must match for "
                                "lhs/rhs";
      }
    }
  }
  return success();
}

namespace {
// Handle the generic case of DotGeneral and convert to a regulat DotOp.
struct DotGeneralToDot : public OpRewritePattern<DotGeneralOp> {
  using OpRewritePattern<DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DotGeneralOp dot,
                                PatternRewriter& rewriter) const override {
    auto lhs = dot.getLhs();
    auto rhs = dot.getRhs();
    auto lhsTy = lhs.getType().cast<ShapedType>();
    auto rhsTy = rhs.getType().cast<ShapedType>();

    if (lhsTy.getRank() != 2) return failure();
    if (rhsTy.getRank() != 2) return failure();

    auto nums = dot.getDotDimensionNumbers();
    if (!nums.getLhsBatchingDimensions().empty()) return failure();
    if (!nums.getRhsBatchingDimensions().empty()) return failure();

    auto lhsContract = nums.getLhsContractingDimensions();
    auto rhsContract = nums.getRhsContractingDimensions();
    if (lhsContract.size() != 1 || rhsContract.size() != 1) return failure();

    if (lhsContract.front() != 1) return failure();
    if (rhsContract.front() != 0) return failure();

    rewriter.replaceOpWithNewOp<mhlo::DotOp>(
        dot, dot.getType(), lhs, rhs,
        dot.getPrecisionConfig().value_or(nullptr));

    return success();
  }
};
}  // namespace

void DotGeneralOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                               MLIRContext* context) {
  results.add<DotGeneralToDot>(context);
}

LogicalResult DotGeneralOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto lhsType = getLhs().getType().dyn_cast<ShapedType>();
  auto rhsType = getRhs().getType().dyn_cast<ShapedType>();
  if (!lhsType || !rhsType) {
    return failure();
  }

  Adaptor adaptor(operands);
  auto dimNumbers = getDotDimensionNumbers();
  SmallVector<Value> dimensions;
  for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions()) {
    dimensions.push_back(
        builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), lhsDim));
  }

  for (int64_t i = 0; i < lhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), i));
    }
  }
  for (int64_t i = 0; i < rhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getRhs(), i));
    }
  }

  reifiedReturnShapes.push_back(
      builder.create<tensor::FromElementsOp>(getLoc(), dimensions));
  return success();
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

// We intend to verify the following properties
// P1. 1 <= rank <= 3
// P2. Element types agree with fft_type
// P3. Operand shape dimensions agree with fft_length for the given fft_type
LogicalResult FftOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  FftOp::Adaptor adaptor(operands, attributes, regions);
  auto fftLength = adaptor.getFftLength().getValues<int64_t>();
  int64_t fftRank = fftLength.size();

  // P1.
  if (fftRank > 3 || fftRank < 1) {
    return emitOptionalError(location, "rank must be between 1 and 3, but got ",
                             fftRank, ".");
  }

  // P2. Element type agreement
  // FFT : C -> C
  // IFFT : C -> C
  // RFFT : R -> C
  // IRFFT : C -> R
  auto fftType = adaptor.getFftType();
  auto operandType = adaptor.getOperand().getType().cast<TensorType>();
  Type operandElementType = operandType.getElementType();
  // Check the input element type and infer return element type
  if (fftType == FftType::RFFT) {
    if (!operandElementType.isF32() && !operandElementType.isF64()) {
      return emitOptionalError(
          location, "RFFT requires f32 or f64 input type, but is given ",
          operandElementType, ".");
    }
  } else {
    if (!operandElementType.isa<ComplexType>()) {
      return emitOptionalError(
          location, stringifyFftType(fftType),
          " takes a complex tensor as input, but is given ", operandType, ".");
    }
  }
  // Generate the output element type
  Type resultElementType = operandElementType;
  if (fftType == FftType::RFFT) {  // RFFT : R -> C
    resultElementType = ComplexType::get(resultElementType);
  } else if (fftType == FftType::IRFFT) {  // IRFFT : C -> R
    resultElementType = operandElementType.cast<ComplexType>().getElementType();
  }

  // P3. Check input shape and infer return shape
  operandType = operandType.dyn_cast<RankedTensorType>();
  if (!operandType) {
    inferredReturnShapes.emplace_back(resultElementType);
    return success();
  }
  auto operandShape = operandType.getShape();
  if (static_cast<int64_t>(operandShape.size()) < fftRank) {
    return emitOptionalError(
        location, "operand rank must not be less than fft rank of ", fftRank,
        " for operand of type ", operandType, ".");
  }

  SmallVector<int64_t> resultShape = to_vector(operandShape);

  if (fftType == FftType::RFFT) {
    auto shapeBack = operandShape.take_back(fftRank);
    for (auto [operandDim, fftDim] : llvm::zip(shapeBack, fftLength)) {
      if (operandDim != fftDim) {
        return emitOptionalError(
            location,
            "RFFT requires innermost dimensions match fft_length. Got: ",
            operandShape, " but wanted ", fftLength, ".");
      }
    }
    if (fftLength[fftRank - 1] != 0) {
      resultShape[resultShape.size() - 1] = fftLength[fftRank - 1] / 2 + 1;
    }
  }
  if (fftType == FftType::IRFFT) {
    auto shapeBack = operandShape.take_back(fftRank).drop_back();
    for (auto [operandDim, fftDim] : llvm::zip(shapeBack, fftLength)) {
      if (operandDim != fftDim) {
        return emitOptionalError(location,
                                 "IRFFT requires non-final dimensions "
                                 "match fft_length. Got: ",
                                 operandShape, " but wanted ", fftLength,
                                 ", and ", operandDim, " != ", fftDim, ".");
      }
    }
    if ((operandShape[operandShape.size() - 1] != 0 ||
         fftLength[fftRank - 1] != 0) &&
        operandShape[operandShape.size() - 1] != fftLength[fftRank - 1] / 2 + 1)
      return emitOptionalError(location,
                               "IRFFT requires innermost dimension match "
                               "fft_length[-1]/2+1. Got: ",
                               operandShape, " but fft_length is ", fftLength,
                               ".");
    resultShape[resultShape.size() - 1] = fftLength[fftRank - 1];
  }

  inferredReturnShapes.emplace_back(resultShape, resultElementType);
  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

// Converts gather ops to slice ops in case we have a single set of constant
// indices.
struct GatherSlice : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter& rewriter) const override {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.getStartIndices(), m_Constant(&index)))
      return failure();

    const auto& dnums = gather.getDimensionNumbers();
    if (dnums.getIndexVectorDim() != 0 || index.getType().getRank() > 1)
      return failure();

    // TODO(tberghammer): Remove when the verifier catches this case what is
    // invalid if all previous condition holds.
    if (index.getNumElements() !=
        static_cast<int64_t>(dnums.getStartIndexMap().size()))
      return failure();

    RankedTensorType operandType =
        gather->getOperand(0).getType().dyn_cast<RankedTensorType>();
    if (!operandType || !operandType.hasStaticShape()) return failure();

    auto sliceEnd =
        llvm::to_vector<8>(gather.getSliceSizes().getValues<int64_t>());
    llvm::SmallVector<int64_t, 8> sliceStart(sliceEnd.size(), 0);
    for (auto it :
         llvm::zip(dnums.getStartIndexMap(), index.getValues<APInt>())) {
      int64_t mapIndex = std::get<0>(it);
      // Clamp the indices within bounds to faithfully mirror gather semantics.
      int64_t offset =
          clamp(std::get<1>(it).getSExtValue(), static_cast<int64_t>(0),
                operandType.getDimSize(mapIndex) - sliceEnd[mapIndex]);
      sliceStart[mapIndex] += offset;
      sliceEnd[mapIndex] += offset;
    }

    llvm::SmallVector<int64_t, 8> sliceStride(sliceEnd.size(), 1);
    llvm::SmallVector<int64_t, 8> sliceShape(sliceEnd.size());
    for (size_t i = 0; i < sliceEnd.size(); ++i) {
      sliceShape[i] = sliceEnd[i] - sliceStart[i];
    }
    Type elementType = gather.getType().cast<TensorType>().getElementType();
    auto sliceType = RankedTensorType::get(sliceShape, elementType);
    Value result = rewriter.create<SliceOp>(
        gather.getLoc(), sliceType, gather.getOperand(),
        rewriter.getI64TensorAttr(sliceStart),
        rewriter.getI64TensorAttr(sliceEnd),
        rewriter.getI64TensorAttr(sliceStride));

    auto collapsedSliceDims = dnums.getCollapsedSliceDims();
    if (!collapsedSliceDims.empty()) {
      llvm::SmallVector<int64_t, 8> reshapeShape;
      for (size_t i = 0; i < sliceShape.size(); ++i) {
        if (llvm::count(collapsedSliceDims, i) == 0) {
          reshapeShape.push_back(sliceShape[i]);
        }
      }
      auto reshapeType = RankedTensorType::get(reshapeShape, elementType);
      result = rewriter.create<ReshapeOp>(gather.getLoc(), reshapeType, result);
    }

    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

void GatherOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<GatherSlice>(context);
}

namespace {

// following https://www.tensorflow.org/xla/operation_semantics#gather
// The bounds for the output array along dimension i is computed as follows:
// (1) If i is present in batch_dims (i.e. is equal to batch_dims[k] for some k)
// then we pick
// the corresponding dimension bounds out of start_indices.shape, skipping
// index_vector_dim
// (i.e. pick start_indices.shape.dims[k] if k < index_vector_dim and
// start_indices.shape.dims[k+1] otherwise).
// (2) If i is present in offset_dims (i.e. equal to offset_dims[k] for some k)
// then we pick
// the corresponding bound out of slice_sizes after accounting for
// collapsed_slice_dims
// (i.e. we pick adjusted_slice_sizes[k] where adjusted_slice_sizes is
// slice_sizes with the bounds at indices collapsed_slice_dims removed).

void getSliceSizeValues(GatherOp* gather, OpBuilder& builder, Location loc,
                        ValueRange operands,
                        SmallVectorImpl<Value>& sliceSizes) {
  for (int64_t val : gather->getSliceSizes().getValues<int64_t>()) {
    sliceSizes.push_back(builder.create<arith::ConstantIndexOp>(loc, val));
  }
}

void getSliceSizeValues(DynamicGatherOp* /*dGather*/, OpBuilder& builder,
                        Location loc, ValueRange operands,
                        SmallVectorImpl<Value>& sliceSizeValues) {
  DynamicGatherOp::Adaptor adaptor(operands);
  Value sliceSizes = adaptor.getSliceSizes();
  auto sliceSizesTy = sliceSizes.getType().cast<ShapedType>();
  for (int64_t i = 0; i < sliceSizesTy.getDimSize(0); ++i) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    sliceSizeValues.push_back(
        builder.create<tensor::ExtractOp>(loc, sliceSizes, idx));
  }
}

// Verify the following properties:
//  P1. Verify no repeat in start_index_map.
//  P2. Verify 0 <= start_index_map[i] < rank(operand), for every i.
//  P3. Verify 0 <= index_vector_dim <= rank(start_indices).
//  P4. Verify size(start_index_map) == shape(start_indices)[index_vector_dim].
//  P5. Verify offset_dims is_sorted and no repeated.
//  P6. Verify collapsed_slice_dims is_sorted and no repeated.
//  P7. Verify rank(operand) == size(offset_dims) + size(collapsed_slice_dims).
//  P8. Verify slice_sizes has rank of 1.
//  P9. Verify size(slice_sizes) == rank(operand).
//  P10. Verify 0 <= collapsed_slice_dims[i] < size(slice_sizes) for all items.
static LogicalResult verifyGather(
    ShapeAdaptor operandShape, ShapeAdaptor startIndicesShape,
    ShapeAdaptor sliceSizesShape, GatherDimensionNumbersAttr dimensionNumbers,
    llvm::function_ref<InFlightDiagnostic()> errorEmitter) {
  int64_t indexVectorDim = dimensionNumbers.getIndexVectorDim();

  // Check startIndexMap
  auto startIndexMap = to_vector(dimensionNumbers.getStartIndexMap());
  // P1.
  if (hasDuplicates(startIndexMap))
    return errorEmitter() << "expects start_index_map to not repeat, got: ["
                          << startIndexMap << "]";

  // P2.
  for (auto i : llvm::seq<int64_t>(0, startIndexMap.size()))
    if (startIndexMap[i] < 0 ||
        (operandShape.hasRank() && startIndexMap[i] >= operandShape.getRank()))
      return errorEmitter()
             << "start_index_map[" << i << "]: " << startIndexMap[i]
             << " is out of bounds for "
             << "operand rank " << operandShape.getRank();

  if (startIndicesShape.hasRank()) {
    // P3.
    // index_vector_dim == start_indices.rank implies a trailing 1 on the shape
    // of start_indices.
    if (indexVectorDim > startIndicesShape.getRank() || indexVectorDim < 0)
      return errorEmitter() << "index_vector_dim " << indexVectorDim
                            << " is out of bounds for start indices with rank "
                            << startIndicesShape.getRank();

    bool impliedTrailingDim = indexVectorDim == startIndicesShape.getRank();
    if (impliedTrailingDim || !startIndicesShape.isDynamicDim(indexVectorDim)) {
      int64_t effectiveDimSize;
      if (impliedTrailingDim)
        effectiveDimSize = 1;
      else
        effectiveDimSize = startIndicesShape.getDimSize(indexVectorDim);
      // P4.
      if (effectiveDimSize !=
          static_cast<int64_t>(dimensionNumbers.getStartIndexMap().size()))
        return errorEmitter() << "start_index_map size ("
                              << dimensionNumbers.getStartIndexMap().size()
                              << ") is not equal to size of index dimension ("
                              << indexVectorDim << ") of start_indices ("
                              << effectiveDimSize << ")";
    }
  }

  // P5.
  auto offsetDims = to_vector(dimensionNumbers.getOffsetDims());
  if (!llvm::is_sorted(offsetDims))
    return errorEmitter() << "expects offset_dims to be sorted, got: ["
                          << offsetDims << "]";
  if (hasDuplicates(offsetDims))
    return errorEmitter() << "expects offset_dims to not repeat, got: ["
                          << offsetDims << "]";

  // P6.
  auto collapsedSliceDims = to_vector(dimensionNumbers.getCollapsedSliceDims());
  if (!llvm::is_sorted(collapsedSliceDims))
    return errorEmitter() << "expects collapsed_slice_dims to be sorted, got: ["
                          << collapsedSliceDims << "]";
  if (hasDuplicates(collapsedSliceDims))
    return errorEmitter()
           << "expects collapsed_slice_dims to not repeat, got: ["
           << collapsedSliceDims << "]";

  // P7.
  int64_t impliedOperandRank = dimensionNumbers.getOffsetDims().size() +
                               dimensionNumbers.getCollapsedSliceDims().size();
  if (operandShape.hasRank() && operandShape.getRank() != impliedOperandRank)
    return errorEmitter() << "offset_dims size ("
                          << dimensionNumbers.getOffsetDims().size()
                          << ") plus collapse_slice_dims size ("
                          << dimensionNumbers.getCollapsedSliceDims().size()
                          << ") is not equal to operand rank ("
                          << operandShape.getRank() << ")";

  // P8.
  // This should be fully expressible with type constraints, but it isn't
  // obvious how to do that with the current infrastructure.
  if (sliceSizesShape.hasRank() && sliceSizesShape.getRank() != 1)
    return errorEmitter() << "slice_sizes.rank != 1";
  if (sliceSizesShape.hasStaticShape()) {
    int64_t sliceSize = sliceSizesShape.getNumElements();

    // P9.
    if (sliceSize != impliedOperandRank)
      return errorEmitter() << "slice_sizes size (" << sliceSize
                            << ") not equal to (implied) operand rank ("
                            << impliedOperandRank << ")";

    // P10.
    for (auto dim : dimensionNumbers.getCollapsedSliceDims())
      if (dim < 0 || dim >= sliceSize)
        return errorEmitter() << "collapsed dimension " << dim
                              << " is out of bounds for slice_sizes.size ("
                              << sliceSize << ")";
  }

  return success();
}

// Verify the following properties:
//  P1. Verifications by verifyGather().
//  P2. Verify slice_sizes[i] <= 1 for i in collapsed_slice_dims.
//  P3. Verify 0 <= slice_sizes[i] < shape(operand)[i], for every i.
static LogicalResult verifyStaticGather(
    ShapeAdaptor operandShape, ShapeAdaptor startIndicesShape,
    DenseIntElementsAttr sliceSizes,
    GatherDimensionNumbersAttr dimensionNumbers,
    llvm::function_ref<InFlightDiagnostic()> errorEmitter) {
  // P1.
  // For some reason the getType call is necessary here
  if (failed(verifyGather(
          /*operandShape=*/operandShape,
          /*startIndicesShape=*/startIndicesShape,
          /*sliceSizesShape=*/sliceSizes.getType(), dimensionNumbers,
          errorEmitter)))
    return failure();

  // P2.
  for (auto dim : dimensionNumbers.getCollapsedSliceDims()) {
    int64_t sliceDimSize = sliceSizes.getValues<int64_t>()[dim];
    if (sliceDimSize > 1) {
      return errorEmitter() << "slice_sizes collapsed dimension " << dim
                            << " should <= 1 but got " << sliceDimSize;
    }
  }

  // P3.
  if (operandShape.hasRank()) {
    for (const auto& it : llvm::enumerate(sliceSizes.getValues<int64_t>())) {
      if (operandShape.isDynamicDim(it.index())) continue;
      auto operandDimSize = operandShape.getDimSize(it.index());
      auto sliceDimSize = it.value();
      if (sliceDimSize < 0 || sliceDimSize > operandDimSize)
        return errorEmitter() << "slice size (" << sliceDimSize
                              << ") is out of bounds for operand dimension ("
                              << operandDimSize << ") at index " << it.index();
    }
  }
  return success();
}

template <typename dimTy>
static void inferGatherShape(
    int64_t resultRank, llvm::function_ref<dimTy(int64_t)> getStartIndicesDim,
    llvm::function_ref<dimTy(int64_t)> getSliceDim,
    GatherDimensionNumbersAttr dimensionNumbers,
    SmallVectorImpl<dimTy>& shape) {
  ArrayRef<int64_t> collapsedSliceDims =
      dimensionNumbers.getCollapsedSliceDims();
  int64_t indexVectorDim = dimensionNumbers.getIndexVectorDim();

  // We don't necessarily know the rank of sliceSizes, but we do know that it
  // can't be larger than the highest collapsed dimension. So go through those
  // and populate the leading dimensions of adjustedSliceSizes. The trailing
  // dimensions can just be adjusted by an offset.
  const auto* maxCollapsedDimIt =
      std::max_element(collapsedSliceDims.begin(), collapsedSliceDims.end());
  int64_t maxCollapsedDim = -1;
  if (maxCollapsedDimIt != collapsedSliceDims.end())
    maxCollapsedDim = *maxCollapsedDimIt;

  SmallVector<dimTy> adjustedSliceSizePrefix;
  for (int dimIndex = 0; dimIndex <= maxCollapsedDim; ++dimIndex) {
    if (llvm::is_contained(collapsedSliceDims, dimIndex)) continue;
    adjustedSliceSizePrefix.push_back(getSliceDim(dimIndex));
  }
  auto getAdjustedSliceDim = [&](int64_t index) -> dimTy {
    if (index < static_cast<int64_t>(adjustedSliceSizePrefix.size()))
      return adjustedSliceSizePrefix[index];
    return getSliceDim(index + collapsedSliceDims.size());
  };

  ArrayRef<int64_t> offsetDims = dimensionNumbers.getOffsetDims();

  // Dimensions in the output that aren't offset dimensions are called batch
  // dimensions.
  SmallVector<int64_t> batchDims;
  for (int dim = 0; dim < resultRank; ++dim)
    if (!llvm::is_contained(offsetDims, dim)) batchDims.push_back(dim);

  for (int i = 0; i < resultRank; ++i) {
    const auto* offsetDimsIt =
        std::find(offsetDims.begin(), offsetDims.end(), i);
    if (offsetDimsIt != offsetDims.end()) {
      auto index = std::distance(offsetDims.begin(), offsetDimsIt);
      shape.push_back(getAdjustedSliceDim(index));
      continue;
    }
    auto* batchDimsIt = std::find(batchDims.begin(), batchDims.end(), i);
    assert(batchDimsIt != batchDims.end());
    auto index = std::distance(batchDims.begin(), batchDimsIt);
    // This can never run into the special case where start_indices gets
    // implicitly expanded with a trailing 1 if
    // index_vector_dim = start_indices.rank because then index would equal
    // index_vector_dim, which means we'd be looking at index+1, which would be
    // out of bounds anyway.
    if (index >= indexVectorDim) ++index;
    shape.push_back(getStartIndicesDim(index));
  }
}

// Verify the following properties:
//  P1. Verify 0 <= offset_dims[i] < output_shape_rank, for every i.
//      (output_shape_rank = size(offset_dims) + rank(start_indices) -1)
static LogicalResult inferGatherReturnTypeComponents(
    ShapeAdaptor operandShape, ShapeAdaptor startIndicesShape,
    llvm::function_ref<int64_t(int64_t)> getSliceDim,
    GatherDimensionNumbersAttr dimensionNumbers,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes,
    llvm::function_ref<InFlightDiagnostic()> errorEmitter) {
  Type elementType = operandShape.getElementType();

  // We need this to determine the result rank. We could still place bounds on
  // the result rank if that was something ShapedTypeComponents could express.
  if (!startIndicesShape.hasRank()) {
    inferredReturnShapes.push_back(elementType);
    return success();
  }

  ArrayRef<int64_t> offsetDims = dimensionNumbers.getOffsetDims();
  int64_t startIndicesRank = startIndicesShape.getRank();
  // If index_vector_dim == start_indices.rank, then an implicit trailing 1 is
  // appended to start_indices shape.
  if (dimensionNumbers.getIndexVectorDim() == startIndicesRank)
    ++startIndicesRank;
  int64_t resultRank = offsetDims.size() + startIndicesRank - 1;
  // P1.
  for (auto i : llvm::seq<int64_t>(0, offsetDims.size()))
    if (offsetDims[i] < 0 || offsetDims[i] >= resultRank)
      return errorEmitter() << "offset_dims[" << i << "]: " << offsetDims[i]
                            << " is out of bounds for "
                            << "implied result rank " << resultRank;

  auto getStartIndicesDim = [&](int64_t index) {
    return startIndicesShape.getDimSize(index);
  };

  SmallVector<int64_t> shape;
  inferGatherShape<int64_t>(resultRank, getStartIndicesDim, getSliceDim,
                            dimensionNumbers, shape);

  inferredReturnShapes.emplace_back(shape, elementType);
  return success();
}

template <typename Op>
LogicalResult reifyGatherShape(Op* op, OpBuilder& builder, ValueRange operands,
                               SmallVectorImpl<Value>& reifiedReturnShapes) {
  // No support for unranked gather output shape a.t.m.
  auto resultTy =
      op->getResult().getType().template dyn_cast<RankedTensorType>();
  if (!resultTy) return failure();

  typename Op::Adaptor adaptor(operands);
  Value startIndices = adaptor.getStartIndices();

  Location loc = op->getLoc();
  int resultRank = resultTy.getRank();
  Type shapeElTy = builder.getIndexType();
  auto toShapeElType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeElTy);
  };

  SmallVector<Value, 4> sliceSizes;
  getSliceSizeValues(op, builder, loc, operands, sliceSizes);
  llvm::transform(sliceSizes, sliceSizes.begin(),
                  [&](Value v) { return toShapeElType(v); });

  auto getStartIndicesDim = [&](int64_t index) {
    return toShapeElType(
        builder.create<tensor::DimOp>(loc, startIndices, index));
  };
  SmallVector<Value, 4> shapeValues;
  auto getSliceDim = [&sliceSizes](int64_t index) -> Value {
    return sliceSizes[index];
  };
  inferGatherShape<Value>(resultRank, getStartIndicesDim, getSliceDim,
                          op->getDimensionNumbers(), shapeValues);

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get({resultRank}, shapeElTy), shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

}  // namespace

LogicalResult GatherOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return reifyGatherShape(this, builder, operands, reifiedReturnShapes);
}

// The following properties are already enforced by the ODS:
//  P0. Verify the start_indices has element type of integer.
// Verify the following properties:
//  Verifications by verifyStaticGather() and verifyGather() inside it.
//  Verifications by inferGatherReturnTypeComponents.
LogicalResult GatherOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  // TODO(zhouxin) remove this comment after the ordering issue is clear.
  // This can get called before other op verify methods, so we have to do a
  // bunch of verification up front. With a better story for ordering and/or
  // multi-phase op verification, this should hopefully all go away.
  Location loc = location.value_or(UnknownLoc::get(context));
  auto errorEmitter = [&loc]() {
    return mlir::emitError(loc)
           << "'" << GatherOp::getOperationName() << "' op ";
  };
  GatherOp::Adaptor adaptor(operands, attributes, regions);
  if (failed(adaptor.verify(loc))) return failure();

  // We want the ShapeAdaptors, so can't route via the adaptor :-/
  ShapeAdaptor operandShape = operands.getShape(0);
  ShapeAdaptor startIndicesShape = operands.getShape(1);
  GatherDimensionNumbersAttr dimensionNumbers = adaptor.getDimensionNumbers();
  DenseIntElementsAttr sliceSizesAttr = adaptor.getSliceSizes();

  if (failed(verifyStaticGather(/*operandShape=*/operandShape,
                                /*startIndicesShape=*/startIndicesShape,
                                /*sliceSizes=*/sliceSizesAttr, dimensionNumbers,
                                errorEmitter)))
    return failure();

  auto getSliceDim = [&sliceSizesAttr](int64_t index) -> int64_t {
    return sliceSizesAttr.getValues<int64_t>()[index] == kDynamicPrintValue
               ? ShapedType::kDynamic
               : sliceSizesAttr.getValues<int64_t>()[index];
  };

  return inferGatherReturnTypeComponents(operandShape, startIndicesShape,
                                         getSliceDim, dimensionNumbers,
                                         inferredReturnShapes, errorEmitter);
}

//===----------------------------------------------------------------------===//
// DynamicGatherOp
//===----------------------------------------------------------------------===//

// Canonicalize mhlo.dynamic_gather to mhlo.gather when slice_sizes is constant.
LogicalResult simplifyDynamicGatherToGather(DynamicGatherOp op,
                                            PatternRewriter& rewriter) {
  DenseIntElementsAttr sliceSizes;
  if (!matchPattern(op.getSliceSizes(), m_Constant(&sliceSizes))) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::GatherOp>(
      op, op.getOperand(), op.getStartIndices(), op.getDimensionNumbersAttr(),
      sliceSizes, op.getIndicesAreSortedAttr());
  return success();
}

void DynamicGatherOp::getCanonicalizationPatterns(RewritePatternSet& result,
                                                  MLIRContext* context) {
  result.add(simplifyDynamicGatherToGather);
}

LogicalResult DynamicGatherOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return reifyGatherShape(this, builder, operands, reifiedReturnShapes);
}

LogicalResult DynamicGatherOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  // This can get called before other op verify methods, so we have to do a
  // bunch of verification up front. With a better story for ordering and/or
  // multi-phase op verification, this should hopefully all go away.
  Location loc = location.value_or(UnknownLoc::get(context));
  auto errorEmitter = [&loc]() {
    return mlir::emitError(loc)
           << "'" << DynamicGatherOp::getOperationName() << "' op ";
  };
  DynamicGatherOp::Adaptor adaptor(operands, attributes, regions);
  if (failed(adaptor.verify(loc))) return failure();

  // We want the ShapeAdaptors, so can't route via the adaptor :-/
  ShapeAdaptor operandShape = operands.getShape(0);
  ShapeAdaptor startIndicesShape = operands.getShape(1);
  ShapeAdaptor sliceSizesShape = operands.getShape(2);
  GatherDimensionNumbersAttr dimensionNumbers = adaptor.getDimensionNumbers();

  if (failed(verifyGather(/*operandShape=*/operandShape,
                          /*startIndicesShape=*/startIndicesShape,
                          /*sliceSizesShape=*/sliceSizesShape, dimensionNumbers,
                          errorEmitter)))
    return failure();

  auto getSliceDim = [](int64_t index) { return ShapedType::kDynamic; };
  return inferGatherReturnTypeComponents(operandShape, startIndicesShape,
                                         getSliceDim, dimensionNumbers,
                                         inferredReturnShapes, errorEmitter);
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//
//
LogicalResult GetDimensionSizeOp::verify() { return verifyDimAttr(*this); }

/// Fold get_dimension_size when the said shape dimension is a constant.
OpFoldResult GetDimensionSizeOp::fold(ArrayRef<Attribute> attrs) {
  RankedTensorType type = getOperand().getType().dyn_cast<RankedTensorType>();
  if (!type) return {};

  int32_t dim = getDimension();
  if (type.isDynamicDim(dim)) return {};
  // The result type is always is a 0-d i32 tensor.
  return DenseIntElementsAttr::get<int32_t>(
      getResult().getType().cast<RankedTensorType>(), type.getDimSize(dim));
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

LogicalResult IotaOp::verify() {
  auto shape = getType().cast<ShapedType>();
  if (!shape.hasRank()) return success();

  if (shape.getRank() == 0) return emitOpError() << "does not support scalars.";

  auto iotaDimension = static_cast<int64_t>(this->getIotaDimension());
  if (iotaDimension >= shape.getRank() || iotaDimension < 0)
    return emitOpError()
           << "iota dimension cannot go beyond the output rank or be negative.";
  return success();
}

// Iota operations across multiple dimensions can be reduced to an iota and a
// ranked broadcast.
struct IotaBroadcast : public OpRewritePattern<IotaOp> {
  using OpRewritePattern<IotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IotaOp iota,
                                PatternRewriter& rewriter) const override {
    auto resultTy = iota.getType().cast<ShapedType>();
    if (!resultTy.hasRank() || resultTy.getRank() < 2) {
      return failure();
    }

    auto iotaDimension = iota.getIotaDimension();

    auto iotaType = RankedTensorType::get({resultTy.getDimSize(iotaDimension)},
                                          resultTy.getElementType());

    auto newIota = rewriter.create<IotaOp>(iota.getLoc(), iotaType,
                                           rewriter.getI64IntegerAttr(0));

    auto broadcastAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iotaDimension});
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(iota, resultTy, newIota,
                                                  broadcastAttr);
    return success();
  }
};

void IotaOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<IotaBroadcast>(context);
}

OpFoldResult IotaOp::fold(ArrayRef<Attribute> operands) {
  auto dimension = getIotaDimension();
  auto resultTy = getResult().getType().cast<ShapedType>();
  if (resultTy.hasRank() && resultTy.getDimSize(dimension) == 1) {
    Builder builder(getContext());
    return builder.getZeroAttr(resultTy);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// DynamicIotaOp
//===----------------------------------------------------------------------===//

// Does the same as PatternRewriter::replaceOpWithNewOp, but with a twist.
//
// Sometimes, we want to replace an op with a new op and simultaneously refine
// the result type from a dynamically-shaped type to a statically-shaped type.
// (Search for usages of this function for examples).
//
// Oftentimes, this works just fine because MHLO is designed to accommodate
// this kind of type refinements. But sometimes, this doesn't work - when
// the op is used outside of the MHLO dialect (e.g. in func.return). In these
// cases, we insert a tensor.cast to smooth things out.
template <typename OpTy, typename... Args>
OpTy refineOpWithNewOp(PatternRewriter& rewriter, Operation* op,
                       Args&&... args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);

  llvm::SmallVector<Value> replacementResults;
  assert(op->getNumResults() == newOp->getNumResults() &&
         "replacement op doesn't match results of original op");
  for (auto [opResult, newOpResult] :
       llvm::zip(op->getResults(), newOp->getResults())) {
    Value replacementResult = newOpResult;
    if (llvm::any_of(opResult.getUsers(), [&](Operation* user) {
          return user->getDialect() != op->getDialect();
        })) {
      replacementResult = rewriter.create<tensor::CastOp>(
          op->getLoc(), opResult.getType(), newOpResult);
    }
    replacementResults.push_back(replacementResult);
  }

  rewriter.replaceOp(op, replacementResults);
  return newOp;
}

namespace {

struct DynamicIotaIsStatic : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter& rewriter) const override {
    // Result type has static shape, replace with iota.
    auto resultTy = iota.getType().cast<ShapedType>();
    if (resultTy.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<IotaOp>(iota, resultTy,
                                          iota.getIotaDimension());
      return success();
    }

    // Output shape is constant, compute result type with static shape, then
    // replace with iota.
    DenseIntElementsAttr outputShapeAttr;
    if (matchPattern(iota.getOutputShape(), m_Constant(&outputShapeAttr))) {
      SmallVector<int64_t> outputShape;
      for (APInt dim : outputShapeAttr.getValues<APInt>()) {
        outputShape.push_back(dim.getSExtValue());
      }
      resultTy = RankedTensorType::get(outputShape, resultTy.getElementType());
      refineOpWithNewOp<IotaOp>(rewriter, iota, resultTy,
                                iota.getIotaDimension());
      return success();
    }

    return rewriter.notifyMatchFailure(
        iota, "requires static shape or constant output shape");
  }
};

// Dynamic Iota operations across multiple dimensions can be reduced to an iota
// and a ranked broadcast.
struct DynamicIotaBroadcast : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter& rewriter) const override {
    auto resultTy = iota.getType().cast<ShapedType>();
    if (!resultTy.hasRank() || resultTy.getRank() < 2) {
      return failure();
    }

    auto iotaDimension = iota.getIotaDimension();
    auto iotaDimensionInt = iotaDimension;

    auto convertedShape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            iota.getOutputShape().getType().cast<ShapedType>().getShape(),
            rewriter.getI64Type()),
        iota.getOutputShape());

    auto slicedShape = rewriter.create<SliceOp>(
        iota.getLoc(), convertedShape,
        rewriter.getI64TensorAttr(iotaDimensionInt),
        rewriter.getI64TensorAttr(iotaDimensionInt + 1),
        rewriter.getI64TensorAttr(1));

    auto convertedSlicedShape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get({1}, iota.getOutputShape()
                                       .getType()
                                       .cast<ShapedType>()
                                       .getElementType()),
        slicedShape);

    auto iotaType = RankedTensorType::get(
        {resultTy.getDimSize(iotaDimensionInt)}, resultTy.getElementType());

    auto newIota = rewriter.create<DynamicIotaOp>(
        iota.getLoc(), iotaType, convertedSlicedShape,
        rewriter.getI64IntegerAttr(0));

    auto broadcastAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iotaDimension});
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        iota, resultTy, newIota, iota.getOutputShape(), broadcastAttr);
    return success();
  }
};

}  // namespace

void DynamicIotaOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<DynamicIotaIsStatic>(context);
  results.add<DynamicIotaBroadcast>(context);
}

static Value castToIndexTensor(OpBuilder& builder, Location loc,
                               Value shapeOp) {
  ShapedType resultTy = shape::getExtentTensorType(
      builder.getContext(), shapeOp.getType().cast<ShapedType>().getDimSize(0));
  if (shapeOp.getType() == resultTy) return shapeOp;  // Nothing to do.
  return builder.create<arith::IndexCastOp>(loc, resultTy, shapeOp);
}

LogicalResult DynamicIotaOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicIotaOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputShape()));
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicUpdateSliceOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicUpdateSliceOp::verify() {
  OperandRange indices = getStartIndices();
  if (indices.size() <= 1) return success();

  // Note: start_indices is constrained to Variadic<HLO_ScalarIntTensor>, so it
  // is OK to cast indices to ShapedType here.
  auto idxTensor = indices.take_front().front().getType().cast<ShapedType>();
  Type firstElemTy = idxTensor.getElementType();
  Type elemTy;

  for (auto idx : llvm::drop_begin(indices, 1)) {
    idxTensor = idx.getType().cast<ShapedType>();
    elemTy = idxTensor.getElementType();

    if (firstElemTy != elemTy) {
      return emitOpError() << "start indices must have same element type "
                              "(encountered mismatch: "
                           << firstElemTy << " vs " << elemTy << ")";
    }
  }
  return success();
}

OpFoldResult DynamicUpdateSliceOp::fold(ArrayRef<Attribute> operands) {
  auto operandShape = this->getOperand().getType().cast<RankedTensorType>();
  auto updateShape = this->getUpdate().getType().cast<RankedTensorType>();

  // If any of the dimensions are length-0, the update does nothing.
  for (auto dim : updateShape.getShape()) {
    if (dim == 0) {
      return this->getOperand();
    }
  }

  if (operandShape != updateShape || !operandShape.hasStaticShape()) {
    return {};
  }

  // Ensure that indices are 0 constants. The 0 check mostly ensures
  // correctness. For non-constants, the pattern does not fold to avoid hiding
  // the behavior of incorrect user input.
  for (Value index : this->getStartIndices()) {
    DenseIntElementsAttr deAttr;
    if (!matchPattern(index, m_Constant(&deAttr))) return {};
    if (!deAttr.getSplatValue<IntegerAttr>().getValue().isZero()) return {};
  }
  return this->getUpdate();
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto operandTy = (*operands.begin()).getType().cast<ShapedType>();
  Type elementTy = operandTy.getElementType();
  if (auto complexTy = elementTy.dyn_cast<ComplexType>()) {
    elementTy = complexTy.getElementType();
  }

  Type resultTy;
  if (auto rankedOperandTy = operandTy.dyn_cast<RankedTensorType>()) {
    resultTy = RankedTensorType::get(operandTy.getShape(), elementTy,
                                     rankedOperandTy.getEncoding());
  } else if (operandTy.hasRank()) {
    resultTy = RankedTensorType::get(operandTy.getShape(), elementTy);
  } else {
    resultTy = UnrankedTensorType::get(elementTy);
  }
  inferredReturnTypes.push_back(resultTy);
  return success();
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

LogicalResult CollectivePermuteOp::verify() {
  return mlir::hlo::verifyCollectivePermuteSourceTargetPairs(
      *this, getSourceTargetPairs());
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

namespace {
// Checks:
//  P1. Same sizes for input, kernel and output spatial_dims.
//  P2. Spatial and non-spatial dimentions (for input,kernel, &output) should
//      be unique and in range [0, num_dims), where num_dims = rank of input
//      (lhs/rhs) tensors.
//
//  Note that the spatial + non-spatial dimensions may not cover all the
//  dimensions in the range [0,num) because of the presence of 'unknown'
//  dimensions (ref. cl/415132294).
LogicalResult isSpatialDimensionsValid(ConvolutionOp op) {
  auto inputSpatialDimensions =
      op.getDimensionNumbers().getInputSpatialDimensions();
  auto kernelSpatialDimensions =
      op.getDimensionNumbers().getKernelSpatialDimensions();
  auto outputSpatialDimensions =
      op.getDimensionNumbers().getOutputSpatialDimensions();

  // P1.
  if ((inputSpatialDimensions.size() != kernelSpatialDimensions.size()) ||
      (inputSpatialDimensions.size() != outputSpatialDimensions.size()))
    return op.emitOpError() << "expects the same size for input, kernel and "
                               "output spatial-dimensions, but got "
                            << inputSpatialDimensions.size() << ", "
                            << kernelSpatialDimensions.size() << ", and "
                            << outputSpatialDimensions.size() << " resp.";

  // P2.
  SmallVector<int64_t> inputDnums(inputSpatialDimensions.size() + 2);
  inputDnums[0] = op.getDimensionNumbers().getInputBatchDimension();
  inputDnums[1] = op.getDimensionNumbers().getInputFeatureDimension();
  std::copy(inputSpatialDimensions.begin(), inputSpatialDimensions.end(),
            inputDnums.begin() + 2);

  SmallVector<int64_t> windowDnums(kernelSpatialDimensions.size() + 2);
  windowDnums[0] = op.getDimensionNumbers().getKernelInputFeatureDimension();
  windowDnums[1] = op.getDimensionNumbers().getKernelOutputFeatureDimension();
  std::copy(kernelSpatialDimensions.begin(), kernelSpatialDimensions.end(),
            windowDnums.begin() + 2);

  SmallVector<int64_t> outputDnums(outputSpatialDimensions.size() + 2);
  outputDnums[0] = op.getDimensionNumbers().getOutputBatchDimension();
  outputDnums[1] = op.getDimensionNumbers().getOutputFeatureDimension();
  std::copy(outputSpatialDimensions.begin(), outputSpatialDimensions.end(),
            outputDnums.begin() + 2);

  auto numDims = op.getLhs().getType().cast<RankedTensorType>().getRank();
  const auto inRange = [numDims](int64_t i) { return 0 <= i && i < numDims; };

  if (!llvm::all_of(inputDnums, inRange) ||
      !llvm::all_of(windowDnums, inRange) ||
      !llvm::all_of(outputDnums, inRange))
    return op.emitOpError() << "expects input, kernel, and output "
                               "dimension-numbers to be in-range [0, "
                            << numDims << ").";

  if (hasDuplicates(inputDnums))
    return op.emitOpError()
           << "expects input dimension-numbers to be unique, got {"
           << inputDnums << "}.";

  if (hasDuplicates(windowDnums))
    return op.emitOpError()
           << "expects kernel dimension-numbers to be unique, got {"
           << windowDnums << "}.";

  if (hasDuplicates(outputDnums))
    return op.emitOpError()
           << "expects output dimension-numbers to be unique, got {"
           << outputDnums << "}.";

  return success();
}

// Verifies the following properties:
//  P1. The input, kernel, and output spatial-dimentions are valid.
//  P2. Given,
//          input-dimensions: b * input-spatial-dims * f
//          kernel-dimensions: kernel-spatial-dims * i * o
//          output-dimensions: b' * out-spatial-dims * f'
//            where b = input-batch-dims
//            where f = input-feature-dims
//            where i = kernel-input-feature-dims
//            where o = kernel-output-feature-dims
//            where b' = output-batch-dims
//            where f' = output-feature-dims
//      Check the following properties w.r.t feature_group_count (fgc) and
//      batch_group_count (bgc).
//        fgc > 0, bgc > 1 and !(fgc > 1 && bgc > 1)
//        b % bgc == 0
//        f % fgc == 0 and i = f / fgc
//        o (or f') % bgc == 0 and o (or f') % fgc == 0
LogicalResult verifyConvolutionAttributes(ConvolutionOp op) {
  // P1.
  if (failed(isSpatialDimensionsValid(op))) return failure();

  // P2.
  const int64_t featureGroupCount = op.getFeatureGroupCount();
  const int64_t batchGroupCount = op.getBatchGroupCount();

  if (featureGroupCount <= 0)
    return op.emitOpError()
           << "expects feature_group_count to be a positive number, got "
           << featureGroupCount << ".";

  if (batchGroupCount <= 0)
    return op.emitOpError()
           << "expects batch_group_count to be a positive number, got "
           << batchGroupCount << ".";

  if (batchGroupCount > 1 && featureGroupCount > 1)
    return op.emitOpError()
           << "expects batch_group_count and feature_group_count not to be "
              "both greater than 1. Got "
           << batchGroupCount << " and " << featureGroupCount << " resp.";

  auto lhsType = op.getLhs().getType().cast<RankedTensorType>();
  const int64_t inputFeatures =
      lhsType.getShape()[op.getDimensionNumbers().getInputFeatureDimension()];
  const int64_t inputBatch =
      lhsType.getShape()[op.getDimensionNumbers().getInputBatchDimension()];

  auto rhsType = op.getRhs().getType().cast<RankedTensorType>();
  const int64_t kernelInputFeatures =
      rhsType.getShape()[op.getDimensionNumbers()
                             .getKernelInputFeatureDimension()];
  const int64_t kernelOutputFeatures =
      rhsType.getShape()[op.getDimensionNumbers()
                             .getKernelOutputFeatureDimension()];

  if (!hlo::isDynamicDimSize(kernelOutputFeatures)) {
    if (kernelOutputFeatures % batchGroupCount != 0)
      return op.emitOpError() << "expects output feature dimension size ("
                              << kernelOutputFeatures
                              << ") to be a multiple of "
                                 "batch_group_count. Got batch_group_count = "
                              << batchGroupCount << ".";

    if (kernelOutputFeatures % featureGroupCount != 0)
      return op.emitOpError()
             << "expects kernel output feature dimension ("
             << kernelOutputFeatures
             << ") to be divisible by "
                "feature_group_count. For feature_group_count = "
             << featureGroupCount << ".";
  }

  if (!hlo::isDynamicDimSize(inputFeatures)) {
    if (inputFeatures % featureGroupCount != 0)
      return op.emitOpError()
             << "expects input feature dimension (" << inputFeatures
             << ") to be a multiple of "
                "feature_group_count. Got feature_group_count = "
             << featureGroupCount << ".";

    if (!hlo::isDynamicDimSize(kernelInputFeatures) &&
        inputFeatures / featureGroupCount != kernelInputFeatures)
      return op.emitOpError()
             << "expects input feature dimension (" << inputFeatures
             << ") / "
                "feature_group_count = kernel input feature dimension ("
             << kernelInputFeatures
             << "). Got feature_group_count = " << featureGroupCount << ".";
  }

  if (!hlo::isDynamicDimSize(inputBatch) && inputBatch % batchGroupCount != 0)
    return op.emitOpError() << "expects input batch dimension (" << inputBatch
                            << ") to be divisible by "
                               "batch_group_count. Got batch_group_count = "
                            << batchGroupCount << ".";

  return success();
}

// Infer the return-shape of ConvolutionOp.
// Precondition:
//  1. Input args to ConvolutionOp 'op' are RankedTypes.
//  2. rank-of(input-type) == rank-of(output-type)
SmallVector<int64_t> inferConvolutionOpReturnShape(
    ConvolutionOp op, const ArrayRef<hlo::WindowDimension> window) {
  // We keep the 'unknown' dimensions (cl/415132294) as it is in the
  // output-shape. To do that we initilize the output dimensions with the shape
  // of the return-type and updates only the spatial + non-spatial dimensions.
  // Precondition 2 ensures that size of output-shape == size of input-shape.
  SmallVector<int64_t> outputDimensions =
      to_vector(op.getResult().getType().cast<ShapedType>().getShape());

  // Infer the output spatial dimensions.
  auto lhsType = op.getLhs().getType().cast<RankedTensorType>();
  auto inputSpatialDims = op.getDimensionNumbers().getInputSpatialDimensions();
  auto numSpatialDims = inputSpatialDims.size();
  SmallVector<int64_t> inputSpatialDimVals(numSpatialDims);
  for (int64_t i = 0; i < static_cast<int64_t>(numSpatialDims); ++i)
    inputSpatialDimVals[i] = lhsType.getShape()[inputSpatialDims[i]];

  auto windowOutputShape = inferWindowOutputShape(inputSpatialDimVals, window);

  for (int64_t i = 0; i < static_cast<int64_t>(window.size()); ++i)
    outputDimensions[op.getDimensionNumbers().getOutputSpatialDimensions()[i]] =
        windowOutputShape[i];

  // Infer the output-batch-dimension and output-feature-dimension.
  auto rhsType = op.getRhs().getType().cast<RankedTensorType>();
  const int64_t inputBatch =
      lhsType.getShape()[op.getDimensionNumbers().getInputBatchDimension()];
  const int64_t kernelOutputFeatures =
      rhsType.getShape()[op.getDimensionNumbers()
                             .getKernelOutputFeatureDimension()];

  outputDimensions[op.getDimensionNumbers().getOutputBatchDimension()] =
      hlo::isDynamicDimSize(inputBatch) ? ShapedType::kDynamic
                                        : inputBatch / op.getBatchGroupCount();
  outputDimensions[op.getDimensionNumbers().getOutputFeatureDimension()] =
      kernelOutputFeatures;

  return outputDimensions;
}

// Some mhlo.convolutions are dot products, specifically when there is no
// padding and no spatial dimensions. DotGeneralOp is general enough that it
// can sufficiently describe it.
struct ConvolutionIsDot : public OpRewritePattern<mhlo::ConvolutionOp> {
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();
    auto resultTy = op.getType().cast<RankedTensorType>();

    if (lhsTy.getRank() != 2) return failure();
    if (rhsTy.getRank() != 2) return failure();

    if (op.getBatchGroupCount() != 1) return failure();

    // There should not be any padding if this is a matmul.
    auto dNums = op.getDimensionNumbers();
    assert(!op.getPadding() || op.getPadding()->empty());
    assert(dNums.getKernelSpatialDimensions().empty());

    auto lhsBatchDim = dNums.getInputBatchDimension();
    auto rhsBatchDim = dNums.getKernelOutputFeatureDimension();
    auto lhsContractDim = dNums.getInputFeatureDimension();
    auto rhsContractDim = dNums.getKernelInputFeatureDimension();
    auto outBatchDim = dNums.getOutputBatchDimension();
    auto outFeatureDim = dNums.getOutputFeatureDimension();

    // If the input features are not grouped then we can directly convert to an
    // mhlo.dot_general.
    if (op.getFeatureGroupCount() == 1) {
      // We can swap the lhs and rhs sides to avoid a transpose.
      if (outBatchDim == 1 && outFeatureDim == 0) {
        std::swap(lhs, rhs);
        std::swap(outBatchDim, outFeatureDim);
        std::swap(lhsContractDim, rhsContractDim);
      }

      auto dotNums = DotDimensionNumbersAttr::get(
          op.getContext(), {}, {}, {lhsContractDim}, {rhsContractDim});
      auto dotOp = rewriter.create<mhlo::DotGeneralOp>(
          op.getLoc(), op.getType(), lhs, rhs, dotNums,
          op.getPrecisionConfig().value_or(nullptr));

      rewriter.replaceOp(op, dotOp.getResult());
      return success();
    }

    int64_t featureGroupCount = op.getFeatureGroupCount();
    int64_t lhsBatchSize = lhsTy.getDimSize(lhsBatchDim);
    int64_t lhsContractSize = lhsTy.getDimSize(lhsContractDim);
    int64_t rhsBatchSize = rhsTy.getDimSize(rhsBatchDim);
    int64_t rhsContractSize = rhsTy.getDimSize(rhsContractDim);

    llvm::SmallVector<int64_t> lhsShape;
    llvm::SmallVector<int64_t> rhsShape;
    lhsShape.resize(3, lhsBatchSize);
    rhsShape.resize(3, rhsContractSize);
    lhsShape[lhsContractDim] = featureGroupCount;
    lhsShape[lhsContractDim + 1] = lhsContractSize / featureGroupCount;
    rhsShape[rhsContractDim] = featureGroupCount;
    rhsShape[rhsContractDim + 1] = rhsBatchSize / featureGroupCount;

    lhsTy = RankedTensorType::get(lhsShape, lhsTy.getElementType());
    rhsTy = RankedTensorType::get(rhsShape, rhsTy.getElementType());

    lhs = rewriter.create<mhlo::ReshapeOp>(op.getLoc(), lhsTy, lhs);
    rhs = rewriter.create<mhlo::ReshapeOp>(op.getLoc(), rhsTy, rhs);

    auto dotTy = RankedTensorType::get(
        {featureGroupCount, lhsBatchSize, rhsBatchSize / featureGroupCount},
        resultTy.getElementType());

    auto dotNums = DotDimensionNumbersAttr::get(
        op.getContext(), {lhsContractDim}, {rhsContractDim},
        {lhsContractDim + 1}, {rhsContractDim == 0 ? 2 : 0});
    auto dotOp = rewriter.create<mhlo::DotGeneralOp>(
        op.getLoc(), dotTy, lhs, rhs, dotNums,
        op.getPrecisionConfig().value_or(nullptr));

    llvm::SmallVector<int64_t> perms;
    perms.resize(3, dNums.getOutputBatchDimension() == 0 ? 0 : 2);
    perms[0] = dNums.getOutputFeatureDimension();
    perms[2] = dNums.getOutputFeatureDimension() + 1;

    auto transposeTy = RankedTensorType::get(
        {dotTy.getDimSize(perms[0]), dotTy.getDimSize(perms[1]),
         dotTy.getDimSize(perms[2])},
        dotTy.getElementType());
    auto transposeOp = rewriter.create<mhlo::TransposeOp>(
        op.getLoc(), transposeTy, dotOp, rewriter.getI64TensorAttr(perms));

    rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, resultTy, transposeOp);
    return success();
  }
};

}  // namespace

void ConvolutionOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<ConvolutionIsDot>(context);
}

/*
 * We intend to verify the following properties
 *  P1. Verify the input, kernel types.
 *  P2. Verify the convolution atributes.
 *  P3. Verify and collect the window atributes.
 *  P4. Verify the return shape.
 *      TODO(b/232574102): Verify the element-type of return-value.
 */
LogicalResult ConvolutionOp::verify() {
  auto lhsType = getLhs().getType().dyn_cast<RankedTensorType>();
  auto rhsType = getRhs().getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType) return success();

  // P1.
  int numDims = lhsType.getRank();
  if (numDims != rhsType.getRank())
    return emitOpError()
           << "expects convolution arguments to have same number of "
              "dimensions. Got: "
           << lhsType << " and " << rhsType << ".";

  if (numDims < 2)
    return emitOpError()
           << "expects convolution arguments to have >= 2 dimensions. "
              "Got: "
           << lhsType << " and " << rhsType << ".";

  // P2.
  if (failed(verifyConvolutionAttributes(*this))) return failure();

  // P3.
  auto kernelSpatialDimensions =
      getDimensionNumbers().getKernelSpatialDimensions();
  SmallVector<int64_t> windowDimensions(kernelSpatialDimensions.size());
  for (size_t i = 0; i < windowDimensions.size(); i++)
    windowDimensions[i] = rhsType.getShape()[kernelSpatialDimensions[i]];

  auto paddingOrErr = convertNx2Attribute(this->getPadding(), getLoc());
  if (failed(paddingOrErr)) return failure();
  SmallVector<std::pair<int64_t, int64_t>> padding = *paddingOrErr;

  auto windowOrErr = hlo::verifyWindowAttributesAndInferWindowDimensions(
      windowDimensions, convertDenseIntAttr(getWindowStrides()), padding,
      convertDenseIntAttr(getLhsDilation()),
      convertDenseIntAttr(getRhsDilation()), getLoc());
  if (failed(windowOrErr)) return failure();

  // P4.
  auto actualReturnType = getResult().getType().cast<TensorType>();
  auto actualReturnElementType = actualReturnType.getElementType();
  if (!actualReturnType.hasRank()) return success();

  auto actualReturnRankedType = actualReturnType.cast<RankedTensorType>();
  if (numDims != actualReturnRankedType.getRank())
    return emitOpError() << "expects rank of convolution return-type to be "
                            "equal to input-ranks ("
                         << numDims << "), but got "
                         << actualReturnRankedType.getRank() << ".";

  auto expectedReturnShape = inferConvolutionOpReturnShape(*this, *windowOrErr);
  auto expectedReturnType =
      RankedTensorType::get(expectedReturnShape, actualReturnElementType);
  if (failed(verifyCompatibleShape(expectedReturnType, actualReturnRankedType)))
    return emitOpError()
           << "has shape mismatch between the expected return-type ("
           << expectedReturnType << ") and actual return-type ("
           << actualReturnRankedType << ").";

  return success();
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void ConvertOp::build(OpBuilder& builder, OperationState& result, Value operand,
                      Type resultElementTy) {
  Type resultTy;
  Type operandTy = operand.getType();
  if (auto rankedTy = operandTy.dyn_cast<RankedTensorType>()) {
    resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy);
  } else {
    resultTy = UnrankedTensorType::get(resultElementTy);
  }
  build(builder, result, resultTy, operand);
}

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
  auto operandTy = getOperand().getType().cast<TensorType>();
  auto resultTy = getResult().getType().cast<TensorType>();
  if (operandTy == resultTy) return getOperand();

  // If the result has non-static shape, a convert op is necessary to go from
  // static shape to non-static shape.
  if (!resultTy.hasStaticShape()) return {};

  // If the operand is constant, we can do the conversion now.
  auto elementsAttr = operands.front().dyn_cast_or_null<ElementsAttr>();
  if (!elementsAttr) return {};

  // Prevent folding if the result is too large.
  if (elementsAttr.getNumElements() > kFoldOpEltLimit) return {};
  return hlo::convertElementsAttr(elementsAttr,
                                  getElementTypeOrSelf(getResult()));
}

namespace {

struct EliminateRedundantConvert : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern<ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter& rewriter) const override {
    auto convertOp = op.getOperand().getDefiningOp<ConvertOp>();
    if (!convertOp) {
      return failure();
    }
    auto firstType =
        convertOp.getOperand().getType().cast<TensorType>().getElementType();
    auto secondType =
        op.getOperand().getType().cast<TensorType>().getElementType();
    auto thirdType =
        op.getResult().getType().cast<TensorType>().getElementType();
    auto loc = rewriter.getFusedLoc({convertOp->getLoc(), op->getLoc()});
    if (firstType.isa<FloatType>() && secondType.isa<FloatType>() &&
        thirdType.isa<FloatType>()) {
      // fold when the second float type's width is longer than first,
      // like fp16 -> fp32 -> fp64, bf16 -> fp32 -> fp16
      if (secondType.cast<FloatType>().getWidth() >
          firstType.cast<FloatType>().getWidth()) {
        Value result = rewriter.create<ConvertOp>(loc, op.getResult().getType(),
                                                  convertOp.getOperand());
        rewriter.replaceOp(op, result);
        return success();
      }
    } else if (firstType.isa<IntegerType>() && secondType.isa<IntegerType>() &&
               thirdType.isa<IntegerType>()) {
      // fold when the second integer type's width is longer than first,
      // like i16 -> i32 -> i64, u16 -> i32 -> u32
      if (secondType.cast<IntegerType>().getWidth() >
          firstType.cast<IntegerType>().getWidth()) {
        Value result = rewriter.create<ConvertOp>(loc, op.getResult().getType(),
                                                  convertOp.getOperand());
        rewriter.replaceOp(op, result);
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void ConvertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<EliminateIdentityConvert>(context);
  results.add<EliminateRedundantConvert>(context);
}

//===----------------------------------------------------------------------===//
// StochasticConvertOp
//===----------------------------------------------------------------------===//

LogicalResult StochasticConvertOp::verify() {
  DataLayout dataLayout = DataLayout::closest(*this);
  unsigned operandElementSize =
      dataLayout.getTypeSizeInBits(getOperand().getType().getElementType());
  unsigned randomElementSize =
      dataLayout.getTypeSizeInBits(getRandom().getType().getElementType());
  if (operandElementSize != randomElementSize) {
    return emitOpError() << "requires the random's bitwidth to match the "
                            "operand's, but got: "
                         << randomElementSize << " and " << operandElementSize;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::verify() {
  auto indexVal = getIndex();
  auto operandType = getOperand().getType().cast<TupleType>();
  if (indexVal >= operandType.size()) {
    return emitOpError(
        llvm::formatv("index {0} is out of bounds of operand with size {1}",
                      indexVal, operandType.size()));
  }

  auto expectedType = operandType.getType(indexVal);
  if (getType() != expectedType) {
    return emitOpError(llvm::formatv("has return type {0}, but expected {1}",
                                     getType(), expectedType));
  }
  return success();
}

OpFoldResult GetTupleElementOp::fold(ArrayRef<Attribute> operands) {
  if (auto tupleOp = getOperand().getDefiningOp<mhlo::TupleOp>()) {
    return tupleOp.getOperand(getIndex());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

LogicalResult TupleOp::verify() {
  auto opType = getType().dyn_cast<TupleType>();
  if (!opType) return emitOpError("tuple op with non-tuple result");
  if (getNumOperands() != opType.size())
    return emitOpError(
        "number of operands to tuple expected to match number of types in "
        "resultant tuple type");
  for (const auto& it :
       llvm::enumerate(llvm::zip_first(getOperandTypes(), opType.getTypes()))) {
    if (std::get<0>(it.value()) != std::get<1>(it.value()))
      return emitOpError("has return type mismatch at ")
             << it.index() << "th value (" << std::get<0>(it.value())
             << " != " << std::get<1>(it.value()) << ")";
  }
  return success();
}

namespace {

// Pattern for unpacking and repacking the same tuple.
struct UnpackRepackSameTuple : public OpRewritePattern<TupleOp> {
  using OpRewritePattern<TupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getVal().empty()) return failure();

    Value firstElement = op.getVal().front();
    auto firstElementOp = firstElement.getDefiningOp<GetTupleElementOp>();
    if (!firstElementOp || firstElementOp.getIndexAttr().getInt() != 0)
      return failure();

    Value tuplePredecessor = firstElementOp.getOperand();
    if (tuplePredecessor.getType() != op.getType()) return failure();

    for (const auto& elementAndIdx :
         llvm::enumerate(op.getVal().drop_front(1))) {
      auto elementOp = elementAndIdx.value().getDefiningOp<GetTupleElementOp>();
      if (!elementOp ||
          elementOp.getIndexAttr().getInt() !=
              static_cast<int64_t>(elementAndIdx.index() + 1) ||
          elementOp.getOperand() != tuplePredecessor)
        return failure();
    }

    rewriter.replaceOp(op, tuplePredecessor);
    return success();
  }
};

}  // namespace

void TupleOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<UnpackRepackSameTuple>(context);
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

LogicalResult AllToAllOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  AllToAllOp::Adaptor adaptor(operands, attributes, regions);

  bool isArrayAllToAll = adaptor.getSplitDimension() &&
                         adaptor.getConcatDimension() &&
                         adaptor.getSplitCount();
  if (!isArrayAllToAll) {
    if (adaptor.getSplitDimension() || adaptor.getConcatDimension() ||
        adaptor.getSplitCount()) {
      return emitOptionalError(location,
                               "TupleAllToAll should not have split_dimension, "
                               "concat_dimension or split_count attributes");
    }

    // TupleAllToAll has identical result and operand shapes.
    for (int64_t i = 0; i < operands.size(); ++i) {
      inferredReturnShapes.emplace_back(
          operands[i].getType().cast<ShapedType>());
    }

    return success();
  }

  if (adaptor.getOperand().size() != 1) {
    return emitOptionalError(location,
                             "ArrayAllToAll should have exactly one operand");
  }

  Type operandType = adaptor.getOperand()[0].getType();
  RankedTensorType operandRankedType = operandType.dyn_cast<RankedTensorType>();
  if (!operandRankedType) {
    inferredReturnShapes.emplace_back(
        operandType.cast<TensorType>().getElementType());
    return success();
  }

  int64_t inputRank = operandRankedType.getRank();
  int64_t splitDimension = static_cast<int64_t>(*adaptor.getSplitDimension());
  int64_t concatDimension = static_cast<int64_t>(*adaptor.getConcatDimension());
  if (splitDimension >= inputRank || splitDimension < 0) {
    return emitOptionalError(location, "AllToAll split_dimension ",
                             splitDimension,
                             " is out-of-bounds for input rank ", inputRank);
  }
  if (concatDimension >= inputRank || concatDimension < 0) {
    return emitOptionalError(location, "AllToAll concat_dimension ",
                             concatDimension,
                             " is out-of-bounds for input rank ", inputRank);
  }

  // If operand is ranked, size of split dimension should be a multiple of split
  // count.
  int64_t splitCount = *adaptor.getSplitCount();
  auto splitDimSize = operandRankedType.getDimSize(splitDimension);
  if (splitDimSize != ShapedType::kDynamic &&
      (splitDimSize % splitCount != 0)) {
    return emitOptionalError(
        location, "split dimension has size ", splitDimSize,
        ", expected to be a multiple of split_count ", splitCount);
  }
  SmallVector<int64_t> resultShape(operandRankedType.getShape().begin(),
                                   operandRankedType.getShape().end());
  if (resultShape[splitDimension] != ShapedType::kDynamic) {
    resultShape[splitDimension] /= splitCount;
  }
  if (resultShape[concatDimension] != ShapedType::kDynamic) {
    resultShape[concatDimension] *= splitCount;
  }
  inferredReturnShapes.emplace_back(resultShape,
                                    operandRankedType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  // If operand and result are both ranked, then the size of the gather
  // dimension in the result should be a multiple of the size of the gather
  // dimension in the operand.
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();
  uint64_t allGatherDimIndex = getAllGatherDim();
  if (!operandType || !resultType ||
      operandType.isDynamicDim(allGatherDimIndex) ||
      resultType.isDynamicDim(allGatherDimIndex))
    return success();
  if (operandType.getDimSize(allGatherDimIndex) == 0)
    return emitOpError() << "operand gather dimension cannot be zero.";
  if ((resultType.getDimSize(allGatherDimIndex) %
       operandType.getDimSize(allGatherDimIndex)) != 0)
    return emitOpError()
           << "result gather dimension has size "
           << resultType.getDimSize(allGatherDimIndex)
           << ", expected to be a multiple of operand gather dimension size "
           << operandType.getDimSize(allGatherDimIndex);

  return success();
}

void AllGatherOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        Type resultType, Value operand,
                        IntegerAttr allGatherDim,
                        DenseIntElementsAttr replicaGroups,
                        ChannelHandleAttr channelHandle) {
  AllGatherOp::build(odsBuilder, odsState, resultType, operand, allGatherDim,
                     replicaGroups, channelHandle,
                     /*use_global_device_ids=*/nullptr);
}

//===----------------------------------------------------------------------===//
// BatchNormGradOp
//===----------------------------------------------------------------------===//

LogicalResult verifyBatchNorm(Location loc, Value operand,
                              int64_t feature_index, Value scale) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  if (feature_index >= operandType.getRank())
    return emitError(loc) << "expects feature_index to be smaller "
                             "than the rank of operand type; got feature_index "
                          << feature_index << ", and rank "
                          << operandType.getRank() << ".";

  if (feature_index < 0)
    return emitError(loc) << "expects feature_index to be a "
                          << "non-negative number, got " << feature_index
                          << ".";

  // Note: the above checks '0 <= feature-index < operandType.getRank()'
  // imply 'operand_type.getRank() >= 1'.

  const int64_t featureCount = operandType.getDimSize(feature_index);
  const int64_t scaleShape =
      scale.getType().cast<RankedTensorType>().getDimSize(0);
  // As ODS enforces `scale`, `mean`, `variance`, `offset` are AllShapesMatch,
  // this also infers that featureCount is aligned with them.
  if (scaleShape != featureCount) {
    auto dimToStr = [](int64_t dim) {
      return ShapedType::isDynamic(dim) ? "?" : std::to_string(dim);
    };
    return emitError(loc) << "expects the size of scale factor to be "
                             "same as the feature count,"
                             " but the size of scale factor is "
                          << dimToStr(scaleShape)
                          << " and the feature count is "
                          << dimToStr(featureCount) << ".";
  }

  return success();
}

// Refer ODS for properties that are already enforced including shapes and
// element types. This verifier includes additional checks.
LogicalResult BatchNormGradOp::verify() {
  if (failed(verifyBatchNorm(getLoc(), getOperand(), getFeatureIndex(),
                             getScale())))
    return failure();
  return success();
}

LogicalResult BatchNormGradOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormGradOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBatchNormGradOp(
      location, adaptor.getOperand(), adaptor.getScale(),
      adaptor.getFeatureIndex(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//

// Refer ODS for properties that are already enforced including shapes and
// element types. This verifier includes additional checks.
LogicalResult BatchNormTrainingOp::verify() {
  if (failed(verifyBatchNorm(getLoc(), getOperand(), getFeatureIndex(),
                             getScale())))
    return failure();
  return success();
}

LogicalResult BatchNormTrainingOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormTrainingOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBatchNormTrainingOp(
      location, adaptor.getOperand(), adaptor.getScale(),
      adaptor.getFeatureIndex(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BatchNormInferenceOp
//===----------------------------------------------------------------------===//

// Refer ODS for properties that are already enforced including shapes and
// element types. This verifier includes additional checks.
LogicalResult BatchNormInferenceOp::verify() {
  if (failed(verifyBatchNorm(getLoc(), getOperand(), getFeatureIndex(),
                             getScale())))
    return failure();
  return success();
}

LogicalResult BatchNormInferenceOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormInferenceOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBatchNormInferenceOp(
      location, adaptor.getOperand(), adaptor.getScale(),
      adaptor.getFeatureIndex(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BitcastOp::fold(ArrayRef<Attribute>) {
  if (getResult().getType() != getOperand().getType()) {
    return {};
  }

  auto sourceLayout =
      getOperation()->getAttrOfType<DenseIntElementsAttr>("source_layout");
  auto resultLayout =
      getOperation()->getAttrOfType<DenseIntElementsAttr>("result_layout");

  if (sourceLayout == resultLayout) {
    return getOperand();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// BitcastConvertOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastConvertOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto operandType = operands[0].getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();

  // Only ranked tensors are supported.
  if (!operandType || !resultType) return failure();

  // Shape-changing bitcast convert is not implemented.
  // TODO(kramerb): This could be done by adjusting the last dimension.
  DataLayout dataLayout = DataLayout::closest(*this);
  unsigned operandElementSize =
      dataLayout.getTypeSizeInBits(operandType.getElementType());
  unsigned resultElementSize =
      dataLayout.getTypeSizeInBits(resultType.getElementType());
  if (operandElementSize != resultElementSize) return failure();

  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

/*
 * We intend to verify the following properties
 * P1. We cannot convert between complex and real types (cf xla)
 * P3. The dimensions of the operand and the target
 * shape must match, except that the shape with the smaller element bitwidth has
 * an appropriately-sized additional innermost dimension, e.g.
 * ... x f32 => [bitcast_convert] => ... x 4 x i8
 * ... x 4 x i8 => [bitcast_convert] => ... x f32
 */
LogicalResult BitcastConvertOp::verify() {
  auto operandTensorType = getOperand().getType().cast<TensorType>();
  auto targetTensorType = getResult().getType().cast<TensorType>();

  // P1.
  auto targetElt = targetTensorType.getElementType();
  auto operandElt = operandTensorType.getElementType();
  if (targetElt.isa<ComplexType>() != operandElt.isa<ComplexType>()) {
    return emitOpError()
           << "cannot convert between real and complex types, but got: "
           << operandTensorType << " and " << targetTensorType;
  }

  auto targetEltBitwidth = hlo::potentiallyComplexBitwidth(targetElt);
  auto operandEltBitwidth = hlo::potentiallyComplexBitwidth(operandElt);

  // P2.
  auto operandType = operandTensorType.dyn_cast<RankedTensorType>();
  auto targetType = targetTensorType.dyn_cast<RankedTensorType>();
  if (!operandType || !targetType) return success();

  auto targetShape = targetType.getShape();
  auto operandShape = operandType.getShape();
  ArrayRef<int64_t> smallerEltShape, biggerEltShape;
  Type smallerElt, biggerElt;
  if (operandEltBitwidth < targetEltBitwidth) {
    smallerEltShape = operandShape;
    smallerElt = operandElt;
    biggerEltShape = targetShape;
    biggerElt = targetElt;
  } else {
    smallerEltShape = targetShape;
    smallerElt = targetElt;
    biggerEltShape = operandShape;
    biggerElt = operandElt;
  }

  ArrayRef<int64_t> smallerEltPrefix;
  auto smallerEltBitwidth = std::min(targetEltBitwidth, operandEltBitwidth);
  auto biggerEltBitwidth = std::max(targetEltBitwidth, operandEltBitwidth);
  if (operandEltBitwidth != targetEltBitwidth) {
    if (smallerEltShape.empty()) {
      return emitOpError() << "does not allow the smaller element type to be "
                              "part of a 0d tensor, but got: "
                           << operandType << " and " << targetType << ".";
    }
    smallerEltPrefix = smallerEltShape.drop_back();
    if (!hlo::isDynamicDimSize(smallerEltShape.back()) &&
        smallerEltShape.back() * smallerEltBitwidth != biggerEltBitwidth) {
      return emitOpError() << "requires compatible bitwidths. "
                           << "Got: " << operandType << " and " << targetType
                           << ", but " << smallerEltBitwidth << " * "
                           << smallerEltShape.back()
                           << " != " << biggerEltBitwidth << ".";
    }
  } else {
    smallerEltPrefix = smallerEltShape;
  }

  for (auto it : llvm::zip(smallerEltPrefix, biggerEltShape)) {
    auto targetDim = std::get<0>(it);
    auto operandDim = std::get<1>(it);
    if (!hlo::isDynamicDimSize(targetDim) &&
        !hlo::isDynamicDimSize(operandDim)) {
      if (targetDim != operandDim) {
        return emitOpError() << "operand and result shapes must match except "
                                "for the innermost dimension of the shape with "
                                "the smaller element type. Got: "
                             << operandType << " and " << targetType << ".";
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// TODO(b/129012527) These should be expressed as type constraints.
LogicalResult BroadcastOp::verify() {
  auto sizes = getBroadcastSizes();
  auto sizesType = sizes.getType();
  auto sizesRank = sizesType.getRank();
  if (sizesRank != 1) {
    return emitOpError(llvm::formatv(
        "broadcast_sizes has rank {0} instead of rank 1", sizesRank));
  }

  return success();
}

OpFoldResult BroadcastOp::fold(ArrayRef<Attribute> attrs) {
  auto type = getType().cast<RankedTensorType>();
  auto sizesType = getBroadcastSizes().getType();
  if (sizesType.getNumElements() == 0) {
    return getOperand();
  }

  // Constant fold when an operand is a splat tensor attribute.
  if (!attrs[0] || !type.hasStaticShape()) return {};
  auto splatOperandAttr = attrs[0].dyn_cast<SplatElementsAttr>();
  if (!splatOperandAttr) return {};

  // Handle complex type
  if (type.getElementType().isa<ComplexType>()) {
    ComplexType complex = type.getElementType().cast<ComplexType>();
    if (complex.getElementType().isa<FloatType>()) {
      return DenseElementsAttr::get(
          type, {splatOperandAttr.getSplatValue<std::complex<APFloat>>()});
    }
    if (complex.getElementType().isa<IntegerType>()) {
      return DenseElementsAttr::get(
          type, {splatOperandAttr.getSplatValue<std::complex<APInt>>()});
    }
    return {};
  }

  return SplatElementsAttr::get(
      type, splatOperandAttr.getSplatValue<mlir::Attribute>());
}

LogicalResult BroadcastOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands, attributes, regions);
  Value operand = adaptor.getOperand();
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  Type elementTy = operandType.getElementType();
  auto dimensionAttr = adaptor.getBroadcastSizes();
  for (int64_t size : dimensionAttr.getValues<int64_t>()) {
    if (size < 0)
      return emitOptionalError(location,
                               "Broadcast with negative dimension size ", size);
  }
  SmallVector<int64_t> shapeValues(dimensionAttr.getValues<int64_t>());
  llvm::append_range(shapeValues, operandType.getShape());

  inferredReturnShapes.emplace_back(shapeValues, elementTy);
  return success();
}

LogicalResult BroadcastOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Unranked tensors are not supported.
  if (!operandType) return failure();

  Location loc = getLoc();
  SmallVector<Value, 4> shapeValues;

  // Collect the broadcast sizes.
  for (const auto& size : getBroadcastSizes()) {
    shapeValues.push_back(
        builder.create<arith::ConstantIndexOp>(loc, size.getZExtValue()));
  }

  // Collect the operand sizes.
  for (auto index : llvm::seq<int64_t>(0, operandType.getRank())) {
    shapeValues.push_back(
        builder.createOrFold<tensor::DimOp>(loc, operand, index));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            builder.getIndexType()),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastInDimOp::verify() {
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();
  if (!operandType) {
    // The following verification checks all depend on knowing the rank of
    // the operand. Bail out now if we don't know the rank of the operand.
    return success();
  }

  auto operandRank = operandType.getRank();
  if (!getBroadcastDimensions()) {
    if (operandRank == 0) {
      return success();
    }
    return emitOpError(
        llvm::formatv("broadcast_dimensions is absent, but required because "
                      "operand has non-zero rank ({0})",
                      operandRank));
  }

  auto dimensionsType = getBroadcastDimensions().getType();
  auto dimensionsRank = dimensionsType.getRank();
  if (dimensionsRank != 1) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions has rank {0} instead of rank 1", dimensionsRank));
  }

  auto dimensionsSize = dimensionsType.getNumElements();
  if (dimensionsSize != operandRank) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        dimensionsSize, operandRank));
  }

  auto dimensions =
      llvm::to_vector(getBroadcastDimensions().getValues<int64_t>());
  if (hasDuplicates(dimensions))
    return emitOpError("broadcast_dimensions should not have duplicates");

  auto resultType = getResult().getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();
  for (int i = 0; i != dimensionsSize; ++i) {
    auto dimIndex = dimensions[i];
    if (dimIndex >= resultRank) {
      return emitOpError(
          llvm::formatv("broadcast_dimensions contains invalid value {0} for "
                        "result with rank {1}",
                        dimIndex, resultRank));
    }

    if (!operandType.isDynamicDim(i)) {
      auto dimSize = operandType.getDimSize(i);
      auto resultDimSize = resultType.getDimSize(dimIndex);
      if (dimSize != 1 && dimSize != resultDimSize) {
        return emitOpError(
            llvm::formatv("size of operand dimension {0} ({1}) is not equal to "
                          "1 or size of result dimension {2} ({3})",
                          i, dimSize, dimIndex, resultDimSize));
      }
    }
  }

  return success();
}

OpFoldResult BroadcastInDimOp::fold(ArrayRef<Attribute> attrs) {
  auto type = getType().cast<RankedTensorType>();
  if (type == getOperand().getType()) {
    auto broadcastValues = getBroadcastDimensions().getValues<int64_t>();
    if (!std::equal(broadcastValues.begin(), broadcastValues.end(),
                    llvm::seq<int64_t>(0, type.getRank()).begin())) {
      return {};
    }
    return getOperand();
  }

  // Constant fold when an operand is a splat tensor attribute.
  if (!attrs[0] || !type.hasStaticShape()) return {};
  auto splatOperandAttr = attrs[0].dyn_cast<SplatElementsAttr>();
  if (!splatOperandAttr) return {};

  // Handle complex type
  if (type.getElementType().isa<ComplexType>()) {
    ComplexType complex = type.getElementType().cast<ComplexType>();
    if (complex.getElementType().isa<FloatType>()) {
      return DenseElementsAttr::get(
          type, {splatOperandAttr.getSplatValue<std::complex<APFloat>>()});
    }
    if (complex.getElementType().isa<IntegerType>()) {
      return DenseElementsAttr::get(
          type, {splatOperandAttr.getSplatValue<std::complex<APInt>>()});
    }
    return {};
  }

  return SplatElementsAttr::get(
      type, splatOperandAttr.getSplatValue<mlir::Attribute>());
}

// Simplify BroadcastInDim has the following behaviors: replace BroadcastInDim
// with Reshape or Transpose if they are equivalent or replace
// BroadcastInDim(BroadcastInDim(X)) with BroadcastInDim(X)
class BroadcastInDimSimplifier : public OpRewritePattern<BroadcastInDimOp> {
 public:
  using OpRewritePattern<BroadcastInDimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getOperand().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!operandType || !resultType) {
      return failure();
    }
    auto bsDimIndices = op.getBroadcastDimensions().getValues<int64_t>();
    if (operandType.hasStaticShape() && resultType.hasStaticShape()) {
      bool sameTotalElements =
          operandType.getNumElements() == resultType.getNumElements();
      // BroadcastInDim equivalent to reshape
      if (llvm::is_sorted(bsDimIndices) && sameTotalElements) {
        rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(),
                                               op.getOperand());
        return success();
      }
      // BroadcastInDim equivalent to transpose
      if (operandType.getRank() == resultType.getRank() && sameTotalElements) {
        rewriter.replaceOpWithNewOp<TransposeOp>(
            op, op.getType(), op.getOperand(), op.getBroadcastDimensions());
        return success();
      }
    }
    // eliminate redundant BroadcastInDim
    if (auto broadcastInDimOp = llvm::dyn_cast_or_null<BroadcastInDimOp>(
            op.getOperand().getDefiningOp())) {
      auto newIndices =
          broadcastInDimOp.getBroadcastDimensions()
              .mapValues(op.getBroadcastDimensions().getElementType(),
                         [&bsDimIndices](const APInt& dim) -> APInt {
                           return APInt(dim.getBitWidth(),
                                        bsDimIndices[dim.getSExtValue()], true);
                         })
              .cast<DenseIntElementsAttr>();
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, op.getType(), broadcastInDimOp.getOperand(), newIndices);
      return success();
    }
    return failure();
  }
};

void BroadcastInDimOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                   MLIRContext* context) {
  results.add<BroadcastInDimSimplifier>(context);
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicBroadcastInDimOp::verify() {
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();

  // If either the operand or result are unranked, there is very little
  // to verify statically.
  if (!operandType || !resultType) {
    return success();
  }

  auto outputDimensionsType =
      getOutputDimensions().getType().cast<RankedTensorType>();
  auto outputDimensionsSize = outputDimensionsType.getDimSize(0);
  auto operandRank = operandType.getRank();
  auto resultRank = resultType.getRank();

  // Verify broadcast_dimensions.
  auto bcastDimensions = getBroadcastDimensions();
  auto bcastDimensionsType = getBroadcastDimensions().getType();
  auto bcastDimensionsRank = bcastDimensionsType.getRank();
  // TODO(laurenzo): Update the BroadcastDimAttr to constrain its rank to 1.
  if (bcastDimensionsRank != 1) {
    return emitOpError(
        llvm::formatv("broadcast_dimensions has rank {0} instead of rank 1",
                      bcastDimensionsRank));
  }

  auto bcastDimensionsSize = bcastDimensionsType.getNumElements();
  if (bcastDimensionsSize != operandRank) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        bcastDimensionsSize, operandRank));
  }

  if (resultRank < operandRank) {
    return emitOpError(
        llvm::formatv("result rank ({0}) is less than operand rank ({1})",
                      resultRank, operandRank));
  }

  for (int i = 0; i != bcastDimensionsSize; ++i) {
    auto dimIndex = bcastDimensions.getValues<int64_t>()[i];
    if (dimIndex >= resultRank) {
      return emitOpError(
          llvm::formatv("broadcast_dimensions contains invalid value {0} for "
                        "result with rank {1}",
                        dimIndex, resultRank));
    }

    auto dimSize = operandType.getDimSize(i);
    auto resultDimSize = resultType.getDimSize(dimIndex);
    // Note: verifyCompatibleShapes doesn't consider size-1 broadcasting, so we
    // add a manual check for this.
    if (dimSize != 1 && failed(verifyCompatibleShape(dimSize, resultDimSize))) {
      return emitOpError(
          llvm::formatv("size of operand dimension {0} ({1}) is not compatible "
                        "with size of result dimension {2} ({3})",
                        i, dimSize, dimIndex, resultDimSize));
    }
  }

  if (outputDimensionsSize != resultRank) {
    return emitOpError(
        llvm::formatv("result rank ({0}) is not equal to number of output "
                      "dimensions ({1})",
                      resultRank, outputDimensionsSize));
  }

  // Verify that the known expanding and non-expanding dimensions are a subset
  // of the operand's dimensions.
  int64_t numKnownExpansionBehavior = 0;
  DenseSet<int64_t> knownExpansionBehavior;
  auto collectExpansionBehaviorDims =
      [&](const Optional<DenseIntElementsAttr>& attr) {
        if (!attr) return;
        for (const APInt& it : *attr) {
          numKnownExpansionBehavior++;
          knownExpansionBehavior.insert(it.getLimitedValue());
        }
      };
  collectExpansionBehaviorDims(getKnownExpandingDimensions());
  collectExpansionBehaviorDims(getKnownNonexpandingDimensions());
  if (knownExpansionBehavior.size() != numKnownExpansionBehavior) {
    return emitOpError(
        "duplicate expansion hint for at least one operand dimension");
  }
  for (int64_t i : knownExpansionBehavior) {
    if (i < 0 || i >= operandRank) {
      return emitOpError(
          llvm::formatv("hint for expanding dimension {0} does not refer to a "
                        "valid operand dimension",
                        i));
    }
  }

  return success();
}

namespace {
// If a DynamicBroadCastInDimOp is not actually dynamic, use an ordinary
// BroadcastInDimOp.
class DynamicBroadcastInDimOpNotActuallyDynamic
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto type = op.getType().dyn_cast<RankedTensorType>();
    auto operandType = op.getOperand().getType().dyn_cast<RankedTensorType>();
    auto* outputDimOp = op.getOutputDimensions().getDefiningOp();
    if (!type || !operandType || !operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires operand static shape");
    }
    // output has static shape, replace with broadcast_in_dim
    if (type.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, type, op.getOperand(), op.getBroadcastDimensions());
      return success();
    }
    // output_dimensions are constant, set output shape with output_dimensions,
    // then replace with broadcast_in_dim
    if (outputDimOp && outputDimOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
      DenseIntElementsAttr shapeAttr;
      if (matchPattern(outputDimOp, m_Constant(&shapeAttr))) {
        SmallVector<int64_t> outputShape;
        for (APInt shape : shapeAttr.getValues<APInt>()) {
          outputShape.push_back(shape.getZExtValue());
        }
        refineOpWithNewOp<BroadcastInDimOp>(
            rewriter, op,
            RankedTensorType::get(outputShape, type.getElementType()),
            op.getOperand(), op.getBroadcastDimensions());
        return success();
      }
    }
    return rewriter.notifyMatchFailure(
        op, "requires output static shape or constant broadcast dimensions");
  }
};

class ChainedDynamicBroadcastInDimCanonicalization
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp bcast,
                                PatternRewriter& rewriter) const override {
    auto precedingBcast =
        bcast.getOperand().getDefiningOp<DynamicBroadcastInDimOp>();
    if (!precedingBcast) return failure();

    // Compose broadcast dimensions.
    DenseIntElementsAttr precedingBcastDims =
        precedingBcast.getBroadcastDimensions();
    DenseIntElementsAttr bcastDims = bcast.getBroadcastDimensions();
    SmallVector<APInt, 4> composition;
    for (APInt precedingDim : precedingBcastDims) {
      composition.push_back(
          bcastDims.getValues<APInt>()[precedingDim.getZExtValue()]);
    }
    auto composedBcastDims =
        DenseIntElementsAttr::get(precedingBcastDims.getType(), composition);

    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        bcast, bcast.getType(), precedingBcast.getOperand(),
        bcast.getOutputDimensions(), composedBcastDims);
    return success();
  }
};
}  // namespace

void DynamicBroadcastInDimOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.add<ChainedDynamicBroadcastInDimCanonicalization,
              DynamicBroadcastInDimOpNotActuallyDynamic,
              DynamicBroadcastToOwnShape_1, DynamicBroadcastToOwnShape_2,
              DynamicBroadcastToOwnShape_3, DynamicBroadcastToOwnShape_4>(
      context);
}

LogicalResult DynamicBroadcastInDimOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicBroadcastInDimOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputDimensions()));
  return success();
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

LogicalResult ClampOp::verify() {
  auto operandType = getOperand().getType().cast<RankedTensorType>();
  auto operandShape = operandType.getShape();
  auto minType = getMin().getType().cast<RankedTensorType>();

  auto minShape = minType.getShape();
  if (failed(verifyCompatibleShape(minType, operandType)) &&
      minType.getRank() != 0) {
    return emitOpError(llvm::formatv(
        "min shape [{0}] is not scalar and is not compatible to operand shape "
        "[{1}]",
        llvm::make_range(minShape.begin(), minShape.end()),
        llvm::make_range(operandShape.begin(), operandShape.end())));
  }

  auto maxType = getMax().getType().cast<RankedTensorType>();
  auto maxShape = maxType.getShape();
  if (failed(verifyCompatibleShape(maxType, operandType)) &&
      maxType.getRank() != 0) {
    return emitOpError(llvm::formatv(
        "max shape [{0}] is not scalar and is not compatible to operand shape "
        "[{1}]",
        llvm::make_range(maxShape.begin(), maxShape.end()),
        llvm::make_range(operandShape.begin(), operandShape.end())));
  }

  return success();
}

LogicalResult ClampOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> /*location*/, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ClampOp::Adaptor adaptor(operands, attributes, regions);
  RankedTensorType operandType =
      adaptor.getOperand().getType().cast<RankedTensorType>();
  inferredReturnShapes.emplace_back(operandType.getShape(),
                                    operandType.getElementType());
  return success();
}

LogicalResult ClampOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  // For `mhlo.clamp`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// ComplexOp
//===----------------------------------------------------------------------===//

LogicalResult ComplexOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  TensorType operandType = operands[0].getType().cast<TensorType>();
  ComplexType elementTy = ComplexType::get(operandType.getElementType());
  inferredReturnTypes.push_back(
      hlo::getSameShapeTensorType(operandType, elementTy));
  return success();
}

OpFoldResult ComplexOp::fold(ArrayRef<Attribute> operands) {
  auto realOp = getOperand(0).getDefiningOp<mhlo::RealOp>();
  auto imagOp = getOperand(1).getDefiningOp<mhlo::ImagOp>();
  if (realOp && imagOp && realOp.getOperand() == imagOp.getOperand()) {
    return realOp.getOperand();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ImagOp
//===----------------------------------------------------------------------===//

namespace {
Type createRealType(TensorType type) {
  auto elementTy = type.getElementType();
  if (auto complexTy = elementTy.dyn_cast<ComplexType>()) {
    elementTy = complexTy.getElementType();
  }
  return hlo::getSameShapeTensorType(type, elementTy);
}
}  // namespace

LogicalResult ImagOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(
      createRealType(operands[0].getType().cast<TensorType>()));
  return success();
}

OpFoldResult ImagOp::fold(ArrayRef<Attribute> operands) {
  if (auto complexOp = getOperand().getDefiningOp<mhlo::ComplexOp>()) {
    return complexOp.getOperand(1);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

LogicalResult IsFiniteOp::inferReturnTypes(
    MLIRContext* ctx, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto argTy = operands.front().getType().cast<TensorType>();
  Builder b(ctx);
  inferredReturnTypes.push_back(
      hlo::getSameShapeTensorType(argTy, b.getI1Type()));
  return success();
}

//===----------------------------------------------------------------------===//
// RealOp
//===----------------------------------------------------------------------===//

LogicalResult RealOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(
      createRealType(operands[0].getType().cast<TensorType>()));
  return success();
}

OpFoldResult RealOp::fold(ArrayRef<Attribute> operands) {
  if (auto complexOp = getOperand().getDefiningOp<mhlo::ComplexOp>()) {
    return complexOp.getOperand(0);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

namespace {
class SingleOperandConcatenateToCast : public OpRewritePattern<ConcatenateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getVal().size() != 1) return failure();

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                                op.getVal().front());
    return success();
  }
};

class ConcatenateOperandRemoval : public OpRewritePattern<ConcatenateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    auto axis = op.getDimension();
    llvm::SmallVector<Value, 6> newOperands;
    for (auto operand : op.getOperands()) {
      auto ty = operand.getType().cast<ShapedType>();
      if (!ty.hasRank() || ty.getDimSize(axis) != 0) {
        newOperands.push_back(operand);
      }
    }

    if (!newOperands.empty() && newOperands.size() < op.getNumOperands()) {
      rewriter.replaceOpWithNewOp<ConcatenateOp>(
          op, op.getResult().getType(), newOperands, op.getDimension());
      return success();
    }

    return failure();
  }
};

class ConcatenateForwarding : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    auto getFlattenedOperands = [&](const Value& val) -> ValueRange {
      auto definingOp = dyn_cast_or_null<ConcatenateOp>(val.getDefiningOp());
      // To avoid inflate the memory footprint, only flatten the ConcatenateOp
      // when it has only one use.
      if (definingOp && definingOp->hasOneUse() &&
          definingOp.getDimension() == op.getDimension())
        return definingOp.getVal();
      return val;
    };

    bool needToFlatten = false;
    int operandCount = 0;
    llvm::for_each(op.getVal(), [&](Value val) {
      auto result = getFlattenedOperands(val);
      if (result.size() != 1 || result[0] != val) needToFlatten = true;
      operandCount += result.size();
    });

    if (!needToFlatten) return failure();

    llvm::SmallVector<Value, 6> newOperands;
    newOperands.reserve(operandCount);

    for (auto operand : op.getVal()) {
      auto flattenedOperands = getFlattenedOperands(operand);
      newOperands.append(flattenedOperands.begin(), flattenedOperands.end());
    }

    rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getResult().getType(),
                                               newOperands, op.getDimension());
    return success();
  }
};

}  // namespace

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext*, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  if (operands.empty()) {
    return failure();
  }

  auto dimensionAttr = attributes.get("dimension").cast<IntegerAttr>();
  auto dimension = dimensionAttr.getInt();

  auto firstType = (*operands.begin()).getType().cast<ShapedType>();
  auto outElement = firstType.getElementType();

  // Find the first ranked input to determine the output rank.
  for (auto type : operands.getTypes()) {
    auto shapedType = type.cast<ShapedType>();
    if (shapedType.hasRank()) {
      firstType = shapedType;
      break;
    }
  }

  // If all inputs are unranked, the result must be unranked.
  if (!firstType.hasRank()) {
    inferredReturnTypes.push_back(UnrankedTensorType::get(outElement));
    return success();
  }

  auto outShape = llvm::to_vector<6>(firstType.getShape());

  // Determine what the non-concatenate dimensions should be.
  for (auto type : operands.getTypes()) {
    auto shapedTy = type.cast<ShapedType>();
    if (!shapedTy.hasRank()) {
      continue;
    }

    for (const auto& it : llvm::enumerate(shapedTy.getShape())) {
      // If a dimension is not dynamic, the output shape should match.
      if (ShapedType::isDynamic(outShape[it.index()])) {
        outShape[it.index()] = it.value();
      }
    }
  }

  outShape[dimension] = 0;

  for (auto operand : operands.getTypes()) {
    auto type = operand.cast<ShapedType>();
    if (!type.hasRank()) {
      inferredReturnTypes.push_back(UnrankedTensorType::get(outElement));
      return success();
    }

    // If the dimension is dynamic we know the output dimension is dynamic.
    auto dim = type.getShape()[dimension];
    if (ShapedType::isDynamic(dim)) {
      outShape[dimension] = ShapedType::kDynamic;
      break;
    }

    outShape[dimension] += dim;
  }

  bool allSparse = llvm::all_of(operands.getTypes(), [](Type t) -> bool {
    return sparse_tensor::getSparseTensorEncoding(t) != nullptr;
  });

  sparse_tensor::SparseTensorEncodingAttr enc;
  if (allSparse) {
    // Picks the encoding from an abitrary input is fine and it will be lowered
    // correctly by the sparse compiler (though efficiency might vary).
    // TODO: Extra rules are needed to infer sparse encoding when inputs have
    // different encodings for better efficiency.
    enc = sparse_tensor::getSparseTensorEncoding(operands.getTypes()[0]);
  }

  inferredReturnTypes.push_back(
      RankedTensorType::get(outShape, outElement, enc));

  return success();
}

void ConcatenateOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<ConcatenateOperandRemoval, ConcatenateForwarding,
              SingleOperandConcatenateToCast>(context);
}

template <typename T>
static Attribute foldConcatenateHelper(ConcatenateOp* op,
                                       ArrayRef<Attribute> operands) {
  auto axis = op->getDimension();
  auto type = op->getType().cast<ShapedType>();
  auto shape = type.getShape();

  size_t topSize = 1;
  for (int i = 0, e = axis; i < e; i++) {
    topSize = topSize * shape[i];
  }

  // Prevent folding if the result is too large.
  if (type.getNumElements() > kFoldOpEltLimit) return {};

  SmallVector<T, 6> values;
  for (size_t i = 0; i < topSize; i++) {
    for (auto operand : operands) {
      DenseElementsAttr attr = operand.cast<DenseElementsAttr>();
      size_t bottomSize = attr.getNumElements() / topSize;
      auto iter = attr.getValues<T>().begin() + i * bottomSize;
      values.append(iter, iter + bottomSize);
    }
  }

  return DenseElementsAttr::get(type, values);
}

static Attribute foldConcatenate(ConcatenateOp* op,
                                 ArrayRef<Attribute> operands) {
  for (auto operand : operands) {
    if (!operand) return {};
  }

  auto type = op->getResult().getType().cast<ShapedType>();
  auto etype = type.getElementType();
  if (etype.isa<IntegerType>()) {
    return foldConcatenateHelper<APInt>(op, operands);
  }

  if (etype.isa<FloatType>()) {
    return foldConcatenateHelper<APFloat>(op, operands);
  }

  return {};
}

OpFoldResult ConcatenateOp::fold(ArrayRef<Attribute> operands) {
  if (getNumOperands() == 1) return getOperand(0);

  ShapedType type = getResult().getType().cast<ShapedType>();
  if (!type.hasStaticShape()) return {};

  auto axis = getDimension();
  if (auto attr = foldConcatenate(this, operands)) {
    return attr;
  }

  for (auto operand : getOperands()) {
    auto ty = operand.getType().cast<ShapedType>();
    if (ty.getDimSize(axis) != 0) {
      return {};
    }
  }

  return DenseElementsAttr::get(type, ArrayRef<Attribute>());
}

LogicalResult ConcatenateOp::verify() {
  RankedTensorType firstRankedType;
  int firstRankedIndex;
  int numOperands = getNumOperands();
  int64_t concatDimension = static_cast<int64_t>(getDimension());
  if (concatDimension < 0) {
    return emitOpError(
        llvm::formatv("dimension {0} is negative", concatDimension));
  }
  for (int i = 0; i < numOperands; i++) {
    auto secondType = getOperand(i).getType().dyn_cast<ShapedType>();
    if (!secondType.hasRank()) {
      continue;
    }

    if (!firstRankedType) {
      firstRankedType = secondType.cast<RankedTensorType>();
      firstRankedIndex = i;
      if (firstRankedType.getRank() == 0)
        return emitOpError(
            llvm::formatv("rank-0 values cannot be concatenated"));
      if (concatDimension >= firstRankedType.getRank()) {
        return emitOpError(
            llvm::formatv("dimension {0} is out-of-bounds for input rank {1}",
                          concatDimension, firstRankedType.getRank()));
      }
      continue;
    }

    if (firstRankedType.getRank() != secondType.getRank()) {
      return emitOpError(llvm::formatv(
          "operands ({0}) and ({1}) do not match rank", firstRankedIndex, i));
    }

    auto firstShape = firstRankedType.getShape();
    auto secondShape = secondType.getShape();
    for (int d = 0; d < firstRankedType.getRank(); ++d) {
      if (!ShapedType::isDynamic(firstShape[d]) &&
          !ShapedType::isDynamic(secondShape[d]) &&
          firstShape[d] != secondShape[d] && d != concatDimension) {
        return emitOpError(llvm::formatv(
            "shapes of operand ({0}) and ({1}) do not match at non-concat "
            "index: ({2}) != ({3}) at non-concat index {4}",
            firstRankedIndex, i,
            llvm::make_range(firstShape.begin(), firstShape.end()),
            llvm::make_range(secondShape.begin(), secondShape.end()), d));
      }
    }
  }
  return success();
}

LogicalResult ConcatenateOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  ConcatenateOp::Adaptor adaptor(operands);
  auto inputs = adaptor.getVal();

  auto operandType = inputs[0].getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  SmallVector<SmallVector<Value, 4>, 4> allShapeValues;
  for (size_t inputId = 0; inputId < inputs.size(); ++inputId) {
    Value operand = inputs[inputId];
    auto operandType = operand.getType().dyn_cast<RankedTensorType>();
    if (!operandType) return failure();

    SmallVector<Value, 4> shapeVals;
    for (const auto& element : llvm::enumerate(operandType.getShape())) {
      Value valueDim = toShapeScalarType(
          builder.create<tensor::DimOp>(loc, operand, element.index()));
      shapeVals.push_back(valueDim);
    }
    allShapeValues.emplace_back(std::move(shapeVals));
  }

  int axis = this->getDimension();
  auto& shapeValues = allShapeValues[0];
  for (size_t vecId = 1; vecId < allShapeValues.size(); ++vecId) {
    auto& otherShapeValues = allShapeValues[vecId];
    if (otherShapeValues.size() != shapeValues.size()) {
      this->emitOpError()
          << "Concatenate expects all operands must be of the same rank";
      return failure();
    }
    shapeValues[axis] = builder.create<arith::AddIOp>(loc, shapeValues[axis],
                                                      otherShapeValues[axis]);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

//===----------------------------------------------------------------------===//
// DynamicReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicReshapeOp::verify() {
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();
  auto outputShapeType =
      getOutputShape().getType().dyn_cast<RankedTensorType>();
  if (resultType && outputShapeType && outputShapeType.hasStaticShape() &&
      outputShapeType.getDimSize(0) != resultType.getRank()) {
    return emitError() << "output should have a rank equal to the number of "
                          "elements in output_shape";
  }
  return success();
}

LogicalResult DynamicReshapeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicReshapeOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputShape()));
  return success();
}

namespace {
class DynamicReshapeOpNotActuallyDynamic
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto type = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!type || !type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires static shape tensor");
    }
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getOperand());
    return success();
  }
};

// Canonicalizes
// %0 = some_op(%tensor)
// %1 = "mhlo.dynamic_reshape"(%0, %shape)
//      (tensor<?xT>, tensor<1xindex>) -> tensor<?xT>
// ... uses of %1.
//
// into
//
// ... uses of %0.
// This canonicalization is only correct if the input is correct!
// TODO(b/178779691): Use a more sophisticated canonicalization that preserves
// errors in input, and still allows us to get rid of redundant reshapes.
class RemoveRedundantRank1DynamicReshape
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto type = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!type || type.getRank() != 1 || type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "requires rank 1 shape tensor with dynamic dimension");
    }
    auto operandType = op.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!operandType || operandType.getRank() != 1 ||
        operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "requires rank 1 shape tensor with dynamic dimension");
    }
    rewriter.replaceOp(op, {op.getOperand()});
    return success();
  }
};

// Canonicalizes
// %0 = "mhlo.dynamic_reshape"(%tensor, %shape)
// %1 = same_operands_and_result_shape_op(%tensor)
// %2 = "mhlo.dynamic_reshape"(%1, %shape)
// ... uses of %2.
//
// into
//
// %0 = "mhlo.dynamic_reshape"(%tensor, %shape)
// %1 = same_operands_and_result_shape_op(%tensor)
// ... uses of %1.
class DynamicReshapeOpSameShapeOpResult
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* defOp = op.getOperand().getDefiningOp();
    if (!defOp ||
        !defOp->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
      return failure();
    }
    Operation* inputDefOp = defOp->getOperand(0).getDefiningOp();
    if (!inputDefOp) {
      return failure();
    }
    auto reshape = dyn_cast<DynamicReshapeOp>(*inputDefOp);
    if (reshape && reshape.getOutputShape() == op.getOutputShape()) {
      rewriter.replaceOp(op, {defOp->getResult(0)});
      return success();
    }
    return failure();
  }
};
}  // namespace

void DynamicReshapeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                   MLIRContext* context) {
  // clang-format off
  results.add<
      DynamicReshapeOpNotActuallyDynamic,
      DynamicReshapeOpSameShapeOpResult,
      RemoveRedundantDynamicBroadcast,
      RemoveRedundantDynamicReshape,
      RemoveRedundantRank1DynamicReshape,
      ShapeOfDynamicReshape
    >(context);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// DynamicSliceOp
//===----------------------------------------------------------------------===//

namespace {
// Canonicalizes DynamicSlice ops that can be replaced instead with Slice ops.
// This canonicalization is applied the case when the `begin` input values are
// compile time constants and thus can be made into a tensor.
struct DynamicSliceToSlice : public OpRewritePattern<DynamicSliceOp> {
  using OpRewritePattern<DynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicSliceOp dynamicSlice,
                                PatternRewriter& rewriter) const override {
    Value input = dynamicSlice.getOperand();
    auto inputTensor = input.getType().dyn_cast<RankedTensorType>();
    if (!inputTensor || !inputTensor.hasStaticShape()) return failure();

    auto sliceSizes = dynamicSlice.getSliceSizes().getValues<int64_t>();
    SmallVector<int64_t, 4> tempStartIndices;
    for (const auto& indexAndSliceStart :
         llvm::enumerate(dynamicSlice.getStartIndices())) {
      APInt val;
      Value start = indexAndSliceStart.value();
      int64_t index = indexAndSliceStart.index();
      if (!matchPattern(start, m_ConstantInt(&val))) {
        return failure();
      }
      // Clamp the indices within bounds to faithfully mirror dynamic slice
      // semantics.
      int64_t clampedStart =
          clamp(val.getSExtValue(), static_cast<int64_t>(0),
                inputTensor.getDimSize(index) - sliceSizes[index]);
      tempStartIndices.push_back(clampedStart);
    }

    // At this point we've determined that the start indices are all constants;
    // pack them into a single tensor.
    auto loc = dynamicSlice.getLoc();
    int64_t inputRank = inputTensor.getRank();
    auto sliceStartIndices = rewriter.getI64TensorAttr(tempStartIndices);
    DenseIntElementsAttr sliceLimits = buildSliceLimits(
        sliceStartIndices, dynamicSlice.getSliceSizes(), &rewriter);
    DenseIntElementsAttr sliceStrides =
        rewriter.getI64TensorAttr(SmallVector<int64_t, 4>(inputRank, 1));
    auto result = rewriter.create<SliceOp>(loc, input, sliceStartIndices,
                                           sliceLimits, sliceStrides);
    rewriter.replaceOp(dynamicSlice, {result});
    return success();
  }
};

}  // namespace

void DynamicSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                 MLIRContext* context) {
  results.add<DynamicSliceToSlice>(context);
}

// Verifies that the number of slice sizes and the number of start indices match
LogicalResult DynamicSliceOp::verify() {
  int numSliceSizes = getSliceSizes().getNumElements();
  int numStartIndices = getStartIndices().size();
  if (numStartIndices != numSliceSizes) {
    return emitOpError() << "has mismatched number of slice sizes ("
                         << numSliceSizes << ") and number of start indices ("
                         << numStartIndices << ")";
  }
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  if (operandType.getRank() != numStartIndices) {
    return emitOpError() << "has mismatched number of start indices ("
                         << numStartIndices << ") and the rank of operand ("
                         << operandType.getRank() << ")";
  }

  for (int i = 0; i < numSliceSizes; ++i) {
    int64_t sliceSize = getSliceSizes().getValues<int64_t>()[i];
    if (sliceSize < 0) {
      return emitOpError() << "has negative size index to dynamic slice: "
                           << sliceSize;
    }
    if (!operandType.isDynamicDim(i)) {
      int64_t dimSize = operandType.getDimSize(i);
      if (sliceSize > dimSize) {
        return emitOpError() << "has slice size " << sliceSize
                             << " greater than dimension size " << dimSize
                             << " in dimension " << i << " of operand";
      }
    }
  }
  return success();
}

LogicalResult DynamicSliceOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, regions);
  Value operand = adaptor.getOperand();
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  auto sliceSizes = adaptor.getSliceSizes();
  Type elementTy = operandType.getElementType();
  inferredReturnShapes.emplace_back(sliceSizes.getValues<int64_t>(), elementTy);
  return success();
}

//===----------------------------------------------------------------------===//
// RealDynamicSliceOp
//===----------------------------------------------------------------------===//
// Verifies that operand rank matches start_indices/limit_indices/strides size
LogicalResult RealDynamicSliceOp::verify() {
  auto inputType = getOperand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!inputType) return success();
  int inputRank = inputType.getRank();

  auto startType = getStartIndices().getType().cast<RankedTensorType>();
  auto limitType = getLimitIndices().getType().cast<RankedTensorType>();
  auto stridesType = getStrides().getType().cast<RankedTensorType>();

  if (inputRank != startType.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << inputRank << ") and start_indices size ("
                         << startType.getNumElements() << ")";
  }

  if (inputRank != limitType.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << inputRank << ") and limit_indices size ("
                         << limitType.getNumElements() << ")";
  }

  if (inputRank != stridesType.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << inputRank << ") and strides size ("
                         << stridesType.getNumElements() << ")";
  }

  return success();
}

namespace {
// Canonicalizes RealDynamicSlice ops that can be replaced instead with Slice
// ops. This canonicalization is applied the case when the `begin` input values
// are compile time constants and thus can be made into a tensor.
struct RealDynamicSliceIsStatic : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern<RealDynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RealDynamicSliceOp realDynamicSlice,
                                PatternRewriter& rewriter) const override {
    Location loc = realDynamicSlice.getLoc();
    Value input = realDynamicSlice.getOperand();
    Value output = realDynamicSlice.getResult();
    auto inputTy = input.getType().dyn_cast<RankedTensorType>();
    auto outputTy = output.getType().dyn_cast<RankedTensorType>();

    if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
        !outputTy.hasStaticShape()) {
      return failure();
    }

    int64_t inputRank = inputTy.getRank();

    auto startVal = realDynamicSlice.getStartIndices();
    auto limitVal = realDynamicSlice.getLimitIndices();
    auto strideVal = realDynamicSlice.getStrides();
    auto startOp = startVal.getDefiningOp<mlir::arith::ConstantOp>();
    auto limitOp = limitVal.getDefiningOp<mlir::arith::ConstantOp>();
    auto strideOp = strideVal.getDefiningOp<mlir::arith::ConstantOp>();
    if (!startOp || !limitOp || !strideOp) return failure();

    auto startAttr =
        startOp.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    auto limitAttr =
        limitOp.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    auto strideAttr =
        strideOp.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    if (!startAttr || !limitAttr || !strideAttr) return failure();

    SmallVector<int64_t, 4> tempStartIndices;
    SmallVector<int64_t, 4> tempLimitIndices;
    SmallVector<int64_t, 4> tempStride;
    for (int64_t dimIdx = 0; dimIdx < inputRank; dimIdx++) {
      int64_t start = startAttr.getValues<IntegerAttr>()[dimIdx].getInt();
      tempStartIndices.push_back(start);
      int64_t limit = limitAttr.getValues<IntegerAttr>()[dimIdx].getInt();
      tempLimitIndices.push_back(limit);
      int64_t end = strideAttr.getValues<IntegerAttr>()[dimIdx].getInt();
      tempStride.push_back(end);
    }

    DenseIntElementsAttr sliceStartIndices =
        rewriter.getI64TensorAttr(tempStartIndices);
    DenseIntElementsAttr sliceLimitIndices =
        rewriter.getI64TensorAttr(tempLimitIndices);
    DenseIntElementsAttr sliceStrides = rewriter.getI64TensorAttr(tempStride);
    auto result = rewriter.create<SliceOp>(loc, input, sliceStartIndices,
                                           sliceLimitIndices, sliceStrides);
    rewriter.replaceOp(realDynamicSlice, {result});
    return success();
  }
};
}  // namespace

void RealDynamicSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                     MLIRContext* context) {
  results.add<RealDynamicSliceIsStatic, RealDSliceToSlice>(context);
}

LogicalResult RealDynamicSliceOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  RealDynamicSliceOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();
  Value startIndices = adaptor.getStartIndices();
  Value limitIndices = adaptor.getLimitIndices();
  Value strides = adaptor.getStrides();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      startIndices.getType().cast<ShapedType>().getElementType();
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  one = maybeCastTo(builder, loc, one, shapeScalarType);
  for (const auto& element : llvm::enumerate(operandType.getShape())) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, element.index());
    Value valueStart =
        builder.create<tensor::ExtractOp>(loc, startIndices, offset);
    Value valueLimit =
        builder.create<tensor::ExtractOp>(loc, limitIndices, offset);
    Value valueStride = builder.create<tensor::ExtractOp>(loc, strides, offset);
    // size = (limit - start + stride - 1) / stride
    shapeValues.push_back(builder.create<arith::DivSIOp>(
        loc,
        builder.create<arith::SubIOp>(
            loc,
            builder.create<arith::AddIOp>(
                loc, valueStride,
                builder.create<arith::SubIOp>(loc, valueLimit, valueStart)),
            one),
        valueStride));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));
  return success();
}

//===----------------------------------------------------------------------===//
// InfeedOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `zero_or_more_type(s),
// mhlo::token`
LogicalResult InfeedOp::verify() {
  auto resultTypes = getResultTypes();
  if (resultTypes.empty())
    return emitOpError()
           << "result is expected to be at least of size 1, but got "
           << resultTypes.size();

  if (!resultTypes[resultTypes.size() - 1].isa<TokenType>())
    return emitOpError() << "last element of result types is expected to "
                            "be of token type, but got "
                         << resultTypes[resultTypes.size() - 1];

  // Verify layout attribute
  constexpr char kLayoutAttr[] = "layout";
  if (!getOperation()->hasAttr(kLayoutAttr)) return success();

  mlir::ArrayAttr layout =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(kLayoutAttr);
  if (!layout)
    return emitOpError() << "layout-attribute expected to be of array-type.";

  if (layout.size() != resultTypes.size() - 1) {
    return emitOpError() << "layout-attribute size must be "
                         << resultTypes.size() - 1
                         << " (which is the number of "
                            "op-results - 1 (for token result)), but got "
                         << layout.size();
  }

  for (auto childLayout : layout) {
    mlir::ArrayAttr childLayoutArr = childLayout.dyn_cast<mlir::ArrayAttr>();
    if (!childLayoutArr) {
      return emitOpError() << "layout-attribute expected to have "
                              "elements of type array, but got "
                           << childLayout;
    }

    for (auto i : childLayoutArr) {
      mlir::IntegerAttr attr = i.dyn_cast<mlir::IntegerAttr>();
      if (!attr) {
        return emitOpError() << "layout-attribute's leaf elements are "
                                "expected to be of type integer, but got "
                             << i;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

LogicalResult MapOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  MapOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferMapOp(location, adaptor.getInputs(), adaptor.getDimensions(),
                         adaptor.getComputation(), inferredReturnShapes);
}

OpFoldResult MapOp::fold(ArrayRef<Attribute> operands) {
  mlir::Block& bb = getComputation().front();
  mlir::Operation& frontOp = bb.front();

  auto retOp = mlir::dyn_cast<ReturnOp>(frontOp);
  if (!retOp) return nullptr;
  if (retOp.getResults().size() != 1) return nullptr;

  for (mlir::BlockArgument barg : bb.getArguments()) {
    if (barg == retOp.getResults()[0])
      return getOperands()[barg.getArgNumber()];
  }
  return nullptr;
}

LogicalResult MapOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `zero_or_more_type(s),
// mhlo::token`
LogicalResult RecvOp::verify() {
  auto resultTypes = getResultTypes();
  if (resultTypes.empty())
    return emitOpError()
           << "result is expected to be at least of size 1, but got "
           << resultTypes.size();
  if (!resultTypes[resultTypes.size() - 1].isa<TokenType>())
    return emitOpError() << "last element of result types is expected to "
                            "be of token type, but got "
                         << resultTypes[resultTypes.size() - 1];
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

OpFoldResult CopyOp::fold(ArrayRef<Attribute> operands) { return getOperand(); }

//===----------------------------------------------------------------------===//
// ReduceWindowOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceWindowOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ReduceWindowOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferReduceWindowOp(
      location, adaptor.getInputs(), adaptor.getInitValues(),
      adaptor.getWindowDimensions(), adaptor.getWindowStrides(),
      adaptor.getBaseDilations(), adaptor.getWindowDilations(),
      adaptor.getPadding(), adaptor.getBody(), inferredReturnShapes);
}

// Get the operation used for reduction applied to `result_index`th result. Its
// expected to be a binary operation that consumes `result_index`th and
// `result_index + getInputs().size`th arguments of the body.
Operation* ReduceWindowOp::getReductionOp(int resultIndex) {
  auto returnOp = cast<ReturnOp>(getBody().front().getTerminator());
  Operation* computeOp = returnOp.getResults()[resultIndex].getDefiningOp();
  if (computeOp->getNumOperands() != 2) return nullptr;
  auto arg0 = computeOp->getOperand(0).dyn_cast<BlockArgument>();
  auto arg1 = computeOp->getOperand(1).dyn_cast<BlockArgument>();
  if (!arg0 || !arg1) return nullptr;
  int64_t arg0Num = arg0.getArgNumber();
  int64_t arg1Num = arg1.getArgNumber();
  int64_t otherArgIndex = resultIndex + getInputs().size();
  if (arg0Num == resultIndex && arg1Num == otherArgIndex) return computeOp;
  if (arg0Num == otherArgIndex && arg1Num == resultIndex &&
      computeOp->hasTrait<mlir::OpTrait::IsCommutative>())
    return computeOp;
  return nullptr;
}

bool isSplatZero(SplatElementsAttr attr) {
  if (!attr) return false;
  if (attr.getElementType().isa<FloatType>()) {
    return attr.getSplatValue<APFloat>().isZero();
  }
  if (attr.getElementType().isa<IntegerType>()) {
    return attr.getSplatValue<APInt>().isZero();
  }
  return false;
}

LogicalResult ReduceWindowOp::fold(ArrayRef<Attribute> operands,
                                   SmallVectorImpl<OpFoldResult>& results) {
  const auto emptyOrAllEq = [](const Optional<DenseIntElementsAttr> opt,
                               const int64_t n) {
    return !opt.has_value() ||
           (opt->isSplat() && opt->getSplatValue<IntegerAttr>().getInt() == n);
  };
  const auto isSumReductionBody = [](mlir::Region& body) {
    if (body.getNumArguments() != 2) return false;
    auto returnOp = dyn_cast_or_null<ReturnOp>(body.back().getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1) return false;
    auto addOp = returnOp.getOperand(0).getDefiningOp<AddOp>();
    if (!addOp) return false;
    return (addOp.getLhs() == body.getArgument(0) &&
            addOp.getRhs() == body.getArgument(1)) ||
           (addOp.getLhs() == body.getArgument(1) &&
            addOp.getRhs() == body.getArgument(0));
  };

  // Fold no-op single input sum reduction.
  if (getInputs().size() == 1 &&
      isSplatZero(operands[1].dyn_cast_or_null<SplatElementsAttr>()) &&
      emptyOrAllEq(getWindowDimensionsAttr(), 1) &&
      emptyOrAllEq(getWindowStrides(), 1) &&
      emptyOrAllEq(getBaseDilations(), 1) &&
      emptyOrAllEq(getWindowDilations(), 1) && emptyOrAllEq(getPadding(), 0) &&
      isSumReductionBody(getBody())) {
    results.push_back(getInputs()[0]);
    return success();
  }

  return failure();
}

// Builder that takes a constructor for its region and infers result types
void ReduceWindowOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, ValueRange inputs,
    ValueRange init_values, DenseIntElementsAttr window_dimensions,
    /*optional*/ DenseIntElementsAttr window_strides,
    /*optional*/ DenseIntElementsAttr base_dilations,
    /*optional*/ DenseIntElementsAttr window_dilations,
    /*optional*/ DenseIntElementsAttr padding,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuilder) {
  odsState.addOperands(inputs);
  odsState.addOperands(init_values);
  odsState.addAttribute(getWindowDimensionsAttrName(odsState.name),
                        window_dimensions);
  if (window_strides) {
    odsState.addAttribute(getWindowStridesAttrName(odsState.name),
                          window_strides);
  }
  if (base_dilations) {
    odsState.addAttribute(getBaseDilationsAttrName(odsState.name),
                          base_dilations);
  }
  if (window_dilations) {
    odsState.addAttribute(getWindowDilationsAttrName(odsState.name),
                          window_dilations);
  }
  if (padding) {
    odsState.addAttribute(getPaddingAttrName(odsState.name), padding);
  }
  Region* region = odsState.addRegion();

  llvm::SmallVector<Type> blockArgTypes;
  llvm::SmallVector<Location> locs;
  auto numValues = inputs.size() + init_values.size();
  blockArgTypes.reserve(numValues);
  locs.reserve(numValues);
  for (auto i : inputs) {
    auto iType = i.getType().cast<ShapedType>();
    blockArgTypes.push_back(iType.cloneWith(
        llvm::makeArrayRef<int64_t>(std::nullopt), iType.getElementType()));
    locs.push_back(i.getLoc());
  }
  for (auto i : init_values) {
    auto iType = i.getType().cast<ShapedType>();
    blockArgTypes.push_back(iType.cloneWith(
        llvm::makeArrayRef<int64_t>(std::nullopt), iType.getElementType()));
    locs.push_back(i.getLoc());
  }

  {
    OpBuilder::InsertionGuard g(odsBuilder);
    Block* body =
        odsBuilder.createBlock(region, /*insertPt=*/{}, blockArgTypes, locs);
    bodyBuilder(odsBuilder, odsState.location, body->getArguments());
  }

  llvm::SmallVector<mlir::Type, 2> inferredReturnTypes;
  if (mlir::succeeded(ReduceWindowOp::inferReturnTypes(
          odsBuilder.getContext(), odsState.location, odsState.operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes)))
    odsState.addTypes(inferredReturnTypes);
  else
    llvm::report_fatal_error("Failed to infer result type(s).");
}

//===----------------------------------------------------------------------===//
// ReducePrecisionOp
//===----------------------------------------------------------------------===//

// The following property is already enforced by the ODS:
//  P0. operand element type is float
//  P1. mantissa_bits >= 0
// We intend to verify the following properties
//  P2. exponent_bits >= 1
LogicalResult ReducePrecisionOp::verify() {
  if (getExponentBits() < 1) {
    return emitOpError() << "exponent_bits must be at least 1.";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

template <typename T>
static Attribute foldReverseHelper(DenseElementsAttr& attr, ShapedType& type,
                                   DenseIntElementsAttr& dims) {
  int64_t numElements = attr.getNumElements();
  // No-op if the tensor has 0 elements.
  // No-op if the result of folding is too large.
  if (numElements == 0 || numElements > kFoldOpEltLimit) return {};

  SmallVector<T> result(attr.getValues<T>().begin(), attr.getValues<T>().end());

  size_t rank = type.getRank();
  SmallVector<int64_t> stride(rank + 1, numElements);
  for (size_t i = 0; i < rank; i++) {
    if (type.getDimSize(i) == 0) return {};
    stride[i + 1] = stride[i] / type.getDimSize(i);
  }

  for (auto dim : dims.getValues<int64_t>()) {
    // For example, given:
    //   * tensor: tensor<2x3x2xi32>
    //     [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9,10], [11, 12]]]
    //   * dim: [1]
    //
    // We're going to reverse the tensor with respect to dim as follows:
    //   1) Split the tensor into blocks, i.e. smaller tensors whose type is
    //   derived from the tensor by dropping the first `dim` dimensions, i.e.
    //   tensor<3x2xi32> for the running example.
    //   2) Split each block into windows, i.e. even smaller tensors whose type
    //   is derived from the block by dropping the first dimension of the
    //   block, i.e. tensor<2xi32> for the running example.
    //   3) Within each block, swap windows but don't change the order of
    //   elements within the windows: 0th window goes to N-1st spot, 1st window
    //   goes to N-2nd spot etc.
    //
    // For the running example, the result will be:
    //   [[[5, 6], [3, 4], [1, 2]], [[11, 12], [9, 10], [7, 8]]].
    //
    // Note how elements within windows haven't changed their order with respect
    // to each other and how blocks haven't changed their order with respect to
    // each other.
    int64_t numWindows = type.getDimSize(dim);
    int64_t windowSize = stride[dim] / numWindows;

    for (int64_t index = 0; index < numElements; index++) {
      int64_t blockNumber = index / stride[dim];
      int64_t windowNumber = (index % stride[dim]) / windowSize;
      int64_t reversedWindowNumber = numWindows - windowNumber - 1;
      if (windowNumber >= reversedWindowNumber) continue;
      int64_t reversedIndex = blockNumber * stride[dim] +
                              reversedWindowNumber * windowSize +
                              index % windowSize;
      std::swap(result[index], result[reversedIndex]);
    }
  }
  return DenseElementsAttr::get(type, result);
}

OpFoldResult ReverseOp::fold(ArrayRef<Attribute> operands) {
  Value input = getOperand();

  // No dimensions to reverse.
  DenseIntElementsAttr dims = getDimensions();
  if (dims.getNumElements() == 0) return input;

  // If size of all dimensions to reverse equals 1, then the reverse is a no-op.
  // Eg. Reverse dimensions {0,1} of a 1x1x2 tensor
  auto shapedType = input.getType().cast<ShapedType>();
  if (llvm::all_of(dims.getValues<int64_t>(), [&](int64_t dim) {
        return shapedType.getDimSize(dim) == 1;
      }))
    return input;

  // If the operand is a static shaped tensor of constants, return reversed
  // tensor
  DenseElementsAttr inputAttr =
      operands.begin()->dyn_cast_or_null<DenseElementsAttr>();
  if (inputAttr && shapedType.hasStaticShape()) {
    auto etype = shapedType.getElementType();
    if (etype.isa<IntegerType>())
      return foldReverseHelper<APInt>(inputAttr, shapedType, dims);
    if (etype.isa<FloatType>())
      return foldReverseHelper<APFloat>(inputAttr, shapedType, dims);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

// Returns the result type after reducing operand of the given type across the
// specified dimensions.
static TensorType getReduceResultType(Type operandTy,
                                      DenseIntElementsAttr dimensions) {
  Type elementTy = getElementTypeOrSelf(operandTy);

  auto rankedTy = operandTy.dyn_cast<RankedTensorType>();
  if (!rankedTy) return UnrankedTensorType::get(elementTy);

  int64_t rank = rankedTy.getRank();
  llvm::SmallVector<bool, 4> dimsMask(rank, false);
  for (int64_t dim : dimensions.getValues<int64_t>()) dimsMask[dim] = true;

  SmallVector<int64_t, 4> shape;
  for (int64_t i = 0; i < rank; ++i) {
    if (!dimsMask[i]) shape.push_back(rankedTy.getDimSize(i));
  }

  return RankedTensorType::get(shape, elementTy);
}

LogicalResult ReduceOp::fold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<OpFoldResult>& results) {
  // No dimensions to reduce.
  if (getDimensions().getNumElements() == 0) {
    for (Value operand : this->getInputs()) {
      results.push_back(operand);
    }
    return success();
  }

  // If all returned values in the ReduceOp region exists outside
  // the region replace the ReduceOp with those values.
  mlir::Block& bb = this->getBody().front();
  SmallVector<Value> replacedResults;
  if (auto retOp = mlir::dyn_cast<ReturnOp>(bb.back())) {
    for (Value result : retOp.getResults()) {
      if (result.getParentRegion() == retOp->getParentRegion())
        return failure();
      replacedResults.push_back(result);
    }

    results.insert(results.end(), replacedResults.begin(),
                   replacedResults.end());
    return success();
  }

  return failure();
}

bool hasSameOperandAndResultTypes(Operation& op) {
  Type expected;
  if (op.getNumResults() != 0) expected = op.getResult(0).getType();
  if (op.getNumOperands() != 0) expected = op.getOperand(0).getType();
  if (!expected) return false;

  auto typeMatch = [&](Type actual) { return actual == expected; };
  return llvm::all_of(op.getOperandTypes(), typeMatch) &&
         llvm::all_of(op.getResultTypes(), typeMatch);
}

// Checks the following eligibility criteria for compact printing of
// mhlo.reduce:
// E1. The reduce-op wraps a single inner-op in the associated region.
// E2. The single operation is a commutative binary-op from mhlo dialect, zero
//     region, producing single result such that the operands and result all
//     have the same type.
// E3. The reduce-op consist of at least one input-operand; The operand-types of
//     inner-op should be derived trivially from the element-type of reduce-op's
//     first input-operand.
// E4. The  arguments of the region's only basic block are forwarded perfectly
//     to inner-op's operands.
// E5. The reduce-op, inner-op, blocks arguments, and the return-op all have the
//     same location.
// E6. The single operation result is perfectly forwarded to the reduce op
//     return.
static bool isEligibleForCompactPrint(ReduceOp op) {
  // Check E1.
  auto& block = op.getBody().front();
  if (!hasSingleElement(block.without_terminator())) return false;

  Operation& innerOp = *block.begin();

  // Check E2.
  if (innerOp.getDialect() != op->getDialect()) return false;

  if (innerOp.getNumOperands() != 2 ||
      !innerOp.hasTrait<mlir::OpTrait::OneResult>() ||
      !hasSameOperandAndResultTypes(innerOp) ||
      !innerOp.hasTrait<mlir::OpTrait::IsCommutative>() ||
      !innerOp.hasTrait<mlir::OpTrait::ZeroRegions>())
    return false;

  // Check E3.
  if (op.getInputs().empty()) return false;

  auto elemType =
      op.getInputs()[0].getType().cast<TensorType>().getElementType();
  auto expectedInnerOpType = RankedTensorType::get(/*shape=*/{}, elemType);
  if (innerOp.getOperands()[0].getType() != expectedInnerOpType) return false;

  // Check E4.
  if (!llvm::equal(block.getArguments(), innerOp.getOperands())) return false;

  // Check E5.
  auto retOp = dyn_cast<ReturnOp>(block.getTerminator());
  if (!retOp) return false;

  auto blockArgLoc = block.getArgument(0).getLoc();
  if (blockArgLoc != block.getArgument(1).getLoc()) return false;

  if (innerOp.getLoc() != op.getLoc() || retOp.getLoc() != op.getLoc() ||
      blockArgLoc != op.getLoc())
    return false;

  // Check E6.
  return llvm::equal(innerOp.getResults(), retOp.getOperands());
}

void ReduceOp::print(OpAsmPrinter& p) {
  {
    // Print the pairs of operands under the form:
    //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
    StringRef comma = "";
    int numOperandPairs = getNumOperands() / 2;
    for (int opId : llvm::seq<int>(0, numOperandPairs)) {
      p << comma << "(" << getOperand(opId)
        << " init: " << getOperand(opId + numOperandPairs) << ")";
      comma = ", ";
    }
  }

  // If the reduce-op is eligible for compact printing, we emit the one-liner:
  //  mhlo.reduce applies <inner-op> across dimensions = [...] : <func-type>
  // Note: We are not printing the function type of reduction operation. We
  // have some simplifying assumptions (refer to IsEligibleForCompactPrint::E3)
  // to derive the type from that of reduce-op.
  if (isEligibleForCompactPrint(*this)) {
    Operation& innerOp = getBody().front().front();
    p << " applies ";
    printEscapedString(innerOp.getName().getStringRef(), p.getStream());

    p << " across dimensions = [";
    llvm::interleaveComma(getDimensions().getValues<int64_t>(), p);
    p << "]";
    p << " : ";
    p.printFunctionalType(*this);
  } else {
    p << " across dimensions = [";
    llvm::interleaveComma(getDimensions().getValues<int64_t>(), p);
    p << "]";
    p.printOptionalAttrDict(getOperation()->getAttrs(), {"dimensions"});
    p << " : ";
    p.printFunctionalType(*this);
    p.printNewline();
    p << " reducer";
    {
      // Print the pairs of block operands under the form:
      //   (%arg0_elt, %arg0_acc) (%arg1_elt, %arg1_acc):
      Block& reducer = getBody().front();
      int numOperandPairs = getNumOperands() / 2;
      for (int opId : llvm::seq<int>(0, numOperandPairs)) {
        p << "(";
        p.printRegionArgument(reducer.getArgument(opId));
        p << ", ";
        p.printRegionArgument(reducer.getArgument(opId + numOperandPairs));
        p << ") ";
      }
    }
    p << ' ';
    p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  }
}

ParseResult ReduceOp::parse(OpAsmParser& parser, OperationState& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Location currLocation = parser.getEncodedSourceLoc(loc);

  // Parse the operands of reduce-op, this is a list of pair under the form:
  //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
  // Each input to reduce is paired with its init value, even though in memory
  // they are stored with the input first and the init values after.
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> initOperands;
  do {
    (void)parser.parseOptionalComma();
    if (parser.parseOptionalLParen()) break;
    OpAsmParser::UnresolvedOperand operand, initOperand;
    if (parser.parseOperand(operand) || parser.parseKeyword("init") ||
        parser.parseColon() || parser.parseOperand(initOperand) ||
        parser.parseRParen())
      return failure();
    operands.push_back(operand);
    initOperands.push_back(initOperand);
  } while (true);
  operands.append(initOperands);

  // Check if we are parsing the compact version of reduce-op:
  //  mhlo.reduce applies <inner-op> across dimensions = [...] : <func-type>
  // else parse the "region-based" variant.
  if (failed(parser.parseOptionalKeyword("applies"))) {
    // Parse the inner-op dimensions, reduce-op's function-type and
    // optional location.
    SmallVector<int64_t> dimensions;
    auto parseDim = [&]() -> ParseResult {
      if (parser.parseInteger(dimensions.emplace_back())) return failure();
      return success();
    };

    FunctionType reduceOpFntype;
    if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
        parser.parseEqual() ||
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       parseDim) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType(reduceOpFntype) ||
        parser.parseKeyword("reducer"))
      return failure();
    OpBuilder builder(parser.getBuilder().getContext());
    result.addAttribute("dimensions", builder.getI64TensorAttr(dimensions));

    // Parse the "reducer" region now.
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerOperands;
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerInitOperands;
    SmallVector<Type, 2> reducerTypes;
    SmallVector<Type, 2> reducerInitTypes;
    SmallVector<Optional<Location>, 2> reducerLocs;
    SmallVector<Optional<Location>, 2> reducerInitLocs;
    auto parseBlockOperand =
        [&](SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
            SmallVectorImpl<Type>& types,
            SmallVectorImpl<Optional<Location>>& locs) -> ParseResult {
      OpAsmParser::UnresolvedOperand operand;
      Type type;
      Optional<Location> loc;
      if (parser.parseOperand(operand, /*allowResultNumber=*/false) ||
          parser.parseColon() || parser.parseType(type) ||
          parser.parseOptionalLocationSpecifier(loc))
        return failure();
      operands.push_back(operand);
      types.push_back(type);
      locs.push_back(loc);
      return success();
    };
    do {
      if (failed(parser.parseOptionalLParen())) break;
      if (parseBlockOperand(reducerOperands, reducerTypes, reducerLocs) ||
          parser.parseComma() ||
          parseBlockOperand(reducerInitOperands, reducerInitTypes,
                            reducerInitLocs) ||
          parser.parseRParen())
        return failure();
    } while (true);
    reducerOperands.append(reducerInitOperands);
    reducerTypes.append(reducerInitTypes);
    reducerLocs.append(reducerInitLocs);
    result.addTypes(reduceOpFntype.getResults());
    SmallVector<OpAsmParser::Argument> reducerArgs;
    createArgs(reducerOperands, reducerTypes, reducerArgs);

    // Derive the SSA-values for reduce-op's operands and parse the region, and
    // the optional trailing location.
    Optional<Location> trailingLoc;
    if (parser.resolveOperands(operands, reduceOpFntype.getInputs(), loc,
                               result.operands) ||
        parser.parseRegion(*result.addRegion(), reducerArgs))
      return failure();
    // Set the individual block arguments.
    for (auto argAndLoc :
         llvm::zip(result.regions.front()->front().getArguments(), reducerLocs))
      if (std::get<1>(argAndLoc))
        std::get<0>(argAndLoc).setLoc(std::get<1>(argAndLoc).value());
    result.location = trailingLoc.value_or(currLocation);
    return success();
  }

  // Parse the inner-op name and check if the contract on inner-op
  // mentioned in "isEligibleForCompactPrint::E2" for pretty-priting is met.
  FailureOr<OperationName> innerOpNameInfo = parser.parseCustomOperationName();
  if (failed(innerOpNameInfo)) return failure();

  StringRef innerOpName = innerOpNameInfo->getStringRef();
  Dialect* innerOpDialect = innerOpNameInfo->getDialect();
  if (!innerOpDialect || !innerOpDialect->getNamespace().equals("mhlo") ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::NOperands<2>::Impl>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::OneResult>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::IsCommutative>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::ZeroRegions>()) {
    parser.emitError(loc,
                     "expected the inner-op to be a commutative binary-op from "
                     "mhlo dialect, zero region, producing single result");
    return failure();
  }

  // Parse the inner-op dimensions, reduce-op's function-type and
  // optional location.
  SmallVector<int64_t> dimensions;
  auto parseDim = [&]() -> ParseResult {
    if (parser.parseInteger(dimensions.emplace_back())) return failure();
    return success();
  };

  Optional<Location> explicitLoc;
  FunctionType reduceOpFntype;
  if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
      parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseDim) ||
      parser.parseColon() || parser.parseType(reduceOpFntype) ||
      parser.parseOptionalLocationSpecifier(explicitLoc))
    return failure();

  if (!reduceOpFntype || reduceOpFntype.getInputs().empty()) {
    if (!reduceOpFntype) return parser.emitError(loc, "expected function type");
    return parser.emitError(loc,
                            "input types missing in reduce-op function type");
  }

  // If location of reduce-op is explicitly provided, then use it; Else use
  // the parser's current location.
  Location reduceOpLoc = explicitLoc.value_or(currLocation);

  // Derive the SSA-values for reduce-op's operands.
  if (parser.resolveOperands(operands, reduceOpFntype.getInputs(), loc,
                             result.operands))
    return failure();

  // Derive the type of inner-op from that of reduce-op's input operand.
  auto innerOpType = RankedTensorType::get(
      /*shape=*/{}, getElementTypeOrSelf(reduceOpFntype.getInput(0)));

  // Add a region for reduce-op.
  Region& region = *result.addRegion();

  // Create a basic-block inside reduce-op's region.
  Block& block = region.emplaceBlock();
  auto lhs = block.addArgument(innerOpType, reduceOpLoc);
  auto rhs = block.addArgument(innerOpType, reduceOpLoc);

  // Create and insert an "inner-op" operation in the block.
  OpBuilder builder(parser.getBuilder().getContext());
  builder.setInsertionPointToStart(&block);

  OperationState innerOpState(reduceOpLoc, innerOpName);
  innerOpState.operands.push_back(lhs);
  innerOpState.operands.push_back(rhs);
  innerOpState.addTypes(innerOpType);

  Operation* innerOp = builder.create(innerOpState);

  // Insert a return statement in the block returning the inner-op's result.
  builder.create<ReturnOp>(innerOp->getLoc(), innerOp->getResults());

  // Populate the reduce-op operation-state with result-type, location, and
  // dimension attribute.
  result.addTypes(reduceOpFntype.getResults());
  result.location = innerOp->getLoc();
  result.addAttribute("dimensions", builder.getI64TensorAttr(dimensions));

  return success();
}

LogicalResult ReduceOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location>, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ReduceOp::Adaptor adaptor(operands, attributes, regions);
  for (auto input : adaptor.getInputs()) {
    ShapedType outputType =
        getReduceResultType(input.getType(), adaptor.getDimensions());
    inferredReturnShapes.emplace_back(outputType);
  }
  return success();
}

LogicalResult ReduceOp::verify() {
  SmallVector<ShapedTypeComponents> unusedReturnShapes;
  return hlo::inferReduceOp(getLoc(), getInputs(), getInitValues(),
                            getDimensions(), getBody(), unusedReturnShapes);
}

// Enable constant folding to occur within the region of the ReduceOp
// by replacing block argument uses with constants if:
//  1. All the ReduceOp operands are splat constants.
//  2. The ReduceOp region consists of a single logical AND or logical OR.
// The pattern leverages the idempotent property of the AND and OR operators
// to determine the value of a reduction on splat constants. Other boolean
// operators do not have this property, and need separate patterns to resolve
// reductions of their splat constants.
struct LowerBoolSplatConstantsIntoRegion : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    mlir::Block& bb = op.getBody().front();

    // Ensure only a compute op and return op exist and the
    // compute op is an AND or OR op.
    if (bb.getOperations().size() != 2) return failure();
    if (!mlir::isa<mhlo::AndOp, mhlo::OrOp>(bb.front())) return failure();

    // Ensure all operands are splat constants.
    SmallVector<DenseElementsAttr, 4> bargCstAttrs;
    for (auto inpAndBarg : llvm::zip(op.getOperands(), bb.getArguments())) {
      Value inp = std::get<0>(inpAndBarg);
      BlockArgument barg = std::get<1>(inpAndBarg);
      ConstantOp cst = inp.getDefiningOp<ConstantOp>();
      if (!cst) return failure();

      auto cstAttr = cst.getValue().dyn_cast_or_null<DenseElementsAttr>();
      if (!cstAttr.isSplat()) {
        return rewriter.notifyMatchFailure(op, "Must be splat constant.");
      }

      auto bargShapedType = barg.getType().dyn_cast<ShapedType>();
      if (!bargShapedType) return failure();

      auto bargCstAttr = DenseElementsAttr::get(
          bargShapedType, cstAttr.getSplatValue<mlir::Attribute>());
      bargCstAttrs.push_back(bargCstAttr);
    }

    // Create new splat constants to replace block arguments.
    for (BlockArgument barg : bb.getArguments()) {
      int argIdx = barg.getArgNumber();
      mhlo::ConstantOp newCst = rewriter.create<mhlo::ConstantOp>(
          bb.front().getLoc(), barg.getType(), bargCstAttrs[argIdx]);
      barg.replaceAllUsesWith(newCst);
    }
    return success();
  }
};

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<LowerBoolSplatConstantsIntoRegion>(context);
}

LogicalResult ReduceOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  ReduceOp::Adaptor adaptor(operands);
  auto inputs = adaptor.getInputs();

  auto operandType = inputs[0].getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  SmallVector<int64_t, 4> dimensions(
      this->getDimensions().getValues<int64_t>());
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto& element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto* it = std::find(dimensions.begin(), dimensions.end(), idx);
    if (it != dimensions.end()) {
      continue;
    }
    Value valueDim = toShapeScalarType(
        builder.create<tensor::DimOp>(loc, inputs[0], element.index()));
    shapeValues.push_back(valueDim);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  for (size_t i = 0; i < inputs.size(); ++i) {
    reifiedReturnShapes.push_back(outputShape);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RngBitGeneratorOp
//===----------------------------------------------------------------------===//

// Verify that input state has the same shape as output shape
LogicalResult RngBitGeneratorOp::verify() {
  auto initialShape = getInitialState().getType().dyn_cast<RankedTensorType>();
  auto outputShape = getOutputState().getType().dyn_cast<RankedTensorType>();
  if (initialShape.getShape() != outputShape.getShape())
    return emitOpError()
           << "output state shape must match initial state shape. Got: "
           << initialShape << " and " << outputShape;
  return success();
}

//===----------------------------------------------------------------------===//
// RngOp
//===----------------------------------------------------------------------===//

LogicalResult RngOp::verify() {
  auto dist = getRngDistribution();
  if (dist == RngDistribution::UNIFORM) {
    return success();
  }
  auto muTy = getA().getType().cast<TensorType>().getElementType();
  auto sigmaTy = getB().getType().cast<TensorType>().getElementType();
  if (muTy.isa<FloatType>() && sigmaTy.isa<FloatType>()) {
    return success();
  }
  return emitOpError() << "mu and sigma must be floats";
}

LogicalResult RngOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  return rngInferReturnTypeComponents(context, location, operands, attributes,
                                      regions, inferredReturnShapes);
}

LogicalResult RngOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  RngOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getShape()));
  return success();
}

//===----------------------------------------------------------------------===//
// XlaRngGetAndUpdateStateOp
//===----------------------------------------------------------------------===//

LogicalResult XlaRngGetAndUpdateStateOp::verify() {
  auto resultTy = getType().cast<RankedTensorType>();
  if (!resultTy) return emitOpError() << "Output is not ranked.";
  if (!resultTy.hasStaticShape())
    return emitOpError() << "Output is not statically shaped.";
  auto rank = resultTy.getRank();
  if (rank != 1)
    return emitOpError() << "Output is of rank " << rank << " instead of 1";
  auto extent = resultTy.getDimSize(0);
  if (extent != 2)
    return emitOpError() << "Output size is " << extent << " instead of 2";

  return success();
}

LogicalResult XlaRngGetAndUpdateStateOp::inferReturnTypes(
    MLIRContext* ctx, Optional<Location>, ValueRange, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      {2}, mlir::IntegerType::get(ctx, 64, IntegerType::Unsigned)));
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  // The operands 'on_true' and 'on_false' should have compatible types, i.e.,
  //   (a) have the same element type, and
  //   (b) have compatible shapes (i.e. the same shape and/or at least one
  //       dynamic shape)
  if (!hlo::compatibleShapeAndElementType(getOnTrue().getType(),
                                          getOnFalse().getType()))
    return emitOpError()
           << "requires compatible types for non-predicate operands";

  // The predicate, if not-scalar, should have the same shape as the remaining
  // operands.
  auto predTy = getPred().getType().dyn_cast<RankedTensorType>();
  bool predMayBeScalar = !predTy || predTy.getRank() == 0;
  if (predMayBeScalar) return success();

  if (failed(verifyCompatibleShape(getPred().getType(), getOnTrue().getType())))
    return emitOpError() << "requires the same shape for all operands";

  return success();
}

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
  if (getOnTrue() == getOnFalse()) {
    return getOnTrue();
  }

  auto predicate = operands[0].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!predicate) {
    return {};
  }

  auto predicateTy = predicate.getType().cast<ShapedType>();
  if (!predicateTy.getElementType().isInteger(1)) {
    return {};
  }

  if (predicate.isSplat()) {
    return predicate.getSplatValue<APInt>().getBoolValue() ? getOnTrue()
                                                           : getOnFalse();
  }

  return {};
}

void SelectOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<FusePredNegIntoSelect, FuseBroadcastedPredNegIntoSelect>(context);
}

// Makes it such that a SelectOp that is a non-root operation in a DRR infers
// the return type based on operand type.
LogicalResult SelectOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SelectOp::Adaptor op(operands, attributes);
  auto trueType = op.getOnTrue().getType().cast<TensorType>();
  auto falseType = op.getOnFalse().getType().cast<TensorType>();

  // The output shape should be the most general of the operand shapes at each
  // dimension.
  ShapedTypeComponents& outputType = inferredReturnShapes.emplace_back();
  if (trueType == falseType || !trueType.hasRank()) {
    outputType = ShapedTypeComponents(trueType.cast<ShapedType>());
  } else if (!falseType.hasRank()) {
    outputType = ShapedTypeComponents(falseType.cast<ShapedType>());
  } else {
    assert(trueType.getRank() == falseType.getRank());
    llvm::SmallVector<int64_t, 4> dims;
    dims.reserve(trueType.getRank());
    for (auto dim : llvm::zip(trueType.getShape(), falseType.getShape())) {
      dims.push_back(std::get<0>(dim) == std::get<1>(dim)
                         ? std::get<0>(dim)
                         : ShapedType::kDynamic);
    }
    outputType = ShapedTypeComponents(dims, trueType.getElementType());
  }
  return success();
}

LogicalResult SelectOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  // For `hlo.select`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult SetDimensionSizeOp::verify() {
  if (auto size = this->getSize().getType().dyn_cast<RankedTensorType>()) {
    if (size.getRank() != 0)
      return emitOpError() << "size operand should be of rank-0";
  }

  return verifyDimAttr(*this);
}

OpFoldResult SetDimensionSizeOp::fold(ArrayRef<Attribute> operands) {
  DenseElementsAttr input = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  if (input) return input;

  DenseElementsAttr size = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  if (!size || !size.isSplat()) return {};

  auto ty = getType().dyn_cast<RankedTensorType>();
  if (!ty) return {};

  int64_t dimSize = ty.getDimSize(getDimension());
  if (dimSize == size.getSplatValue<IntegerAttr>().getInt())
    return getOperand();
  return {};
}

// TODO(b/238903565): Switch to inferReturnTypeComponents after adding support
// for the encoding upstream.
LogicalResult SetDimensionSizeOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  Location loc = location.value_or(UnknownLoc::get(context));

  SetDimensionSizeOp::Adaptor adaptor(operands, attributes, regions);
  if (failed(adaptor.verify(loc))) return failure();

  auto inputType = adaptor.getOperand().getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    inferredReturnTypes.push_back(adaptor.getOperand().getType());
    return success();
  }

  int64_t dim = adaptor.getDimension();
  int64_t rank = inputType.getRank();
  if (dim < 0 || dim >= rank) {
    return mlir::emitError(loc) << "expects dimension to be in range [0, "
                                << rank << "); got: [" << dim << "].";
  }

  auto shape = llvm::to_vector<4>(inputType.getShape());
  llvm::SmallVector<int64_t, 4> bounds(rank, ShapedType::kDynamic);
  if (auto encoding =
          inputType.getEncoding().dyn_cast_or_null<TypeExtensionsAttr>())
    bounds = llvm::to_vector<4>(encoding.getBounds());

  if (shape[dim] != ShapedType::kDynamic) bounds[dim] = shape[dim];
  shape[dim] = ShapedType::kDynamic;

  DenseIntElementsAttr sizeAttr;
  if (matchPattern(adaptor.getSize(), m_Constant(&sizeAttr))) {
    int64_t splat =
        sizeAttr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
    if (splat == bounds[dim]) {
      shape[dim] = splat;
      bounds[dim] = ShapedType::kDynamic;
    }
  }

  auto extensions = TypeExtensionsAttr::get(context, bounds);
  auto resultType =
      llvm::all_of(bounds, [](int64_t v) { return v == ShapedType::kDynamic; })
          ? RankedTensorType::get(shape, inputType.getElementType())
          : RankedTensorType::get(shape, inputType.getElementType(),
                                  extensions);
  inferredReturnTypes.push_back(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult PadOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  PadOp::Adaptor adaptor(operands, attributes, regions);
  auto inputType = adaptor.getOperand().getType().cast<RankedTensorType>();
  auto padType = adaptor.getPaddingValue().getType().cast<RankedTensorType>();

  if (padType.getRank() != 0) {
    return emitOptionalError(
        location, llvm::formatv("padding value type should be a rank-0 "
                                "tensor, is rank {0}",
                                padType.getRank()));
  }

  const auto& paddingLow = adaptor.getEdgePaddingLow();
  if (paddingLow.getType().getNumElements() != inputType.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "edge_padding_low length ({0}) must match operand rank ({1})",
            paddingLow.getType().getNumElements(), inputType.getRank()));
  }

  const auto& paddingHigh = adaptor.getEdgePaddingHigh();
  if (paddingHigh.getType().getNumElements() != inputType.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "edge_padding_high length ({0}) must match operand rank ({1})",
            paddingHigh.getType().getNumElements(), inputType.getRank()));
  }

  const auto& paddingInterior = adaptor.getInteriorPadding();
  if (paddingInterior.getType().getNumElements() != inputType.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "interior_padding length ({0}) must match operand rank ({1})",
            paddingInterior.getType().getNumElements(), inputType.getRank()));
  }

  auto inputShape = inputType.getShape();
  SmallVector<int64_t> resultShape;
  for (int i = 0, e = inputShape.size(); i < e; i++) {
    if (hlo::isDynamicDimSize(inputShape[i])) {
      resultShape.push_back(ShapedType::kDynamic);
      continue;
    }

    int64_t paddingLowVal = paddingLow.getValues<APInt>()[i].getSExtValue();
    int64_t paddingHighVal = paddingHigh.getValues<APInt>()[i].getSExtValue();
    int64_t paddingInteriorVal =
        paddingInterior.getValues<APInt>()[i].getSExtValue();
    if (paddingInteriorVal < 0) {
      return emitOptionalError(
          location, llvm::formatv("Interior padding cannot be negative: {0}",
                                  paddingInteriorVal));
    }
    int64_t expectedOutput =
        inputShape[i] + paddingLowVal + paddingHighVal +
        std::max<int64_t>(inputShape[i] - 1, 0LL) * paddingInteriorVal;
    if (expectedOutput < 0) {
      return emitOptionalError(
          location,
          llvm::formatv("Padding result in negative size for dimension {0}",
                        i));
    }
    resultShape.push_back(expectedOutput);
  }
  inferredReturnShapes.emplace_back(resultShape, inputType.getElementType());

  return success();
}

template <typename T>
OpFoldResult padOpFoldHelper(DenseElementsAttr input, DenseElementsAttr padding,
                             RankedTensorType returnType,
                             DenseIntElementsAttr edgePaddingLow,
                             DenseIntElementsAttr /*edgePaddingHigh*/,
                             DenseIntElementsAttr interiorPadding) {
  // Prevent folding if the result is too large.
  if (returnType.getNumElements() > kFoldOpEltLimit) return {};

  // Fill the full result tensor with the padding value.
  llvm::SmallVector<T, 4> result(returnType.getNumElements(),
                                 padding.getValues<T>()[0]);

  auto nextIndex = [](llvm::SmallVector<uint64_t, 8>& index,
                      llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (static_cast<int64_t>(index[i]) < shape[i]) return;
      index[i] = 0;
    }
  };

  // Iterate over all elements of the input tensor and copy it to the correct
  // location in the output tensor.
  llvm::SmallVector<uint64_t, 8> index(input.getType().getRank(), 0);
  uint64_t numElements = input.getNumElements();
  for (uint64_t operandIdx = 0; operandIdx < numElements; operandIdx++) {
    uint64_t resultIdx = 0;
    uint64_t idxMultiplyer = 1;
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      resultIdx += (edgePaddingLow.getValues<int64_t>()[i] +
                    index[i] * (interiorPadding.getValues<int64_t>()[i] + 1)) *
                   idxMultiplyer;
      idxMultiplyer *= returnType.getDimSize(i);
    }
    result[resultIdx] = input.getValues<T>()[index];
    nextIndex(index, input.getType().getShape());
  }
  return DenseElementsAttr::get(returnType, result);
}

OpFoldResult PadOp::fold(ArrayRef<Attribute> operands) {
  // If all padding is zero then it is an identity pad.
  auto isZero = [](const APInt& i) { return i == 0; };
  if (llvm::all_of(getEdgePaddingLow().getValues<APInt>(), isZero) &&
      llvm::all_of(getEdgePaddingHigh().getValues<APInt>(), isZero) &&
      llvm::all_of(getInteriorPadding().getValues<APInt>(), isZero))
    return getOperand();

  // If any padding is negative then it isn't supported by the folder (yet).
  auto isNegative = [](const APInt& i) { return i.slt(0); };
  if (llvm::any_of(getEdgePaddingLow().getValues<APInt>(), isNegative) ||
      llvm::any_of(getEdgePaddingHigh().getValues<APInt>(), isNegative) ||
      llvm::any_of(getInteriorPadding().getValues<APInt>(), isNegative))
    return {};

  DenseElementsAttr input = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  DenseElementsAttr padding = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  RankedTensorType returnType = getType().dyn_cast_or_null<RankedTensorType>();
  if (!input || !input.getType().hasRank() || !padding || !returnType ||
      !returnType.hasStaticShape())
    return {};

  if (returnType.getElementType().isa<IntegerType>())
    return padOpFoldHelper<APInt>(input, padding, returnType,
                                  getEdgePaddingLow(), getEdgePaddingHigh(),
                                  getInteriorPadding());
  if (returnType.getElementType().isa<FloatType>())
    return padOpFoldHelper<APFloat>(input, padding, returnType,
                                    getEdgePaddingLow(), getEdgePaddingHigh(),
                                    getInteriorPadding());
  if (ComplexType complex =
          returnType.getElementType().dyn_cast_or_null<ComplexType>()) {
    // TODO(atondwal): Allow int types in HLO_complex
    if (complex.getElementType().isa<FloatType>())
      return padOpFoldHelper<std::complex<APFloat>>(
          input, padding, returnType, getEdgePaddingLow(), getEdgePaddingHigh(),
          getInteriorPadding());
  }
  return {};
}

LogicalResult PadOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  PadOp::Adaptor adaptor(operands, this->getOperation()->getAttrDictionary());
  auto loc = this->getLoc();
  Value operand = adaptor.getOperand();
  auto operandTy = operand.getType().cast<RankedTensorType>();

  llvm::SmallVector<int32_t> padHigh;
  llvm::SmallVector<int32_t> padLow;
  llvm::SmallVector<int32_t> padInterior;

  auto padHighAttr = adaptor.getEdgePaddingHigh();
  auto padLowAttr = adaptor.getEdgePaddingLow();
  auto padInteriorAttr = adaptor.getInteriorPadding();

  padHigh.reserve(padHighAttr.getNumElements());
  padLow.reserve(padLowAttr.getNumElements());
  padInterior.reserve(padInteriorAttr.getNumElements());

  for (const APInt& val : padHighAttr.getValues<APInt>())
    padHigh.push_back(val.getSExtValue());

  for (const APInt& val : padLowAttr.getValues<APInt>())
    padLow.push_back(val.getSExtValue());

  for (const APInt& val : padInteriorAttr.getValues<APInt>())
    padInterior.push_back(val.getSExtValue());

  Value one = builder.create<arith::ConstantIndexOp>(loc, 1).getResult();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0).getResult();

  llvm::SmallVector<Value> dimensions;
  dimensions.reserve(operandTy.getRank());
  for (int i = 0, s = operandTy.getRank(); i < s; ++i) {
    Value padEdge =
        builder.create<arith::ConstantIndexOp>(loc, padHigh[i] + padLow[i]);

    // First we grab the initial interior size.
    Value dim = builder.create<tensor::DimOp>(loc, operand, i).getResult();

    // Compute the interior of the tensor and determine padding size.
    if (padInterior[i] > 0) {
      Value padInter =
          builder.create<arith::ConstantIndexOp>(loc, padInterior[i])
              .getResult();
      Value interior = builder.create<arith::SubIOp>(loc, dim, one).getResult();
      interior = builder.create<arith::MaxSIOp>(loc, interior, zero);
      interior = builder.create<arith::MulIOp>(loc, interior, padInter);
      dim = builder.create<arith::AddIOp>(loc, dim, interior).getResult();
    }

    // Then we add the padding on the edge of the tensor.
    dim = builder.create<arith::AddIOp>(loc, dim, padEdge).getResult();
    dimensions.push_back(dim);
  }

  Value dimensionTensor =
      builder.create<tensor::FromElementsOp>(loc, dimensions).getResult();
  reifiedReturnShapes.push_back(dimensionTensor);
  return success();
}

// If the input tensor has a dimension of length-0, the input tensor is
// irrelevant. Instead we can broadcast the pad value to the output size rather
// than pad the input tensor.
struct PadEmptyTensor : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter& rewriter) const override {
    auto operand = op.getOperand();
    auto padVal = op.getPaddingValue();

    auto operandTy = operand.getType().cast<RankedTensorType>();
    auto resultTy = op.getType().cast<RankedTensorType>();

    if (llvm::all_of(operandTy.getShape(), [](int64_t d) { return d != 0; })) {
      return failure();
    }

    if (resultTy.hasStaticShape()) {
      auto dimsType = RankedTensorType::get({0}, rewriter.getIntegerType(64));
      auto dims =
          DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
      rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(op, resultTy, padVal,
                                                          dims);
      return success();
    }

    llvm::SmallVector<Value> reifiedShapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op.getOperands(),
                                        reifiedShapes)))
      return failure();

    auto dimsType = RankedTensorType::get({0}, rewriter.getIntegerType(64));
    auto broadcastDims =
        DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
    rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op, op.getType(), padVal, reifiedShapes.front(), broadcastDims);

    return failure();
  }
};

void PadOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<PadEmptyTensor>(context);
}

//===----------------------------------------------------------------------===//
// DynamicPadOp
//===----------------------------------------------------------------------===//

// If the input tensor has a dimension of length-0, the input tensor is
// irrelevant. Instead we can broadcast the pad value to the output size rather
// than pad the input tensor.
struct DynamicPadEmptyTensor : public OpRewritePattern<DynamicPadOp> {
  using OpRewritePattern<DynamicPadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicPadOp op,
                                PatternRewriter& rewriter) const override {
    // auto loc = op.getLoc();
    auto operand = op.getOperand();
    auto padVal = op.getPaddingValue();

    auto operandTy = operand.getType().cast<RankedTensorType>();

    if (llvm::all_of(operandTy.getShape(), [](int64_t d) { return d != 0; })) {
      return failure();
    }

    llvm::SmallVector<Value> reifiedShapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op->getOperands(),
                                        reifiedShapes)))
      return failure();

    auto dimsType = RankedTensorType::get({0}, rewriter.getIntegerType(64));
    auto broadcastDims =
        DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
    rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op, op.getType(), padVal, reifiedShapes.front(), broadcastDims);

    return failure();
  }
};

void DynamicPadOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                               MLIRContext* context) {
  results.add<DPadToPad, DynamicPadEmptyTensor>(context);
}

LogicalResult DynamicPadOp::verify() {
  auto inputType = getOperand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!inputType) return success();
  int inputRank = inputType.getRank();

  auto padType = getPaddingValue().getType().cast<RankedTensorType>();
  if (padType.getRank() != 0) {
    return emitOpError() << "padding value type should be a rank-0";
  }

  auto paddingLowType = getEdgePaddingLow().getType().cast<RankedTensorType>();
  if (paddingLowType.getNumElements() != inputRank) {
    return emitOpError() << "edge_padding_low length("
                         << paddingLowType.getNumElements()
                         << ") must match operand rank(" << inputRank << ").";
  }

  auto paddingHighType =
      getEdgePaddingHigh().getType().cast<RankedTensorType>();
  if (paddingHighType.getNumElements() != inputRank) {
    return emitOpError() << "edge_padding_high length("
                         << paddingHighType.getNumElements()
                         << ") must match operand rank(" << inputRank << ").";
  }

  auto interiorPaddingType =
      getInteriorPadding().getType().cast<RankedTensorType>();
  if (interiorPaddingType.getNumElements() != inputRank) {
    return emitOpError() << "edge_padding_interior length("
                         << interiorPaddingType.getNumElements()
                         << ") must match operand rank(" << inputRank << ").";
  }

  auto outputType = getResult().getType().dyn_cast<RankedTensorType>();
  // If result is unranked, there is very little to verify statically.
  if (!outputType) return success();
  int outputRank = outputType.getRank();
  if (inputRank != outputRank) {
    return emitOpError() << "operand rank(" << inputRank
                         << ") must match result(" << outputRank << ").";
  }

  return success();
}

LogicalResult DynamicPadOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicPadOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();
  Value edgePaddingLow = adaptor.getEdgePaddingLow();
  Value edgePaddingHigh = adaptor.getEdgePaddingHigh();
  Value interiorPadding = adaptor.getInteriorPadding();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked pad a.t.m.
  if (!operandType) return failure();

  auto loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      edgePaddingLow.getType().cast<ShapedType>().getElementType();

  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  Value zero =
      toShapeScalarType(builder.create<arith::ConstantIndexOp>(loc, 0));
  Value one = toShapeScalarType(builder.create<arith::ConstantIndexOp>(loc, 1));

  for (int idx : llvm::seq<int>(0, operandType.getShape().size())) {
    Value valueDim =
        toShapeScalarType(builder.create<tensor::DimOp>(loc, operand, idx));
    Value offset = builder.create<arith::ConstantIndexOp>(loc, idx);
    Value valueLow =
        builder.create<tensor::ExtractOp>(loc, edgePaddingLow, offset);
    Value valueHigh =
        builder.create<tensor::ExtractOp>(loc, edgePaddingHigh, offset);
    Value valueInterior =
        builder.create<tensor::ExtractOp>(loc, interiorPadding, offset);
    // output_size = input_size + padding_low + padding_high + interior *
    // max(input_size - 1, 0)
    Value valueDimLessThanOne = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, valueDim, one);
    Value interiorSize = builder.create<arith::MulIOp>(
        loc, valueInterior,
        builder.create<mlir::arith::SelectOp>(
            loc, valueDimLessThanOne, zero,
            builder.create<arith::SubIOp>(loc, valueDim, one)));
    shapeValues.push_back(builder.create<arith::AddIOp>(
        loc,
        builder.create<arith::AddIOp>(
            loc, builder.create<arith::AddIOp>(loc, interiorSize, valueDim),
            valueLow),
        valueHigh));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  // If the operand type is dynamically shaped there is nothing to verify.
  auto operandTy = getOperand().getType().dyn_cast<RankedTensorType>();
  if (!operandTy || !operandTy.hasStaticShape()) return success();

  // If the operand type is statically shaped (not required) the number of
  // elements must match that of the result type.
  auto resultTy = getType().cast<RankedTensorType>();
  assert(resultTy && resultTy.hasStaticShape() &&
         "result type must be statically shaped");
  int64_t numResultElements = resultTy.getNumElements();
  int64_t numOperandElements = operandTy.getNumElements();
  if (numResultElements != numOperandElements)
    return emitOpError() << "number of output elements (" << numResultElements
                         << ") doesn't match expected number of elements ("
                         << numOperandElements << ")";

  return success();
}

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand().getType() == getType()) {
    return getOperand();
  }

  if (auto prevOp = getOperand().getDefiningOp<ReshapeOp>()) {
    setOperand(prevOp.getOperand());
    return getResult();
  }

  if (auto elements = operands.front().dyn_cast_or_null<DenseElementsAttr>()) {
    return reshape(elements, getResult().getType().cast<ShapedType>());
  }

  return {};
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<IdentityBroadcastReshape, IdentityBroadcastInDimReshape,
              EliminateRedundantReshape, EliminateIdentityReshape>(context);
}

//===----------------------------------------------------------------------===//
// ReplicaId Op
//===----------------------------------------------------------------------===//

LogicalResult ReplicaIdOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(RankedTensorType::get(
      /*shape=*/{}, IntegerType::get(context, 32, IntegerType::Unsigned)));
  return success();
}

//===----------------------------------------------------------------------===//
// PartitionId Op
//===----------------------------------------------------------------------===//

LogicalResult PartitionIdOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange /*operands*/,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(RankedTensorType::get(
      /*shape=*/{}, IntegerType::get(context, 32, IntegerType::Unsigned)));
  return success();
}

//===----------------------------------------------------------------------===//
// AddDependency Op
//===----------------------------------------------------------------------===//

LogicalResult AddDependencyOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(operands.getTypes()[0]);
  return success();
}

//===----------------------------------------------------------------------===//
// If Op
//===----------------------------------------------------------------------===//

LogicalResult IfOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  IfOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferIfOp(location, adaptor.getRegions(), inferredReturnTypes);
}

static LogicalResult inlineIfConstantCondition(IfOp ifOp,
                                               PatternRewriter& rewriter) {
  DenseIntElementsAttr predAttr;
  if (!matchPattern(ifOp.getPred(), m_Constant(&predAttr))) return failure();

  if (predAttr.getSplatValue<BoolAttr>().getValue()) {
    replaceOpWithRegion(rewriter, ifOp, ifOp.getTrueBranch());
  } else {
    replaceOpWithRegion(rewriter, ifOp, ifOp.getFalseBranch());
  }
  return success();
}

void IfOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add(&inlineIfConstantCondition);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

LogicalResult CaseOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  CaseOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferCaseOp(location, adaptor.getRegions(), inferredReturnTypes);
}

static LogicalResult inlineCaseConstantCondition(CaseOp caseOp,
                                                 PatternRewriter& rewriter) {
  DenseIntElementsAttr indexAttr;
  if (!matchPattern(caseOp.getIndex(), m_Constant(&indexAttr))) {
    return failure();
  }
  int64_t index =
      indexAttr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
  // For an OOB index, the last branch is executed as the default branch:
  // https://www.tensorflow.org/xla/operation_semantics#conditional
  if (index < 0 || index >= caseOp.getNumRegions())
    index = caseOp.getNumRegions() - 1;

  Region& region = caseOp.getRegion(index);
  if (!llvm::hasSingleElement(region)) return failure();
  replaceOpWithRegion(rewriter, caseOp, region);
  return success();
}

void CaseOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add(&inlineCaseConstantCondition);
}

//===----------------------------------------------------------------------===//
// UnaryOps
//===----------------------------------------------------------------------===//

template <typename ValType>
struct AnyValue {
  bool operator()(const ValType&) { return true; }
};

template <typename ValType>
struct NonNegativeValue {
  bool operator()(const ValType& v) { return !v.isNegative(); }
};

template <typename ValType>
struct PositiveValue {
  bool operator()(const ValType& v) { return !v.isNegative() && !v.isZero(); }
};

static const APFloat& addSign(const APFloat& v, Type) { return v; }
static APSInt addSign(const APInt& v, Type t) {
  // Add signedness information to the value, treating signless as signed,
  // unless it's i1.
  return APSInt(v, t.isUnsignedInteger() || t.isSignlessInteger(1));
}

template <typename Op, typename ElementType, typename ValType, typename Convert,
          typename Validate = AnyValue<ValType>>
static Attribute UnaryFolder(Op* op, ArrayRef<Attribute> attrs) {
  if (!attrs[0]) return {};

  DenseElementsAttr val = attrs[0].dyn_cast<DenseElementsAttr>();
  if (!val) return {};

  ShapedType type = op->getType().template cast<ShapedType>();
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!etype.isa<ElementType>()) {
    return {};
  }

  // Prevent folding if the result is too large.
  if (val.getNumElements() > kFoldOpEltLimit) return {};

  SmallVector<ValType, 6> values;
  values.reserve(val.getNumElements());
  for (const auto v : val.getValues<ValType>()) {
    if (!Validate()(v)) return {};
    Optional<ValType> r = Convert()(addSign(v, type));
    if (!r) return {};
    values.push_back(r.value());
  }

  return DenseElementsAttr::get(type, values);
}

struct Round {
  Optional<APFloat> operator()(const APFloat& f) {
    APFloat r = f;
    r.roundToIntegral(llvm::RoundingMode::NearestTiesToAway);
    return r;
  }
};

struct RoundNearestEven {
  Optional<APFloat> operator()(const APFloat& f) {
    APFloat r = f;
    r.roundToIntegral(llvm::RoundingMode::NearestTiesToEven);
    return r;
  }
};

template <typename FloatOrInt>
struct Sign {
  APFloat compute(const APFloat& f) {
    if (f.isZero() || f.isNaN()) return f;
    double value = f.isNegative() ? -1.0 : 1.0;
    APFloat val(value);
    bool unused;
    val.convert(f.getSemantics(), APFloat::rmNearestTiesToEven, &unused);
    return val;
  }

  APInt compute(const APInt& i) {
    APInt r = i;
    if (r == 0) return r;
    if (r.isNegative()) {
      return APInt(r.getBitWidth(), -1, /*isSigned=*/true);
    }
    return APInt(r.getBitWidth(), 1, /*isSigned=*/true);
  }

  Optional<FloatOrInt> operator()(const FloatOrInt& fi) { return compute(fi); }
};

template <typename FloatOrInt>
struct Abs {
  APFloat compute(const APFloat& f) { return abs(f); }

  APInt compute(const APInt& i) { return i.abs(); }

  Optional<FloatOrInt> operator()(const FloatOrInt& fi) { return compute(fi); }
};

double rsqrt(double d) { return 1.0 / std::sqrt(d); }

double logistic(double d) { return 1.0 / (1.0 + std::exp(-d)); }

// NOLINTBEGIN(bugprone-macro-parentheses)
#define UNARY_FOLDER(Op, Func)                                                \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                          \
    /* AbsOp could take complex but return float */                           \
    if (getElementTypeOrSelf(getOperation()->getOperand(0).getType()) !=      \
        getElementTypeOrSelf(getType())) {                                    \
      return {};                                                              \
    }                                                                         \
    if (getElementTypeOrSelf(getType()).isa<FloatType>())                     \
      return UnaryFolder<Op, FloatType, APFloat, Func<APFloat>>(this, attrs); \
    if (getElementTypeOrSelf(getType()).isa<IntegerType>())                   \
      return UnaryFolder<Op, IntegerType, APInt, Func<APInt>>(this, attrs);   \
    return {};                                                                \
  }

#define UNARY_FOLDER_INT(Op, Func)                                          \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                        \
    if (getElementTypeOrSelf(getType()).isa<IntegerType>())                 \
      return UnaryFolder<Op, IntegerType, APInt, Func<APInt>>(this, attrs); \
    return {};                                                              \
  }

#define UNARY_FOLDER_FLOAT(Op, Func)                                 \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                 \
    if (getElementTypeOrSelf(getType()).isa<FloatType>())            \
      return UnaryFolder<Op, FloatType, APFloat, Func>(this, attrs); \
    return {};                                                       \
  }

#define UNARY_FOLDER_UPCAST_TO_F64(Op, Func, Validate)               \
  struct Op##Folder {                                                \
    Optional<APFloat> operator()(const APFloat& input) {             \
      APFloat f = input;                                             \
      const llvm::fltSemantics& oldSemantics = f.getSemantics();     \
                                                                     \
      bool unusedLoseInfo;                                           \
      f.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, \
                &unusedLoseInfo);                                    \
                                                                     \
      APFloat result(Func(f.convertToDouble()));                     \
      result.convert(oldSemantics, APFloat::rmNearestTiesToEven,     \
                     &unusedLoseInfo);                               \
      return result;                                                 \
    }                                                                \
  };                                                                 \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                 \
    if (getElementTypeOrSelf(getType()).isa<FloatType>())            \
      return UnaryFolder<Op, FloatType, APFloat, Op##Folder,         \
                         Validate<APFloat>>(this, attrs);            \
    return {};                                                       \
  }
// NOLINTEND(bugprone-macro-parentheses)

UNARY_FOLDER(NegOp, std::negate)
UNARY_FOLDER(SignOp, Sign)
UNARY_FOLDER(AbsOp, Abs)
UNARY_FOLDER_INT(NotOp, std::bit_not)
UNARY_FOLDER_FLOAT(RoundNearestEvenOp, RoundNearestEven)
UNARY_FOLDER_FLOAT(RoundOp, Round)

UNARY_FOLDER_UPCAST_TO_F64(CosineOp, std::cos, AnyValue)
UNARY_FOLDER_UPCAST_TO_F64(ExpOp, std::exp, AnyValue)
UNARY_FOLDER_UPCAST_TO_F64(LogisticOp, logistic, AnyValue)
UNARY_FOLDER_UPCAST_TO_F64(LogOp, std::log, PositiveValue)
UNARY_FOLDER_UPCAST_TO_F64(RsqrtOp, rsqrt, PositiveValue)
UNARY_FOLDER_UPCAST_TO_F64(SineOp, std::sin, AnyValue)
UNARY_FOLDER_UPCAST_TO_F64(SqrtOp, std::sqrt, NonNegativeValue)
UNARY_FOLDER_UPCAST_TO_F64(TanhOp, std::tanh, AnyValue)

#undef UNARY_FOLDER
#undef UNARY_FOLDER_INT
#undef UNARY_FOLDER_FLOAT
#undef UNARY_FOLDER_UPCAST_TO_F64

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
static Attribute BinaryFolder(Op* op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1]) return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs) return {};

  ShapedType type = op->getType().template cast<ShapedType>();
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!etype.isa<ElementType>()) {
    return {};
  }

  // Special case for folding splats no matter how large.
  // Only covers the case of both attrs being splats; operation-specific cases
  // like adding a zero or multiplying by one are handled elsewhere.
  SplatElementsAttr splatLhs = lhs.dyn_cast<SplatElementsAttr>();
  SplatElementsAttr splatRhs = rhs.dyn_cast<SplatElementsAttr>();
  if (splatLhs && splatRhs) {
    auto signedLhs = addSign(splatLhs.getSplatValue<ValType>(), etype);
    auto signedRhs = addSign(splatRhs.getSplatValue<ValType>(), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    return succeeded(result) ? SplatElementsAttr::get(type, *result)
                             : Attribute();
  }

  // Prevent folding if the result is too large.
  if (lhs.getNumElements() > kFoldOpEltLimit) return {};

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    auto signedLhs = addSign(std::get<0>(zip), etype);
    auto signedRhs = addSign(std::get<1>(zip), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    if (failed(result)) {
      return {};
    }
    values.push_back(std::move(*result));
  }

  return DenseElementsAttr::get(type, values);
}

template <typename T>
struct Divide : std::divides<T> {};

template <>
struct Divide<APSInt> {
  FailureOr<APSInt> operator()(const APSInt& a, const APSInt& b) const {
    if (b.isZero()) return failure();
    return a / b;
  }
};

template <typename T>
struct Remainder : std::modulus<T> {};

template <>
struct Remainder<APSInt> {
  FailureOr<APSInt> operator()(const APSInt& a, const APSInt& b) const {
    if (b.isZero()) return failure();
    return a % b;
  }
};

template <>
struct Remainder<APFloat> {
  APFloat operator()(const APFloat& a, const APFloat& b) const {
    APFloat result(a);
    // Using .mod instead of .remainder is important for behavior around signed
    // zeros
    result.mod(b);
    return result;
  }
};

template <typename T>
struct Max {
  T operator()(const T& a, const T& b) const { return std::max<T>(a, b); }
};

template <>
struct Max<APFloat> {
  // maximum on APFloat is required for NaN propagation logic
  APFloat operator()(const APFloat& a, const APFloat& b) const {
    return llvm::maximum(a, b);
  }
};

template <typename T>
struct Min {
  T operator()(const T& a, const T& b) const { return std::min<T>(a, b); }
};

template <>
struct Min<APFloat> {
  // minimum on APFloat is required for NaN propagation logic
  APFloat operator()(const APFloat& a, const APFloat& b) const {
    return llvm::minimum(a, b);
  }
};

#define BINARY_FOLDER_INTERNAL(Op, Func)                                     \
  if (getElementTypeOrSelf(getType()).isa<FloatType>())                      \
    return BinaryFolder<Op, FloatType, APFloat, Func<APFloat>>(this, attrs); \
  if (getElementTypeOrSelf(getType()).isa<IntegerType>())                    \
    return BinaryFolder<Op, IntegerType, APInt, Func<APSInt>>(this, attrs);  \
  return {};

#define BINARY_FOLDER(Op, Func)                      \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) { \
    BINARY_FOLDER_INTERNAL(Op, Func)                 \
  }

// Addition, subtraction and multiplication use the std:: versions of the ops.
// Due to the other ops behaving differently in signed vs unsigned integers,
// APInts need a special implementation. Currently, it replicates signed int
// op behavior.
BINARY_FOLDER(SubtractOp, std::minus)
BINARY_FOLDER(DivOp, Divide)
BINARY_FOLDER(RemOp, Remainder)
BINARY_FOLDER(MaxOp, Max)
BINARY_FOLDER(MinOp, Min)

OpFoldResult AddOp::fold(ArrayRef<Attribute> attrs) {
  // Handle special case where one operand is 0:  x + 0 => x
  if (attrs[0] || attrs[1]) {
    SplatElementsAttr splatLhs = attrs[0].dyn_cast_or_null<SplatElementsAttr>();
    SplatElementsAttr splatRhs = attrs[1].dyn_cast_or_null<SplatElementsAttr>();
    if (isSplatZero(splatLhs))
      return splatRhs ? (OpFoldResult)splatRhs : getRhs();
    if (isSplatZero(splatRhs))
      return splatLhs ? (OpFoldResult)splatLhs : getLhs();
  }
  if (attrs[0] && attrs[1]) {
    BINARY_FOLDER_INTERNAL(AddOp, std::plus)
  }
  return {};
}

bool isSplatOne(SplatElementsAttr attr) {
  if (!attr) return false;
  if (attr.getElementType().isa<FloatType>()) {
    return attr.getSplatValue<APFloat>().convertToDouble() == 1.0;
  }
  if (attr.getElementType().isa<IntegerType>()) {
    return attr.getSplatValue<APInt>().getSExtValue() == 1;
  }
  return false;
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> attrs) {
  // Handle special case where one operand is 1: x * 1 => x
  if (attrs[0] || attrs[1]) {
    SplatElementsAttr splatLhs = attrs[0].dyn_cast_or_null<SplatElementsAttr>();
    SplatElementsAttr splatRhs = attrs[1].dyn_cast_or_null<SplatElementsAttr>();
    if (isSplatOne(splatLhs))
      return splatRhs ? (OpFoldResult)splatRhs : getRhs();
    if (isSplatOne(splatRhs))
      return splatLhs ? (OpFoldResult)splatLhs : getLhs();
  }
  if (attrs[0] && attrs[1]) {
    BINARY_FOLDER_INTERNAL(MulOp, std::multiplies);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Logical Ops
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(ArrayRef<Attribute> operands) {
  if (getLhs() == getRhs()) return getLhs();

  auto lhsVal = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsVal = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnesValue()) {
      return getRhs();
    }

    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return lhsVal;
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnesValue()) {
      return getLhs();
    }

    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return rhsVal;
    }
  }

  if (!rhsVal || !lhsVal) return {};
  return BinaryFolder<AndOp, IntegerType, APInt, std::bit_and<APSInt>>(
      this, operands);
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> operands) {
  if (getLhs() == getRhs()) return getLhs();

  auto lhsVal = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsVal = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnesValue()) {
      return lhsVal;
    }

    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return getRhs();
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnesValue()) {
      return rhsVal;
    }

    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return getLhs();
    }
  }

  if (!rhsVal || !lhsVal) return {};
  return BinaryFolder<OrOp, IntegerType, APInt, std::bit_or<APSInt>>(this,
                                                                     operands);
}

OpFoldResult XorOp::fold(ArrayRef<Attribute> operands) {
  // Fold x^x to 0. Attributes only support static shapes.
  auto rType = getType().cast<ShapedType>();
  if (getLhs() == getRhs() && rType.hasStaticShape()) {
    Builder builder(getContext());
    return builder.getZeroAttr(rType);
  }

  auto lhsVal = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsVal = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return getRhs();
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return getLhs();
    }
  }

  if (!rhsVal || !lhsVal) return {};
  return BinaryFolder<XorOp, IntegerType, APInt, std::bit_xor<APSInt>>(
      this, operands);
}

#undef BINARY_FOLDER_INTERNAL
#undef BINARY_FOLDER

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// Returns output dimension size for slice result for the given arguments.
// Returns -1 if arguments are illegal.
static int64_t inferSliceDim(int64_t inputDim, int64_t start, int64_t end,
                             int64_t stride) {
  if (inputDim == -1 || start < 0 || start > end || end > inputDim ||
      stride == 0)
    return -1;

  return llvm::divideCeil(end - start, stride);
}

// The following properties are already enforced by the ODS:
//  type(start_indices) == type(limit_indices) == type(strides).
// Verify the following properties:
//  P1. Verify rank(start_indices) == 1.
//  P2. Verify size(start_indices) == rank(operand).
//  P3~5. Verify 0 <= start_indices[i] <= limit_indices[i] <= shape(operand)[i].
//  P6. Verify stride[i] > 0.
LogicalResult SliceOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SliceOpAdaptor slice(operands, attributes);
  Type ty = slice.getOperand().getType();
  RankedTensorType rankedTy = ty.dyn_cast<RankedTensorType>();
  if (!rankedTy) {
    // The operand type is unranked, so the best we can infer for the result
    // type is an unranked tensor with the same element type as the operand
    // type.
    inferredReturnTypes.assign({ty});
    return success();
  }

  ShapedType attrTy = slice.getStartIndices().getType();
  // P1.
  // Note: ODS has type(start_indices) == type(limit_indices) == type(strides)
  // So this implies rank(limit_indices) == rank(strides) == 1 also.
  if (attrTy.getRank() != 1) {
    return emitOptionalError(location, "start_indices has rank ",
                             attrTy.getRank(), " instead of required rank 1");
  }

  // P2.
  int64_t rank = rankedTy.getRank();
  if (attrTy.getNumElements() != rank) {
    return emitOptionalError(
        location, "the number of elements in start_indices (",
        attrTy.getNumElements(), ") does not match the rank of the operand (",
        rank, ")");
  }

  SmallVector<int64_t, 4> start(slice.getStartIndices().getValues<int64_t>());
  SmallVector<int64_t, 4> limit(slice.getLimitIndices().getValues<int64_t>());
  SmallVector<int64_t, 4> strideVals(slice.getStrides().getValues<int64_t>());

  SmallVector<int64_t, 4> shape;
  shape.reserve(rank);
  for (int64_t i = 0, e = rank; i != e; i++) {
    if (hlo::isDynamicDimSize(rankedTy.getDimSize(i))) {
      shape.push_back(ShapedType::kDynamic);
      continue;
    }
    // P3.
    if (start[i] < 0)
      return emitOptionalError(location, "negative start index ", start[i],
                               " in dimension ", i);
    // P4.
    if (limit[i] > rankedTy.getDimSize(i))
      return emitOptionalError(location, "limit index ", limit[i],
                               " is larger than dimension size ",
                               rankedTy.getDimSize(i), " in dimension ", i);
    // P5.
    if (start[i] > limit[i])
      return emitOptionalError(location, "start index ", start[i],
                               " is larger than limit index ", limit[i],
                               " in dimension ", i);
    // P6.
    if (strideVals[i] <= 0)
      return emitOptionalError(location, "stride must be positive but got ",
                               strideVals[i], " in dimension ", i);

    shape.push_back(inferSliceDim(rankedTy.getDimSize(i), start[i], limit[i],
                                  strideVals[i]));
  }
  inferredReturnTypes.assign(
      {RankedTensorType::get(shape, rankedTy.getElementType())});
  return success();
}

template <typename I, typename E>
static void sliceElements(I values, ArrayRef<int64_t> sizes,
                          ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                          ArrayRef<int64_t> strides,
                          llvm::SmallVectorImpl<E>* outValues) {
  assert(starts.size() == limits.size());
  assert(starts.size() == strides.size());
  if (starts.empty()) return;

  int64_t start = starts.front();
  int64_t limit = limits.front();
  int64_t stride = strides.front();
  if (starts.size() == 1) {
    for (int i = start; i < limit; i += stride) {
      outValues->push_back(*(values + i));
    }
    return;
  }

  for (; start < limit; start += stride) {
    auto begin = values + start * sizes.front();
    sliceElements<I, E>(begin, sizes.drop_front(), starts.drop_front(),
                        limits.drop_front(), strides.drop_front(), outValues);
  }
}

template <typename I, typename E>
static Attribute foldSlice(SliceOp* op, I values) {
  auto start = llvm::to_vector<6>(op->getStartIndices().getValues<int64_t>());
  auto limit = llvm::to_vector<6>(op->getLimitIndices().getValues<int64_t>());
  auto stride = llvm::to_vector<6>(op->getStrides().getValues<int64_t>());

  // TODO(b/235903849): This should be op->getType().case<ShapedType>().
  auto resultType = op->getOperand().getType().cast<ShapedType>();
  if (!resultType.hasStaticShape()) return {};

  auto shape = resultType.getShape();
  int64_t count = resultType.getNumElements();
  if (count == 0) {
    return DenseElementsAttr::get<E>(
        op->getResult().getType().cast<ShapedType>(),
        /*list=*/{});
  }

  // Compute the striding for each dimension.
  llvm::SmallVector<int64_t, 6> sizes;
  sizes.reserve(shape.size());
  for (auto v : shape) {
    count = count / v;
    sizes.push_back(count);
  }

  // Prevent folding if the result is too large.
  if (resultType.getNumElements() > kFoldOpEltLimit) return {};

  llvm::SmallVector<E, 6> outValues;
  outValues.reserve(resultType.getNumElements());
  sliceElements<I, E>(values, sizes, start, limit, stride, &outValues);

  return DenseElementsAttr::get(op->getResult().getType().cast<ShapedType>(),
                                outValues);
}

OpFoldResult SliceOp::fold(ArrayRef<Attribute> operands) {
  // Check if the SliceOp is a NoOp operation.
  auto operandType = getOperand().getType().cast<ShapedType>();
  auto resultType = getResult().getType().cast<ShapedType>();

  if (operandType.hasStaticShape() && resultType.hasStaticShape() &&
      (operandType.getShape() == resultType.getShape())) {
    return getOperand();
  }

  if (operands.empty() || !operands.front()) return {};

  // Evaluate for statically valued inputs.
  DenseElementsAttr elements = operands.front().dyn_cast<DenseElementsAttr>();
  if (!elements) return {};

  auto etype = elements.getType().getElementType();
  if (etype.isa<IntegerType>()) {
    return foldSlice<DenseElementsAttr::IntElementIterator, APInt>(
        this, elements.value_begin<APInt>());
  }
  if (etype.isa<FloatType>()) {
    return foldSlice<DenseElementsAttr::FloatElementIterator, APFloat>(
        this, elements.value_begin<APFloat>());
  }

  return {};
}

namespace {
// In cases where a concat is fed into a slice, it is possible the concat
// can be simplified or bypassed. This checks which inputs to the concat are
// used by the slice, either reducing the number of concatenated values or
// entirely removes the concat.
struct SimplifyConcatSlice : public OpRewritePattern<SliceOp> {
  using OpRewritePattern<SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp slice,
                                PatternRewriter& rewriter) const override {
    auto resultTy = slice.getType().cast<ShapedType>();
    if (!resultTy.hasStaticShape()) {
      return failure();
    }

    auto sliceInput = slice.getOperand();
    auto sliceInputTy = sliceInput.getType().cast<ShapedType>();
    auto concat = sliceInput.getDefiningOp<ConcatenateOp>();
    if (!concat) {
      return failure();
    }

    auto dimension = concat.getDimension();

    auto start = slice.getStartIndices().getValues<APInt>();
    auto limit = slice.getLimitIndices().getValues<APInt>();

    auto sliceStart = (*(start.begin() + dimension)).getSExtValue();
    auto sliceLimit = (*(limit.begin() + dimension)).getSExtValue();

    // We need to determine what inputs from the concat affect the slice, and
    // how the bounds of the slice need to be updated for the minimally required
    // inputs.
    int64_t runningSize = 0;
    int64_t frontOffset = sliceInputTy.getShape()[dimension];

    auto subsetStart = concat.operand_end();
    auto subsetEnd = concat.operand_end();
    for (auto it = concat.operand_begin(); it < concat.operand_end(); ++it) {
      auto input = *it;
      ShapedType inputTy = input.getType().cast<ShapedType>();
      if (inputTy.isDynamicDim(dimension)) {
        return failure();
      }
      auto dimSize = inputTy.getShape()[dimension];

      // If this position is in the slice its the start of the subset and we
      // need to update the start and limit values.
      if (runningSize + dimSize > sliceStart &&
          subsetStart == concat.operand_end()) {
        subsetStart = it;
        frontOffset = runningSize;
      }

      // Determine the last required offset.
      if (runningSize < sliceLimit) {
        subsetEnd = it + 1;
      }

      runningSize += dimSize;
    }

    auto subsetSize = subsetEnd - subsetStart;
    // We need all inputs so no optimization.
    if (subsetSize == concat.getNumOperands()) {
      return failure();
    }

    // If there's nothing to slice that means the output is an empty tensor and
    // there is dead code. We do nothing here and rely on other passes to clean
    // this up.
    if (subsetSize == 0) {
      return failure();
    }

    if (subsetSize > 1 && !concat.getResult().hasOneUse()) {
      return failure();
    }

    auto concatRange = OperandRange(subsetStart, subsetEnd);
    auto newConcat = rewriter.create<ConcatenateOp>(
        concat.getLoc(), concatRange, concat.getDimension());

    llvm::SmallVector<APInt, 6> newStart(start);
    llvm::SmallVector<APInt, 6> newLimit(limit);
    newStart[dimension] -= frontOffset;
    newLimit[dimension] -= frontOffset;

    auto attrType = slice.getStartIndices().getType().cast<ShapedType>();
    auto create = rewriter.create<SliceOp>(
        slice.getLoc(), newConcat,
        DenseIntElementsAttr::get(attrType, newStart),
        DenseIntElementsAttr::get(attrType, newLimit), slice.getStrides());
    rewriter.replaceOp(slice, create.getResult());
    return success();
  }
};
}  // namespace

void SliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<SimplifyConcatSlice>(context);
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::build(OpBuilder& builder, OperationState& state,
                   ValueRange operands, int64_t dimension, bool isStable) {
  state.addOperands(operands);
  state.addAttribute("dimension", builder.getI64IntegerAttr(dimension));
  state.addAttribute("is_stable", builder.getBoolAttr(isStable));

  for (Value operand : operands) state.addTypes(operand.getType());

  state.addRegion();
}

LogicalResult SortOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location>, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SortOp::Adaptor adaptor(operands, attributes, regions);
  for (auto resultType : adaptor.getInputs().getTypes())
    inferredReturnShapes.emplace_back(resultType.cast<ShapedType>());
  return success();
}

LogicalResult SortOp::verify() {
  SmallVector<ShapedTypeComponents> unusedReturnShapes;
  return hlo::inferSortOp(getLoc(), getInputs(), getDimension(),
                          getComparator(), unusedReturnShapes);
}

/// Drops the operands if the results are not used and they are not used in
/// op.comparator().
static LogicalResult sortDropEmptyUseArgs(SortOp op,
                                          PatternRewriter& rewriter) {
  DenseSet<unsigned> erasedArgs;
  unsigned numOperands = op.getNumOperands();
  for (unsigned i = 0; i < numOperands; ++i) {
    if (!op.getResult(i).use_empty()) continue;
    Block& block = op.getComparator().front();
    if (!block.getArgument(i * 2).use_empty()) continue;
    if (!block.getArgument(i * 2 + 1).use_empty()) continue;
    erasedArgs.insert(i);
  }
  if (erasedArgs.empty()) return failure();

  SmallVector<Value> newOperands;
  BitVector erasedBlockArgs(op.getNumOperands() * 2);
  for (const auto& en : llvm::enumerate(op.getInputs())) {
    if (erasedArgs.contains(en.index())) {
      erasedBlockArgs.set(en.index() * 2);
      erasedBlockArgs.set(en.index() * 2 + 1);
    } else {
      newOperands.push_back(en.value());
    }
  }

  auto newOp = rewriter.create<SortOp>(op.getLoc(), newOperands,
                                       op.getDimension(), op.getIsStable());
  Region& region = newOp.getComparator();
  rewriter.inlineRegionBefore(op.getComparator(), region, region.end());
  region.front().eraseArguments(erasedBlockArgs);

  SmallVector<Value> results;
  for (unsigned i = 0, j = 0; i < numOperands; ++i) {
    if (erasedArgs.contains(i)) {
      results.push_back({});
    } else {
      results.push_back(newOp.getResult(j++));
    }
  }
  rewriter.replaceOp(op, results);

  return success();
}

/// Set the sorting dimension to the last dimension if it's not set and the rank
/// is known.
static LogicalResult sortOpInferDefaultDimension(SortOp op,
                                                 PatternRewriter& rewriter) {
  auto ty = op.getResultTypes()[0].dyn_cast<ShapedType>();
  if (!ty) {
    return failure();
  }
  if (static_cast<int64_t>(op.getDimension()) != -1) {
    return failure();
  }

  IntegerAttr dim = rewriter.getI64IntegerAttr(ty.getRank() - 1);
  auto newOp =
      rewriter.create<SortOp>(op.getLoc(), op.getResultTypes(), op.getInputs(),
                              dim, op.getIsStableAttr());
  Region& region = newOp.getComparator();
  rewriter.inlineRegionBefore(op.getComparator(), region, region.end());
  rewriter.replaceOp(op, newOp.getResults());

  return success();
}

void SortOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* /*context*/) {
  results.add(sortDropEmptyUseArgs);
  results.add(sortOpInferDefaultDimension);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  if (auto elements = operands.front().dyn_cast_or_null<SplatElementsAttr>()) {
    return reshape(elements, getResult().getType().cast<ShapedType>());
  }
  for (const auto& it : llvm::enumerate(getPermutation().getValues<APInt>())) {
    if (it.index() != it.value()) {
      return {};
    }
  }
  return getOperand();
}

// transpose(transpose(X)) => transpose(X)
static LogicalResult eliminateRedundantTranspse(TransposeOp op,
                                                PatternRewriter& rewriter) {
  auto tranposeOperand = op.getOperand().getDefiningOp<TransposeOp>();
  if (!tranposeOperand) {
    return failure();
  }
  auto operandPermutation = tranposeOperand.getPermutation().getValues<APInt>();
  auto newPermutation =
      op.getPermutation()
          .mapValues(op.getPermutation().getElementType(),
                     [&operandPermutation](const APInt& index) -> APInt {
                       return operandPermutation[index.getSExtValue()];
                     })
          .cast<DenseIntElementsAttr>();
  rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getResult().getType(),
                                           tranposeOperand.getOperand(),
                                           newPermutation);
  return success();
}

// transpose(broadcast_in_dim(X)) => broadcast_in_dim(X)
static LogicalResult eliminateBroadcastInDimTranspose(
    TransposeOp op, PatternRewriter& rewriter) {
  auto broadcastInDimOp = op.getOperand().getDefiningOp<BroadcastInDimOp>();
  if (!broadcastInDimOp) {
    return failure();
  }
  DenseIntElementsAttr broadcastDimensions =
      broadcastInDimOp.getBroadcastDimensions();
  DenseIntElementsAttr permutation = op.getPermutation();
  SmallVector<int64_t> newBroadcastDimensions;
  for (auto dimension : broadcastDimensions.getValues<int64_t>()) {
    int64_t index = 0;
    for (auto p : permutation.getValues<int64_t>()) {
      if (p == dimension) {
        newBroadcastDimensions.push_back(index);
        break;
      }
      index++;
    }
  }
  rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
      op, op->getResultTypes(), broadcastInDimOp.getOperand(),
      rewriter.getI64TensorAttr(newBroadcastDimensions));
  return success();
}

// simplify Transpose: replace Transpose with Reshape if they are equivalent
static LogicalResult simplifyTranspose(TransposeOp op,
                                       PatternRewriter& rewriter) {
  auto operandType = op.getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!operandType || !resultType) {
    return failure();
  }
  // Not support dynamic shape a.t.m. BTW, when it's dynamic shape,
  // maybe Transpose should be replaced by DynamicReshape.
  if (!operandType.hasStaticShape() || !resultType.hasStaticShape()) {
    return failure();
  }
  auto permutation = op.getPermutation().getValues<int64_t>();
  llvm::SmallVector<int64_t> sortedPermutation;
  for (int64_t i = 0, e = resultType.getRank(); i < e; i++) {
    if (resultType.getDimSize(i) != 1) {
      sortedPermutation.push_back(permutation[i]);
    }
  }
  if (llvm::is_sorted(sortedPermutation)) {
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getOperand());
    return success();
  }
  return failure();
}

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* /*context*/) {
  results.add(eliminateRedundantTranspse);
  results.add(eliminateBroadcastInDimTranspose);
  results.add(simplifyTranspose);
}

LogicalResult TransposeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  TransposeOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  SmallVector<int64_t, 4> permutation(
      this->getPermutation().getValues<int64_t>());
  SmallVector<Value, 4> shapeValues(permutation.size());

  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto& element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto* it = std::find(permutation.begin(), permutation.end(), idx);
    Value valueDim = toShapeScalarType(
        builder.createOrFold<tensor::DimOp>(loc, operand, element.index()));
    shapeValues[std::distance(permutation.begin(), it)] = valueDim;
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

// Method for InferTypeOpInterface: infer the return type from the operand type
// and the permutation.
LogicalResult TransposeOp::inferReturnTypes(
    MLIRContext* /*context*/, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto type = operands[0].getType();
  auto rankedTy = type.dyn_cast<RankedTensorType>();
  if (!rankedTy) {
    auto shapedTy = type.dyn_cast<ShapedType>();
    inferredReturnTypes.emplace_back(shapedTy);
    return success();
  }
  auto permutation = attributes.getAs<DenseIntElementsAttr>("permutation");
  int64_t rank = rankedTy.getRank();
  if (permutation.getType().getRank() != 1)
    return emitOptionalError(loc, "TransposeOp permutation has rank ",
                             permutation.getType().getRank(),
                             " instead of rank 1");

  if (permutation.size() != rank)
    return emitOptionalError(loc, "TransposeOp operand rank ", rank,
                             " does not match permutation size ",
                             permutation.size());

  std::vector<int64_t> range(rank);
  std::iota(range.begin(), range.end(), 0);
  if (!std::is_permutation(range.begin(), range.end(), permutation.begin()))
    return emitOptionalError(loc,
                             "attribute permutation must be a permutation"
                             " of [",
                             range, "] but got ", permutation);

  SmallVector<int64_t> resultShape;
  ArrayRef<int64_t> inputShape = rankedTy.getShape();
  for (int64_t dim : permutation.getValues<int64_t>()) {
    resultShape.push_back(inputShape[dim]);
  }
  inferredReturnTypes.emplace_back(RankedTensorType::get(
      resultShape, rankedTy.getElementType(), rankedTy.getEncoding()));
  return success();
}

//===----------------------------------------------------------------------===//
// TriangularSolveOp
//===----------------------------------------------------------------------===//

LogicalResult TriangularSolveOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  TriangularSolveOp::Adaptor adaptor(operands, attributes, regions);
  bool isTransposeAInvalid =
      (adaptor.getTransposeA() == Transpose::TRANSPOSE_INVALID);
  return hlo::inferTriangularSolveOp(location, adaptor.getA(), adaptor.getB(),
                                     adaptor.getLeftSide(), isTransposeAInvalid,
                                     inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto tupleType = operands[0].getType().dyn_cast<TupleType>();
  if (!tupleType) return failure();

  auto indexAttr = attributes.get("index").cast<IntegerAttr>();
  auto index = indexAttr.getInt();
  if (index < 0 || index >= static_cast<int64_t>(tupleType.size()))
    return failure();

  inferredReturnTypes.push_back(tupleType.getType(index));
  return success();
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(TupleType::get(context, TypeRange(operands)));
  return success();
}

//===----------------------------------------------------------------------===//
// UnaryEinsumOp
//===----------------------------------------------------------------------===//

void UnaryEinsumOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<UnaryEinsumToEinsum>(context);
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

void CompareOp::build(OpBuilder& builder, OperationState& result, Value lhs,
                      Value rhs, ComparisonDirection comparisonDirection,
                      ComparisonType compareType) {
  build(builder, result, lhs, rhs,
        ComparisonDirectionAttr::get(builder.getContext(), comparisonDirection),
        ComparisonTypeAttr::get(builder.getContext(), compareType));
}

LogicalResult CompareOp::inferReturnTypeComponents(
    mlir::MLIRContext* ctx, llvm::Optional<mlir::Location>,
    ValueShapeRange operands, mlir::DictionaryAttr, mlir::RegionRange,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnTypes) {
  ShapedTypeComponents& components =
      inferredReturnTypes.emplace_back(IntegerType::get(ctx, /*width=*/1));
  auto argTy = operands.front().getType().cast<TensorType>();
  if (argTy.hasRank()) {
    components =
        ShapedTypeComponents(argTy.getShape(), components.getElementType());
  }
  return success();
}

LogicalResult CompareOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

template <typename Op, typename ElementType, typename SrcType, typename Convert>
static Attribute CompareFolder(CompareOp op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1]) return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs) return {};

  ShapedType operandType =
      op.getOperand(0).getType().template cast<ShapedType>();
  if (!operandType.hasStaticShape()) {
    return {};
  }

  auto etype = operandType.getElementType();
  if (!etype.isa<ElementType>()) {
    return {};
  }

  // Prevent folding if the result is too large.
  if (lhs.getNumElements() > kFoldOpEltLimit) return {};

  SmallVector<bool, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<SrcType>(), rhs.getValues<SrcType>())) {
    values.push_back(
        Convert()(addSign(std::get<0>(zip), lhs.getElementType()),
                  addSign(std::get<1>(zip), rhs.getElementType())));
  }

  auto resultTy = op.getType().cast<ShapedType>();
  return DenseElementsAttr::get(resultTy, values);
}

OpFoldResult CompareOp::fold(ArrayRef<Attribute> operands) {
  auto resultTy = getType().cast<ShapedType>();
  if (!resultTy.hasStaticShape()) return {};

  auto direction = getComparisonDirection();
  auto lhsTy = getElementTypeOrSelf(getLhs());
  if (getLhs() == getRhs() && !lhsTy.isa<FloatType>() &&
      (!lhsTy.isa<ComplexType>() ||
       !lhsTy.cast<ComplexType>().getElementType().isa<FloatType>())) {
    if (direction == ComparisonDirection::LE ||
        direction == ComparisonDirection::EQ ||
        direction == ComparisonDirection::GE) {
      return DenseIntElementsAttr::get(resultTy, {true});
    }
    return DenseIntElementsAttr::get(resultTy, {false});
  }

  auto opElType = getLhs().getType().cast<ShapedType>().getElementType();
  // Fold tensor<*xi1> != false to just return tensor<*xi1>
  if (direction == ComparisonDirection::NE && opElType.isInteger(1)) {
    DenseIntElementsAttr cstAttr;
    if (matchPattern(getLhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && !cstAttr.getSplatValue<bool>()) {
        return getRhs();
      }
    }

    if (matchPattern(getRhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && !cstAttr.getSplatValue<bool>()) {
        return getLhs();
      }
    }
  }

  // Fold tensor<*xi1> == True to just return tensor<*xi1>
  if (direction == ComparisonDirection::EQ && opElType.isInteger(1)) {
    DenseIntElementsAttr cstAttr;
    if (matchPattern(getLhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && cstAttr.getSplatValue<bool>()) {
        return getRhs();
      }
    }

    if (matchPattern(getRhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && cstAttr.getSplatValue<bool>()) {
        return getLhs();
      }
    }
  }

  if (!operands[0] || !operands[1]) {
    return {};
  }

#define COMPARE_FOLDER(Op, comparison, Func)                                \
  if (direction == comparison) {                                            \
    if (auto folded = CompareFolder<Op, FloatType, APFloat, Func<APFloat>>( \
            *this, operands))                                               \
      return folded;                                                        \
    if (auto folded = CompareFolder<Op, IntegerType, APInt, Func<APSInt>>(  \
            *this, operands))                                               \
      return folded;                                                        \
  }

  COMPARE_FOLDER(CompareOp, ComparisonDirection::EQ, std::equal_to);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::NE, std::not_equal_to);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::LT, std::less);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::LE, std::less_equal);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::GT, std::greater);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::GE, std::greater_equal);
#undef COMPARE_FOLDER

  return {};
}

//===----------------------------------------------------------------------===//
// SelectAndScatterOp
//===----------------------------------------------------------------------===//

namespace {
// Infer the return-type of SelectAndScatterOp.
TensorType inferSelectAndScatterOpReturnType(
    TensorType operandType, const ArrayRef<hlo::WindowDimension> window) {
  if (!operandType.hasRank())
    return UnrankedTensorType::get(operandType.getElementType());

  return RankedTensorType::get(
      inferWindowOutputShape(operandType.getShape(), window),
      operandType.getElementType());
}
}  // namespace

//  We intend to verify the following properties:
//   P1. Check if the select function has a proper shape of (T,T) -> PRED, where
//        T is a 0-D tensor with element-type same as 'operand' element-type.
//   P2. Verify scatter-computation type.
//   P3. size-of(window_dimension) == rank-of(input),
//         where input is an element of 'inputs'.
//   P4. Verify and collect the window attributes.
//   P5. Verify the return type matches the operand-type.
//   P6. Check if the result type of window operation matches the source type.
LogicalResult SelectAndScatterOp::verify() {
  auto operandType = getOperand().getType().cast<TensorType>();
  auto initValueType = getInitValue().getType().cast<TensorType>();
  auto sourceType = getSource().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();

  // P1.
  Block& selectBlock = getSelect().front();

  if (selectBlock.getArguments().size() != 2)
    return emitOpError()
           << "expects the select-region to take 2 parameters, but takes "
           << selectBlock.getArguments().size();

  Type expectedSelectArgType =
      RankedTensorType::get({}, operandType.getElementType());
  for (const auto& selectArgIt : llvm::enumerate(selectBlock.getArguments()))
    if (!hlo::compatibleShapeAndElementType(expectedSelectArgType,
                                            selectArgIt.value().getType(),
                                            /*ignoreFpPrecision=*/true))
      return emitOpError()
             << "expects the type of select-region's parameter at index "
             << selectArgIt.index() << " to be " << expectedSelectArgType
             << ", but got " << selectArgIt.value().getType();

  auto selectResult = selectBlock.getTerminator()->getOperands();
  if (selectResult.size() != 1)
    return emitOpError()
           << "expects select-region to return single value, but got: "
           << selectResult.size();

  auto selectResultType = selectResult[0].getType().dyn_cast<TensorType>();
  if (!selectResultType || !selectResultType.getElementType().isInteger(1) ||
      (selectResultType.hasRank() &&
       selectResultType.cast<RankedTensorType>().getRank() != 0))
    return emitOpError() << "expects the return-type of select-region to be "
                            "tensor<i1>, but got: "
                         << selectResult[0].getType();

  // P2.
  Block& scatterBlock = getScatter().front();
  if (failed(hlo::verifyReducerShape(
          this->getLoc(), scatterBlock,
          {RankedTensorType::get({}, sourceType.getElementType())},
          {initValueType},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/false)))
    return failure();

  // P3.
  SmallVector<int64_t> windowDims =
      convertDenseIntAttr(this->getWindowDimensions());
  if (operandType.hasRank()) {
    if (operandType.getRank() != static_cast<int64_t>(windowDims.size()))
      return emitOpError()
             << "expects window-dimensions size == operand rank, but got "
                "window-dimensions size: "
             << windowDims.size() << " and operand-type: " << operandType
             << " with rank = " << operandType.getRank() << ".";
  }

  // P4.
  auto paddingOrErr = convertNx2Attribute(this->getPadding(), getLoc());
  if (failed(paddingOrErr)) return failure();
  SmallVector<std::pair<int64_t, int64_t>> padding = *paddingOrErr;

  auto windowOrErr = hlo::verifyWindowAttributesAndInferWindowDimensions(
      windowDims, convertDenseIntAttr(getWindowStrides()), padding,
      /*lhs_dilation=*/{}, /*rhs_dilation=*/{}, getLoc());
  if (failed(windowOrErr)) return failure();

  // P5.
  if (!hlo::compatibleShapeAndElementType(operandType, resultType))
    return emitOpError()
           << "expects the return-type to match the operand-type, but got "
           << resultType << " and " << operandType << " resp.";

  // P6.
  auto windowResultType =
      inferSelectAndScatterOpReturnType(operandType, *windowOrErr);

  if (!hlo::compatibleShapeAndElementType(windowResultType, sourceType,
                                          /*ignoreFpPrecision=*/true))
    return emitOpError() << "expects source-type to be " << windowResultType
                         << ", but got" << sourceType;

  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

/*
 * We intend to verify the following properties:
 * P1. The 'update_window_dims' must be valid indices of 'updates' tensor.
 * P2. The 'inserted_window_dims' must be valid indices of 'operand' tensor.
 * P3. Check if the rank-of('operand') == size-of('update_window_dims') +
 *     size-of('inserted_window_dims')
 * P4. size-of('scatter_dims_to_operand_dims') =
 *         'scatter_indices'['index_vector_dim'] &
 *     'scatter_dims_to_operand_dims' must be valid indices of 'operand' tensor.
 */
LogicalResult validateScatterDimensionNumbers(
    ShapedType operandType, ArrayRef<int64_t> scatterIndicesShape,
    ShapedType updateType, bool operandTypeRanked,
    bool scatterIndicesTypeRanked, bool updatesTypeRanked,
    ScatterDimensionNumbersAttr dimNumbers, Location loc) {
  // P1.
  auto updateWindowDims = to_vector(dimNumbers.getUpdateWindowDims());
  if (!llvm::is_sorted(updateWindowDims))
    return mlir::emitError(loc)
           << "Expects update_window_dims to be sorted; got: ["
           << updateWindowDims << "].";

  if (hasDuplicates(updateWindowDims))
    return mlir::emitError(loc)
           << "Expects update_window_dims to not repeat; got: ["
           << updateWindowDims << "].";

  if (updatesTypeRanked) {
    for (int64_t windowDim : updateWindowDims) {
      if (windowDim < 0 || windowDim >= updateType.getRank()) {
        return mlir::emitError(loc)
               << "Expects each element of update_window_dims to be in range "
                  "[0, "
                  "rank-of('updates') i.e. [0, "
               << updateType.getRank() << "). got: " << windowDim << ".";
      }
    }
  }

  // P2.
  auto insertedWindowDims = to_vector(dimNumbers.getInsertedWindowDims());
  if (!llvm::is_sorted(insertedWindowDims))
    return mlir::emitError(loc)
           << "Expects inserted_window_dims to be sorted; got: ["
           << insertedWindowDims << "].";

  if (hasDuplicates(insertedWindowDims))
    return mlir::emitError(loc)
           << "Expects inserted_window_dims to not repeat; got: ["
           << insertedWindowDims << "].";

  if (operandTypeRanked) {
    for (int64_t insertedDim : insertedWindowDims) {
      if (insertedDim < 0 || insertedDim >= operandType.getRank()) {
        return mlir::emitError(loc)
               << "Expects each element of inserted_window_dims to be in range "
                  "[0, rank-of('operand') i.e. [0, "
               << operandType.getRank() << "). got: " << insertedDim << ".";
      }
    }
  }

  // P3.
  if (operandTypeRanked) {
    auto windowSize = updateWindowDims.size() + insertedWindowDims.size();
    if (operandType.getRank() != static_cast<int64_t>(windowSize))
      return mlir::emitError(loc)
             << "Expects rank-of operand to match "
                "size-of('update_window_dims')  + "
                "size-of('inserted_window_dims') i.e. "
             << windowSize << " but got " << operandType.getRank() << ".";
  }

  // P4.
  auto scatterDimsToOperandDims =
      to_vector(dimNumbers.getScatterDimsToOperandDims());
  auto indexVectorDim = dimNumbers.getIndexVectorDim();
  if (scatterIndicesTypeRanked) {
    if (!hlo::isDynamicDimSize(scatterIndicesShape[indexVectorDim]) &&
        static_cast<int64_t>(scatterDimsToOperandDims.size()) !=
            scatterIndicesShape[dimNumbers.getIndexVectorDim()])
      return mlir::emitError(loc)
             << "Scatter op has " << scatterDimsToOperandDims.size()
             << " elements in scatter_dims_to_operand_dims and the bound of "
                "dimension index_vector_dim="
             << dimNumbers.getIndexVectorDim() << " of scatter_indices is "
             << scatterIndicesShape[dimNumbers.getIndexVectorDim()]
             << ". These two numbers must be equal.";
  }

  if (operandTypeRanked) {
    for (int64_t i = 0;
         i < static_cast<int64_t>(scatterDimsToOperandDims.size()); ++i) {
      int64_t scatterDimToOperandDim = scatterDimsToOperandDims[i];
      if (scatterDimToOperandDim < 0 ||
          scatterDimToOperandDim >= operandType.getRank())
        return mlir::emitError(loc)
               << "Invalid scatter_dims_to_operand_dims mapping; domain is [0, "
               << operandType.getRank() << "), got: " << i << "->"
               << scatterDimToOperandDim << ".";
    }
  }

  if (hasDuplicates(scatterDimsToOperandDims))
    return mlir::emitError(loc)
           << "Expects scatter_dims_to_operand_dims to not repeat; got: ["
           << scatterDimsToOperandDims << "].";

  return success();
}
/*
 * We intend to verify the following properties:
 *  P0. scatter_indices argument must be an integral tensor. Enforced by ODS.
 *  P1. Scatter index leaf dimension must be within [0, rank(scatter_indices)"
 *      " + 1).
 *  P2. Verify reducer shape.
 *  P3. rank-of('updates[i]') == size-of('update_window_dims') +
 *      rank-of('scatter_indices') - 1, where 'scatter_indices' is expanded by a
 *      trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')
 *      for all values of `i`.
 *  P4. Validate the scatter-dimensions-numbers.
 *  P5. Valide the bounds of each of the 'updates' w.r.t the operands.
 *  P6. Validate the bounds of each of the 'updates' w.r.t the
 * 'scatter_indices'.
 *  P7. Check return types.
 */
LogicalResult ScatterOp::verify() {
  // Get the first operand and update, since variadic Scatter is not yet
  // implemented
  auto numOperands = getInputs().size();
  auto scatterIndicesType =
      getScatterIndices().getType().dyn_cast<TensorType>();

  SmallVector<TensorType, 1> operandTypes =
      llvm::to_vector(llvm::map_range(getInputs().getTypes(), [](Type type) {
        return type.cast<TensorType>();
      }));
  SmallVector<TensorType, 1> updatesTypes =
      llvm::to_vector(llvm::map_range(getUpdates().getTypes(), [](Type type) {
        return type.cast<TensorType>();
      }));
  bool allOperandTypesRanked =
      llvm::all_of(getInputs().getTypes(),
                   [](Type type) { return type.isa<RankedTensorType>(); });
  bool scatterIndicesTypeRanked = scatterIndicesType.isa<RankedTensorType>();

  // P1.
  int64_t indexVectorDim = getScatterDimensionNumbers().getIndexVectorDim();
  if (scatterIndicesTypeRanked) {
    if (indexVectorDim > scatterIndicesType.getRank() || indexVectorDim < 0)
      return emitOpError()
             << "expects scatter index leaf dimension to be within [0, "
                "rank(scatter_indices) + 1."
                " rank(scatter_indices) is "
             << scatterIndicesType.getRank()
             << " and scatter index leaf dimension is " << indexVectorDim
             << ".";
  }

  // P2.
  Block& block = getUpdateComputation().front();
  SmallVector<TensorType> inputTypes, initValueTypes;
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    inputTypes.push_back(operandTypes[i]);
    initValueTypes.push_back(
        RankedTensorType::get({}, updatesTypes[i].getElementType()));
  }
  if (failed(hlo::verifyReducerShape(
          this->getLoc(), block, inputTypes, initValueTypes, numOperands,
          /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!allOperandTypesRanked)))
    return failure();

  // P3.
  auto updateWindowDims = getScatterDimensionNumbers().getUpdateWindowDims();
  SmallVector<int64_t> expandedScatterIndicesShape;
  if (scatterIndicesTypeRanked) {
    expandedScatterIndicesShape =
        llvm::to_vector(scatterIndicesType.getShape());
    if (static_cast<int64_t>(expandedScatterIndicesShape.size()) ==
        indexVectorDim)
      expandedScatterIndicesShape.push_back(1);
  }

  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    if (scatterIndicesTypeRanked && updatesTypes[i].isa<RankedTensorType>()) {
      int64_t expectedUpdatesRank =
          expandedScatterIndicesShape.size() - 1 + updateWindowDims.size();
      if (updatesTypes[i].getRank() != expectedUpdatesRank)
        return emitOpError()
               << "expects updates tensor must be of rank "
               << expectedUpdatesRank
               << " ( == rank-of('scatter_indices') - 1 + "
                  "size-of('update_window_dims'), where 'scatter_indices' is "
                  "expanded by a trailing 1 dimension if 'index_vector_dim' == "
                  "rank-of('scatter_indices')), but got "
               << updatesTypes[i].getRank() << ".";
    }
  }

  // P4.
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    if (failed(validateScatterDimensionNumbers(
            operandTypes[i], expandedScatterIndicesShape, updatesTypes[i],
            operandTypes[i].isa<RankedTensorType>(), scatterIndicesTypeRanked,
            updatesTypes[i].isa<RankedTensorType>(),
            getScatterDimensionNumbers(), getLoc())))
      return failure();
  }

  // P5.
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    if (updatesTypes[i].isa<RankedTensorType>()) {
      auto updatesShape = updatesTypes[i].getShape();
      if (operandTypes[i].isa<RankedTensorType>()) {
        auto operandShape = operandTypes[i].getShape();
        auto insertedWindowDims =
            getScatterDimensionNumbers().getInsertedWindowDims();

        int64_t insertedDimsSeen = 0;
        SmallVector<int64_t> maxUpdateSliceSizes;
        const auto dimensionsSize = operandTypes[i].getRank();
        maxUpdateSliceSizes.reserve(dimensionsSize);
        for (int i = 0; i < dimensionsSize; ++i) {
          if (insertedDimsSeen <
                  static_cast<int64_t>(insertedWindowDims.size()) &&
              insertedWindowDims[insertedDimsSeen] == i) {
            ++insertedDimsSeen;
          } else {
            maxUpdateSliceSizes.push_back(operandShape[i]);
          }
        }

        for (int64_t i = 0; i < static_cast<int64_t>(updateWindowDims.size());
             ++i) {
          auto updateWindowDim = updateWindowDims[i];

          if (hlo::isDynamicDimSize(updatesShape[updateWindowDim]) ||
              hlo::isDynamicDimSize(maxUpdateSliceSizes[i]))
            continue;

          if (updatesShape[updateWindowDim] > maxUpdateSliceSizes[i]) {
            return emitOpError()
                   << "expects bounds of the window dimensions of "
                      "updates to not exceed the "
                      "bounds of the corresponding dimensions of "
                      "operand. For dimension "
                   << updateWindowDim << ", updates bound is "
                   << updatesShape[updateWindowDim] << ", operand bound is "
                   << maxUpdateSliceSizes[i] << ".";
          }
        }
      }

      // P6.
      if (scatterIndicesTypeRanked) {
        int64_t scatterDimsSeen = 0;
        for (int64_t i = 0; i < static_cast<int64_t>(updatesShape.size());
             ++i) {
          bool isUpdateWindowDim = std::binary_search(
              updateWindowDims.begin(), updateWindowDims.end(), i);

          if (isUpdateWindowDim) continue;
          if (scatterDimsSeen == indexVectorDim) ++scatterDimsSeen;

          if (!hlo::isDynamicDimSize(updatesShape[i]) &&
              !hlo::isDynamicDimSize(
                  expandedScatterIndicesShape[scatterDimsSeen]) &&
              (updatesShape[i] !=
               expandedScatterIndicesShape[scatterDimsSeen])) {
            return emitOpError()
                   << "expects bounds of the scatter dimensions of "
                      "updates to be same as the "
                      "bounds of the corresponding dimensions of "
                      "scatter indices. For "
                      "scatter dimension "
                   << i << ", updates bound is " << updatesShape[i]
                   << " , scatter_indices "
                      "bound is "
                   << expandedScatterIndicesShape[scatterDimsSeen] << ".";
          }
          ++scatterDimsSeen;
        }
      }
    }
  }

  // P7.
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    if (!hlo::compatibleShapeAndElementType(operandTypes[i],
                                            getResult(i).getType()))
      return emitOpError()
             << "expects the return type to be same as the operand type: "
             << operandTypes[i] << ", but got " << getResult(i).getType()
             << ".";
  }

  return success();
}

llvm::SmallVector<Attribute, 4> evaluateMhloRegion(Region& region,
                                                   ArrayRef<Attribute> inputs) {
  if (region.getNumArguments() != inputs.size()) return {};

  llvm::DenseMap<Value, Attribute> values;
  values.reserve(region.getNumArguments());
  for (auto it : llvm::zip(region.getArguments(), inputs)) {
    values.try_emplace(std::get<0>(it), std::get<1>(it));
  }

  for (auto& op : region.getOps()) {
    llvm::SmallVector<Attribute, 4> inputs;
    for (auto& operand : op.getOpOperands()) {
      inputs.push_back(values.lookup(operand.get()));
    }
    if (isa<ReturnOp>(op)) return inputs;

    llvm::SmallVector<OpFoldResult, 4> results;
    if (failed(op.fold(inputs, results))) return {};
    for (auto it : llvm::zip(op.getResults(), results)) {
      if (!std::get<1>(it).is<Attribute>()) return {};
      values.insert({std::get<0>(it), std::get<1>(it).get<Attribute>()});
    }
  }
  return {};
}

LogicalResult ScatterOp::fold(
    ArrayRef<Attribute> args,
    llvm::SmallVectorImpl<OpFoldResult>& foldResults) {
  // Variadic Scatter not yet implemented
  if (getInputs().size() != 1 || getUpdates().size() != 1) return failure();
  auto index = args[1].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!index) return failure();

  auto baseType = getInputs().getTypes()[0].dyn_cast<RankedTensorType>();
  auto updateType = getUpdates().getTypes()[0].dyn_cast<RankedTensorType>();
  auto indexType = index.getType().cast<RankedTensorType>();
  if (!baseType || !indexType || !updateType) return failure();

  // TODO(b/228310289): Work around canonicalization crash for complex types.
  // Remove after upstream MLIR has been fixed.
  if (baseType.getElementType().isa<ComplexType>()) return failure();

  // Catch a trivial full replacement of base with update, this does not require
  // these to be constant: just that we know the type.
  if (updateType == baseType && updateType.hasStaticShape() &&
      baseType.hasStaticShape() && index.isSplat() &&
      index.getSplatValue<uint32_t>() == 0 &&
      llvm::hasSingleElement(getUpdateComputation().front())) {
    foldResults.push_back(getUpdates()[0]);
    return success();
  }
  auto base = args[0].dyn_cast_or_null<DenseElementsAttr>();
  auto update = args[2].dyn_cast_or_null<DenseElementsAttr>();
  if (!base || !update) return failure();

  // Add the virtual trailing dimension of size 1 if indexVectorDim equals to
  // indexType.rank.
  const int64_t indexVectorDim =
      getScatterDimensionNumbers().getIndexVectorDim();
  if (indexVectorDim == indexType.getRank()) {
    auto indexShape = indexType.getShape().vec();
    indexShape.push_back(1);
    indexType = RankedTensorType::get(indexShape, indexType.getElementType());
    index = reshape(index, indexType).cast<DenseIntElementsAttr>();
  }

  // Increment the multi-dimensional index vector based on the limits for each
  // dimension specified by shape and returns false if the index rolled around
  // with true otherwise.
  auto nextIndex = [](llvm::SmallVector<uint64_t, 8>& index,
                      llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (index[i] < static_cast<unsigned long>(shape[i])) return true;
      index[i] = 0;
    }
    return false;
  };

  // Prevent folding if the result is too large.
  if (base.getNumElements() > kFoldOpEltLimit) return failure();

  // Iterate over all elements of the update tensor, then find the corresponding
  // value in the indices tensor to determine which location we have to update
  // in the base/result tensor.
  llvm::SmallVector<Attribute, 8> results(base.getValues<Attribute>());
  llvm::SmallVector<uint64_t, 8> updateIndex(updateType.getRank(), 0);
  llvm::SmallVector<uint64_t, 8> indexIndex;
  indexIndex.reserve(indexType.getRank());
  llvm::SmallVector<int64_t, 8> baseIndex;
  baseIndex.reserve(baseType.getRank());
  do {
    // Compute the index for the slice of the indices tensor for this update
    // value.
    indexIndex.clear();
    if (indexVectorDim == 0) indexIndex.push_back(0);
    for (int64_t i = 0; i < static_cast<int64_t>(updateIndex.size()); ++i) {
      if (llvm::count(getScatterDimensionNumbers().getUpdateWindowDims(), i) ==
          0)
        indexIndex.push_back(updateIndex[i]);
      if (static_cast<int64_t>(indexIndex.size()) == indexVectorDim)
        indexIndex.push_back(0);
    }

    // Compute the index for the given update value in the base tensor.
    baseIndex.assign(baseType.getRank(), 0);
    uint64_t indexCount = indexType.getShape()[indexVectorDim];
    for (uint64_t i = 0; i < indexCount; ++i) {
      uint64_t operandDim =
          getScatterDimensionNumbers().getScatterDimsToOperandDims()[i];
      indexIndex[indexVectorDim] = i;
      baseIndex[operandDim] +=
          index.getValues<APInt>()[indexIndex].getSExtValue();
    }
    uint64_t updateWindowDimIndex = 0;
    auto insertedWindowDims =
        getScatterDimensionNumbers().getInsertedWindowDims();
    auto updateWindowDims = getScatterDimensionNumbers().getUpdateWindowDims();
    for (uint64_t i = 0; i < baseIndex.size(); ++i) {
      if (llvm::count(insertedWindowDims, i)) continue;
      baseIndex[i] += updateIndex[updateWindowDims[updateWindowDimIndex]];
      updateWindowDimIndex++;
    }

    // Compute the linear index for the index into the base tensor.
    int64_t linearBaseIndex = 0;
    int64_t linearBaseIndexMultiplyer = 1;
    for (int64_t i = baseIndex.size() - 1; i >= 0; --i) {
      // Out of bound index have backend specific behaviour so avoid folding it.
      if (baseIndex[i] < 0 || baseIndex[i] >= baseType.getShape()[i])
        return failure();
      linearBaseIndex += baseIndex[i] * linearBaseIndexMultiplyer;
      linearBaseIndexMultiplyer *= baseType.getShape()[i];
    }

    // Evaluate update computation and update the value with the newly computed
    // attribute in the base tensor.
    auto lhs = DenseElementsAttr::get(
        RankedTensorType::get({}, baseType.getElementType()),
        results[linearBaseIndex]);
    auto rhs = DenseElementsAttr::get(
        RankedTensorType::get({}, baseType.getElementType()),
        update.getValues<Attribute>()[updateIndex]);
    auto newValue = evaluateMhloRegion(getUpdateComputation(), {lhs, rhs});
    if (newValue.size() != 1 || !newValue[0]) return failure();
    results[linearBaseIndex] =
        newValue[0].cast<DenseElementsAttr>().getValues<Attribute>()[0];
  } while (nextIndex(updateIndex, updateType.getShape()));

  foldResults.push_back(DenseElementsAttr::get(baseType, results));
  return success();
}

// Replace mhlo.scatter overwriting the entire input with mhlo.map.
struct ScatterFullReplace : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp scatter,
                                PatternRewriter& rewriter) const override {
    // Variadic Scatter not yet implemented
    if (scatter.getInputs().size() != 1 || scatter.getUpdates().size() != 1)
      return failure();

    auto baseType =
        scatter.getInputs().getTypes()[0].dyn_cast<RankedTensorType>();
    auto updateType =
        scatter.getUpdates().getTypes()[0].dyn_cast<RankedTensorType>();
    auto indexType =
        scatter.getScatterIndices().getType().dyn_cast<RankedTensorType>();
    if (!baseType || !indexType || !updateType) return failure();

    // If updates is an empty shape, scatter overwrites the entire tensor.
    // Transform it into a map with the combiner function.
    if (!indexType.hasStaticShape() || indexType.getNumElements() > 0)
      return failure();

    // Require the same shape for base and updates. This isn't strictly
    // necessary, but handling other cases would require turning scatter options
    // into the appropriate reshapes and transposes.
    if (!baseType.hasStaticShape() || !updateType.hasStaticShape() ||
        baseType != updateType)
      return failure();

    auto dimensions =
        llvm::to_vector(llvm::seq<int64_t>(0, baseType.getRank()));
    auto map = rewriter.create<mhlo::MapOp>(
        scatter.getLoc(), scatter->getResultTypes(),
        ValueRange{scatter.getOperands()[0], scatter.getUpdates()[0]},
        rewriter.getI64TensorAttr(dimensions));
    rewriter.inlineRegionBefore(scatter.getRegion(), map.getRegion(),
                                map.getRegion().begin());
    rewriter.replaceOp(scatter, map->getResults());
    return success();
  }
};

void ScatterOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<ScatterFullReplace>(context);
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  WhileOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferWhileOp(location, adaptor.getOperand(), adaptor.getCond(),
                           adaptor.getBody(), inferredReturnTypes);
}

/// Print a `while` op.
///
/// op ::= `mhlo.while` `(` assignment-list `)` `:` types attribute-dict
///         `cond` region
///         `do` region
/// assignment-list ::= assignment | assignment `,` assignment-list
/// assignment ::= ssa-value `=` ssa-value
void WhileOp::print(OpAsmPrinter& p) {
  p << '(';
  llvm::interleaveComma(
      llvm::zip(SingleBlock::getBody()->getArguments(), getOperands()), p,
      [&](auto zip) {
        p.printOperand(std::get<0>(zip));
        p << " = ";
        p.printOperand(std::get<1>(zip));
      });
  p << ")";
  if (getNumOperands()) {
    p << " : ";
    llvm::interleaveComma(getOperandTypes(), p);
  }
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p.printNewline();
  p << " cond ";
  p.printRegion(getRegion(0), /*printEntryBlockArgs=*/false);
  p << " do ";
  p.printRegion(getRegion(1), /*printEntryBlockArgs=*/false);
}

ParseResult WhileOp::parse(OpAsmParser& parser, OperationState& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse the operands of the while: these are of the form:
  //   %iter_arg = %init_val
  // where %iter_arg is the name of the block argument in the cond/body blocks
  // and %init_val is the actual operand.
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<OpAsmParser::UnresolvedOperand> iterArgs;
  if (parser.parseLParen()) return failure();
  do {
    if (succeeded(parser.parseOptionalRParen())) break;
    OpAsmParser::UnresolvedOperand operand, iterArg;
    if (parser.parseOperand(iterArg) || parser.parseEqual() ||
        parser.parseOperand(operand))
      return failure();
    iterArgs.push_back(iterArg);
    operands.push_back(operand);
    if (succeeded(parser.parseOptionalRParen())) break;
    if (failed(parser.parseComma())) return failure();
  } while (true);
  if (!operands.empty()) {
    if (parser.parseColon() || parser.parseTypeList(result.types))
      return failure();
  }

  SmallVector<OpAsmParser::Argument> args;
  createArgs(iterArgs, result.types, args);
  if (parser.resolveOperands(operands, result.types, loc, result.operands) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseKeyword("cond") ||
      parser.parseRegion(*result.addRegion(), args) ||
      parser.parseKeyword("do") ||
      parser.parseRegion(*result.addRegion(), args))
    return failure();
  return success();
}

LogicalResult WhileOp::fold(ArrayRef<Attribute> /*operands*/,
                            SmallVectorImpl<OpFoldResult>& results) {
  DenseIntElementsAttr condValue;
  auto condReturnOp = cast<ReturnOp>(getCond().front().back());
  if (!matchPattern(condReturnOp.getOperand(0), m_Constant(&condValue)))
    return failure();
  if (condValue.getSplatValue<BoolAttr>().getValue())
    return failure();  // TODO(mhlo): this is an infinite loop, should we fold?

  results.append(getOperands().begin(), getOperands().end());
  return success();
}

static LogicalResult whileCanonicalization(WhileOp whileOp,
                                           PatternRewriter& rewriter) {
  // Turn loop invariant values into implicit capture.
  // Check if there is at least one value is forwarded from one iteration to the
  // next, or one of the yielded value is an implicit capture already. Otherwise
  // there is nothing to do here.
  Block* cond = whileOp.SingleBlock::getBody(0);
  Block* body = whileOp.SingleBlock::getBody(1);
  auto bodyReturnOp = cast<ReturnOp>(body->getTerminator());
  if (!llvm::any_of(llvm::zip(whileOp->getOperands(), body->getArguments(),
                              bodyReturnOp->getOperands()),
                    [&](auto zip) {
                      return (std::get<0>(zip) == std::get<2>(zip) ||
                              std::get<1>(zip) == std::get<2>(zip));
                    }))
    return rewriter.notifyMatchFailure(whileOp, "no loop invariant found");

  SmallVector<Value> newOperands, resultsToReplace;
  SmallVector<unsigned> invariantArgIdxs;
  BitVector invariantArgIdxBitVector(cond->getNumArguments());
  for (const auto& enumeratedOperands : llvm::enumerate(llvm::zip(
           whileOp.getOperands(), cond->getArguments(), body->getArguments(),
           bodyReturnOp->getOperands(), whileOp->getResults()))) {
    const auto& operands = enumeratedOperands.value();
    Value whileOperand = std::get<0>(operands);
    BlockArgument condBlockArg = std::get<1>(operands);
    BlockArgument bodyBlockArg = std::get<2>(operands);
    Value bodyReturnOperand = std::get<3>(operands);
    Value whileResult = std::get<4>(operands);

    bool forwarded = (whileOperand == bodyReturnOperand ||
                      bodyBlockArg == bodyReturnOperand);
    if (forwarded) {
      invariantArgIdxs.push_back(enumeratedOperands.index());
      invariantArgIdxBitVector.set(enumeratedOperands.index());
      condBlockArg.replaceAllUsesWith(whileOperand);
      bodyBlockArg.replaceAllUsesWith(whileOperand);
      whileResult.replaceAllUsesWith(whileOperand);
      continue;
    }
    newOperands.push_back(whileOperand);
    resultsToReplace.push_back(whileResult);
  }
  cond->eraseArguments(invariantArgIdxBitVector);
  body->eraseArguments(invariantArgIdxBitVector);
  for (int idx : llvm::reverse(invariantArgIdxs))
    bodyReturnOp->eraseOperand(idx);

  WhileOp newWhileOp = rewriter.create<WhileOp>(
      whileOp.getLoc(), bodyReturnOp->getOperandTypes(), newOperands);
  newWhileOp.getBodyRegion(0).takeBody(whileOp.getBodyRegion(0));
  newWhileOp.getBodyRegion(1).takeBody(whileOp.getBodyRegion(1));
  for (auto results : llvm::zip(resultsToReplace, newWhileOp->getResults()))
    std::get<0>(results).replaceAllUsesWith(std::get<1>(results));
  rewriter.eraseOp(whileOp);
  return success();
}

void WhileOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add(&whileCanonicalization);
}

LogicalResult UniformDequantizeOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> /*location*/, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  UniformDequantizeOp::Adaptor adaptor(operands, attributes, regions);
  auto operandType = (*operands.begin()).getType().cast<ShapedType>();
  // Trait HLO_QuantizedIntTensor in ODS guarantees QuantizedType;
  auto quantType = operandType.getElementType().cast<quant::QuantizedType>();
  auto shape = operandType.dyn_cast<ShapedType>().getShape();
  inferredReturnShapes.emplace_back(shape, quantType.getExpressedType());
  return success();
}

using mlir::hlo::parseWindowAttributes;
using mlir::hlo::printWindowAttributes;

}  // namespace mhlo
}  // namespace mlir

// clang-format off
using mlir::hlo::printSameOperandsAndResultType;
using mlir::hlo::parseSameOperandsAndResultType;
using mlir::hlo::printVariadicSameOperandsAndResultType;
using mlir::hlo::parseVariadicSameOperandsAndResultType;
using mlir::hlo::printComplexOpType;
using mlir::hlo::parseComplexOpType;
using mlir::hlo::printPairwiseOpType;
using mlir::hlo::parsePairwiseOpType;
using mlir::hlo::printSelectOpType;
using mlir::hlo::parseSelectOpType;
using mlir::hlo::printTupleOpType;
using mlir::hlo::parseTupleOpType;
using mlir::hlo::printCustomCallTarget;
using mlir::hlo::parseCustomCallTarget;
using mlir::hlo::printExponentMantissa;
using mlir::hlo::parseExponentMantissa;
// clang-format on

#define GET_OP_CLASSES
#include "mhlo/IR/hlo_ops.cc.inc"

namespace mlir {
namespace mhlo {

//===----------------------------------------------------------------------===//
// mhlo Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct HLOInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       BlockAndValueMapping& valueMapping) const final {
    return true;
  }
  // Operations in mhlo dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation*, Region*, bool,
                       BlockAndValueMapping&) const final {
    return true;
  }
};

struct HLOBoundedDialectInterface : public hlo::BoundedDialectInterface {
  using BoundedDialectInterface::BoundedDialectInterface;

  Attribute createBoundedAttr(ArrayRef<int64_t> bounds) const override {
    return TypeExtensionsAttr::get(getDialect()->getContext(), bounds);
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// mhlo Dialect Constructor
//===----------------------------------------------------------------------===//

MhloDialect::MhloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mhlo/IR/hlo_ops.cc.inc"
      >();
  addInterfaces<HLOBoundedDialectInterface>();
  addInterfaces<HLOInlinerInterface>();
  addBytecodeInterface(this);
  addTypes<TokenType, AsyncBundleType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mhlo/IR/hlo_ops_attrs.cc.inc"
      >();
  context->loadDialect<tensor::TensorDialect>();
}

Type MhloDialect::parseType(DialectAsmParser& parser) const {
  StringRef mnemonic;
  Type parsedType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, parsedType);
  if (parseResult.has_value()) return parsedType;
  if (mnemonic == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc()) << "unknown mhlo type: " << mnemonic;
  return nullptr;
}

void MhloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<TokenType>()) {
    os << "token";
    return;
  }
  if (succeeded(generatedTypePrinter(type, os))) return;
  os << "<unknown mhlo type>";
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute MhloDialect::parseAttribute(DialectAsmParser& parser,
                                      Type type) const {
  StringRef attrTag;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) return attr;
  parser.emitError(parser.getNameLoc(), "unknown mhlo attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void MhloDialect::printAttribute(Attribute attr, DialectAsmPrinter& os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  (void)result;
  assert(succeeded(result));
}

/// Helpers for attributes parsing.
static ParseResult parseDims(AsmParser& parser, SmallVector<int64_t>& dims) {
  dims.clear();
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
    dims.emplace_back();
    return parser.parseInteger(dims.back());
  });
}

static ParseResult parseDimsWithMinimumElements(AsmParser& parser,
                                                SmallVector<int64_t>& dims,
                                                int minElements) {
  if (failed(parseDims(parser, dims))) return failure();
  if (static_cast<int64_t>(dims.size()) < minElements)
    return parser.emitError(parser.getCurrentLocation())
           << "expected at least " << minElements << " element(s), found "
           << dims.size();
  return success();
}

FailureOr<SmallVector<int64_t>> parseIntArray(AsmParser& parser) {
  SmallVector<int64_t> ints;
  if (failed(parseDims(parser, ints))) return failure();
  return ints;
}

void printIntArray(AsmPrinter& printer, ArrayRef<int64_t> ints) {
  printer << '[';
  llvm::interleaveComma(ints, printer);
  printer << ']';
}

/// Parse a custom attribute that resembles a struct of the form
/// <
///   foo = something_parsed_by_custom_parser,
///   bar = something_parsed_by_different_custom_parser,
///   baz something_parsed_by_another_custom_parser
/// >
/// The optional argument `parse_equal` array can be used to denote if
/// '=' follows the keyword (see baz in the example above) for a field. If
/// not provided, all fields must be followed by a '='.
static ParseResult parseStruct(
    AsmParser& parser, ArrayRef<StringRef> keywords,
    ArrayRef<llvm::function_ref<ParseResult()>> parseFuncs,
    ArrayRef<bool> parseEqual = {}) {
  assert(keywords.size() == parseFuncs.size());
  assert(parseEqual.empty() || parseEqual.size() == keywords.size());
  SmallVector<bool> seen(keywords.size(), false);
  while (failed(parser.parseOptionalGreater())) {
    bool foundOne = false;
    for (const auto& it : llvm::enumerate(keywords)) {
      size_t index = it.index();
      StringRef keyword = it.value();
      if (succeeded(parser.parseOptionalKeyword(keyword))) {
        if (seen[index]) {
          return parser.emitError(parser.getCurrentLocation())
                 << "duplicated `" << keyword << "` entry";
        }
        if (parseEqual.empty() || parseEqual[index]) {
          if (failed(parser.parseEqual())) return failure();
        }
        if (failed(parseFuncs[index]())) return failure();
        if (failed(parser.parseOptionalComma())) return parser.parseGreater();
        seen[index] = true;
        foundOne = true;
      }
    }
    if (!foundOne) {
      auto parseError = parser.emitError(parser.getCurrentLocation())
                        << "expected one of: ";
      llvm::interleaveComma(keywords, parseError, [&](StringRef kw) {
        parseError << '`' << kw << '`';
      });
      return parseError;
    }
  }
  return success();
}

// Helpers to print an optional array or integer field, to simplify writing
// attribute printers.
template <typename T>
static void printField(AsmPrinter& printer, StringRef name, T field,
                       StringRef& separator) {
  if (field != 0) {
    printer << separator << name << " = " << field;
    separator = ", ";
  }
}
template <typename T>
static void printField(AsmPrinter& printer, StringRef name, ArrayRef<T> field,
                       StringRef& separator) {
  if (!field.empty()) {
    printer << separator << name << " = [";
    llvm::interleaveComma(field, printer);
    printer << "]";
    separator = ", ";
  }
}

template <typename... Ts>
static void printStruct(AsmPrinter& printer, StringRef name,
                        Ts... printFields) {
  printer << "<";
  StringRef separator = "";
  // Fold expression to print each entry in the parameter pack.
  // TODO(mhlo-team): this can be simplified when TF moves to C++17.
  using unused = int[];
  (void)unused{0, (printField(printer, std::get<0>(printFields),
                              std::get<1>(printFields), separator),
                   0)...};
  printer << ">";
}

// Custom printer and parser for ScatterDimensionNumbersAttr.
void ScatterDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(printer, "scatter",
              std::make_pair("update_window_dims", getUpdateWindowDims()),
              std::make_pair("inserted_window_dims", getInsertedWindowDims()),
              std::make_pair("scatter_dims_to_operand_dims",
                             getScatterDimsToOperandDims()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}
Attribute ScatterDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  SmallVector<int64_t> updateWindowDims;
  SmallVector<int64_t> insertedWindowDims;
  SmallVector<int64_t> scatterDimsToOperandDims;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"update_window_dims", "inserted_window_dims",
           "scatter_dims_to_operand_dims", "index_vector_dim"},
          {[&]() { return parseDims(parser, updateWindowDims); },
           [&]() { return parseDims(parser, insertedWindowDims); },
           [&]() { return parseDims(parser, scatterDimsToOperandDims); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing scatter dimension numbers attribute";
    return {};
  }

  return ScatterDimensionNumbersAttr::get(
      parser.getContext(), updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims, indexVectorDim);
}

// Custom printer and parser for GatherDimensionNumbersAttr.
void GatherDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(printer, "gather", std::make_pair("offset_dims", getOffsetDims()),
              std::make_pair("collapsed_slice_dims", getCollapsedSliceDims()),
              std::make_pair("start_index_map", getStartIndexMap()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute GatherDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};

  SmallVector<int64_t> offsetDims;
  SmallVector<int64_t> collapsedSliceDims;
  SmallVector<int64_t> startIndexMap;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"offset_dims", "collapsed_slice_dims", "start_index_map",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, offsetDims); },
           [&]() { return parseDims(parser, collapsedSliceDims); },
           [&]() { return parseDims(parser, startIndexMap); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing gather dimension numbers attribute";
    return {};
  }

  return GatherDimensionNumbersAttr::get(parser.getContext(), offsetDims,
                                         collapsedSliceDims, startIndexMap,
                                         indexVectorDim);
}

namespace {

void printCommaSeparatedDynamicShapes(AsmPrinter& printer,
                                      llvm::ArrayRef<int64_t> shape) {
  printer << '[';
  auto printIntOrQuestion = [&](int64_t value) {
    if (ShapedType::isDynamic(value))
      printer << '?';
    else
      printer << value;
  };
  llvm::interleaveComma(shape, printer, printIntOrQuestion);
  printer << ']';
}

ParseResult parseCommaSeparatedDynamicShapes(AsmParser& parser,
                                             SmallVectorImpl<int64_t>& shape) {
  auto parseElt = [&]() -> ParseResult {
    if (!parser.parseOptionalQuestion()) {
      shape.push_back(ShapedType::kDynamic);
      return success();
    }
    return parser.parseInteger(shape.emplace_back());
  };
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseElt);
}

}  // namespace

void TypeExtensionsAttr::print(AsmPrinter& printer) const {
  printer << "<bounds = ";
  printCommaSeparatedDynamicShapes(printer, getBounds());
  printer << ">";
}

Attribute TypeExtensionsAttr::parse(AsmParser& parser, mlir::Type) {
  if (parser.parseLess() || parser.parseKeyword("bounds") ||
      parser.parseEqual())
    return {};

  SmallVector<int64_t> resultBounds;
  if (parseCommaSeparatedDynamicShapes(parser, resultBounds)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse TypeExtensions parameter 'bounds' which "
                     "is to be a `::llvm::ArrayRef<int64_t>`");
    return {};
  }
  if (parser.parseGreater()) return {};
  return TypeExtensionsAttr::get(parser.getContext(), resultBounds);
}

// Custom printer and parser for DotDimensionNumbersAttr.
void DotDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(
      printer, "dot",
      std::make_pair("lhs_batching_dimensions", getLhsBatchingDimensions()),
      std::make_pair("rhs_batching_dimensions", getRhsBatchingDimensions()),
      std::make_pair("lhs_contracting_dimensions",
                     getLhsContractingDimensions()),
      std::make_pair("rhs_contracting_dimensions",
                     getRhsContractingDimensions()));
}

Attribute DotDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};

  SmallVector<int64_t> lhsBatchingDimensions;
  SmallVector<int64_t> rhsBatchingDimensions;
  SmallVector<int64_t> lhsContractingDimensions;
  SmallVector<int64_t> rhsContractingDimensions;

  if (failed(parseStruct(
          parser,
          {"lhs_batching_dimensions", "rhs_batching_dimensions",
           "lhs_contracting_dimensions", "rhs_contracting_dimensions"},
          {[&]() { return parseDims(parser, lhsBatchingDimensions); },
           [&]() { return parseDims(parser, rhsBatchingDimensions); },
           [&]() { return parseDims(parser, lhsContractingDimensions); },
           [&]() { return parseDims(parser, rhsContractingDimensions); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return {};
  }
  return DotDimensionNumbersAttr::get(
      parser.getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

namespace {
enum NonSpatialDim : int64_t {
  IOBatch = -1,    // Input or output batch dimension
  IOFeature = -2,  // Input or output feature dimension
  KIFeature = -3,  // Kernel input feature dimension
  KOFeature = -4,  // Kernel output feature dimensions.
};

struct DenseMapInfoNonSpatialDim {
  static inline NonSpatialDim getEmptyKey() {
    return NonSpatialDim(DenseMapInfo<int64_t>::getEmptyKey());
  }

  static inline NonSpatialDim getTombstoneKey() {
    return NonSpatialDim(DenseMapInfo<int64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const NonSpatialDim& key) {
    return DenseMapInfo<int64_t>::getHashValue(key);
  }

  static bool isEqual(const NonSpatialDim& lhs, const NonSpatialDim& rhs) {
    return lhs == rhs;
  }
};

char nonSpatialDimToString(NonSpatialDim dim) {
  switch (dim) {
    case IOBatch:
      return 'b';
    case IOFeature:
      return 'f';
    case KIFeature:
      return 'i';
    case KOFeature:
      return 'o';
  }
  llvm_unreachable("Unknown NonSpatialDim");
}
}  // namespace

// Custom printer and parser for convolution attribute.
void printConvolutionDimensions(AsmPrinter& p, ConvDimensionNumbersAttr dnums) {
  // TODO(b/202040055): we should check the attribute invariant and print the
  // "raw" form if they are violated, otherwise we'll crash here.
  constexpr int64_t kUnknownDim = std::numeric_limits<int64_t>::min();
  auto printDim =
      [&](ArrayRef<int64_t> spatialDims,
          ArrayRef<std::pair<int64_t, NonSpatialDim>> nonSpatialDims) {
        int64_t numDims = 0;
        if (!spatialDims.empty()) {
          numDims =
              *std::max_element(spatialDims.begin(), spatialDims.end()) + 1;
        }
        for (const auto& dim : nonSpatialDims) {
          numDims = std::max(numDims, dim.first + 1);
        }

        llvm::SmallVector<int64_t> dims(numDims, kUnknownDim);
        // Fill each element of dims with a (< 0) NonSpatialDim enum or a (>=0)
        // spatial dimension index.
        for (const std::pair<int64_t, NonSpatialDim>& nonSpatialDim :
             nonSpatialDims) {
          dims[nonSpatialDim.first] = nonSpatialDim.second;
        }
        for (const auto& spatialDim : llvm::enumerate(spatialDims)) {
          dims[spatialDim.value()] = static_cast<int64_t>(spatialDim.index());
        }

        // Each dimension numbers will be printed as a comma separated list
        // surrounded by square brackets, e.g., [b, 0, 1, 2, f]
        p << '[';
        llvm::interleaveComma(dims, p, [&](int64_t dim) {
          if (dim == kUnknownDim) {
            p << "?";
          } else if (dim >= 0) {
            p << dim;
          } else {
            p << nonSpatialDimToString(static_cast<NonSpatialDim>(dim));
          }
        });
        p << ']';
      };

  printDim(dnums.getInputSpatialDimensions(),
           {{dnums.getInputBatchDimension(), IOBatch},
            {dnums.getInputFeatureDimension(), IOFeature}});
  p << "x";
  printDim(dnums.getKernelSpatialDimensions(),
           {{dnums.getKernelInputFeatureDimension(), KIFeature},
            {dnums.getKernelOutputFeatureDimension(), KOFeature}});
  p << "->";
  printDim(dnums.getOutputSpatialDimensions(),
           {{dnums.getOutputBatchDimension(), IOBatch},
            {dnums.getOutputFeatureDimension(), IOFeature}});
}

void printConvolutionDimensions(AsmPrinter& p, Operation*,
                                ConvDimensionNumbersAttr dnums) {
  printConvolutionDimensions(p, dnums);
}

// Custom printer and parser for ConvDimensionNumbersAttr.
void ConvDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printer << "<";
  printConvolutionDimensions(printer, *this);
  printer << ">";
}

// If the attribute is written with `#mhlo.conv raw<`, we parse it as a struct
// instead of the compressed format. This enables writing tests covering
// impossible/invalid internal representation for the attribute.
static ParseResult parseConvolutionDimensionsRaw(
    AsmParser& parser, ConvDimensionNumbersAttr& dnums) {
  int64_t inputBatchDimension = 0;
  int64_t inputFeatureDimension = 0;
  SmallVector<int64_t> inputSpatialDimensions;
  int64_t kernelInputFeatureDimension = 0;
  int64_t kernelOutputFeatureDimension = 0;
  SmallVector<int64_t> kernelSpatialDimensions;
  int64_t outBatchDimension = 0;
  int64_t outputFeatureDimension = 0;
  SmallVector<int64_t> outputSpatialDimensions;
  if (failed(parseStruct(
          parser,
          {"input_batch_dimension", "input_feature_dimension",
           "input_spatial_dimensions", "kernel_input_feature_dimension",
           "kernel_output_feature_dimension", "kernel_spatial_dimensions",
           "output_batch_dimension", "output_feature_dimension",
           "output_spatial_dimensions"},
          {
              [&]() { return parser.parseInteger(inputBatchDimension); },
              [&]() { return parser.parseInteger(inputFeatureDimension); },
              [&]() { return parseDims(parser, inputSpatialDimensions); },
              [&]() {
                return parser.parseInteger(kernelInputFeatureDimension);
              },
              [&]() {
                return parser.parseInteger(kernelOutputFeatureDimension);
              },
              [&]() { return parseDims(parser, kernelSpatialDimensions); },
              [&]() { return parser.parseInteger(outBatchDimension); },
              [&]() { return parser.parseInteger(outputFeatureDimension); },
              [&]() { return parseDims(parser, outputSpatialDimensions); },
          }))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return failure();
  }
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), inputBatchDimension,
      inputFeatureDimension, inputSpatialDimensions,
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      kernelSpatialDimensions, outBatchDimension, outputFeatureDimension,
      outputSpatialDimensions);
  return success();
}

ParseResult parseConvolutionDimensions(AsmParser& parser,
                                       ConvDimensionNumbersAttr& dnums) {
  // Parsing a single set of dim numbers gives the spatial dimensions as a
  // single ArrayRef<int64_t> and a list of non-spatial dimensions as
  // IntegerAttrs (indexed by the NonSpatialDim enum).
  using parse_dim_result_t =
      std::pair<llvm::SmallVector<int64_t>,
                llvm::SmallDenseMap<NonSpatialDim, int64_t, 4,
                                    DenseMapInfoNonSpatialDim>>;

  // Note that the allowed_non_spatial_dims is a set (as opposed to unordered
  // set) because its used to print a list of allowed non spatial dims in the
  // error messages, so making it a set keeps the error messages deterministic.
  auto parseDims =
      [&](std::set<NonSpatialDim, std::greater<>> allowedNonSpatialDims,
          parse_dim_result_t& parsedDims) -> ParseResult {
    auto& spatialDims = std::get<0>(parsedDims);
    auto& nonSpatialDims = std::get<1>(parsedDims);
    spatialDims.clear();
    nonSpatialDims.clear();

    // Parse the starting [
    if (parser.parseLSquare()) {
      return failure();
    }

    llvm::SmallDenseMap<int64_t, int64_t> spatialDimsMap;
    constexpr int64_t kInvalidDimension = -1;
    // Keep track of the maximum spatial dimension parsed as we expect to see
    // all the dimensions from 0 to maximum dimension parsed.
    int64_t maxParsedSpatialDim = kInvalidDimension;

    int64_t index = 0;
    do {
      int64_t spatialDim;
      auto dimLocation = parser.getCurrentLocation();
      OptionalParseResult parseResult = parser.parseOptionalInteger(spatialDim);
      if (parseResult.has_value()) {
        if (parseResult.value().failed()) {
          return failure();
        }
        // We were successful in parsing an integer. Check if it is a valid
        // dimension (non-negative and no duplicate) and add its index to the
        // spatial dims map.
        if (spatialDim < 0)
          return parser.emitError(dimLocation)
                 << "Unexpected dimension " << spatialDim;
        if (!spatialDimsMap
                 .insert(std::pair<int64_t, int64_t>(spatialDim, index))
                 .second)
          return parser.emitError(dimLocation)
                 << "Duplicate entries for spatial dimension " << spatialDim;
        maxParsedSpatialDim = std::max(spatialDim, maxParsedSpatialDim);
      } else if (!parser.parseOptionalQuestion()) {
        // Do nothing other than increment `index` at the bottom of the loop;
        // '?' means "unknown dimension", and it's not represented in the
        // return value of this function.
      } else {
        // We did not parse an integer or question mark. We expect a keyword
        // token.
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) {
          return failure();
        }
        if (keyword.size() != 1 || allowedNonSpatialDims.empty()) {
          return parser.emitError(dimLocation, "Unexpected keyword ")
                 << keyword;
        }
        // Check if the keyword matches one of the allowed non-spatial dims.
        // If so, add it to the non_spatial dims and remove it from the
        // allowed set so that it won't be allowed again.
        bool isAllowed = false;
        for (NonSpatialDim allowed : allowedNonSpatialDims) {
          if (keyword[0] == nonSpatialDimToString(allowed)) {
            nonSpatialDims.insert({allowed, index});
            allowedNonSpatialDims.erase(allowed);
            isAllowed = true;
            break;
          }
        }

        if (!isAllowed) {
          mlir::InFlightDiagnostic diag =
              parser.emitError(dimLocation, "Unexpected dimension ");
          diag << keyword << ", expecting ";
          llvm::interleaveComma(
              allowedNonSpatialDims, diag,
              [&](NonSpatialDim dim) { diag << nonSpatialDimToString(dim); });
          return diag;
        }
      }
      index++;
    } while (parser.parseOptionalComma().succeeded());

    // Make sure all expected non-spatial dimensions are parsed.
    if (!allowedNonSpatialDims.empty()) {
      mlir::InFlightDiagnostic diag =
          parser.emitError(parser.getCurrentLocation(), "Expected dimensions ");
      llvm::interleaveComma(
          allowedNonSpatialDims, diag,
          [&](NonSpatialDim dim) { diag << nonSpatialDimToString(dim); });
      diag << " not specified";
      return diag;
    }

    // parse ending ]
    if (parser.parseRSquare()) {
      return failure();
    }

    // Number of expected spatial dimensions is one more than the maximum parsed
    // spatial dimension. For example, if we parse [0, 3, 2, b, i, 1], then the
    // maximum parsed spatial dimension is 3 and the number of expected spatial
    // dimensions is 4.
    int64_t numSpatialDimensions = maxParsedSpatialDim + 1;
    spatialDims.resize(numSpatialDimensions);
    // Store spatial dimensions in a vector which maps spatial dim (vector
    // index) -> index in the tensor dimensions. For example, for parsed
    // dimension numbers [0, 3, 2, b, i, 1] the spatial dimension vector would
    // be [0, 5, 2, 1].
    //
    // Get all the unspecified spatial dimensions to throw a more descriptive
    // error later.
    llvm::SmallVector<int64_t> unspecifiedSpatialDims;
    constexpr int kPrintUnspecifiedDimsMax = 10;
    for (int dim = 0; dim < numSpatialDimensions; ++dim) {
      auto it = spatialDimsMap.find(dim);
      if (it == spatialDimsMap.end()) {
        // Have an upper bound on the number of unspecified dimensions to print
        // in the error message.
        if (unspecifiedSpatialDims.size() < kPrintUnspecifiedDimsMax)
          unspecifiedSpatialDims.push_back(dim);
        continue;
      }
      spatialDims[dim] = it->second;
    }

    // Verify that we got all spatial dimensions between 0 and maximum parsed
    // spatial dimension.
    if (!unspecifiedSpatialDims.empty()) {
      mlir::InFlightDiagnostic diag = parser.emitError(
          parser.getCurrentLocation(), "Expected spatial dimensions ");
      llvm::interleaveComma(unspecifiedSpatialDims, diag);
      diag << " not specified";
      return diag;
    }

    return success();
  };

  parse_dim_result_t parsedDims;
  if (parseDims({IOBatch, IOFeature}, parsedDims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> inputSpatialDimensions = parsedDims.first;
  int64_t inputBatchDimension = parsedDims.second[IOBatch];
  int64_t inputFeatureDimension = parsedDims.second[IOFeature];
  if (parser.parseKeyword("x")) return failure();
  if (parseDims({KIFeature, KOFeature}, parsedDims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> kernelSpatialDimensions = parsedDims.first;
  int64_t kernelInputFeatureDimension = parsedDims.second[KIFeature];
  int64_t kernelOutputFeatureDimension = parsedDims.second[KOFeature];
  if (parser.parseArrow()) {
    return failure();
  }
  if (parseDims({IOBatch, IOFeature}, parsedDims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> outputSpatialDimensions = parsedDims.first;
  const int64_t outBatchDimension = parsedDims.second[IOBatch];
  const int64_t outputFeatureDimension = parsedDims.second[IOFeature];
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), inputBatchDimension,
      inputFeatureDimension, inputSpatialDimensions,
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      kernelSpatialDimensions, outBatchDimension, outputFeatureDimension,
      outputSpatialDimensions);

  return success();
}

Attribute ConvDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  ConvDimensionNumbersAttr dnums;
  if (succeeded(parser.parseOptionalKeyword("raw"))) {
    if (failed(parseConvolutionDimensionsRaw(parser, dnums))) return {};
    return dnums;
  }
  if (failed(parseConvolutionDimensions(parser, dnums))) return {};
  if (failed(parser.parseGreater())) return {};
  return dnums;
}

// Custom printer and parser for ArgResultAliasAttr.
constexpr char kMustAlias[] = "must_alias";
constexpr char kResult[] = "result_index";
constexpr char kArgTupleIndices[] = "tuple_indices";

void ArgResultAliasAttr::print(AsmPrinter& printer) const {
  printer << "<";

  // The attribute can have empty tuple indices. Only print argument tuple
  // indices if they are non-empty.
  if (!getArgTupleIndices().empty())
    printer << kArgTupleIndices << " = [" << getArgTupleIndices() << "], ";

  // Print the result index followed by any result tuple indices if present.
  printer << kResult << " = [";
  printer << getResultIndex();
  if (!getResultTupleIndices().empty()) {
    printer << ", " << getResultTupleIndices();
  }
  printer << "]";

  // Print the "must_alias" keyword if this is a must alias, otherwise skip.
  if (getIsMustAlias()) printer << ", " << kMustAlias;

  printer << ">";
}

Attribute ArgResultAliasAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  llvm::SmallVector<int64_t> argTupleIndices;
  // The first element of result indices holds the aliased result index and the
  // remaining elements are the result tuple indices.
  llvm::SmallVector<int64_t> resultIndices;
  bool isMustAlias = false;

  // This conveys to parseStruct that keyword "must_alias" (3rd field) is not
  // followed by a "=", but other fields are.
  llvm::SmallVector<bool, 3> parseEqual = {true, true, false};

  if (failed(parseStruct(parser, {kArgTupleIndices, kResult, kMustAlias},
                         {[&]() { return parseDims(parser, argTupleIndices); },
                          [&]() {
                            // Since the first element is the index of result,
                            // at least one element is expected.
                            return parseDimsWithMinimumElements(
                                parser, resultIndices, /*minElements=*/1);
                          },
                          [&]() {
                            // always succeeds if the keyword "must_alias" was
                            // parsed
                            isMustAlias = true;
                            return success();
                          }},
                         parseEqual))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing argument-result alias attribute";
    return {};
  }

  int64_t resultIndex = resultIndices[0];
  auto resultTupleIndices =
      ArrayRef<int64_t>{resultIndices.begin() + 1, resultIndices.end()};

  return ArgResultAliasAttr::get(parser.getContext(), argTupleIndices,
                                 resultIndex, resultTupleIndices, isMustAlias);
}

// Returns the element type pointed to by `indices` in type `t`. If the indices
// are invalid, returns nullptr.
static Type getTypeFromTupleIndices(Type type, ArrayRef<int64_t> indices) {
  Type current = type;
  for (auto index : indices) {
    TupleType tupleType = current.dyn_cast<TupleType>();
    if (!tupleType || index >= static_cast<int64_t>(tupleType.size()))
      return {};
    current = tupleType.getType(index);
  }
  return current;
}

static LogicalResult verifyArgResultAliasAttr(StringAttr attrName,
                                              ArgResultAliasAttr aliasAttr,
                                              unsigned argIndex,
                                              Operation* op) {
  // The attribute can only be applied to function-like operations.
  if (!isa<mlir::FunctionOpInterface>(op))
    return op->emitOpError() << "attribute " << attrName
                             << " can only be used on function-like operations";

  // Verify there are no negative indices.
  auto tupleIndices = llvm::concat<const int64_t>(
      aliasAttr.getArgTupleIndices(), aliasAttr.getResultTupleIndices());
  if (llvm::any_of(tupleIndices, [](const int64_t val) { return val < 0; }) ||
      aliasAttr.getResultIndex() < 0)
    return op->emitOpError()
           << "attribute " << attrName
           << " expects all argument and result indices to be >= 0";

  // Verify that the result index is not out of range. Since the attribute is a
  // function argument attribute, the argument index is always correct when this
  // verifier is called.
  FunctionOpInterface funcOp = cast<FunctionOpInterface>(op);
  ArrayRef<Type> argTypes = funcOp.getArgumentTypes();
  ArrayRef<Type> resultTypes = funcOp.getResultTypes();
  if (aliasAttr.getResultIndex() >= static_cast<int64_t>(resultTypes.size()))
    return op->emitOpError()
           << "attribute " << attrName
           << " result index is out of range, must be <" << resultTypes.size();

  // Verify that argument and result types pointed to by the indices are valid
  // and compatible.
  Type argType = getTypeFromTupleIndices(argTypes[argIndex],
                                         aliasAttr.getArgTupleIndices());
  if (!argType)
    return op->emitOpError()
           << "attribute " << attrName << " argument tuple indices are invalid";
  Type resultType =
      getTypeFromTupleIndices(resultTypes[aliasAttr.getResultIndex()],
                              aliasAttr.getResultTupleIndices());
  if (!resultType)
    return op->emitOpError()
           << "attribute " << attrName << " result tuple indices are invalid";

  if (failed(mlir::verifyCompatibleShape(argType, resultType)) ||
      getElementTypeOrSelf(argType) != getElementTypeOrSelf(resultType))
    return op->emitOpError() << "attribute " << attrName
                             << " aliases do not have compatible types, "
                             << argType << " vs. " << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// Builder utilities
//===----------------------------------------------------------------------===//

// Builds the region `body` for mhlo.sort's comparator: for each type in
// `element_types`, create two block arguments, one for lhs and one for rhs, and
// generates mhlo.compare op to compare them with the given `direction`.
//
// Note that this right now only does comparision on the first pair of block
// arguments.
static void buildSortComparisonBody(llvm::ArrayRef<Type> elementTypes,
                                    ComparisonDirection direction,
                                    llvm::Optional<StringRef> compareType,
                                    Region* body, OpBuilder* builder) {
  OpBuilder::InsertionGuard insertionPointGurad(*builder);

  Location loc = body->getLoc();
  Block* block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (Type elementType : elementTypes) {
    TensorType tensorType = RankedTensorType::get({}, elementType);
    block->addArguments({tensorType, tensorType},
                        SmallVector<Location, 2>(2, loc));
  }

  ComparisonType typeAttr;
  if (compareType)
    typeAttr = symbolizeComparisonType(*compareType).value();
  else
    typeAttr = ComparisonType::NOTYPE;
  Value compare = builder->create<mhlo::CompareOp>(
      loc, block->getArgument(0), block->getArgument(1), direction, typeAttr);

  builder->create<mhlo::ReturnOp>(loc, compare);
}

SortOp createSortOp(PatternRewriter* rewriter, const Location& loc,
                    const llvm::ArrayRef<Value>& operands,
                    const llvm::ArrayRef<Type>& elementTypes, int64_t dimension,
                    bool isStable, ComparisonDirection direction) {
  assert(!operands.empty() && "No operands to sort");
  // Create the sort op.
  auto sortOp =
      rewriter->create<mhlo::SortOp>(loc, operands, dimension, isStable);

  // Use TOTALORDER comparison type instead of the default comparison if the
  // element type is of type float.
  llvm::Optional<StringRef> compareType = llvm::None;
  for (auto const& elementType : elementTypes)
    if (elementType.isa<FloatType>()) {
      compareType.emplace("TOTALORDER");
      break;
    }
  buildSortComparisonBody(elementTypes, direction, compareType,
                          &sortOp.getComparator(), rewriter);
  return sortOp;
}

//===----------------------------------------------------------------------===//
// MHLO Dialect Hooks
//===----------------------------------------------------------------------===//

Operation* MhloDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc) {
  auto elementsAttr = value.dyn_cast<ElementsAttr>();
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (!elementsAttr) return nullptr;
  // HLO dialect constants require the type of value and result to match.
  if (type != elementsAttr.getType()) return nullptr;

  return builder.create<mhlo::ConstantOp>(loc, type, elementsAttr);
}

LogicalResult MhloDialect::verifyRegionArgAttribute(Operation* op,
                                                    unsigned /*regionIndex*/,
                                                    unsigned argIndex,
                                                    NamedAttribute attr) {
  if (auto aliasAttr = attr.getValue().dyn_cast<ArgResultAliasAttr>()) {
    if (failed(
            verifyArgResultAliasAttr(attr.getName(), aliasAttr, argIndex, op)))
      return failure();
  }
  return success();
}

LogicalResult MhloDialect::verifyOperationAttribute(Operation* op,
                                                    NamedAttribute attr) {
  if (auto aliasAttr = attr.getValue().dyn_cast<ArgResultAliasAttr>()) {
    if (!isa<mlir::FunctionOpInterface>(op))
      return op->emitOpError()
             << "attribute " << attr.getName()
             << " can only be used on function-like operations";
  }
  return success();
}

}  // namespace mhlo
}  // namespace mlir
