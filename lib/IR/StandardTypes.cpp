//===- StandardTypes.cpp - MLIR Standard Type Classes ---------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/IR/StandardTypes.h"
#include "TypeDetail.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Integer Type
//===----------------------------------------------------------------------===//

// static constexpr must have a definition (until in C++17 and inline variable).
constexpr unsigned IntegerType::kMaxWidth;

/// Verify the construction of an integer type.
LogicalResult IntegerType::verifyConstructionInvariants(
    llvm::Optional<Location> loc, MLIRContext *context, unsigned width) {
  if (width > IntegerType::kMaxWidth) {
    if (loc)
      context->emitError(*loc) << "integer bitwidth is limited to "
                               << IntegerType::kMaxWidth << " bits";
    return failure();
  }
  return success();
}

unsigned IntegerType::getWidth() const { return getImpl()->width; }

//===----------------------------------------------------------------------===//
// Float Type
//===----------------------------------------------------------------------===//

unsigned FloatType::getWidth() {
  switch (getKind()) {
  case StandardTypes::BF16:
  case StandardTypes::F16:
    return 16;
  case StandardTypes::F32:
    return 32;
  case StandardTypes::F64:
    return 64;
  default:
    llvm_unreachable("unexpected type");
  }
}

/// Returns the floating semantics for the given type.
const llvm::fltSemantics &FloatType::getFloatSemantics() {
  if (isBF16())
    // Treat BF16 like a double. This is unfortunate but BF16 fltSemantics is
    // not defined in LLVM.
    // TODO(jpienaar): add BF16 to LLVM? fltSemantics are internal to APFloat.cc
    // else one could add it.
    //  static const fltSemantics semBF16 = {127, -126, 8, 16};
    return APFloat::IEEEdouble();
  if (isF16())
    return APFloat::IEEEhalf();
  if (isF32())
    return APFloat::IEEEsingle();
  if (isF64())
    return APFloat::IEEEdouble();
  llvm_unreachable("non-floating point type used");
}

unsigned Type::getIntOrFloatBitWidth() {
  assert(isIntOrFloat() && "only ints and floats have a bitwidth");
  if (auto intType = dyn_cast<IntegerType>()) {
    return intType.getWidth();
  }

  auto floatType = cast<FloatType>();
  return floatType.getWidth();
}

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//

Type ShapedType::getElementType() const {
  return static_cast<ImplType *>(impl)->elementType;
}

unsigned ShapedType::getElementTypeBitWidth() const {
  return getElementType().getIntOrFloatBitWidth();
}

int64_t ShapedType::getNumElements() const {
  assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
  auto shape = getShape();
  int64_t num = 1;
  for (auto dim : shape)
    num *= dim;
  return num;
}

int64_t ShapedType::getRank() const { return getShape().size(); }

bool ShapedType::hasRank() const { return !isa<UnrankedTensorType>(); }

int64_t ShapedType::getDimSize(int64_t i) const {
  assert(i >= 0 && i < getRank() && "invalid index for shaped type");
  return getShape()[i];
}

/// Get the number of bits require to store a value of the given shaped type.
/// Compute the value recursively since tensors are allowed to have vectors as
/// elements.
int64_t ShapedType::getSizeInBits() const {
  assert(hasStaticShape() &&
         "cannot get the bit size of an aggregate with a dynamic shape");

  auto elementType = getElementType();
  if (elementType.isIntOrFloat())
    return elementType.getIntOrFloatBitWidth() * getNumElements();

  // Tensors can have vectors and other tensors as elements, other shaped types
  // cannot.
  assert(isa<TensorType>() && "unsupported element type");
  assert((elementType.isa<VectorType>() || elementType.isa<TensorType>()) &&
         "unsupported tensor element type");
  return getNumElements() * elementType.cast<ShapedType>().getSizeInBits();
}

ArrayRef<int64_t> ShapedType::getShape() const {
  switch (getKind()) {
  case StandardTypes::Vector:
    return cast<VectorType>().getShape();
  case StandardTypes::RankedTensor:
    return cast<RankedTensorType>().getShape();
  case StandardTypes::MemRef:
    return cast<MemRefType>().getShape();
  default:
    llvm_unreachable("not a ShapedType or not ranked");
  }
}

int64_t ShapedType::getNumDynamicDims() const {
  return llvm::count_if(getShape(), isDynamic);
}

bool ShapedType::hasStaticShape() const {
  return hasRank() && llvm::none_of(getShape(), isDynamic);
}

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

VectorType VectorType::get(ArrayRef<int64_t> shape, Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::Vector, shape,
                   elementType);
}

VectorType VectorType::getChecked(ArrayRef<int64_t> shape, Type elementType,
                                  Location location) {
  return Base::getChecked(location, elementType.getContext(),
                          StandardTypes::Vector, shape, elementType);
}

LogicalResult VectorType::verifyConstructionInvariants(
    llvm::Optional<Location> loc, MLIRContext *context, ArrayRef<int64_t> shape,
    Type elementType) {
  if (shape.empty()) {
    if (loc)
      context->emitError(*loc, "vector types must have at least one dimension");
    return failure();
  }

  if (!isValidElementType(elementType)) {
    if (loc)
      context->emitError(*loc, "vector elements must be int or float type");
    return failure();
  }

  if (any_of(shape, [](int64_t i) { return i <= 0; })) {
    if (loc)
      context->emitError(*loc,
                         "vector types must have positive constant sizes");
    return failure();
  }
  return success();
}

ArrayRef<int64_t> VectorType::getShape() const { return getImpl()->getShape(); }

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

// Check if "elementType" can be an element type of a tensor. Emit errors if
// location is not nullptr.  Returns failure if check failed.
static inline LogicalResult checkTensorElementType(Optional<Location> location,
                                                   MLIRContext *context,
                                                   Type elementType) {
  if (!TensorType::isValidElementType(elementType)) {
    if (location)
      context->emitError(*location, "invalid tensor element type");
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RankedTensorType
//===----------------------------------------------------------------------===//

RankedTensorType RankedTensorType::get(ArrayRef<int64_t> shape,
                                       Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::RankedTensor, shape,
                   elementType);
}

RankedTensorType RankedTensorType::getChecked(ArrayRef<int64_t> shape,
                                              Type elementType,
                                              Location location) {
  return Base::getChecked(location, elementType.getContext(),
                          StandardTypes::RankedTensor, shape, elementType);
}

LogicalResult RankedTensorType::verifyConstructionInvariants(
    llvm::Optional<Location> loc, MLIRContext *context, ArrayRef<int64_t> shape,
    Type elementType) {
  for (int64_t s : shape) {
    if (s < -1) {
      if (loc)
        context->emitError(*loc, "invalid tensor dimension size");
      return failure();
    }
  }
  return checkTensorElementType(loc, context, elementType);
}

ArrayRef<int64_t> RankedTensorType::getShape() const {
  return getImpl()->getShape();
}

//===----------------------------------------------------------------------===//
// UnrankedTensorType
//===----------------------------------------------------------------------===//

UnrankedTensorType UnrankedTensorType::get(Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::UnrankedTensor,
                   elementType);
}

UnrankedTensorType UnrankedTensorType::getChecked(Type elementType,
                                                  Location location) {
  return Base::getChecked(location, elementType.getContext(),
                          StandardTypes::UnrankedTensor, elementType);
}

LogicalResult UnrankedTensorType::verifyConstructionInvariants(
    llvm::Optional<Location> loc, MLIRContext *context, Type elementType) {
  return checkTensorElementType(loc, context, elementType);
}

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

/// Get or create a new MemRefType based on shape, element type, affine
/// map composition, and memory space.  Assumes the arguments define a
/// well-formed MemRef type.  Use getChecked to gracefully handle MemRefType
/// construction failures.
MemRefType MemRefType::get(ArrayRef<int64_t> shape, Type elementType,
                           ArrayRef<AffineMap> affineMapComposition,
                           unsigned memorySpace) {
  auto result = getImpl(shape, elementType, affineMapComposition, memorySpace,
                        /*location=*/llvm::None);
  assert(result && "Failed to construct instance of MemRefType.");
  return result;
}

/// Get or create a new MemRefType based on shape, element type, affine
/// map composition, and memory space declared at the given location.
/// If the location is unknown, the last argument should be an instance of
/// UnknownLoc.  If the MemRefType defined by the arguments would be
/// ill-formed, emits errors (to the handler registered with the context or to
/// the error stream) and returns nullptr.
MemRefType MemRefType::getChecked(ArrayRef<int64_t> shape, Type elementType,
                                  ArrayRef<AffineMap> affineMapComposition,
                                  unsigned memorySpace, Location location) {
  return getImpl(shape, elementType, affineMapComposition, memorySpace,
                 location);
}

/// Get or create a new MemRefType defined by the arguments.  If the resulting
/// type would be ill-formed, return nullptr.  If the location is provided,
/// emit detailed error messages.  To emit errors when the location is unknown,
/// pass in an instance of UnknownLoc.
MemRefType MemRefType::getImpl(ArrayRef<int64_t> shape, Type elementType,
                               ArrayRef<AffineMap> affineMapComposition,
                               unsigned memorySpace,
                               Optional<Location> location) {
  auto *context = elementType.getContext();

  for (int64_t s : shape) {
    // Negative sizes are not allowed except for `-1` that means dynamic size.
    if (s < -1) {
      if (location)
        context->emitError(*location, "invalid memref size");
      return {};
    }
  }

  // Check that the structure of the composition is valid, i.e. that each
  // subsequent affine map has as many inputs as the previous map has results.
  // Take the dimensionality of the MemRef for the first map.
  auto dim = shape.size();
  unsigned i = 0;
  for (const auto &affineMap : affineMapComposition) {
    if (affineMap.getNumDims() != dim) {
      if (location)
        context->emitError(*location)
            << "memref affine map dimension mismatch between "
            << (i == 0 ? Twine("memref rank") : "affine map " + Twine(i))
            << " and affine map" << i + 1 << ": " << dim
            << " != " << affineMap.getNumDims();
      return nullptr;
    }

    dim = affineMap.getNumResults();
    ++i;
  }

  // Drop identity maps from the composition.
  // This may lead to the composition becoming empty, which is interpreted as an
  // implicit identity.
  llvm::SmallVector<AffineMap, 2> cleanedAffineMapComposition;
  for (const auto &map : affineMapComposition) {
    if (map.isIdentity())
      continue;
    cleanedAffineMapComposition.push_back(map);
  }

  return Base::get(context, StandardTypes::MemRef, shape, elementType,
                   cleanedAffineMapComposition, memorySpace);
}

ArrayRef<int64_t> MemRefType::getShape() const { return getImpl()->getShape(); }

ArrayRef<AffineMap> MemRefType::getAffineMaps() const {
  return getImpl()->getAffineMaps();
}

unsigned MemRefType::getMemorySpace() const { return getImpl()->memorySpace; }

//===----------------------------------------------------------------------===//
/// ComplexType
//===----------------------------------------------------------------------===//

ComplexType ComplexType::get(Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::Complex,
                   elementType);
}

ComplexType ComplexType::getChecked(Type elementType, Location location) {
  return Base::getChecked(location, elementType.getContext(),
                          StandardTypes::Complex, elementType);
}

/// Verify the construction of an integer type.
LogicalResult ComplexType::verifyConstructionInvariants(
    llvm::Optional<Location> loc, MLIRContext *context, Type elementType) {
  if (!elementType.isa<FloatType>() && !elementType.isa<IntegerType>()) {
    if (loc)
      context->emitError(*loc, "invalid element type for complex");
    return failure();
  }
  return success();
}

Type ComplexType::getElementType() { return getImpl()->elementType; }

//===----------------------------------------------------------------------===//
/// TupleType
//===----------------------------------------------------------------------===//

/// Get or create a new TupleType with the provided element types. Assumes the
/// arguments define a well-formed type.
TupleType TupleType::get(ArrayRef<Type> elementTypes, MLIRContext *context) {
  return Base::get(context, StandardTypes::Tuple, elementTypes);
}

/// Return the elements types for this tuple.
ArrayRef<Type> TupleType::getTypes() const { return getImpl()->getTypes(); }

/// Accumulate the types contained in this tuple and tuples nested within it.
/// Note that this only flattens nested tuples, not any other container type,
/// e.g. a tuple<i32, tensor<i32>, tuple<f32, tuple<i64>>> is flattened to
/// (i32, tensor<i32>, f32, i64)
void TupleType::getFlattenedTypes(SmallVectorImpl<Type> &types) {
  for (Type type : getTypes()) {
    if (auto nestedTuple = type.dyn_cast<TupleType>())
      nestedTuple.getFlattenedTypes(types);
    else
      types.push_back(type);
  }
}

/// Return the number of element types.
size_t TupleType::size() const { return getImpl()->size(); }
