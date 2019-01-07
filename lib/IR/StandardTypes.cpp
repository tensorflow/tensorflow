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
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::detail;

/// Integer Type.

/// Verify the construction of an integer type.
bool IntegerType::verifyConstructionInvariants(llvm::Optional<Location> loc,
                                               MLIRContext *context,
                                               unsigned width) {
  if (width > IntegerType::kMaxWidth) {
    if (loc)
      context->emitError(*loc, "integer bitwidth is limited to " +
                                   Twine(IntegerType::kMaxWidth) + " bits");
    return true;
  }
  return false;
}

IntegerType IntegerType::get(unsigned width, MLIRContext *context) {
  return Base::get(context, StandardTypes::Integer, width);
}

IntegerType IntegerType::getChecked(unsigned width, MLIRContext *context,
                                    Location location) {
  return Base::getChecked(location, context, StandardTypes::Integer, width);
}

unsigned IntegerType::getWidth() const {
  return static_cast<ImplType *>(type)->width;
}

/// Float Type.

unsigned FloatType::getWidth() const {
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
const llvm::fltSemantics &FloatType::getFloatSemantics() const {
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

unsigned Type::getIntOrFloatBitWidth() const {
  assert(isIntOrFloat() && "only ints and floats have a bitwidth");
  if (auto intType = dyn_cast<IntegerType>()) {
    return intType.getWidth();
  }

  auto floatType = cast<FloatType>();
  return floatType.getWidth();
}

/// VectorOrTensorType

Type VectorOrTensorType::getElementType() const {
  return static_cast<ImplType *>(type)->elementType;
}

unsigned VectorOrTensorType::getElementTypeBitWidth() const {
  return getElementType().getIntOrFloatBitWidth();
}

unsigned VectorOrTensorType::getNumElements() const {
  switch (getKind()) {
  case StandardTypes::Vector:
  case StandardTypes::RankedTensor: {
    auto shape = getShape();
    unsigned num = 1;
    for (auto dim : shape)
      num *= dim;
    return num;
  }
  default:
    llvm_unreachable("not a VectorOrTensorType or not ranked");
  }
}

/// If this is ranked tensor or vector type, return the rank. If it is an
/// unranked tensor, return -1.
int VectorOrTensorType::getRank() const {
  switch (getKind()) {
  case StandardTypes::Vector:
  case StandardTypes::RankedTensor:
    return getShape().size();
  case StandardTypes::UnrankedTensor:
    return -1;
  default:
    llvm_unreachable("not a VectorOrTensorType");
  }
}

int VectorOrTensorType::getDimSize(unsigned i) const {
  switch (getKind()) {
  case StandardTypes::Vector:
  case StandardTypes::RankedTensor:
    return getShape()[i];
  default:
    llvm_unreachable("not a VectorOrTensorType or not ranked");
  }
}

// Get the number of number of bits require to store a value of the given vector
// or tensor types.  Compute the value recursively since tensors are allowed to
// have vectors as elements.
long VectorOrTensorType::getSizeInBits() const {
  assert(hasStaticShape() &&
         "cannot get the bit size of an aggregate with a dynamic shape");

  auto elementType = getElementType();
  if (elementType.isIntOrFloat())
    return elementType.getIntOrFloatBitWidth() * getNumElements();

  // Tensors can have vectors and other tensors as elements, vectors cannot.
  assert(!isa<VectorType>() && "unsupported vector element type");
  auto elementVectorOrTensorType = elementType.dyn_cast<VectorOrTensorType>();
  assert(elementVectorOrTensorType && "unsupported tensor element type");
  return getNumElements() * elementVectorOrTensorType.getSizeInBits();
}

ArrayRef<int> VectorOrTensorType::getShape() const {
  switch (getKind()) {
  case StandardTypes::Vector:
    return cast<VectorType>().getShape();
  case StandardTypes::RankedTensor:
    return cast<RankedTensorType>().getShape();
  default:
    llvm_unreachable("not a VectorOrTensorType or not ranked");
  }
}

bool VectorOrTensorType::hasStaticShape() const {
  if (isa<UnrankedTensorType>())
    return false;
  auto dims = getShape();
  return !std::any_of(dims.begin(), dims.end(), [](int i) { return i < 0; });
}

/// VectorType

VectorType VectorType::get(ArrayRef<int> shape, Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::Vector, shape,
                   elementType);
}

VectorType VectorType::getChecked(ArrayRef<int> shape, Type elementType,
                                  Location location) {
  return Base::getChecked(location, elementType.getContext(),
                          StandardTypes::Vector, shape, elementType);
}

bool VectorType::verifyConstructionInvariants(llvm::Optional<Location> loc,
                                              MLIRContext *context,
                                              ArrayRef<int> shape,
                                              Type elementType) {
  if (shape.empty()) {
    if (loc)
      context->emitError(*loc, "vector types must have at least one dimension");
    return true;
  }

  if (!isValidElementType(elementType)) {
    if (loc)
      context->emitError(*loc, "vector elements must be int or float type");
    return true;
  }

  if (any_of(shape, [](int i) { return i < 0; })) {
    if (loc)
      context->emitError(*loc, "vector types must have static shape");
    return true;
  }
  return false;
}

ArrayRef<int> VectorType::getShape() const {
  return static_cast<ImplType *>(type)->getShape();
}

/// TensorType

// Check if "elementType" can be an element type of a tensor. Emit errors if
// location is not nullptr.  Returns true if check failed.
static inline bool checkTensorElementType(Optional<Location> location,
                                          MLIRContext *context,
                                          Type elementType) {
  if (!TensorType::isValidElementType(elementType)) {
    if (location)
      context->emitError(*location, "invalid tensor element type");
    return true;
  }
  return false;
}

/// RankedTensorType

RankedTensorType RankedTensorType::get(ArrayRef<int> shape, Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::RankedTensor, shape,
                   elementType);
}

RankedTensorType RankedTensorType::getChecked(ArrayRef<int> shape,
                                              Type elementType,
                                              Location location) {
  return Base::getChecked(location, elementType.getContext(),
                          StandardTypes::RankedTensor, shape, elementType);
}

bool RankedTensorType::verifyConstructionInvariants(
    llvm::Optional<Location> loc, MLIRContext *context, ArrayRef<int> shape,
    Type elementType) {
  return checkTensorElementType(loc, context, elementType);
}

ArrayRef<int> RankedTensorType::getShape() const {
  return static_cast<ImplType *>(type)->getShape();
}

ArrayRef<int> MemRefType::getShape() const {
  return static_cast<ImplType *>(type)->getShape();
}

/// UnrankedTensorType

UnrankedTensorType UnrankedTensorType::get(Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::UnrankedTensor,
                   elementType);
}

UnrankedTensorType UnrankedTensorType::getChecked(Type elementType,
                                                  Location location) {
  return Base::getChecked(location, elementType.getContext(),
                          StandardTypes::UnrankedTensor, elementType);
}

bool UnrankedTensorType::verifyConstructionInvariants(
    llvm::Optional<Location> loc, MLIRContext *context, Type elementType) {
  return checkTensorElementType(loc, context, elementType);
}

/// MemRefType

/// Get or create a new MemRefType defined by the arguments.  If the resulting
/// type would be ill-formed, return nullptr.  If the location is provided,
/// emit detailed error messages.  To emit errors when the location is unknown,
/// pass in an instance of UnknownLoc.
MemRefType MemRefType::getImpl(ArrayRef<int> shape, Type elementType,
                               ArrayRef<AffineMap> affineMapComposition,
                               unsigned memorySpace,
                               Optional<Location> location) {
  auto *context = elementType.getContext();

  // Check that the structure of the composition is valid, i.e. that each
  // subsequent affine map has as many inputs as the previous map has results.
  // Take the dimensionality of the MemRef for the first map.
  auto dim = shape.size();
  unsigned i = 0;
  for (const auto &affineMap : affineMapComposition) {
    if (affineMap.getNumDims() != dim) {
      if (location)
        context->emitDiagnostic(
            *location,
            "memref affine map dimension mismatch between " +
                (i == 0 ? Twine("memref rank") : "affine map " + Twine(i)) +
                " and affine map" + Twine(i + 1) + ": " + Twine(dim) +
                " != " + Twine(affineMap.getNumDims()),
            MLIRContext::DiagnosticKind::Error);
      return nullptr;
    }

    dim = affineMap.getNumResults();
    ++i;
  }

  // Drop the unbounded identity maps from the composition.
  // This may lead to the composition becoming empty, which is interpreted as an
  // implicit identity.
  llvm::SmallVector<AffineMap, 2> cleanedAffineMapComposition;
  for (const auto &map : affineMapComposition) {
    if (map.isIdentity() && !map.isBounded())
      continue;
    cleanedAffineMapComposition.push_back(map);
  }

  return Base::get(context, StandardTypes::MemRef, shape, elementType,
                   cleanedAffineMapComposition, memorySpace);
}

Type MemRefType::getElementType() const {
  return static_cast<ImplType *>(type)->elementType;
}

ArrayRef<AffineMap> MemRefType::getAffineMaps() const {
  return static_cast<ImplType *>(type)->getAffineMaps();
}

unsigned MemRefType::getMemorySpace() const {
  return static_cast<ImplType *>(type)->memorySpace;
}

unsigned MemRefType::getNumDynamicDims() const {
  unsigned numDynamicDims = 0;
  for (int dimSize : getShape()) {
    if (dimSize == -1)
      ++numDynamicDims;
  }
  return numDynamicDims;
}

// Define type identifiers.
char FloatType::typeID = 0;
char IntegerType::typeID = 0;
char VectorType::typeID = 0;
char RankedTensorType::typeID = 0;
char UnrankedTensorType::typeID = 0;
char MemRefType::typeID = 0;
