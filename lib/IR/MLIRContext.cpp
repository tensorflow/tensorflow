//===- MLIRContext.cpp - MLIR Type Classes --------------------------------===//
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

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
using namespace mlir;
using namespace llvm;

namespace {
struct FunctionTypeKeyInfo : DenseMapInfo<FunctionType*> {
  // Functions are uniqued based on their inputs and results.
  using KeyTy = std::pair<ArrayRef<Type*>, ArrayRef<Type*>>;
  using DenseMapInfo<FunctionType*>::getHashValue;
  using DenseMapInfo<FunctionType*>::isEqual;

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(hash_combine_range(key.first.begin(), key.first.end()),
                        hash_combine_range(key.second.begin(),
                                           key.second.end()));
  }

  static bool isEqual(const KeyTy &lhs, const FunctionType *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == KeyTy(rhs->getInputs(), rhs->getResults());
  }
};

struct AffineMapKeyInfo : DenseMapInfo<AffineMap *> {
  // Affine maps are uniqued based on their dim/symbol counts and affine
  // expressions.
  using KeyTy =
      std::pair<std::pair<unsigned, unsigned>, ArrayRef<AffineExpr *>>;
  using DenseMapInfo<AffineMap *>::getHashValue;
  using DenseMapInfo<AffineMap *>::isEqual;

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        key.first.first, key.first.second,
        hash_combine_range(key.second.begin(), key.second.end()));
  }

  static bool isEqual(const KeyTy &lhs, const AffineMap *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == KeyTy(std::pair<unsigned, unsigned>(rhs->getNumDims(),
                                                      rhs->getNumSymbols()),
                        rhs->getResults());
  }
};

struct VectorTypeKeyInfo : DenseMapInfo<VectorType*> {
  // Vectors are uniqued based on their element type and shape.
  using KeyTy = std::pair<Type*, ArrayRef<unsigned>>;
  using DenseMapInfo<VectorType*>::getHashValue;
  using DenseMapInfo<VectorType*>::isEqual;

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(DenseMapInfo<Type*>::getHashValue(key.first),
                        hash_combine_range(key.second.begin(),
                                           key.second.end()));
  }

  static bool isEqual(const KeyTy &lhs, const VectorType *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == KeyTy(rhs->getElementType(), rhs->getShape());
  }
};
struct RankedTensorTypeKeyInfo : DenseMapInfo<RankedTensorType*> {
  // Ranked tensors are uniqued based on their element type and shape.
  using KeyTy = std::pair<Type*, ArrayRef<int>>;
  using DenseMapInfo<RankedTensorType*>::getHashValue;
  using DenseMapInfo<RankedTensorType*>::isEqual;

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(DenseMapInfo<Type*>::getHashValue(key.first),
                        hash_combine_range(key.second.begin(),
                                           key.second.end()));
  }

  static bool isEqual(const KeyTy &lhs, const RankedTensorType *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == KeyTy(rhs->getElementType(), rhs->getShape());
  }
};
} // end anonymous namespace.


namespace mlir {
/// This is the implementation of the MLIRContext class, using the pImpl idiom.
/// This class is completely private to this file, so everything is public.
class MLIRContextImpl {
public:
  /// We put immortal objects into this allocator.
  llvm::BumpPtrAllocator allocator;

  /// These are identifiers uniqued into this MLIRContext.
  llvm::StringMap<char, llvm::BumpPtrAllocator&> identifiers;

  // Primitive type uniquing.
  PrimitiveType *primitives[int(TypeKind::LAST_PRIMITIVE_TYPE)+1] = { nullptr };

  // Affine map uniquing.
  using AffineMapSet = DenseSet<AffineMap *, AffineMapKeyInfo>;
  AffineMapSet affineMaps;

  // Affine binary op expression uniquing. We don't need to unique dimensional
  // or symbolic identifiers.
  // std::tuple doesn't work with DesnseMap!
  // DenseSet<AffineBinaryOpExpr *, AffineBinaryOpExprKeyInfo>;
  // AffineExprSet affineExprs;
  DenseMap<std::pair<unsigned, std::pair<AffineExpr *, AffineExpr *>>,
           AffineBinaryOpExpr *>
      affineExprs;

  /// Integer type uniquing.
  DenseMap<unsigned, IntegerType*> integers;

  /// Function type uniquing.
  using FunctionTypeSet = DenseSet<FunctionType*, FunctionTypeKeyInfo>;
  FunctionTypeSet functions;

  /// Vector type uniquing.
  using VectorTypeSet = DenseSet<VectorType*, VectorTypeKeyInfo>;
  VectorTypeSet vectors;

  /// Ranked tensor type uniquing.
  using RankedTensorTypeSet = DenseSet<RankedTensorType*,
                                       RankedTensorTypeKeyInfo>;
  RankedTensorTypeSet rankedTensors;

  /// Unranked tensor type uniquing.
  DenseMap<Type*, UnrankedTensorType*> unrankedTensors;


public:
  MLIRContextImpl() : identifiers(allocator) {}

  /// Copy the specified array of elements into memory managed by our bump
  /// pointer allocator.  This assumes the elements are all PODs.
  template<typename T>
  ArrayRef<T> copyInto(ArrayRef<T> elements) {
    auto result = allocator.Allocate<T>(elements.size());
    std::uninitialized_copy(elements.begin(), elements.end(), result);
    return ArrayRef<T>(result, elements.size());
  }
};
} // end namespace mlir

MLIRContext::MLIRContext() : impl(new MLIRContextImpl()) {
}

MLIRContext::~MLIRContext() {
}


//===----------------------------------------------------------------------===//
// Identifier
//===----------------------------------------------------------------------===//

/// Return an identifier for the specified string.
Identifier Identifier::get(StringRef str, const MLIRContext *context) {
  assert(!str.empty() && "Cannot create an empty identifier");
  assert(str.find('\0') == StringRef::npos &&
         "Cannot create an identifier with a nul character");

  auto &impl = context->getImpl();
  auto it = impl.identifiers.insert({str, char()}).first;
  return Identifier(it->getKeyData());
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

PrimitiveType *PrimitiveType::get(TypeKind kind, MLIRContext *context) {
  assert(kind <= TypeKind::LAST_PRIMITIVE_TYPE && "Not a primitive type kind");
  auto &impl = context->getImpl();

  // We normally have these types.
  if (impl.primitives[(int)kind])
    return impl.primitives[(int)kind];

  // On the first use, we allocate them into the bump pointer.
  auto *ptr = impl.allocator.Allocate<PrimitiveType>();

  // Initialize the memory using placement new.
  new(ptr) PrimitiveType(kind, context);

  // Cache and return it.
  return impl.primitives[(int)kind] = ptr;
}

IntegerType *IntegerType::get(unsigned width, MLIRContext *context) {
  auto &impl = context->getImpl();

  auto *&result = impl.integers[width];
  if (!result) {
    result = impl.allocator.Allocate<IntegerType>();
    new (result) IntegerType(width, context);
  }

  return result;
}

FunctionType *FunctionType::get(ArrayRef<Type*> inputs, ArrayRef<Type*> results,
                                MLIRContext *context) {
  auto &impl = context->getImpl();

  // Look to see if we already have this function type.
  FunctionTypeKeyInfo::KeyTy key(inputs, results);
  auto existing = impl.functions.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<FunctionType>();

  // Copy the inputs and results into the bump pointer.
  SmallVector<Type*, 16> types;
  types.reserve(inputs.size()+results.size());
  types.append(inputs.begin(), inputs.end());
  types.append(results.begin(), results.end());
  auto typesList = impl.copyInto(ArrayRef<Type*>(types));

  // Initialize the memory using placement new.
  new (result) FunctionType(typesList.data(), inputs.size(), results.size(),
                            context);

  // Cache and return it.
  return *existing.first = result;
}

VectorType *VectorType::get(ArrayRef<unsigned> shape, Type *elementType) {
  assert(!shape.empty() && "vector types must have at least one dimension");
  assert((isa<PrimitiveType>(elementType) || isa<IntegerType>(elementType)) &&
         "vectors elements must be primitives");

  auto *context = elementType->getContext();
  auto &impl = context->getImpl();

  // Look to see if we already have this vector type.
  VectorTypeKeyInfo::KeyTy key(elementType, shape);
  auto existing = impl.vectors.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<VectorType>();

  // Copy the shape into the bump pointer.
  shape = impl.copyInto(shape);

  // Initialize the memory using placement new.
  new (result) VectorType(shape, cast<PrimitiveType>(elementType), context);

  // Cache and return it.
  return *existing.first = result;
}


TensorType::TensorType(TypeKind kind, Type *elementType, MLIRContext *context)
  : Type(kind, context), elementType(elementType) {
  assert((isa<PrimitiveType>(elementType) || isa<VectorType>(elementType) ||
          isa<IntegerType>(elementType)) &&
         "tensor elements must be primitives or vectors");
  assert(isa<TensorType>(this));
}

RankedTensorType *RankedTensorType::get(ArrayRef<int> shape,
                                        Type *elementType) {
  auto *context = elementType->getContext();
  auto &impl = context->getImpl();

  // Look to see if we already have this ranked tensor type.
  RankedTensorTypeKeyInfo::KeyTy key(elementType, shape);
  auto existing = impl.rankedTensors.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<RankedTensorType>();

  // Copy the shape into the bump pointer.
  shape = impl.copyInto(shape);

  // Initialize the memory using placement new.
  new (result) RankedTensorType(shape, elementType, context);

  // Cache and return it.
  return *existing.first = result;
}

UnrankedTensorType *UnrankedTensorType::get(Type *elementType) {
  auto *context = elementType->getContext();
  auto &impl = context->getImpl();

  // Look to see if we already have this unranked tensor type.
  auto existing = impl.unrankedTensors.insert({elementType, nullptr});

  // If we already have it, return that value.
  if (!existing.second)
    return existing.first->second;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<UnrankedTensorType>();

  // Initialize the memory using placement new.
  new (result) UnrankedTensorType(elementType, context);

  // Cache and return it.
  return existing.first->second = result;
}

AffineMap *AffineMap::get(unsigned dimCount, unsigned symbolCount,
                          ArrayRef<AffineExpr *> results,
                          MLIRContext *context) {
  // The number of results can't be zero.
  assert(!results.empty());

  auto &impl = context->getImpl();

  // Check if we already have this affine map.
  AffineMapKeyInfo::KeyTy key(
      std::pair<unsigned, unsigned>(dimCount, symbolCount), results);
  auto existing = impl.affineMaps.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *res = impl.allocator.Allocate<AffineMap>();

  // Copy the results into the bump pointer.
  results = impl.copyInto(ArrayRef<AffineExpr *>(results));

  // Initialize the memory using placement new.
  new (res) AffineMap(dimCount, symbolCount, results.size(), results.data());

  // Cache and return it.
  return *existing.first = res;
}

// TODO(bondhugula): complete uniqu'ing of remaining AffinExpr sub-classes
AffineBinaryOpExpr *AffineBinaryOpExpr::get(AffineExpr::Kind kind,
                                            AffineExpr *lhsOperand,
                                            AffineExpr *rhsOperand,
                                            MLIRContext *context) {
  auto &impl = context->getImpl();

  // Check if we already have this affine expression.
  auto key = std::pair<unsigned, std::pair<AffineExpr *, AffineExpr *>>(
      (unsigned)kind,
      std::pair<AffineExpr *, AffineExpr *>(lhsOperand, rhsOperand));
  auto *&result = impl.affineExprs[key];

  // If we already have it, return that value.
  if (!result) {
    // On the first use, we allocate them into the bump pointer.
    result = impl.allocator.Allocate<AffineBinaryOpExpr>();

    // Initialize the memory using placement new.
    new (result) AffineBinaryOpExpr(kind, lhsOperand, rhsOperand);
  }
  return result;
}

AffineAddExpr *AffineAddExpr::get(AffineExpr *lhsOperand,
                                  AffineExpr *rhsOperand,
                                  MLIRContext *context) {
  return cast<AffineAddExpr>(
      AffineBinaryOpExpr::get(Kind::Add, lhsOperand, rhsOperand, context));
}

AffineSubExpr *AffineSubExpr::get(AffineExpr *lhsOperand,
                                  AffineExpr *rhsOperand,
                                  MLIRContext *context) {
  return cast<AffineSubExpr>(
      AffineBinaryOpExpr::get(Kind::Sub, lhsOperand, rhsOperand, context));
}

AffineMulExpr *AffineMulExpr::get(AffineExpr *lhsOperand,
                                  AffineExpr *rhsOperand,
                                  MLIRContext *context) {
  return cast<AffineMulExpr>(
      AffineBinaryOpExpr::get(Kind::Mul, lhsOperand, rhsOperand, context));
}

AffineFloorDivExpr *AffineFloorDivExpr::get(AffineExpr *lhsOperand,
                                            AffineExpr *rhsOperand,
                                            MLIRContext *context) {
  return cast<AffineFloorDivExpr>(
      AffineBinaryOpExpr::get(Kind::FloorDiv, lhsOperand, rhsOperand, context));
}

AffineCeilDivExpr *AffineCeilDivExpr::get(AffineExpr *lhsOperand,
                                          AffineExpr *rhsOperand,
                                          MLIRContext *context) {
  return cast<AffineCeilDivExpr>(
      AffineBinaryOpExpr::get(Kind::CeilDiv, lhsOperand, rhsOperand, context));
}

AffineModExpr *AffineModExpr::get(AffineExpr *lhsOperand,
                                  AffineExpr *rhsOperand,
                                  MLIRContext *context) {
  return cast<AffineModExpr>(
      AffineBinaryOpExpr::get(Kind::Mod, lhsOperand, rhsOperand, context));
}

AffineDimExpr *AffineDimExpr::get(unsigned position, MLIRContext *context) {
  // TODO(bondhugula): complete this
  // FIXME: this should be POD
  return new AffineDimExpr(position);
}

AffineSymbolExpr *AffineSymbolExpr::get(unsigned position,
                                        MLIRContext *context) {
  // TODO(bondhugula): complete this
  // FIXME: this should be POD
  return new AffineSymbolExpr(position);
}

AffineConstantExpr *AffineConstantExpr::get(int64_t constant,
                                            MLIRContext *context) {
  // TODO(bondhugula): complete this
  // FIXME: this should be POD
  return new AffineConstantExpr(constant);
}
