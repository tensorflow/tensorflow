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
#include "AttributeListStorage.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/OperationSet.h"
#include "mlir/IR/StandardOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/STLExtras.h"
#include "third_party/llvm/llvm/include/llvm/ADT/STLExtras.h"
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
  using KeyTy = std::tuple<unsigned, unsigned, ArrayRef<AffineExpr *>,
                           ArrayRef<AffineExpr *>>;
  using DenseMapInfo<AffineMap *>::getHashValue;
  using DenseMapInfo<AffineMap *>::isEqual;

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        std::get<0>(key), std::get<1>(key),
        hash_combine_range(std::get<2>(key).begin(), std::get<2>(key).end()),
        hash_combine_range(std::get<3>(key).begin(), std::get<3>(key).end()));
  }

  static bool isEqual(const KeyTy &lhs, const AffineMap *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_tuple(rhs->getNumDims(), rhs->getNumSymbols(),
                                  rhs->getResults(), rhs->getRangeSizes());
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

struct ArrayAttrKeyInfo : DenseMapInfo<ArrayAttr*> {
  // Array attributes are uniqued based on their elements.
  using KeyTy = ArrayRef<Attribute*>;
  using DenseMapInfo<ArrayAttr*>::getHashValue;
  using DenseMapInfo<ArrayAttr*>::isEqual;

  static unsigned getHashValue(KeyTy key) {
    return hash_combine_range(key.begin(), key.end());
  }

  static bool isEqual(const KeyTy &lhs, const ArrayAttr *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == rhs->getValue();
  }
};

struct AttributeListKeyInfo : DenseMapInfo<AttributeListStorage *> {
  // Array attributes are uniqued based on their elements.
  using KeyTy = ArrayRef<NamedAttribute>;
  using DenseMapInfo<AttributeListStorage *>::getHashValue;
  using DenseMapInfo<AttributeListStorage *>::isEqual;

  static unsigned getHashValue(KeyTy key) {
    return hash_combine_range(key.begin(), key.end());
  }

  static bool isEqual(const KeyTy &lhs, const AttributeListStorage *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == rhs->getElements();
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

  /// This is the set of all operations that are registered with the system.
  OperationSet operationSet;

  /// These are identifiers uniqued into this MLIRContext.
  llvm::StringMap<char, llvm::BumpPtrAllocator&> identifiers;

  // Primitive type uniquing.
  PrimitiveType *primitives[int(Type::Kind::LAST_PRIMITIVE_TYPE)+1] = {nullptr};

  // Affine map uniquing.
  using AffineMapSet = DenseSet<AffineMap *, AffineMapKeyInfo>;
  AffineMapSet affineMaps;

  // Affine binary op expression uniquing. Figure out uniquing of dimensional
  // or symbolic identifiers.
  DenseMap<std::tuple<unsigned, AffineExpr *, AffineExpr *>, AffineExpr *>
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

  // Attribute uniquing.
  BoolAttr *boolAttrs[2] = { nullptr };
  DenseMap<int64_t, IntegerAttr*> integerAttrs;
  DenseMap<int64_t, FloatAttr*> floatAttrs;
  StringMap<StringAttr*> stringAttrs;
  using ArrayAttrSet = DenseSet<ArrayAttr*, ArrayAttrKeyInfo>;
  ArrayAttrSet arrayAttrs;
  using AttributeListSet =
      DenseSet<AttributeListStorage *, AttributeListKeyInfo>;
  AttributeListSet attributeLists;

public:
  MLIRContextImpl() : identifiers(allocator) {
    registerStandardOperations(operationSet);
  }

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

/// Return the operation set associated with the specified MLIRContext object.
OperationSet &OperationSet::get(MLIRContext *context) {
  return context->getImpl().operationSet;
}

/// If this operation has a registered operation description in the
/// OperationSet, return it.  Otherwise return null.
/// TODO: Shouldn't have to pass a Context here.
const AbstractOperation *
Operation::getAbstractOperation(MLIRContext *context) const {
  return OperationSet::get(context).lookup(getName().str());
}

//===----------------------------------------------------------------------===//
// Identifier uniquing
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
// Type uniquing
//===----------------------------------------------------------------------===//

PrimitiveType *PrimitiveType::get(Kind kind, MLIRContext *context) {
  assert(kind <= Kind::LAST_PRIMITIVE_TYPE && "Not a primitive type kind");
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


TensorType::TensorType(Kind kind, Type *elementType, MLIRContext *context)
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
  auto *&result = impl.unrankedTensors[elementType];

  // If we already have it, return that value.
  if (result)
    return result;

  // On the first use, we allocate them into the bump pointer.
  result = impl.allocator.Allocate<UnrankedTensorType>();

  // Initialize the memory using placement new.
  new (result) UnrankedTensorType(elementType, context);
  return result;
}

//===----------------------------------------------------------------------===//
// Attribute uniquing
//===----------------------------------------------------------------------===//

BoolAttr *BoolAttr::get(bool value, MLIRContext *context) {
  auto *&result = context->getImpl().boolAttrs[value];
  if (result)
    return result;

  result = context->getImpl().allocator.Allocate<BoolAttr>();
  new (result) BoolAttr(value);
  return result;
}

IntegerAttr *IntegerAttr::get(int64_t value, MLIRContext *context) {
  auto *&result = context->getImpl().integerAttrs[value];
  if (result)
    return result;

  result = context->getImpl().allocator.Allocate<IntegerAttr>();
  new (result) IntegerAttr(value);
  return result;
}

FloatAttr *FloatAttr::get(double value, MLIRContext *context) {
  // We hash based on the bit representation of the double to ensure we don't
  // merge things like -0.0 and 0.0 in the hash comparison.
  union {
    double floatValue;
    int64_t intValue;
  };
  floatValue = value;

  auto *&result = context->getImpl().floatAttrs[intValue];
  if (result)
    return result;

  result = context->getImpl().allocator.Allocate<FloatAttr>();
  new (result) FloatAttr(value);
  return result;
}

StringAttr *StringAttr::get(StringRef bytes, MLIRContext *context) {
  auto it = context->getImpl().stringAttrs.insert({bytes, nullptr}).first;

  if (it->second)
    return it->second;

  auto result = context->getImpl().allocator.Allocate<StringAttr>();
  new (result) StringAttr(it->first());
  it->second = result;
  return result;
}

ArrayAttr *ArrayAttr::get(ArrayRef<Attribute*> value, MLIRContext *context) {
  auto &impl = context->getImpl();

  // Look to see if we already have this.
  auto existing = impl.arrayAttrs.insert_as(nullptr, value);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *result = impl.allocator.Allocate<ArrayAttr>();

  // Copy the elements into the bump pointer.
  value = impl.copyInto(value);

  // Initialize the memory using placement new.
  new (result) ArrayAttr(value);

  // Cache and return it.
  return *existing.first = result;
}

/// Perform a three-way comparison between the names of the specified
/// NamedAttributes.
static int compareNamedAttributes(const NamedAttribute *lhs,
                                  const NamedAttribute *rhs) {
  return lhs->first.str().compare(rhs->first.str());
}

/// Given a list of NamedAttribute's, canonicalize the list (sorting
/// by name) and return the unique'd result.  Note that the empty list is
/// represented with a null pointer.
AttributeListStorage *AttributeListStorage::get(ArrayRef<NamedAttribute> attrs,
                                                MLIRContext *context) {
  // We need to sort the element list to canonicalize it, but we also don't want
  // to do a ton of work in the super common case where the element list is
  // already sorted.
  SmallVector<NamedAttribute, 8> storage;
  switch (attrs.size()) {
  case 0:
    // An empty list is represented with a null pointer.
    return nullptr;
  case 1:
    // A single element is already sorted.
    break;
  case 2:
    // Don't invoke a general sort for two element case.
    if (attrs[0].first.str() > attrs[1].first.str()) {
      storage.push_back(attrs[1]);
      storage.push_back(attrs[0]);
      attrs = storage;
    }
    break;
  default:
    // Check to see they are sorted already.
    bool isSorted = true;
    for (unsigned i = 0, e = attrs.size() - 1; i != e; ++i) {
      if (attrs[i].first.str() > attrs[i + 1].first.str()) {
        isSorted = false;
        break;
      }
    }
    // If not, do a general sort.
    if (!isSorted) {
      storage.append(attrs.begin(), attrs.end());
      llvm::array_pod_sort(storage.begin(), storage.end(),
                           compareNamedAttributes);
      attrs = storage;
    }
  }

  // Ok, now that we've canonicalized our attributes, unique them.
  auto &impl = context->getImpl();

  // Look to see if we already have this.
  auto existing = impl.attributeLists.insert_as(nullptr, attrs);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // Otherwise, allocate a new AttributeListStorage, unique it and return it.
  auto byteSize =
      AttributeListStorage::totalSizeToAlloc<NamedAttribute>(attrs.size());
  auto rawMem = impl.allocator.Allocate(byteSize, alignof(NamedAttribute));

  //  Placement initialize the AggregateSymbolicValue.
  auto result = ::new (rawMem) AttributeListStorage(attrs.size());
  std::uninitialized_copy(attrs.begin(), attrs.end(),
                          result->getTrailingObjects<NamedAttribute>());
  return *existing.first = result;
}

//===----------------------------------------------------------------------===//
// AffineMap and AffineExpr uniquing
//===----------------------------------------------------------------------===//

AffineMap *AffineMap::get(unsigned dimCount, unsigned symbolCount,
                          ArrayRef<AffineExpr *> results,
                          ArrayRef<AffineExpr *> rangeSizes,
                          MLIRContext *context) {
  // The number of results can't be zero.
  assert(!results.empty());

  assert(rangeSizes.empty() || results.size() == rangeSizes.size());

  auto &impl = context->getImpl();

  // Check if we already have this affine map.
  auto key = std::make_tuple(dimCount, symbolCount, results, rangeSizes);
  auto existing = impl.affineMaps.insert_as(nullptr, key);

  // If we already have it, return that value.
  if (!existing.second)
    return *existing.first;

  // On the first use, we allocate them into the bump pointer.
  auto *res = impl.allocator.Allocate<AffineMap>();

  // Copy the results into the bump pointer.
  results = impl.copyInto(ArrayRef<AffineExpr *>(results));

  // Copy the results into the bump pointer.
  rangeSizes = impl.copyInto(ArrayRef<AffineExpr *>(rangeSizes));

  // Initialize the memory using placement new.
  new (res) AffineMap(dimCount, symbolCount, results.size(), results.data(),
                      rangeSizes.empty() ? nullptr : rangeSizes.data());

  // Cache and return it.
  return *existing.first = res;
}

/// Return a binary affine op expression with the specified op type and
/// operands: if it doesn't exist, create it and store it; if it is already
/// present, return from the list. The stored expressions are unique: they are
/// constructed and stored in a simplified/canonicalized form. The result after
/// simplification could be any form of affine expression.
AffineExpr *AffineBinaryOpExpr::get(AffineExpr::Kind kind, AffineExpr *lhs,
                                    AffineExpr *rhs, MLIRContext *context) {
  auto &impl = context->getImpl();

  // Check if we already have this affine expression.
  auto keyValue = std::make_tuple((unsigned)kind, lhs, rhs);
  auto *&result = impl.affineExprs[keyValue];

  // If we already have it, return that value.
  if (result)
    return result;

  // Simplify the expression if possible.
  AffineExpr *simplified;
  switch (kind) {
  case Kind::Add:
    simplified = AffineBinaryOpExpr::simplifyAdd(lhs, rhs, context);
    break;
  case Kind::Sub:
    simplified = AffineBinaryOpExpr::simplifySub(lhs, rhs, context);
    break;
  case Kind::Mul:
    simplified = AffineBinaryOpExpr::simplifyMul(lhs, rhs, context);
    break;
  case Kind::FloorDiv:
    simplified = AffineBinaryOpExpr::simplifyFloorDiv(lhs, rhs, context);
    break;
  case Kind::CeilDiv:
    simplified = AffineBinaryOpExpr::simplifyCeilDiv(lhs, rhs, context);
    break;
  case Kind::Mod:
    simplified = AffineBinaryOpExpr::simplifyMod(lhs, rhs, context);
    break;
  default:
    llvm_unreachable("unexpected binary affine expr");
  }

  // If simplified to a non-binary affine op expr, don't store it.
  if (simplified && !isa<AffineBinaryOpExpr>(simplified)) {
    // 'affineExprs' only contains uniqued AffineBinaryOpExpr's.
    return simplified;
  }

  if (simplified)
    // We know that it's a binary op expression.
    return result = simplified;

  // On the first use, we allocate them into the bump pointer.
  result = impl.allocator.Allocate<AffineBinaryOpExpr>();
  // Initialize the memory using placement new.
  new (result) AffineBinaryOpExpr(kind, lhs, rhs);
  return result;
}

AffineExpr *AffineAddExpr::get(AffineExpr *lhs, AffineExpr *rhs,
                               MLIRContext *context) {
  return AffineBinaryOpExpr::get(Kind::Add, lhs, rhs, context);
}

AffineExpr *AffineSubExpr::get(AffineExpr *lhs, AffineExpr *rhs,
                               MLIRContext *context) {
  return AffineBinaryOpExpr::get(Kind::Sub, lhs, rhs, context);
}

AffineExpr *AffineMulExpr::get(AffineExpr *lhs, AffineExpr *rhs,
                               MLIRContext *context) {
  return AffineBinaryOpExpr::get(Kind::Mul, lhs, rhs, context);
}

AffineExpr *AffineFloorDivExpr::get(AffineExpr *lhs, AffineExpr *rhs,
                                    MLIRContext *context) {
  return AffineBinaryOpExpr::get(Kind::FloorDiv, lhs, rhs, context);
}

AffineExpr *AffineCeilDivExpr::get(AffineExpr *lhs, AffineExpr *rhs,
                                   MLIRContext *context) {
  return AffineBinaryOpExpr::get(Kind::CeilDiv, lhs, rhs, context);
}

AffineExpr *AffineModExpr::get(AffineExpr *lhs, AffineExpr *rhs,
                               MLIRContext *context) {
  return AffineBinaryOpExpr::get(Kind::Mod, lhs, rhs, context);
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
