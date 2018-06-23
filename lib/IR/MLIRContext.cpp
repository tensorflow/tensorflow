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
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
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
} // end anonymous namespace.


namespace mlir {
/// This is the implementation of the MLIRContext class, using the pImpl idiom.
/// This class is completely private to this file, so everything is public.
class MLIRContextImpl {
public:
  /// We put immortal objects into this allocator.
  llvm::BumpPtrAllocator allocator;

  // Primitive type uniquing.
  PrimitiveType *primitives[int(TypeKind::LAST_PRIMITIVE_TYPE)+1] = { nullptr };

  /// Function type uniquing.
  using FunctionTypeSet = DenseSet<FunctionType*, FunctionTypeKeyInfo>;
  FunctionTypeSet functions;

  /// Vector type uniquing.
  using VectorTypeSet = DenseSet<VectorType*, VectorTypeKeyInfo>;
  VectorTypeSet vectors;


public:
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


PrimitiveType::PrimitiveType(TypeKind kind, MLIRContext *context)
  : Type(kind, context) {

}

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

FunctionType::FunctionType(Type *const *inputsAndResults, unsigned numInputs,
                           unsigned numResults, MLIRContext *context)
  : Type(TypeKind::Function, context, numInputs),
    numResults(numResults), inputsAndResults(inputsAndResults) {
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



VectorType::VectorType(ArrayRef<unsigned> shape, PrimitiveType *elementType,
                       MLIRContext *context)
  : Type(TypeKind::Vector, context, shape.size()),
    shapeElements(shape.data()), elementType(elementType) {
}


VectorType *VectorType::get(ArrayRef<unsigned> shape, Type *elementType) {
  assert(!shape.empty() && "vector types must have at least one dimension");
  assert(isa<PrimitiveType>(elementType) &&
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
