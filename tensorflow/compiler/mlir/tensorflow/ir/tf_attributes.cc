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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"

#include "mlir/IR/Attributes.h"  // from @llvm-project

namespace mlir {
namespace TF {

namespace detail {

// The storage class for ShapeAttr.
struct ShapeAttrStorage : public AttributeStorage {
  using KeyTy = std::pair<ArrayRef<int64_t>, bool>;

  explicit ShapeAttrStorage(ArrayRef<int64_t> shape, bool unranked = false)
      : shape(shape), unranked(unranked) {}

  bool operator==(const KeyTy& key) const {
    return key == KeyTy(shape, unranked);
  }
  static unsigned hashKey(const KeyTy& key) {
    return llvm::hash_combine(key.first, static_cast<char>(key.second));
  }

  // NOLINTNEXTLINE
  static ShapeAttrStorage* construct(mlir::AttributeStorageAllocator& allocator,
                                     const KeyTy& key) {
    return new (allocator.allocate<ShapeAttrStorage>())
        ShapeAttrStorage(allocator.copyInto(key.first), key.second);
  }

  ArrayRef<int64_t> shape;
  bool unranked = false;
};

// The storage class for FuncAttr.
struct FuncAttrStorage : public AttributeStorage {
  using KeyTy = std::pair<Attribute, Attribute>;

  explicit FuncAttrStorage(Attribute name, Attribute attrs)
      : name(name), attrs(attrs) {}

  bool operator==(const KeyTy& key) const { return key == KeyTy(name, attrs); }
  static unsigned hashKey(const KeyTy& key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static FuncAttrStorage* construct(mlir::AttributeStorageAllocator& allocator,
                                    const KeyTy& key) {
    return new (allocator.allocate<FuncAttrStorage>())
        FuncAttrStorage(key.first, key.second);
  }

  Attribute name;
  Attribute attrs;
};

}  // namespace detail

// Get or create a shape attribute.
ShapeAttr ShapeAttr::get(mlir::MLIRContext* context,
                         llvm::Optional<ArrayRef<int64_t>> shape) {
  if (shape)
    return Base::get(context, AttrKind::SHAPE, *shape,
                     /*unranked=*/false);

  return Base::get(context, AttrKind::SHAPE, ArrayRef<int64_t>(),
                   /*unranked=*/true);
}

llvm::Optional<ArrayRef<int64_t>> ShapeAttr::getValue() const {
  if (hasRank()) return getShape();
  return llvm::None;
}

bool ShapeAttr::hasRank() const { return !getImpl()->unranked; }

int64_t ShapeAttr::getRank() const {
  assert(hasRank());
  return getImpl()->shape.size();
}

ArrayRef<int64_t> ShapeAttr::getShape() const {
  assert(hasRank());
  return getImpl()->shape;
}

bool ShapeAttr::hasStaticShape() const {
  if (!hasRank()) return false;

  for (auto dim : getShape()) {
    if (dim < 0) return false;
  }

  return true;
}

FuncAttr FuncAttr::get(mlir::MLIRContext* context, llvm::StringRef name,
                       DictionaryAttr attr) {
  auto symbol = SymbolRefAttr::get(name, context);
  return Base::get(context, AttrKind::FUNC, symbol, attr);
}

FuncAttr FuncAttr::get(mlir::MLIRContext* context, SymbolRefAttr symbol,
                       DictionaryAttr attr) {
  return Base::get(context, AttrKind::FUNC, symbol, attr);
}

SymbolRefAttr FuncAttr::GetName() const {
  return getImpl()->name.cast<SymbolRefAttr>();
}

DictionaryAttr FuncAttr::GetAttrs() const {
  return getImpl()->attrs.cast<DictionaryAttr>();
}

}  // namespace TF
}  // namespace mlir
