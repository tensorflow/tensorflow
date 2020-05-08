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

}  // namespace TF
}  // namespace mlir
