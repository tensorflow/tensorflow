/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"

#include <utility>

#include "llvm/ADT/Hashing.h"
#include "mlir/IR/AttributeSupport.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"

namespace mlir {
namespace dtensor {

// Storage class for MeshAttr.
namespace detail {
struct MeshAttrStorage : public AttributeStorage {
  using Mesh = tensorflow::dtensor::Mesh;
  using KeyTy = Mesh;

  explicit MeshAttrStorage(Mesh mesh) : mesh(std::move(mesh)) {}

  bool operator==(const KeyTy& key) const { return key == KeyTy(mesh); }

  static llvm::hash_code hashKey(const KeyTy& key) {
    const Mesh& mesh = key;
    return llvm::hash_value(mesh.ToString());
  }

  static MeshAttrStorage* construct(mlir::AttributeStorageAllocator& allocator,
                                    const KeyTy& key) {
    return new (allocator.allocate<MeshAttrStorage>()) MeshAttrStorage(key);
  }
  Mesh mesh;
};
}  // namespace detail

MeshAttr MeshAttr::get(MLIRContext* context, const Mesh& mesh) {
  return Base::get(context, mesh);
}

const MeshAttr::Mesh& MeshAttr::getValue() const { return getImpl()->mesh; }

// The storage class for LayoutAttr.
namespace detail {
struct LayoutAttrStorage : public AttributeStorage {
  using Layout = tensorflow::dtensor::Layout;
  using KeyTy = Layout;

  explicit LayoutAttrStorage(Layout layout) : layout(std::move(layout)) {}

  bool operator==(const KeyTy& key) const { return key == KeyTy(layout); }

  static llvm::hash_code hashKey(const KeyTy& key) {
    const Layout& layout = key;
    return llvm::hash_value(layout.ToString());
  }

  static LayoutAttrStorage* construct(
      mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {
    const Layout& layout = key;
    return new (allocator.allocate<LayoutAttrStorage>())
        LayoutAttrStorage(layout);
  }
  Layout layout;
};
}  // namespace detail

LayoutAttr LayoutAttr::get(mlir::MLIRContext* context,
                           tensorflow::dtensor::Layout layout) {
  return Base::get(context, std::move(layout));
}

const LayoutAttr::Layout& LayoutAttr::getValue() const {
  return getImpl()->layout;
}

void DTensorDialect::registerAttributes() {
  addAttributes<dtensor::MeshAttr, dtensor::LayoutAttr>();
}

}  // namespace dtensor
}  // namespace mlir
