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

// This file defines attributes for DTensor.

#ifndef TENSORFLOW_DTENSOR_MLIR_DTENSOR_DIALECT_IR_DTENSOR_ATTRIBUTES_H_
#define TENSORFLOW_DTENSOR_MLIR_DTENSOR_DIALECT_IR_DTENSOR_ATTRIBUTES_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace mlir {
namespace dtensor {

namespace detail {
struct LayoutAttrStorage;
struct MeshAttrStorage;
}  // namespace detail

// Attribute to keep track of a mesh.
class MeshAttr
    : public Attribute::AttrBase<MeshAttr, Attribute, detail::MeshAttrStorage> {
 public:
  using Base::Base;
  using Mesh = tensorflow::dtensor::Mesh;

  static constexpr StringLiteral name = "dtensor.mesh";

  // Constructor of attribute
  static MeshAttr get(MLIRContext* context, const Mesh& mesh);

  // Returns Mesh
  const Mesh& getValue() const;
};

// Custom attribute to keep track of dtensor layouts.
class LayoutAttr : public Attribute::AttrBase<LayoutAttr, Attribute,
                                              detail::LayoutAttrStorage> {
 public:
  using Base::Base;
  using Layout = tensorflow::dtensor::Layout;
  using Mesh = tensorflow::dtensor::Mesh;

  static constexpr StringLiteral name = "dtensor.layout";

  // Create a layout attribute.
  static LayoutAttr get(MLIRContext* context, Layout layout);

  // Get layout.
  const Layout& getValue() const;
};

}  // namespace dtensor
}  // namespace mlir

#endif  // TENSORFLOW_DTENSOR_MLIR_DTENSOR_DIALECT_IR_DTENSOR_ATTRIBUTES_H_
