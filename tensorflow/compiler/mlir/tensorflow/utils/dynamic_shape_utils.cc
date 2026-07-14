/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"

#include "llvm/ADT/SmallVector.h"

namespace tensorflow {

llvm::SmallVector<int64_t> ConvertTFShapeToMlir(
    llvm::ArrayRef<int64_t> shapes) {
  return llvm::to_vector(llvm::map_range(shapes, [](int64_t shape) {
    return shape == kTFDynamicSize ? mlir::ShapedType::kDynamic : shape;
  }));
}

llvm::SmallVector<int64_t> ConvertMlirShapeToTF(
    llvm::ArrayRef<int64_t> shapes) {
  return llvm::to_vector(llvm::map_range(shapes, [](int64_t shape) {
    return mlir::ShapedType::isDynamic(shape) ? kTFDynamicSize : shape;
  }));
}

mlir::RankedTensorType GetTypeFromTFTensorShape(llvm::ArrayRef<int64_t> shape,
                                                mlir::Type elementType,
                                                mlir::Attribute encoding) {
  return mlir::RankedTensorType::get(ConvertTFShapeToMlir(shape), elementType,
                                     encoding);
}

}  // namespace tensorflow
