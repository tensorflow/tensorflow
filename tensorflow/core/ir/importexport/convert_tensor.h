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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_CONVERT_TENSOR_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_CONVERT_TENSOR_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace mlir {
namespace tfg {

// Converts an TensorFlow tensor proto into an MLIR elements attribute.
tensorflow::StatusOr<ElementsAttr> ConvertTensorProto(
    const tensorflow::TensorProto& input_tensor, Builder builder,
    TFGraphDialect* tfgDialect);

// Converts an TensorFlow tensor into an MLIR elements attribute.
tensorflow::StatusOr<ElementsAttr> ConvertTensor(
    const tensorflow::Tensor& input_tensor, Builder builder,
    TFGraphDialect* tfgDialect);

// Converts a shape from MLIR to a TensorFlow tensor shape proto.
void ConvertToTensorShapeProto(ArrayRef<int64_t> shape,
                               tensorflow::TensorShapeProto* output_shape);

// Converts an MLIR type to a TensorFlow tensor shape.
tensorflow::PartialTensorShape ConvertTypeToTensorShape(const Type& type);

// Converts a TensorFlow shape attribute to an MLIR shape attribute.
tensorflow::StatusOr<ShapeAttr> ConvertTensorShapeProto(
    const tensorflow::TensorShapeProto& shape, MLIRContext* context);

// Fill in the contents of TensorShapeProto for the given shape.
// ShapeContainerT is any type with the following methods:
//   bool hasRank()
//   ArrayRef<int64_t> getShape()
// This includes TF::ShapeAttr and ShapedType.
template <typename ShapeContainerT>
void SetTensorShapeProto(ShapeContainerT shape,
                         tensorflow::TensorShapeProto* proto) {
  if (shape.hasRank()) {
    for (int64_t dim : shape.getShape()) {
      proto->add_dim()->set_size(dim);
    }
  } else {
    proto->set_unknown_rank(true);
  }
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto.
tensorflow::Status ConvertToTensorProto(ElementsAttr attr,
                                        tensorflow::TensorProto* output_tensor);

// Converts an MLIR elements attribute to a TensorFlow tensor.
tensorflow::Status ConvertToTensor(ElementsAttr attr,
                                   tensorflow::Tensor* output_tensor);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_CONVERT_TENSOR_H_
