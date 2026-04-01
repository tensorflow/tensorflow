/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_TYPES_H_

#include <optional>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace ifrt_serving {

// TODO - b/445480506: Refactor this struct to a class so that accesses are
// always through accessors.
struct DtypeAndShape {
  tensorflow::DataType dtype;
  tensorflow::TensorShape shape;
  // If available, use static shape (the upper bound of the actual input shapes)
  // for compilation and caching.
  std::optional<tensorflow::TensorShape> static_shape;

  bool operator==(const DtypeAndShape& other) const {
    return dtype == other.dtype && shape == other.shape &&
           static_shape == other.static_shape;
  }

  // Returns the shape to be used for compilation. If static shape is
  // available, use static shape, so that the same executable can be reused for
  // all input shapes with the same static shape.
  tensorflow::TensorShape& GetMutableShapeForCompilation() {
    // Avoid `static_shape.value_or(shape)`. Since `value_or` returns by value,
    // using it here would return a reference to a temporary object, leading to
    // a dangling reference.
    return static_shape.has_value() ? *static_shape : shape;
  }
  const tensorflow::TensorShape& GetShapeForCompilation() const {
    // Avoid `static_shape.value_or(shape)`. Since `value_or` returns by value,
    // using it here would return a reference to a temporary object, leading to
    // a dangling reference.
    return static_shape.has_value() ? *static_shape : shape;
  }
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_TYPES_H_
