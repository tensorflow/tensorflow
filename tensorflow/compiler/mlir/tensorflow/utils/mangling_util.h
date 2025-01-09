/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_MANGLING_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_MANGLING_UTIL_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace mangling_util {
// The type of a mangled string.
enum class MangledKind { kUnknown, kDataType, kTensorShape, kTensor };

// Mangles an attribute name, marking the attribute as a TensorFlow attribute.
string MangleAttributeName(absl::string_view str);

// Returns true if 'str' was mangled with MangleAttributeName.
bool IsMangledAttributeName(absl::string_view str);

// Demangles an attribute name that was manged with MangleAttributeName.
// REQUIRES: IsMangledAttributeName returns true.
absl::string_view DemangleAttributeName(absl::string_view str);

// Returns the type of a mangled string, or kUnknown.
MangledKind GetMangledKind(absl::string_view str);

// Return a TensorShapeProto mangled as a string.
string MangleShape(const TensorShapeProto& shape);
// Demangle a string mangled with MangleShape.
absl::Status DemangleShape(absl::string_view str, TensorShapeProto* proto);

// Return a TensorProto mangled as a string.
string MangleTensor(const TensorProto& tensor);
// Demangle a string mangled with MangleTensor.
absl::Status DemangleTensor(absl::string_view str, TensorProto* proto);

// Return a DataType mangled as a string.
string MangleDataType(const DataType& dtype);
// Demangle a string mangled with MangleDataType.
absl::Status DemangleDataType(absl::string_view str, DataType* proto);

}  // namespace mangling_util
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_MANGLING_UTIL_H_
