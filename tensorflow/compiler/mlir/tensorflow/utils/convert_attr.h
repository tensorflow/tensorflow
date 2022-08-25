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
#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_ATTR_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_ATTR_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/core/framework/attr_value.pb.h"

namespace tensorflow {

using stream_executor::port::StatusOr;

// Converts non func AttrValue proto into an MLIR attribute. Func attribute is
// exclused in this function because the function might be renamed when the
// function definition is imported.
StatusOr<mlir::Attribute> ConvertNonFuncAttributeValue(const AttrValue& value,
                                                       mlir::Builder* builder);

// Converts all kinds of AttrValue proto into an MLIR attribute.
StatusOr<mlir::Attribute> ConvertAttributeValue(const AttrValue& value,
                                                mlir::Builder* builder);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_ATTR_H_
