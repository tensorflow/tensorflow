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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_ATTR_LOWERING_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_ATTR_LOWERING_UTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project

namespace tensorflow {

// TODO(chky): attributes "_output_shapes" should be removed by any tool that
// generates TF MLIR dialect, as they are not used by CoreRuntime. Remove this
// filtering logic once unused attributes are cleaned up in the upper layer.
bool IsUnusedTfrtAttribute(llvm::StringRef name);

// Create a single attribute that contains the named attribute lists. It is an
// array of pairs. The key must be a string attribute, and the value can be
// any attribute that is supported by CoreRuntime.
mlir::ArrayAttr CreateTfrtOpAttrs(llvm::ArrayRef<mlir::NamedAttribute> attrs,
                                  mlir::Builder& builder);

bool IsSupportedTfrtNumericDType(mlir::Type type);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_ATTR_LOWERING_UTILS_H_
