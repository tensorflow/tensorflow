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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DYNAMIC_SHAPE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DYNAMIC_SHAPE_UTILS_H_

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project

namespace tensorflow {

llvm::SmallVector<int64_t> ConvertTFShapeToMlir(llvm::ArrayRef<int64_t> shapes);

llvm::SmallVector<int64_t> ConvertMlirShapeToTF(llvm::ArrayRef<int64_t> shape);

static constexpr int64_t kTFDynamicSize = -1;
mlir::RankedTensorType GetTypeFromTFTensorShape(llvm::ArrayRef<int64_t> shape,
                                                mlir::Type elementType,
                                                mlir::Attribute encoding = {});

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DYNAMIC_SHAPE_UTILS_H_
