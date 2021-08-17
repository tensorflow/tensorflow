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

#include "tensorflow/compiler/mlir/tensorflow/utils/verification_utils.h"

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace TF {

LogicalResult VerifyShapeOfReshapeOp(ArrayRef<int64_t> shape) {
  bool has_dynamic_dim = false;
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamicSize) {
      if (dim < 0) return failure();
      continue;
    }
    if (has_dynamic_dim) return failure();
    has_dynamic_dim = true;
  }
  return success();
}

}  // namespace TF
}  // namespace mlir
