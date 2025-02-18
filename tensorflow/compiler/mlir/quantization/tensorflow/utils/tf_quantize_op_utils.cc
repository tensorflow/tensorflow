/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_quantize_op_utils.h"

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace quant {

UnrankedTensorType CreateUnknownShapeFromElementType(Type tensor_type) {
  if (!mlir::cast<TensorType>(tensor_type)) return UnrankedTensorType();
  return UnrankedTensorType::get(
      mlir::cast<TensorType>(tensor_type).getElementType());
}

}  // namespace quant
}  // namespace mlir
