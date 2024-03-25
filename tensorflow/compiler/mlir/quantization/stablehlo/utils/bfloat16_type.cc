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

#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/bfloat16_type.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::quant::stablehlo {

bool IsLargeFloatType(Type type) {
  type = getElementTypeOrSelf(type);
  return isa<FloatType>(type) && type.getIntOrFloatBitWidth() > 16;
}

Type ToBfloat16Type(Type type) {
  if (auto shaped = type.dyn_cast<ShapedType>()) {
    const Type elem = shaped.getElementType();
    if (IsLargeFloatType(elem)) {
      return shaped.clone(BFloat16Type::get(type.getContext()));
    }
  } else if (IsLargeFloatType(type)) {
    return BFloat16Type::get(type.getContext());
  }
  return type;
}

}  // namespace mlir::quant::stablehlo
