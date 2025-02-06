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

#include "tensorflow/compiler/mlir/lite/utils/size_utils.h"

#include <cstdint>

#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project

namespace mlir {
namespace TFL {

int32_t ConvertToTfliteSize(int64_t size) {
  return mlir::ShapedType::isDynamic(size) ? -1 : static_cast<int32_t>(size);
}

}  // namespace TFL
}  // namespace mlir
