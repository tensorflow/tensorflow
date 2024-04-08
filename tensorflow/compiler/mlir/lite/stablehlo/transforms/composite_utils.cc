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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_utils.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project

namespace mlir {
namespace odml {
// Changes a DenseIntElementsAttr **containing I64** elements to an I32 Vector.
bool DenseI64AttrToI32Vector(const DenseIntElementsAttr& dense_attr,
                             std::vector<int32_t>* out_vec) {
  std::vector<int32_t> ret(dense_attr.getNumElements());
  auto range = dense_attr.getValues<int64_t>();
  std::transform(range.begin(), range.end(), ret.begin(),
                 [](int64_t attr) { return static_cast<int32_t>(attr); });
  *out_vec = std::move(ret);
  return true;
}

bool GetI32VectorFromDenseI64CompositeAttr(
    const DictionaryAttr& composite_attrs, const std::string& attr_name,
    std::vector<int32_t>* out_vec) {
  DenseIntElementsAttr attr;
  if (!EnsureAttribute<DenseIntElementsAttr>(composite_attrs, attr_name,
                                             &attr)) {
    return false;
  }

  return DenseI64AttrToI32Vector(attr, out_vec);
}
}  // namespace odml
}  // namespace mlir
