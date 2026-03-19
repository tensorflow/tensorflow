/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#include "stablehlo_ext/IR/base.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hlo {

DenseIntElementsAttr getPaddingAttr(MLIRContext* context,
                                    ArrayRef<int64_t> values) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(values.size()) / 2, 2},
                            IntegerType::get(context, 64)),
      values);
}

DenseIntElementsAttr getPaddingAttr(Builder* builder,
                                    ArrayRef<int64_t> values) {
  return getPaddingAttr(builder->getContext(), values);
}

}  // namespace hlo
}  // namespace mlir
