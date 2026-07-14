/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#ifndef STABLEHLO_EXT_DIALECT_BASE_H
#define STABLEHLO_EXT_DIALECT_BASE_H

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace hlo {

DenseIntElementsAttr getPaddingAttr(MLIRContext *context,
                                    ArrayRef<int64_t> value);
DenseIntElementsAttr getPaddingAttr(Builder *builder, ArrayRef<int64_t> value);

}  // namespace hlo
}  // namespace mlir

#endif  // STABLEHLO_EXT_DIALECT_BASE_H
