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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_COMPOSITE_AVG_POOL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_COMPOSITE_AVG_POOL_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {

// Given a Composite op that wraps a core.aten.avg_pool2d, returns the padding
// configuration required for the `tfl.pad` if the padding part of the op is
// to be done before average pooling.
DenseIntElementsAttr GetPaddingArrayAttr(Builder& builder, Operation* old_op);

// Given a Composite op that wraps a core.aten.avg_pool2d, and assuming that
// the padding part is extracted into a tfl.pad op prior to a
// tfl.average_pool_2d, this function finds the return type of the needed
// tfl.pad .
ShapedType GetPaddedType(Operation* old_op);

// Given a Composite op that wraps a core.aten.avg_pool2d, finds the padding
// attribute to be passed to the a tfl.average_pool_2d that can fully replace
// this composite (here, padding is done directly by the tfl.average_pool_2d as
// opposed to being extracted into a separate tfl.pad).
StringAttr GetPaddingStringAttr(Builder& builder, Operation* old_op);

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_COMPOSITE_AVG_POOL_H_
