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

#include "tensorflow/compiler/mlir/tensorflow/ir/tfrt_ops.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

//===----------------------------------------------------------------------===//
// _TfrtGetResourceOp
//===----------------------------------------------------------------------===//

namespace mlir {
namespace TF {

static LogicalResult Verify(_TfrtGetResourceOp get_resource_op) {
  // The sizes of indices, shared_name and container must be equal.
  int32_t indices_size =
      get_resource_op->getAttrOfType<mlir::ArrayAttr>("indices").size();
  int32_t shared_name_size =
      get_resource_op->getAttrOfType<mlir::ArrayAttr>("shared_name").size();
  int32_t container_size =
      get_resource_op->getAttrOfType<mlir::ArrayAttr>("container").size();

  if (!(indices_size == shared_name_size &&
        shared_name_size == container_size)) {
    return get_resource_op->emitError()
           << "Length of attribute arrays do not match. indices = "
           << indices_size << ", shared_name = " << shared_name_size
           << ", container = " << container_size;
  }

  return success();
}

}  // namespace TF
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tfrt_ops.cc.inc"
