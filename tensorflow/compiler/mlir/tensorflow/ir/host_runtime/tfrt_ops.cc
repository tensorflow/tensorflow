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

#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"

#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

//===----------------------------------------------------------------------===//
// _TfrtGetResourceOp
//===----------------------------------------------------------------------===//

namespace mlir {
namespace TF {

llvm::SmallVector<ResourceHandleValueAndId, 4>
_TfrtGetResourceOp::GetResourceHandleValueAndIdList(
    llvm::SmallDenseMap<ResourceHandle, int64_t> &resource_handle_id_map,
    int64_t &next_id) {
  llvm::SmallVector<ResourceHandleValueAndId, 4> resource_vec;
  llvm::StringRef device = GetDeviceOrEmpty(getOperation());

  for (const auto &iter : llvm::enumerate(getResults())) {
    auto index = iter.index();
    if (getElementTypeOrSelf(iter.value().getType()).isa<TF::ResourceType>()) {
      resource_vec.push_back(GetResourceHandleValueAndIdBase(
          getContainer()[index].cast<mlir::StringAttr>().getValue(),
          getSharedName()[index].cast<mlir::StringAttr>().getValue(), device,
          getResults()[index], resource_handle_id_map, next_id));
    }
  }
  return resource_vec;
}

LogicalResult _TfrtGetResourceOp::verify() {
  _TfrtGetResourceOp get_resource_op = *this;
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
           << "length of attribute arrays do not match. indices = "
           << indices_size << ", shared_name = " << shared_name_size
           << ", container = " << container_size;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PwStreamResults
//===----------------------------------------------------------------------===//

mlir::LogicalResult PwStreamResultsOp::verify() {
  if (getArgs().size() != getNames().size()) {
    return emitOpError()
           << "has a mismatch between the number of arguments and their names ("
           << getArgs().size() << " vs. " << getNames().size() << ")";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IfrtCall
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtCallOp::verify() {
  auto func = getOperation()->getParentOfType<mlir::func::FuncOp>();
  if (func != nullptr && func->hasAttr("tfrt_ifrt_serving.program_id")) {
    return emitOpError() << "cannot be nested inside an IFRT program";
  }

  for (mlir::Value arg : getArgs()) {
    if (mlir::getElementTypeOrSelf(arg.getType())
            .isa<mlir::TF::ResourceType>()) {
      return emitOpError()
             << "does not support passing '!tf.resource' values as arguments";
    }
  }

  for (mlir::Value result : getResults()) {
    if (mlir::getElementTypeOrSelf(result.getType())
            .isa<mlir::TF::ResourceType>()) {
      return emitOpError()
             << "does not support returning '!tf.resource' values as results";
    }
  }

  // Verify variable_arg_indices is sorted in ascending order.
  int64_t prev_index = -1;
  for (auto arg_index_attr : getVariableArgIndicesAttr()) {
    if (!arg_index_attr.isa_and_nonnull<mlir::IntegerAttr>()) {
      return emitOpError() << "variable_arg_indices must be an integer";
    }

    int64_t index =
        arg_index_attr.dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
    if (index < 0) {
      return emitOpError() << "variable_arg_indices must be positive";
    }

    if (index <= prev_index) {
      return emitOpError()
             << "variable_arg_indices must be sorted in ascending order";
    }
    prev_index = index;
  }

  return mlir::success();
}

}  // namespace TF
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.cc.inc"
