/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/ifrt_interfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/constants.h"

// Generated definitions.

#define GET_ATTR_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_attr_interfaces.cc.inc"

#define GET_OP_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_op_interfaces.cc.inc"

namespace mlir {
namespace OpTrait {
namespace xla {
namespace ifrt {
namespace impl {

LogicalResult verifyNestedInIfrtFunc(Operation* op) {
  auto func_op = op->getParentOfType<func::FuncOp>();
  if (func_op != nullptr &&
      !func_op->hasAttr(::xla::ifrt::kIfrtFunctionAttrName) &&
      !func_op->hasAttr(::xla::ifrt::kIfrtReshardFunctionAttrName)) {
    return op->emitOpError()
           << "must be in a FuncOp with attr `"
           << ::xla::ifrt::kIfrtFunctionAttrName << "` or atttr `"
           << ::xla::ifrt::kIfrtReshardFunctionAttrName << "`";
  }
  return success();
}

}  // namespace impl
}  // namespace ifrt
}  // namespace xla
}  // namespace OpTrait
}  // namespace mlir
