/* Copyright 2026 The OpenXLA Authors.

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

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/WalkResult.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTADDCTRLDEPENDENCIESPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

class IfrtAddCtrlDependenciesPass
    : public impl::IfrtAddCtrlDependenciesPassBase<
          IfrtAddCtrlDependenciesPass> {
 public:
  void runOnOperation() override;
};

void IfrtAddCtrlDependenciesPass::runOnOperation() {
  mlir::func::FuncOp func_op = getOperation();
  // Mapping between IfrtDevicesAttr and the control output of the last
  // CallOp traversed that has that devices attribute.
  llvm::DenseMap<IfrtDevicesAttr, mlir::TypedValue<IfrtControlType>>
      call_op_to_control_output;
  func_op.walk([&](CallOp call_op) {
    IfrtDevicesAttr devices_attr = call_op.getDevicesAttr();
    if (mlir::TypedValue<IfrtControlType> ctrl_input =
            call_op_to_control_output.lookup(devices_attr)) {
      call_op.getControlInputsMutable().append(ctrl_input);
    }
    call_op_to_control_output[devices_attr] = call_op.getControlOutput();
    return mlir::WalkResult::skip();
  });
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
