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

// This pass replicates the tf._TPUCompileMlir op on each host that needs the
// compiled program. It helps avoid transferring the compiled binary between
// hosts.

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFTPU {
namespace {

using DeviceNameUtils = ::tensorflow::DeviceNameUtils;
using ParsedName = ::tensorflow::DeviceNameUtils::ParsedName;

constexpr char kDeviceAttr[] = "device";
constexpr int kStatusResultIndex = 0;
constexpr int kProgramResultIndex = 1;

static std::string GetHost(Operation *op) {
  if (StringAttr device = op->getAttrOfType<StringAttr>(kDeviceAttr)) {
    ParsedName parsed_name;
    DeviceNameUtils::ParseFullName(device.getValue().str(), &parsed_name);
    return DeviceNameUtils::ParsedNameToString(
        DeviceNameUtils::AddressSpace(parsed_name));
  }
  return "";
}

class TPUCompileOpReplicationPass
    : public PassWrapper<TPUCompileOpReplicationPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    getOperation().walk([&](TF::_TPUCompileMlirOp tpu_compile_op) {
      Value compiled_program = tpu_compile_op.getResult(kProgramResultIndex);
      std::string tpu_compile_op_host = GetHost(tpu_compile_op.getOperation());
      llvm::StringMap<Operation *> compile_op_by_host;
      llvm::SmallVector<OpOperand *, 4> usages;

      for (OpOperand &usage : compiled_program.getUses()) {
        usages.push_back(&usage);
      }

      // For any op which uses the program compiled on a different host than the
      // original tf._TPUCompileMlir op, replicate the tf._TPUCompileMlir op on
      // that host and update the op to use the program compiled on the same
      // host.
      for (OpOperand *usage : usages) {
        std::string usage_op_host = GetHost(usage->getOwner());
        if (usage_op_host == tpu_compile_op_host) continue;

        Operation *&new_compile_op = compile_op_by_host[usage_op_host];
        // If it is not already created, create a tf._TPUCompileMlir op and a
        // tf.TPUCompileSucceededAssert op on the first CPU of the target host.
        if (!new_compile_op) {
          std::string device_name = usage_op_host + "/device:CPU:0";
          OpBuilder builder(tpu_compile_op);
          new_compile_op = builder.clone(*tpu_compile_op.getOperation());
          new_compile_op->setAttr(kDeviceAttr,
                                  StringAttr::get(&getContext(), device_name));
          TF::TPUCompileSucceededAssertOp new_assert_op =
              builder.create<TF::TPUCompileSucceededAssertOp>(
                  new_compile_op->getLoc(),
                  new_compile_op->getResult(kStatusResultIndex));
          new_assert_op->setAttr(kDeviceAttr,
                                 new_compile_op->getAttr(kDeviceAttr));
        }
        // Updates the operand to use the result of the newly created
        // tf._TPUCompileMlir op.
        usage->set(new_compile_op->getResult(kProgramResultIndex));
      }
      return WalkResult::advance();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUCompileOpReplicationPass() {
  return std::make_unique<TPUCompileOpReplicationPass>();
}

static PassRegistration<TPUCompileOpReplicationPass> pass(
    "tf-tpu-compile-replication",
    "Replicate the TPU compile op to avoid sending the compiled binary between "
    "hosts.");

}  // namespace TFTPU
}  // namespace mlir
