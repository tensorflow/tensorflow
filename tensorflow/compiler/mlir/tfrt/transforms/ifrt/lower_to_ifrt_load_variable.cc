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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_constants.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_LOWERTOIFRTLOADVARIABLEPASS
#define GEN_PASS_DECL_LOWERTOIFRTLOADVARIABLEPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class LowerToIfrtLoadVariablePass
    : public impl::LowerToIfrtLoadVariablePassBase<
          LowerToIfrtLoadVariablePass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(&getContext());
    std::vector<mlir::TF::ReadVariableOp> read_variable_ops;
    module.walk(
        [&](mlir::TF::ReadVariableOp op) { read_variable_ops.push_back(op); });

    for (auto& read_variable_op : read_variable_ops) {
      auto used_by_tpu_attr = read_variable_op->getAttrOfType<mlir::BoolAttr>(
          kVariableUsedByDeviceAttr);

      // No need to convert to IFRT array if not used by device.
      if (!used_by_tpu_attr || !used_by_tpu_attr.getValue()) continue;
      auto sharding_config_attr =
          read_variable_op->getAttrOfType<mlir::StringAttr>(
              kVariableShardingConfigTextAttr);

      if (!sharding_config_attr ||
          sharding_config_attr.getValue().str().empty()) {
        read_variable_op->emitError()
            << "missing valid sharding config for variable used by device";
        return signalPassFailure();
      }

      auto variable_tensor_name_attr =
          read_variable_op->getAttrOfType<mlir::StringAttr>(
              kVariableArrayNameAttr);
      if (!variable_tensor_name_attr ||
          variable_tensor_name_attr.getValue().str().empty()) {
        read_variable_op->emitError()
            << "missing valid variable tensor name for variable used by device";
        return signalPassFailure();
      }

      builder.setInsertionPointAfter(read_variable_op);
      builder.create<mlir::TF::IfrtLoadVariableOp>(
          read_variable_op->getLoc(), read_variable_op.getValue(),
          sharding_config_attr, variable_tensor_name_attr);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateLowerToIfrtLoadVariablePass() {
  return std::make_unique<LowerToIfrtLoadVariablePass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
