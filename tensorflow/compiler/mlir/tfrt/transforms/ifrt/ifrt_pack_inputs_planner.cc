/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_constants.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf_ifrt_passes.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DECL_IFRTPACKINPUTSPLANNERPASS
#define GEN_PASS_DEF_IFRTPACKINPUTSPLANNERPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class IfrtPackInputsPlannerPass
    : public impl::IfrtPackInputsPlannerPassBase<IfrtPackInputsPlannerPass> {
 public:
  explicit IfrtPackInputsPlannerPass(int threshold) {
    size_threshold_bytes = threshold;
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(&getContext());

    auto process_call = [&](auto call) -> mlir::WalkResult {
      // Extract variable indices
      llvm::SmallVector<int64_t> variable_arg_indices;
      if (auto var_indices_attr = call.getVariableArgIndicesAttr()) {
        for (auto attr : var_indices_attr) {
          if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
            variable_arg_indices.push_back(int_attr.getInt());
          }
        }
      }

      auto args = call.getArgs();
      llvm::SmallVector<int64_t> pack_group_ids;
      pack_group_ids.reserve(args.size());

      for (const auto& [arg_idx, input] : llvm::enumerate(args)) {
        if (llvm::is_contained(variable_arg_indices, arg_idx)) {
          // Skip packed groups for variable tensors
          pack_group_ids.push_back(-1);
          continue;
        }

        auto tensor_type =
            llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
        if (!tensor_type) {
          pack_group_ids.push_back(-1);
          continue;
        }

        if (!tensor_type.hasStaticShape()) {
          pack_group_ids.push_back(-1);
          continue;
        }

        int64_t bitwidth = tensor_type.getElementType().getIntOrFloatBitWidth();
        if (bitwidth < 8 || bitwidth % 8 != 0) {
          pack_group_ids.push_back(-1);
          continue;
        }

        int64_t byte_size = tensor_type.getNumElements() * (bitwidth / 8);
        if (byte_size <= size_threshold_bytes) {
          pack_group_ids.push_back(0);
        } else {
          pack_group_ids.push_back(-1);
        }
      }

      // Update attributes
      call->setAttr(kIfrtPackGroupIdsAttr,
                    builder.getI64ArrayAttr(pack_group_ids));
      return mlir::WalkResult::advance();
    };

    // Walk execution call ops
    mlir::WalkResult walk_ifrt = module.walk(
        [&](mlir::TF::IfrtCallOp call) { return process_call(call); });
    if (walk_ifrt.wasInterrupted()) {
      return signalPassFailure();
    }

    mlir::WalkResult walk_async = module.walk(
        [&](mlir::TF::AsyncIfrtCallOp call) { return process_call(call); });
    if (walk_async.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtPackInputsPlannerPass(int size_threshold_bytes) {
  return std::make_unique<IfrtPackInputsPlannerPass>(size_threshold_bytes);
}

}  // namespace ifrt_serving
}  // namespace tensorflow
