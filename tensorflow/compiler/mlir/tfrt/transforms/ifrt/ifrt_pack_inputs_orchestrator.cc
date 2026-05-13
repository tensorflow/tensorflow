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
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_constants.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/pack_inputs_pass.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf_ifrt_passes.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DECL_IFRTPACKINPUTSORCHESTRATORPASS
#define GEN_PASS_DEF_IFRTPACKINPUTSORCHESTRATORPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class IfrtPackInputsOrchestratorPass
    : public impl::IfrtPackInputsOrchestratorPassBase<
          IfrtPackInputsOrchestratorPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(&getContext());

    // Build lookup map from program_id to callee FuncOp
    llvm::DenseMap<int64_t, mlir::func::FuncOp> program_to_func;
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      if (auto attr = func->getAttrOfType<mlir::IntegerAttr>(
              "tfrt_ifrt_serving.program_id")) {
        program_to_func[attr.getInt()] = func;
      }
    }

    auto process_call = [&](auto call) -> mlir::WalkResult {
      auto group_ids_attr =
          call->template getAttrOfType<mlir::ArrayAttr>(kIfrtPackGroupIdsAttr);
      if (!group_ids_attr) {
        return mlir::WalkResult::advance();
      }

      llvm::SmallVector<int64_t> group_ids;
      for (auto attr : group_ids_attr) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
          group_ids.push_back(int_attr.getInt());
        } else {
          group_ids.push_back(-1);
        }
      }

      bool has_packed = false;
      for (auto gid : group_ids) {
        if (gid >= 0) {
          has_packed = true;
          break;
        }
      }
      if (!has_packed) {
        return mlir::WalkResult::advance();
      }

      int64_t program_id = call.getProgramId();
      auto func_it = program_to_func.find(program_id);
      if (func_it == program_to_func.end()) {
        call->emitError() << "No callee function found with program_id: "
                          << program_id;
        return mlir::WalkResult::interrupt();
      }
      mlir::func::FuncOp callee = func_it->second;

      auto args = call.getArgs();
      llvm::SmallVector<int64_t> pack_offsets(args.size(), 0);
      std::vector<SliceInfo> slices;

      int64_t offset = 0;
      for (size_t i = 0; i < args.size(); ++i) {
        if (group_ids[i] != 0) {
          continue;
        }

        auto tensor_type =
            llvm::dyn_cast<mlir::RankedTensorType>(args[i].getType());
        if (!tensor_type) continue;

        int64_t bitwidth = tensor_type.getElementType().getIntOrFloatBitWidth();
        int64_t byte_size = tensor_type.getNumElements() * (bitwidth / 8);

        // Align offsets to 16-byte boundaries
        offset = (offset + 15) & ~15;
        pack_offsets[i] = offset;

        slices.push_back({.arg_index = static_cast<unsigned>(i),
                          .start = offset,
                          .size = byte_size});

        offset += byte_size;
      }

      // Temporarily rename callee to "main" to satisfy PackInputsPass
      // expectation
      std::string callee_name = callee.getSymName().str();
      callee.setName("main");
      callee.setSymName("main");

      mlir::PassManager pm(module.getContext());
      pm.addPass(CreatePackInputsPass(slices));
      if (mlir::failed(pm.run(module))) {
        call->emitError() << "Failed to run PackInputsPass upstream rewriter";
        return mlir::WalkResult::interrupt();
      }

      // Restore callee name
      callee.setName(callee_name);
      callee.setSymName(callee_name);

      call->setAttr(kIfrtPackOffsetsAttr,
                    builder.getI64ArrayAttr(pack_offsets));

      return mlir::WalkResult::advance();
    };

    // Process both CallOp types
    if (module
            .walk([&](mlir::TF::IfrtCallOp call) { return process_call(call); })
            .wasInterrupted()) {
      return signalPassFailure();
    }
    if (module
            .walk([&](mlir::TF::AsyncIfrtCallOp call) {
              return process_call(call);
            })
            .wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtPackInputsOrchestratorPass() {
  return std::make_unique<IfrtPackInputsOrchestratorPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
