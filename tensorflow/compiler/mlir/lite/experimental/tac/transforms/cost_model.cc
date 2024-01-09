/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/cost.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// These are just fake costs.
constexpr float kDequantCost = 2.0;
constexpr float kQuantCost = 2.0;
constexpr float kRequantCost = 2.0;

// TODO(renjieliu): Ideally this should consider different kinds of SOCs as
// well.

// Get total bytes transferred.
int64_t GetTransferredTensorBytes(func::CallOp from_graph,
                                  func::CallOp to_graph) {
  int64_t total_size_transferred = 0;
  for (auto input : to_graph.getOperands()) {
    Operation* input_op = input.getDefiningOp();
    if (input_op && input_op == from_graph.getOperation()) {
      auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
      if (input_type == nullptr || !input_type.hasStaticShape()) continue;
      // Quantized type does not support getSizeInBits.
      if (IsQUI8Type(input_type) || IsQI8Type(input_type)) {
        total_size_transferred += input_type.getNumElements() * 8;
      } else {
        auto s_type = input_type.cast<ShapedType>();
        total_size_transferred +=
            s_type.getNumElements() * s_type.getElementTypeBitWidth();
      }
    }
  }
  return total_size_transferred;
}

// Get total tensor element size transferred.
int64_t GetTransferredElementCount(func::CallOp from_graph,
                                   func::CallOp to_graph) {
  int64_t total_element_count = 0;
  for (auto input : to_graph.getOperands()) {
    Operation* input_op = input.getDefiningOp();
    if (input_op && input_op == from_graph.getOperation()) {
      auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
      if (input_type == nullptr || !input_type.hasStaticShape()) continue;
      total_element_count += input_type.getNumElements();
    }
  }
  return total_element_count;
}

struct GetOpCostPass
    : mlir::PassWrapper<GetOpCostPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GetOpCostPass)

  llvm::StringRef getArgument() const final { return "tfl-get-op-cost"; }
  llvm::StringRef getDescription() const final {
    return "Get cost for every op";
  }
  void runOnOperation() override;
};

void GetOpCostPass::runOnOperation() {
  auto func = getOperation();
  OpBuilder builder(func);
  func.walk([&](Operation* op) {
    if (IsNonConstOp(op) && !IsTerminatorOp(op) &&
        !llvm::isa<func::ReturnOp, func::FuncOp, CallOpInterface>(op)) {
      auto hardware = GetTargetAnnotation(op);
      if (!hardware) return;
      float cost = GetCostForOp(op, *hardware);
      UpdateCost(op, cost, &builder);
    }
  });
}

}  // namespace

float GetCostForOp(Operation* op, const std::string& hardware) {
  auto* device_hardware = GetTargetHardware(hardware);
  if (device_hardware == nullptr) {
    return kDefaultFixedValuedCost;
  }

  return device_hardware->GetOpCost(op);
}

float GetCostForFunc(func::FuncOp* func, const std::string& hardware) {
  auto* device_hardware = GetTargetHardware(hardware);
  if (device_hardware == nullptr) {
    return kDefaultFixedValuedCost;
  }

  return device_hardware->GetFuncCost(func);
}

float GetTransferCost(const std::string& from_hardware_str,
                      const std::string& to_hardware_str,
                      func::CallOp from_graph, func::CallOp to_graph) {
  auto from_hardware = GetTargetHardware(from_hardware_str);
  auto to_hardware = GetTargetHardware(to_hardware_str);
  if (from_hardware == nullptr) {
    from_graph.emitError(absl::StrCat(
        "we cannot find the registered hardware: ", from_hardware_str));
  }

  if (to_hardware == nullptr) {
    to_graph.emitError(absl::StrCat("we cannot find the registered hardware: ",
                                    to_hardware_str));
  }

  const int64_t total_size_transferred =
      GetTransferredTensorBytes(from_graph, to_graph);
  return to_hardware->GetHardwareSwitchingCost(from_hardware,
                                               total_size_transferred);
}

float GetQuantDequantCost(InferenceType from_inference_type,
                          InferenceType to_inference_type,
                          func::CallOp from_graph, func::CallOp to_graph) {
  // Same inference type, no dequant/quant happens.
  if (from_inference_type == to_inference_type) return 0;

  const int64_t total_element_count_transferred =
      GetTransferredElementCount(from_graph, to_graph);

  if (from_inference_type == FLOAT || from_inference_type == HYBRID) {
    // FLOAT <-> HYBRID will have no quant/dequant as well.
    if (to_inference_type == FLOAT || to_inference_type == HYBRID) {
      return 0;
    } else if (to_inference_type == QUANTIZED_INT8 ||
               to_inference_type == QUANTIZED_UINT8) {
      // QUANT path.
      return kQuantCost * total_element_count_transferred;
    }
  }

  if (from_inference_type == QUANTIZED_INT8 ||
      from_inference_type == QUANTIZED_UINT8) {
    // Dequant path.
    if (to_inference_type == FLOAT || to_inference_type == HYBRID) {
      return kDequantCost * total_element_count_transferred;
    } else if (to_inference_type == QUANTIZED_INT8 ||
               to_inference_type == QUANTIZED_UINT8) {
      // Requant path.
      return kRequantCost * total_element_count_transferred;
    }
  }

  // Default quant/dequant/requant cost.
  return kDefaultFixedValuedCost;
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateGetOpCostPass() {
  return std::make_unique<GetOpCostPass>();
}

static PassRegistration<GetOpCostPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
