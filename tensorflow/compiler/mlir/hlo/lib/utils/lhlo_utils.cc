/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/utils/lhlo_utils.h"

namespace mlir {
namespace lmhlo {

std::string mapLhloOpToHloOpName(const std::string& lhlo_op_name) {
  return ("mhlo." + lhlo_op_name.substr(9));
}

std::pair<Operation*, int> getArbitraryLhloUser(Value memref) {
  assert(memref.getType().isa<MemRefType>() && "unexpected non memref type");
  // TODO: Add visited check.
  // There might be recursive call in the ir in theory
  for (auto user : memref.getUsers()) {
    if (isa<FusionOp>(user)) {
      auto fusion_op = cast<FusionOp>(user);
    }
    if (user->getDialect()->getNamespace() == "lmhlo") {
      int operand_index = -1;
      for (auto indexed_operand : llvm::enumerate(user->getOperands())) {
        if (indexed_operand.value() == memref) {
          operand_index = indexed_operand.index();
          break;
        }
      }
      assert(operand_index != -1);
      return std::make_pair(user, operand_index);

    } else if (isa<memref::CastOp>(user)) {
      auto res = getArbitraryLhloUser(user->getResult(0));
      if (res.first == nullptr) {
        continue;
      } else {
        return res;
      }

    } else if (isa<CallOp>(user)) {
      auto call_op = cast<CallOp>(user);

      auto module = call_op->getParentOfType<ModuleOp>();
      auto callee = module.lookupSymbol<FuncOp>(call_op.getCallee());
      if (callee == nullptr) {
        emitError(module.getLoc(), "callee not found.");
      }

      Block* entry_block = &callee.getBody().front();
      bool has_fusion_op = false;
      entry_block->walk([&](Operation* op) {
        if (dyn_cast<FusionOp>(op)) {
          has_fusion_op = true;
        }
      });
      if (has_fusion_op) {
        int operand_index = -1;
        for (auto indexed_operand : llvm::enumerate(user->getOperands())) {
          if (indexed_operand.value() == memref) {
            operand_index = indexed_operand.index();
            break;
          }
        }
        assert(operand_index != -1);
        return std::make_pair(user, operand_index);
      }

      Operation::operand_range operands = call_op.operands();

      int operand_index = -1;
      for (auto indexed_operand : llvm::enumerate(operands)) {
        if (indexed_operand.value() == memref) {
          operand_index = indexed_operand.index();
          break;
        }
      }
      assert(operand_index != -1);

      if (callee.getCallableRegion() == nullptr) {
        continue;
      }
      auto memref_in_nested_func = callee.getArgument(operand_index);

      auto res = getArbitraryLhloUser(memref_in_nested_func);
      if (res.first == nullptr) {
        continue;
      } else {
        return res;
      }
    } else if (isa<ReturnOp>(user)) {
      int return_idx = -1;
      for (auto indexed_operand : llvm::enumerate(user->getOperands())) {
        if (indexed_operand.value() == memref) {
          return_idx = indexed_operand.index();
          break;
        }
      }
      assert(return_idx != -1);
      auto module_op = user->getParentOfType<ModuleOp>();
      std::pair<Operation*, int> result = std::make_pair(nullptr, -1);
      module_op.walk([&](CallOp parent_call) {
        if (parent_call.getCallee().str() !=
            cast<FuncOp>(user->getParentOp()).getName().str()) {
          return;
        }
        auto res = getArbitraryLhloUser(parent_call.getResult(return_idx));
        if (res.first == nullptr) {
          return;
        } else {
          // TODO: to break the walker
          result = res;
        }
      });
      if (result.first != nullptr) {
        return result;
      }
    }
  }
  return std::make_pair(nullptr, -1);
}

bool isDeviceAlloc(memref::AllocOp alloc) {
  auto memref = alloc.getResult();
  auto res = getArbitraryLhloUser(memref);
  auto lhlo_op = res.first;
  if (lhlo_op == nullptr) {
    return false;
  }
  int operand_index = res.second;
  if (auto d2h = dyn_cast<::mlir::lmhlo::D2HOp>(lhlo_op)) {
    return operand_index == 0;
  } else if (auto h2d = dyn_cast<::mlir::lmhlo::H2DOp>(lhlo_op)) {
    return operand_index == 1;
  }
  auto attr = lhlo_op->getAttrOfType<StringAttr>(hlo::kPlaceTyAttr);
  if ((attr != nullptr) && (attr.getValue() == hlo::kTypeHost)) {
    return false;
  }
  std::string name_str = lhlo_op->getName().getStringRef().str();
  // it contains fusionOp
  if (name_str == "std.call") {
    return true;
  }
  name_str = lmhlo::mapLhloOpToHloOpName(name_str);
  if (hlo::kShapeCalcOperandMap.find(name_str) ==
      hlo::kShapeCalcOperandMap.end()) {
    return true;
  }
  auto shape_operand_indices = hlo::kShapeCalcOperandMap.at(name_str);
  return (shape_operand_indices.find(operand_index) ==
          shape_operand_indices.end());
}

}  // namespace lmhlo
}  // namespace mlir
