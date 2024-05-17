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

#include "tensorflow/compiler/mlir/lite/quantization/quantization_context.h"

#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/device_target.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"

#define DEBUG_TYPE "quantization-context"

namespace mlir {
namespace quant {

QuantizeContext::QuantizeContext(func::FuncOp func, const DeviceTarget &spec)
    : func_(func), target_spec_(spec) {
  llvm::DenseMap<Value, int> value_to_state;
  func.walk([&](quantfork::QuantizeRegionOp op) {
    for (int i = 0, e = op.getNumOperands(); i != e; ++i) {
      states_manager_.InitializeOperandState(op, i, &value_to_state);
    }

    for (int res = 0, e = op.getNumResults(); res != e; ++res) {
      states_manager_.InitializeResultState(op, res, &value_to_state);
    }
  });
}

std::vector<quantfork::QuantizeRegionOp> QuantizeContext::GetAllOps() {
  std::vector<quantfork::QuantizeRegionOp> all_ops;
  all_ops.reserve(128);
  func_.walk([&](quantfork::QuantizeRegionOp op) { all_ops.push_back(op); });
  return all_ops;
}

KernelSpecs::Signature QuantizeContext::GetSignature(
    quantfork::QuantizeRegionOp op) {
  KernelSpecs::Signature signature;
  signature.reserve(op.getInputSpecs().size() + op.getOutputSpecs().size());
  for (int i = 0; i < op.getNumOperands(); ++i) {
    DeviceTarget::AppendToSignature(GetOperandParams(op, i), &signature);
  }
  for (int i = 0; i < op.getNumResults(); ++i) {
    DeviceTarget::AppendToSignature(GetResultParams(op, i), &signature);
  }
  return signature;
}

LogicalResult QuantizeContext::Handle(
    quantfork::QuantizeRegionOp op,
    llvm::SmallVectorImpl<Operation *> *new_items, bool *changed) {
  auto signature = GetSignature(op);
  auto spec = target_spec_.GetKernelSpec(op.getLogicalKernel(), signature);
  if (!spec.has_value()) {
    op.emitWarning(
        "Couldn't find kernel from the registration for quantization.");
    return success();
  }
  switch (spec->type) {
    case ScaleConstraintType::OutputInputFreeScale: {
      // no propagation.
      *changed |= false;
      break;
    }
    case ScaleConstraintType::CustomScale: {
      if (failed(spec->scale_fn(this, op, new_items, changed))) {
        return failure();
      }
      break;
    }
    case ScaleConstraintType::OutputInputSameScale: {
      auto params = GetQuantParamsForSameScaleConstraint(op);
      if (EmptyParams(params)) {
        *changed |= false;
        break;
      }
      // propagate this params to all the quantizable ports.
      if (failed(PropagateQuantParams(op, params, new_items, changed))) {
        return failure();
      }
      break;
    }
    default: {
      // TODO(fengliuai): implement the other types.
      llvm_unreachable("no implementation.");
      return failure();
    }
  }
  return success();
}

LogicalResult QuantizeContext::Finalize() {
  MLIRContext *context = func_.getContext();
  func_.walk([&](quantfork::QuantizeRegionOp op) {
    llvm::SmallVector<Attribute, 4> input_specs;
    auto original_input_specs = op.getInputSpecs().getValue();
    for (int i = 0, e = op.getNumOperands(); i != e; ++i) {
      auto &state = states_manager_.GetOperandQuantState(op, i);
      auto &requantize = states_manager_.GetOperandRequantizeState(op, i);
      if (state.IsEmpty() && requantize.pos == RequantizeState::NO_REQUANTIZE) {
        input_specs.push_back(original_input_specs[i]);
      } else if (requantize.pos == RequantizeState::ON_OUTPUT) {
        input_specs.push_back(TypeAttr::get(requantize.params));
      } else {
        input_specs.push_back(TypeAttr::get(state.params));
      }
    }
    op->setAttr("input_specs", ArrayAttr::get(context, input_specs));

    llvm::SmallVector<Attribute, 4> output_specs;
    auto original_output_specs = op.getOutputSpecs().getValue();
    for (int res = 0, e = op.getNumResults(); res != e; ++res) {
      auto &state = states_manager_.GetResultQuantState(op, res);
      auto &requantize = states_manager_.GetResultRequantizeState(op, res);
      if (state.IsEmpty() && requantize.pos == RequantizeState::NO_REQUANTIZE) {
        output_specs.push_back(original_output_specs[res]);
      } else if (requantize.pos == RequantizeState::ON_INPUT) {
        output_specs.push_back(TypeAttr::get(requantize.params));
      } else {
        output_specs.push_back(TypeAttr::get(state.params));
      }
    }
    op->setAttr("output_specs", ArrayAttr::get(context, output_specs));
  });
  return success();
}

void QuantizeContext::DumpStates(quantfork::QuantizeRegionOp current_op) {
  if (current_op) {
    llvm::errs() << "\n\n\n" << current_op.getLogicalKernel() << "\n";
  }
  func_.walk([&](quantfork::QuantizeRegionOp op) {
    if (current_op == op) llvm::errs() << "===>>>";
    llvm::errs() << op.getLogicalKernel() << " : (";
    for (auto i = 0; i < op.getNumOperands(); ++i) {
      if (auto params = GetOperandParams(op, i))
        params.print(llvm::errs());
      else
        llvm::errs() << "_";
      llvm::errs() << ",";
    }
    llvm::errs() << ") -> (";
    for (auto i = 0; i < op.getNumResults(); ++i) {
      if (auto params = GetResultParams(op, i))
        params.print(llvm::errs());
      else
        llvm::errs() << "_";
      llvm::errs() << ",";
    }
    llvm::errs() << ")\n";
  });
}

// A heuristic to get quantization parameters satisfies the same scale
// constraints:
// - If there are immutable states,
//   - use the single input, or,
//   - use the single output, or,
//   - use the first one in the collection,
// - use the single input if it is ready, or,
// - use the single output if it is ready, or,
// - use the first ready one in the collection.
QuantParams QuantizeContext::GetQuantParamsForSameScaleConstraint(
    Operation *op) {
  // Two vector to collect Non-empty operands and results states.
  std::vector<quant::QuantState *> mutable_states, immutable_states;
  for (int i = 0, e = op->getNumOperands(); i != e; ++i) {
    auto &state = states_manager_.GetOperandQuantState(op, i);
    if (state.immutable) {
      immutable_states.push_back(&state);
    } else if (!state.IsEmpty()) {
      mutable_states.push_back(&state);
    }
  }

  int immutable_operands_num = immutable_states.size();
  int mutable_operands_num = mutable_states.size();
  // Use the operand's state if it is immutable and it is the only one
  // operand.
  if (op->getNumOperands() == 1 && immutable_operands_num == 1) {
    return immutable_states.front()->params;
  }

  for (int i = 0, e = op->getNumResults(); i != e; ++i) {
    auto &state = states_manager_.GetResultQuantState(op, i);
    if (state.immutable) {
      immutable_states.push_back(&state);
    } else if (!state.IsEmpty()) {
      mutable_states.push_back(&state);
    }
  }

  int immutable_results_num = immutable_states.size() - immutable_operands_num;
  int mutable_results_num = mutable_states.size() - mutable_operands_num;
  // Use the result's state if it is immutable and it is the only one result.
  if (op->getNumResults() == 1 && immutable_results_num == 1) {
    return immutable_states.back()->params;
  }

  LLVM_DEBUG(llvm::dbgs()
             << "Quantization parameters are not collected in an ideal place. "
                "Has to fallback values which might introduce errors.\n");

  // Use the first immutable state to quantize the rest operands and results.
  if (!immutable_states.empty()) return immutable_states.front()->params;

  // If there are no immutable states, use the operand's state if it is the
  // only one operand and has parameters propagated.
  if (op->getNumOperands() == 1 && mutable_operands_num == 1) {
    return mutable_states.front()->params;
  }

  // If there are no immutable states, use the result's state if it is the
  // only one result and has parameters propagated.
  if (op->getNumResults() == 1 && mutable_results_num == 1) {
    return mutable_states.back()->params;
  }

  // Use the first propagated state to quantize the rest operands and results.
  if (!mutable_states.empty()) return mutable_states.front()->params;

  // None operands/results have parameters propagated, skip this node for now.
  return {};
}

LogicalResult QuantizeContext::PropagateQuantParams(
    Operation *op, const QuantParams params,
    quant::AdjacentOperations *new_items, bool *changed) {
  // Use the final state to set all the operands' parameters.
  for (int i = 0, e = op->getNumOperands(); i != e; ++i) {
    auto ele = op->getOperand(i).getType().cast<ShapedType>().getElementType();
    if (ele.isa<FloatType>() && SetOperandParams(op, i, params)) {
      *changed |= true;
      new_items->push_back(op->getOperand(i).getDefiningOp());
    }
  }

  // Use the final state to set all the results' parameters.
  for (int res = 0, e = op->getNumResults(); res != e; ++res) {
    auto ele = op->getResult(res).getType().cast<ShapedType>().getElementType();
    if (ele.isa<FloatType>() && SetResultParams(op, res, params)) {
      auto users = op->getResult(res).getUsers();
      *changed |= !users.empty();
      new_items->append(users.begin(), users.end());
    }
  }
  return success();
}

int QuantizeContext::StatesManager::InitializeState(
    quantfork::QuantizeRegionOp op, int index, bool as_result) {
  Attribute params_attr;
  if (as_result) {
    params_attr = op.getOutputSpecs()[index];
  } else {
    params_attr = op.getInputSpecs()[index];
  }
  QuantParams params =
      params_attr.cast<TypeAttr>().getValue().dyn_cast<QuantParams>();
  bool immutable = !EmptyParams(params);
  int next_state_index = states_.size();
  states_.push_back({params, immutable});
  if (as_result) {
    result_states_.insert({{op, index}, next_state_index});
  } else {
    operand_states_.insert({{op, index}, next_state_index});
  }
  return next_state_index;
}

void QuantizeContext::StatesManager::InitializeOperandState(
    quantfork::QuantizeRegionOp op, int index,
    llvm::DenseMap<Value, int> *cache) {
  Value in = op.getOperand(index);
  auto cached = cache->insert({in, 0});
  if (!cached.second) {
    operand_states_.insert({{op, index}, cached.first->second});
    return;
  }
  cached.first->second = InitializeState(op, index, /*as_result=*/false);
}

void QuantizeContext::StatesManager::InitializeResultState(
    quantfork::QuantizeRegionOp op, int index,
    llvm::DenseMap<Value, int> *cache) {
  auto res = op.getResult(index);
  auto cached = cache->insert({res, 0});
  if (!cached.second) {
    result_states_.insert({{op, index}, cached.first->second});
    return;
  }
  cached.first->second = InitializeState(op, index, /*as_result=*/true);
}

bool QuantizeContext::StatesManager::SetConstantResultParams(Operation *op) {
  llvm_unreachable("no implementation.");
  return false;
}

bool QuantizeContext::StatesManager::SetResultParams(Operation *op,
                                                     int res_index,
                                                     QuantParams params) {
  auto &state = GetResultQuantState(op, res_index);
  if (state.params == params) {
    return false;
  }
  if (!state.IsEmpty()) {
    auto &rescale = GetResultRequantizeState(op, res_index);
    rescale.params = params;
    rescale.pos = RequantizeState::ON_INPUT;
    return false;
  }
  state.params = params;
  return true;
}

bool QuantizeContext::StatesManager::SetOperandParams(Operation *op, int index,
                                                      QuantParams params) {
  auto &state = GetOperandQuantState(op, index);
  if (state.params == params) {
    return false;
  }

  if (!state.IsEmpty()) {
    auto &rescale = GetOperandRequantizeState(op, index);
    rescale.params = params;
    rescale.pos = RequantizeState::ON_OUTPUT;
    return false;
  }
  state.params = params;
  return true;
}
}  //  namespace quant
}  // namespace mlir
