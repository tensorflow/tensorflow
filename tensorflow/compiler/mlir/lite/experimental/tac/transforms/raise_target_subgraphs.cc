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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/subgraph.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// Subgraph here is actually an intermediate data structure holder for the ops:
// The ops within share the same "target", they're topologically sorted.
// The subgraph here will be later populated to generate func ops.
// All the subgraphs should not create cyclic dependencies:
// So we should not have:
//     subgraph1
//             \
//            subgraph2
//            /
//       subgraph1
struct Subgraph {
  // All ops must be inserted in it's topological order.
  llvm::SetVector<Operation*> all_ops;
  int subgraph_id;
  InferenceDeviceType inference_device_type;
};

// This will exclude arguments & consts & quantize/dequantize ops.
inline bool IsTFLNonConstQuatnizeOp(Operation* op) {
  return IsTFLDialectNonConstOp(op) && IsTFLNonQuantDequantizeOp(op);
}

inline bool IsTFLNonConstQuatnizeOp(const Value& value) {
  auto* op = value.getDefiningOp();
  if (op == nullptr) return false;
  return IsTFLNonConstQuatnizeOp(op);
}

// This pass will group those ops (non-const TFL dialect ops) have the same
// target together and raise them as FuncOps.
// See the following Example:
//
//     op1 (GPU)
//       \       op2 (GPU)
//       \        |
//        \      op3 (GPU)
//         \     /
//         op4 (CPU)
//
// will be raised as 3 subgraphs:
// Subgraph 1: {op1}, GPU -> Func_1_GPU
// Subgraph 2: {op2, op3}, GPU -> Func_2_GPU
// Subgraph 3: {op4} CPU -> Func_3_CPU
//
// MainFunc:
//   %0 = call @Func_1_GPU
//   %1 = call @Func_2_GPU
//   %2 = call @Func_3_CPU(%0, %1)
class RaiseTargetSubgraphsPass
    : public mlir::PassWrapper<RaiseTargetSubgraphsPass,
                               mlir::OperationPass<ModuleOp>> {
 private:
  void runOnOperation() override;

  void RaiseTargetSubgraphsForBlock(Block* block, OpBuilder* builder,
                                    ModuleOp module);

  void CreateNewSubgraphOrUpdate(
      Operation* op, const InferenceDeviceType& inference_device_type,
      llvm::SetVector<Operation*>* unprocessed_ops,
      llvm::DenseMap<int, Subgraph>* subraphs,
      llvm::DenseMap<Operation*, int>* op_subgraph_mapping);

  void ExtractSubgraphToFunc(Subgraph* subgraph, OpBuilder* builder,
                             ModuleOp module);

  FuncOp BuildFuncOp(Subgraph* subgraph, OpBuilder* builder, ModuleOp module_op,
                     SmallVector<Value, 4>* inputs,
                     SmallVector<Value, 4>* outputs,
                     InferenceDeviceType* inference_device_type);

  int subgraph_count_ = 0;
};

// Currently if & only if all inputs (exclude arguments, constant, etc.) of the
// op belong to the same subgraph, and the op can be considered as part of the
// graph.
// since the unprocessed_ops are assumed to be already topologically sorted,
// we may recursively call this function.
// We will revisit this logic later.
void RaiseTargetSubgraphsPass::CreateNewSubgraphOrUpdate(
    Operation* op, const InferenceDeviceType& inference_device_type,
    llvm::SetVector<Operation*>* unprocessed_ops,
    llvm::DenseMap<int, Subgraph>* subraphs,
    llvm::DenseMap<Operation*, int>* op_subgraph_mapping) {
  llvm::SetVector<int> input_subgraph_ids;
  for (auto input : op->getOperands()) {
    if (IsTFLNonConstQuatnizeOp(input)) {
      Operation* input_op = input.getDefiningOp();

      // If the input op is generated from a different device, we will skip.
      auto input_op_device = GetInferenceDeviceTypeForOp(input_op);
      if (!input_op_device.hasValue()) {
        input_op->emitError("cannot get hardware or inference type for the op");
        signalPassFailure();
      }

      if (!(input_op_device.getValue() == inference_device_type)) continue;

      // If the input has more than one use, we will skip it.
      // The initiative here is if we have the input has more than one nodes
      // consume it, it will be very tricky for us to find the correct insertion
      // point for the call op, there may not be even a legal insertion point.
      // TODO(renjieliu): May revisit this later.
      if (!input.hasOneUse()) continue;

      // If the input op is not processed yet, we will need to process that
      // first (to get the input op's subgraph_id).
      if (unprocessed_ops->count(input_op) == 1) {
        unprocessed_ops->remove(input_op);
        CreateNewSubgraphOrUpdate(input_op, input_op_device.getValue(),
                                  unprocessed_ops, subraphs,
                                  op_subgraph_mapping);
      }
      // Assume we have got the correct subgraph id for this one.
      input_subgraph_ids.insert(op_subgraph_mapping->find(input_op)->second);
    }
  }

  // This op belongs to the input subgraph.
  if (input_subgraph_ids.size() == 1) {
    const int input_graph_id = input_subgraph_ids.front();
    op_subgraph_mapping->insert({op, input_graph_id});
    subraphs->find(input_graph_id)->second.all_ops.insert(op);
  } else {
    // We need to start a new subgraph.
    Subgraph new_subgaph;
    new_subgaph.inference_device_type = inference_device_type;
    new_subgaph.subgraph_id = subgraph_count_++;
    new_subgaph.all_ops.insert(op);
    subraphs->insert({new_subgaph.subgraph_id, new_subgaph});
    op_subgraph_mapping->insert({op, new_subgaph.subgraph_id});
  }
}

// This is to collect input arguments for the given set of ops.
// See the example:
//
//   value1  value2
//    \     /
//      op1
//        \     value3
//        \   /
//         op2
//         |
//         op3
//
//  Then the arguments will be {value1, value2, value3}
void CollectInputs(const llvm::SetVector<Operation*>& all_ops,
                   SmallVector<Value, 4>* inputs) {
  for (Operation* op : all_ops) {
    for (Value input : op->getOperands()) {
      Operation* input_op = input.getDefiningOp();
      const bool input_within_subgraph =
          (input_op && all_ops.count(input_op) == 1);
      if (!input_within_subgraph) {
        inputs->push_back(input);
      }
    }
  }
}

// This is to collect outputs arguments for the given set of ops.
// See the example:
//
//      op1
//      /    \
//   value1   \
//           op2
//           |  \
//         op3  value2
//         |
//       value3
//
//  Then the arguments will be {value1, value2, value3}
void CollectOutputs(const llvm::SetVector<Operation*>& all_ops,
                    SmallVector<Value, 4>* outputs) {
  for (Operation* op : all_ops) {
    for (Value output : op->getResults()) {
      bool output_consumed_outside_subgraph = false;
      for (Operation* consumer : output.getUsers()) {
        if (all_ops.count(consumer) == 0) {
          output_consumed_outside_subgraph = true;
        }
      }
      if (output_consumed_outside_subgraph) {
        outputs->push_back(output);
      }
    }
  }
}

void BuildTypes(const SmallVector<Value, 4>& values,
                SmallVector<Type, 4>* types) {
  for (auto value : values) {
    types->push_back(value.getType());
  }
}

void GetFunctionName(const Subgraph& subgrpah, std::string* function_name,
                     std::string* interface_name) {
  *interface_name = absl::StrCat("func_", std::to_string(subgrpah.subgraph_id));
  *function_name = absl::StrCat(
      (*interface_name), "_", subgrpah.inference_device_type.hardware, "_",
      GetInferenceString(subgrpah.inference_device_type.inference_type));
}

FuncOp RaiseTargetSubgraphsPass::BuildFuncOp(
    Subgraph* subgraph, OpBuilder* builder, ModuleOp module_op,
    SmallVector<Value, 4>* inputs, SmallVector<Value, 4>* outputs,
    InferenceDeviceType* inference_device_type) {
  CollectInputs(subgraph->all_ops, inputs);
  CollectOutputs(subgraph->all_ops, outputs);

  SmallVector<Type, 4> input_types;
  SmallVector<Type, 4> return_types;

  BuildTypes(*inputs, &input_types);
  BuildTypes(*outputs, &return_types);

  FunctionType function_type =
      builder->getFunctionType(input_types, return_types);

  SmallVector<NamedAttribute, 4> attrs;
  // Function name.
  std::string function_name;
  std::string interface_name;
  GetFunctionName(*subgraph, &function_name, &interface_name);
  attrs.push_back(builder->getNamedAttr(
      kInterfaceNameAttr, builder->getStringAttr(interface_name)));

  // Inference Device type.
  attrs.push_back(builder->getNamedAttr(
      kDevice,
      builder->getStringAttr(subgraph->inference_device_type.hardware)));
  attrs.push_back(builder->getNamedAttr(
      kInferenceType, builder->getStringAttr(GetInferenceString(
                          subgraph->inference_device_type.inference_type))));
  *inference_device_type = subgraph->inference_device_type;

  FuncOp new_func = FuncOp::create(builder->getUnknownLoc(), function_name,
                                   function_type, llvm::makeArrayRef(attrs));
  new_func.setPrivate();

  new_func.addEntryBlock();

  // Function argument mapping.
  llvm::DenseMap<Value, int> function_argument_mapping;
  for (int i = 0; i < inputs->size(); ++i) {
    function_argument_mapping.insert({(*inputs)[i], i});
  }

  OpBuilder function_builder(new_func.getBody());

  llvm::DenseMap<Operation*, Operation*> op_cloned_op_mapping;
  llvm::DenseMap<Value, Value> output_cloned_op_output_mapping;
  for (Operation* op : subgraph->all_ops) {
    Operation* cloned_op = function_builder.clone(*op);
    op_cloned_op_mapping.insert({op, cloned_op});
    for (int i = 0; i < op->getNumResults(); ++i) {
      Value op_output = op->getResult(i);
      Value cloned_op_output = cloned_op->getResult(i);
      output_cloned_op_output_mapping.insert({op_output, cloned_op_output});
    }
  }

  for (Operation* op : subgraph->all_ops) {
    Operation* cloned_op = op_cloned_op_mapping.find(op)->second;
    for (int i = 0; i < op->getNumOperands(); ++i) {
      Value input = op->getOperand(i);
      Value cloned_op_input;
      // If the input is actually a function argument.
      if (function_argument_mapping.count(input) > 0) {
        int function_argument = function_argument_mapping.find(input)->second;
        cloned_op_input = new_func.getArgument(function_argument);
      } else {
        // The input is actually with in the subgraph.
        cloned_op_input = output_cloned_op_output_mapping.find(input)->second;
      }
      cloned_op->setOperand(i, cloned_op_input);
    }
  }

  SmallVector<Value, 4> final_outputs;
  for (auto output : *outputs) {
    auto cloned_output = output_cloned_op_output_mapping.find(output)->second;
    final_outputs.push_back(cloned_output);
  }
  function_builder.create<mlir::ReturnOp>(new_func.getLoc(), final_outputs);

  module_op.push_back(new_func);
  return new_func;
}

void RaiseTargetSubgraphsPass::ExtractSubgraphToFunc(Subgraph* subgraph,
                                                     OpBuilder* builder,
                                                     ModuleOp module) {
  SmallVector<Value, 4> func_inputs;
  SmallVector<Value, 4> func_outputs;

  InferenceDeviceType inference_device_type;
  FuncOp func = BuildFuncOp(subgraph, builder, module, &func_inputs,
                            &func_outputs, &inference_device_type);

  // We just use the location of the last ops in the subgraph as the location
  // for the call_op.
  Operation* last_output = subgraph->all_ops.back();

  // TODO(renjieliu): we should add func attributes to the call op.
  builder->setInsertionPoint(last_output);
  auto call_op =
      builder->create<CallOp>(last_output->getLoc(), func, func_inputs);

  auto interface_name = GetInterFaceName(func);

  // Set call op attribute: interface_name, hardware.
  call_op->setAttr(kInterfaceNameAttr,
                   builder->getStringAttr(interface_name.getValue()));
  call_op->setAttr(kDevice,
                   builder->getStringAttr(inference_device_type.hardware));
  call_op->setAttr(kInferenceType, builder->getStringAttr(GetInferenceString(
                                       inference_device_type.inference_type)));

  // Rewire the outputs.
  if (call_op.getNumResults() != func_outputs.size()) {
    module.emitError("the constructed func op has mismatched returns");
    signalPassFailure();
  }

  for (int i = 0; i < func_outputs.size(); ++i) {
    Value output = func_outputs[i];
    output.replaceAllUsesWith(call_op.getResult(i));
  }

  // Clear the subgraph.
  // Those ops should be removed.
  for (auto* op : subgraph->all_ops) {
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  }
}

// TODO(renjieliu): We may need to consider about side effect ops: we may leave
// those ops alone when building the subgraph.
void RaiseTargetSubgraphsPass::RaiseTargetSubgraphsForBlock(Block* block,
                                                            OpBuilder* builder,
                                                            ModuleOp module) {
  llvm::SetVector<Operation*> unprocessed_ops;
  block->walk([&](Operation* op) {
    // We only care about TFL dialect.
    if (IsTFLNonConstQuatnizeOp(op)) {
      unprocessed_ops.insert(op);
    }
  });

  // Create a new subgraph or add to existing subgrpahs.
  std::unordered_map<InferenceDeviceType, llvm::DenseMap<int, Subgraph>,
                     InferenceDeviceType::inference_device_type_hash>
      all_subgraphs;
  llvm::DenseMap<Operation*, int> op_subgraph_mapping;

  while (!unprocessed_ops.empty()) {
    Operation* current = unprocessed_ops.front();
    unprocessed_ops.remove(current);

    auto current_device = GetInferenceDeviceTypeForOp(current);
    if (!current_device.hasValue()) {
      current->emitError("cannot get hardware or inference type for the op");
      signalPassFailure();
    }

    auto device_subgraphs_it = all_subgraphs.find(current_device.getValue());
    if (device_subgraphs_it == all_subgraphs.end()) {
      auto device_subgraph_insert = all_subgraphs.insert(
          {current_device.getValue(), llvm::DenseMap<int, Subgraph>()});
      device_subgraphs_it = device_subgraph_insert.first;
    }
    CreateNewSubgraphOrUpdate(current, current_device.getValue(),
                              &unprocessed_ops, &device_subgraphs_it->second,
                              &op_subgraph_mapping);
  }

  // Create FuncOp & replace with current uses based on those subgraphs.
  for (auto& device_subgraphs : all_subgraphs) {
    for (auto& ids_subgraph : device_subgraphs.second) {
      ExtractSubgraphToFunc(&ids_subgraph.second, builder, module);
    }
  }
}

void RaiseTargetSubgraphsPass::runOnOperation() {
  auto module = getOperation();
  SmallVector<FuncOp, 16> funcs(module.getOps<FuncOp>());
  for (auto func : funcs) {
    for (auto& block : func) {
      auto builder = OpBuilder::atBlockBegin(&block);
      RaiseTargetSubgraphsForBlock(&block, &builder, module);
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateRaiseTargetSubgraphsPass() {
  return std::make_unique<RaiseTargetSubgraphsPass>();
}

static PassRegistration<RaiseTargetSubgraphsPass> pass(
    "tfl-raise-target-subgraphs",
    "This pass will merge those have target-annotated TFL IRs together & raise "
    "them as a function.");

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
