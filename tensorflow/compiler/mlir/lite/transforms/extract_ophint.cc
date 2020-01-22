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
#include <map>
#include <queue>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/LoopAnalysis.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/SymbolTable.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace TFL {
namespace {

constexpr char kTfLiteFunctionName[] = "_tflite_function_name";
constexpr char kTfLiteFunctionUUID[] = "_tflite_function_uuid";
constexpr char kTfLiteFunctionInputIndex[] = "_tflite_function_input_index";
constexpr char kTfLiteFunctionOutputIndex[] = "_tflite_function_output_index";
constexpr char kTfLiteFunctionSortIndex[] = "_tflite_function_sort_index";
constexpr char kTfLiteFunctionAggregate[] = "_tflite_function_aggregate";

constexpr char kStrategyNone[] = "None";
constexpr char kStrategyStack[] = "stack";
constexpr char kStrategyFirst[] = "first";
constexpr char kStrategyLast[] = "last";

//  A Ophinted op typically looks like below"
//
//     InputOp1        InputOp2    InputOp3
//       /  \            |             |
//    val1  val2        val3         val4
//      |    |           |             |
//  identOp1 identOp2  identOp3      identOp4
//     \     |           |            /
//      \    |           |           /
//  ....   a bunch of operations (needs to be fused) ...
//                   /       \
//                  /         \
//      identOp1 (output)    identOp2 (output)
//           |                  |
//       Other ops           Other ops
//
//
//  In this pass, we are trying to convert them into the following format:
//
//                     ||
//                     ||
//                    \ /
//
//     InputOp1        InputOp2    InputOp3
//       /  \            |             /
//    val1  val2        val3         val4
//      \    |           |            /
//       PackOp          |           /
//       \    |          |          /
//        \   |          |         /
//           Call funcOp (fusedOp - name like 'UnidirectionalSequenceRNN')
//            (The funcOp will be inserted at the bottom of the module, also
// .          note every funcOp will be unique.)
//                   |
//                  UnpackOp
//                 /      \
//                /        \
//       Other ops         Other ops
struct OphintCompositeOp {
  // OphintCompositeOp is a conceptually "composite op" which will be converted
  // to a "fused op" later.
  //
  // As a "composite op", it has "inputs" and "outputs", and all the inputs
  // and outputs are annotated by special-annotated identity ops.
  //
  // All inputs and outputs need to be processed based on different strategies,
  // See all the different strategies under
  // tensorflow/lite/python/op_hint.py
  //
  // For example, "stack" strategy means we need to pack the inputs together
  // or unpack the outputs.
 public:
  OphintCompositeOp(StringRef uuid, StringRef function_name)
      : uuid(uuid), function_name(function_name) {}

  void AddInput(int index, Operation* op, StringRef aggregation,
                int sort_index) {
    auto it = inputs.find(index);
    if (it == inputs.end()) {
      AggregatedOperand operand;
      operand.aggregation = aggregation;
      it = inputs.insert({index, operand}).first;
    }
    // TODO(renjieliu): check aggregation strategy stays the same.
    // Also needs to make sure if aggregation strategy is "None" we should not
    // have more than one op.
    it->second.ops[sort_index] = op;
  }

  void AddOutput(int index, Operation* op, llvm::StringRef aggregation,
                 int sort_index) {
    auto it = outputs.find(index);
    if (it == outputs.end()) {
      AggregatedOperand operand;
      operand.aggregation = aggregation;
      it = outputs.insert({index, operand}).first;
    }
    // TODO(renjieliu): check aggregation strategy stays the same.
    // Also needs to make sure if aggregation strategy is "None" we should not
    // have more than one op.
    it->second.ops[sort_index] = op;
  }

  std::vector<Operation*> GetAllInputOps() {
    std::vector<Operation*> all_input_ops;
    for (const auto& kv : inputs) {
      if (kv.second.aggregation == kStrategyFirst) {
        all_input_ops.push_back(kv.second.ops.at(0));
        continue;
      }
      for (const auto& operandKv : kv.second.ops) {
        all_input_ops.push_back(operandKv.second);
      }
    }
    return all_input_ops;
  }

  std::vector<Operation*> GetAllOutputOps() {
    std::vector<Operation*> all_output_ops;
    for (const auto& kv : outputs) {
      for (const auto& operand_kv : kv.second.ops) {
        all_output_ops.push_back(operand_kv.second);
      }
    }
    return all_output_ops;
  }

  std::vector<Operation*> GetAllInUseOutputOps() {
    std::vector<Operation*> all_output_ops;
    for (const auto& kv : outputs) {
      auto& aggregated_operand = kv.second;
      if (aggregated_operand.aggregation != kStrategyStack) {
        continue;
      }
      for (const auto& operand_kv : aggregated_operand.ops) {
        all_output_ops.push_back(operand_kv.second);
      }
    }
    return all_output_ops;
  }

  // This function will process the aggregated inputs based on different
  // strategies like "first", "last", "stack".
  std::map<int, Value*> GetAggregatedInputs(OpBuilder* builder) {
    std::map<int, Value*> aggregated_inputs;
    for (const auto& kv : inputs) {
      Value* op_input = nullptr;
      const AggregatedOperand& operand = kv.second;
      // Dealing with "stack" strategy:
      // This breaks into two parts:
      // 1) If the ops only has one element, we only add a reshape op to expand
      // the dim.
      // 2) If the ops contain more than one element, we need to append a
      // pack_op after the input ops.
      if (operand.aggregation == kStrategyStack) {
        if (operand.ops.size() == 1) {
          // If ops size is 1, it will be simply expanding dimensions at dim 0.
          Operation* current_identity_op = operand.ops.begin()->second;
          Value* input = current_identity_op->getOperand(0);
          RankedTensorType input_type =
              input->getType().cast<RankedTensorType>();
          // The Reshape will be {1, (original_shape)}
          SmallVector<int64_t, 4> reshape_op_shape;
          reshape_op_shape.push_back(1);
          for (const auto& dim : input_type.getShape()) {
            reshape_op_shape.push_back(dim);
          }

          Operation* first_use = current_identity_op->getNextNode();
          builder->setInsertionPoint(first_use);
          Location loc = first_use->getLoc();
          auto shape_type = RankedTensorType::get({input_type.getRank() + 1},
                                                  builder->getIntegerType(32));
          SmallVector<Attribute, 4> result_shape_data(reshape_op_shape.size());
          for (int i = 0; i < reshape_op_shape.size(); ++i) {
            result_shape_data[i] = builder->getI32IntegerAttr(
                static_cast<int32_t>(reshape_op_shape[i]));
          }
          auto shape_attr =
              DenseElementsAttr::get(shape_type, result_shape_data);
          auto shape = builder->create<ConstantOp>(loc, shape_type, shape_attr);
          auto reshape_output_type = RankedTensorType::get(
              reshape_op_shape, input_type.getElementType());
          Operation* reshape = builder->create<TFL::ReshapeOp>(
              first_use->getLoc(), reshape_output_type, input, shape);
          op_input = reshape->getResult(0);

        } else {
          // Insert a pack op to pack all the inputs together.
          std::vector<Value*> pack_input_operands;
          std::vector<Value*> packed_input_consumers;
          for (int i = 0, e = operand.ops.size(); i < e; ++i) {
            pack_input_operands.push_back(operand.ops.at(i)->getOperand(0));
            packed_input_consumers.push_back(operand.ops.at(i)->getResult(0));
          }
          // Find the first op that consumes the last value of the aggregated
          // inputs.
          Operation* first_use = *(packed_input_consumers.back()->user_begin());
          // The pack reshape will be {N, (original_shape)}
          SmallVector<int64_t, 4> pack_shape;
          pack_shape.push_back(pack_input_operands.size());
          RankedTensorType type = operand.ops.at(0)
                                      ->getResult(0)
                                      ->getType()
                                      .cast<RankedTensorType>();
          for (const auto& dim : type.getShape()) {
            pack_shape.push_back(dim);
          }
          auto pack_input_type =
              RankedTensorType::get(pack_shape, type.getElementType());
          builder->setInsertionPoint(first_use);
          Operation* pack_op = builder->create<TFL::PackOp>(
              first_use->getLoc(), pack_input_type, pack_input_operands,
              builder->getI32IntegerAttr(pack_input_operands.size()),
              builder->getI32IntegerAttr(0));
          op_input = pack_op->getResult(0);
        }
      } else if (operand.aggregation == kStrategyLast) {
        // This handle the strategy "last", if simply takes the last input.
        op_input = operand.ops.at(operand.ops.size() - 1)->getOperand(0);
      } else {
        // This handle the strategy "first" and default, if simply takes the
        // first input.
        op_input = operand.ops.at(0)->getOperand(0);
      }
      aggregated_inputs[kv.first] = op_input;
    }
    return aggregated_inputs;
  }

  // For now, we just return the first output's location which the fused op will
  // be inserted in.
  Operation* GetFirstOutputOp() { return outputs.begin()->second.ops.at(0); }

  // Since we have different aggregation strategies, e.g., "first", "last",
  // "stack". We don't somehow aggregated to get the outputs for the funcOp.
  // This function is simply compute the RankedTensorType (shape & element type)
  std::map<int, Type> GetAggregatedOuputTypes(OpBuilder* builder) {
    std::map<int, Type> aggregated_output_types;
    for (const auto& kv : outputs) {
      const AggregatedOperand& operand = kv.second;
      if (operand.aggregation == kStrategyStack) {
        const int output_numer = operand.ops.size();
        Value* first_output = operand.ops.at(0)->getOperand(0);
        RankedTensorType first_output_type =
            first_output->getType().cast<RankedTensorType>();
        // The aggregated output shape will be {N, original_shape}.
        SmallVector<int64_t, 4> shape;
        shape.push_back(output_numer);
        for (const auto& dim : first_output_type.getShape()) {
          shape.push_back(dim);
        }
        aggregated_output_types[kv.first] =
            RankedTensorType::get(shape, first_output_type.getElementType());
      } else if (operand.aggregation == kStrategyLast) {
        Value* last_output =
            operand.ops.at(operand.ops.size() - 1)->getOperand(0);
        aggregated_output_types[kv.first] = last_output->getType();
      } else {
        Value* first_output = operand.ops.at(0)->getOperand(0);
        aggregated_output_types[kv.first] = first_output->getType();
      }
    }
    return aggregated_output_types;
  }

  void AggregateAndRewireOutputs(OpBuilder* builder, Operation* fused_op) {
    // TODO(renjieliu): Consider get rid of the ophinted identity nodes here
    // as well or just rely on the general path to get rid of the identity
    // nodes.
    int output_index = 0;
    for (const auto& kv : outputs) {
      const AggregatedOperand& operand = kv.second;
      // This handles the "stack" strategy. It push a unpack_op before all the
      // outputs and make all the outputs point to the unpack_op.
      if (operand.aggregation == kStrategyStack) {
        // TODO(renjieliu): Revisit here if we need to handle
        // operand.ops().size() == 1 case. Insert a unpack op to unpack the
        // outputs.
        const int output_number = operand.ops.size();
        // Find the first output.
        Operation* first_output = operand.ops.at(0);
        Location insert_loc = first_output->getLoc();
        SmallVector<Type, 4> unpack_output_types(
            output_number, first_output->getOperand(0)->getType());

        builder->setInsertionPoint(first_output);
        Operation* unpack_op = builder->create<TFL::UnpackOp>(
            insert_loc, unpack_output_types, fused_op->getResult(output_index),
            builder->getI32IntegerAttr(output_number),
            builder->getI32IntegerAttr(0));
        // For every unpack output, make sure they point to the right ones.
        for (int i = 0; i < output_number; ++i) {
          Operation* to_be_replaced_op = operand.ops.at(i);
          to_be_replaced_op->replaceUsesOfWith(to_be_replaced_op->getOperand(0),
                                               unpack_op->getResult(i));
        }
      } else if (operand.aggregation == kStrategyLast) {
        // This handles the strategy "last", it simply takes the last output.
        Operation* op = operand.ops.at(operand.ops.size() - 1);
        op->replaceUsesOfWith(op->getOperand(0),
                              fused_op->getResult(output_index));
      } else {
        // This handles the strategy "first" and default, it simply takes the
        // first output.
        Operation* op = operand.ops.at(0);
        op->replaceUsesOfWith(op->getOperand(0),
                              fused_op->getResult(output_index));
      }

      output_index++;
    }
  }

  LogicalResult VerifyOphint() const {
    if (inputs.empty() || outputs.empty()) return failure();
    return success();
  }

  StringRef uuid;
  StringRef function_name;

 private:
  // The AggregatedOperand is used to hold one "aggregated operand".
  // For example, this can be
  // {
  //    aggregation = "stack",
  //    {0: ident_op1, 1: ident_op2, 2: ident_op3}
  // }
  struct AggregatedOperand {
    StringRef aggregation;
    std::map<int, Operation*> ops;
  };

  std::map<int, AggregatedOperand> inputs;
  std::map<int, AggregatedOperand> outputs;
};

// Preprocess the graph for topo sort. (each operation is a node, while
// inputs/outputs indicate edges) Assume the graph is acyclic. The preprocess
// does the following:
//   Compute each operations's in-degress (how many input nodes they're taken)
//   Get all consumer operations for every operations. (operation_to_ouputs)
//   Get the init_queue (those operations will be processed first).
void PreprocessTopoSortGraph(
    Block* block, std::queue<Operation*>* init_queue,
    llvm::DenseMap<Operation*, llvm::DenseSet<Operation*>>* operation_to_ouputs,
    llvm::DenseMap<Operation*, int>* operation_to_in_degrees) {
  for (auto& op : *block) {
    if (&op == block->getTerminator()) continue;
    if (op.getNumOperands() == 0) {
      init_queue->push(&op);
    } else {
      // The operand of the ops is not a direct indication of the "edge" as we
      // can have a pack op after a unpack op (they have multiple edges), we
      // should only count as one.
      llvm::DenseSet<Operation*> input_ops;
      for (int i = 0; i < op.getNumOperands(); ++i) {
        Operation* input_op = op.getOperand(i)->getDefiningOp();
        if (input_op) input_ops.insert(input_op);
      }
      if (input_ops.empty()) {
        init_queue->push(&op);
        continue;
      }
      operation_to_in_degrees->try_emplace(&op, input_ops.size());
      for (auto* input_op : input_ops) {
        auto preceeding_op_it = operation_to_ouputs->find(input_op);
        if (preceeding_op_it == operation_to_ouputs->end()) {
          auto result = operation_to_ouputs->try_emplace(
              input_op, llvm::DenseSet<Operation*>());
          preceeding_op_it = result.first;
        }
        preceeding_op_it->second.insert(&op);
      }
    }
  }
}

bool IsSideEffectOp(Operation* op) {
  if (op->hasNoSideEffect()) return false;

  // Identity op has no side effect.
  // Check the OperationName maybe more elegant here.
  auto tf_identity_op = dyn_cast_or_null<TF::IdentityOp>(op);
  if (tf_identity_op) return false;
  return true;
}

// It's possible other transformations can benefit from this util function, but
// since currently there's none, so we only limit this function to the ophint
// extraction pass. We may refactor this function to extend the usage in future.
//
// Assume the graph is disconnected from outside.
// Also assume the block has no arguments.
LogicalResult TopoSortOperations(OpBuilder* builder) {
  std::queue<Operation*> init_queue;
  llvm::DenseMap<Operation*, llvm::DenseSet<Operation*>> operation_to_ouputs;
  llvm::DenseMap<Operation*, int> operation_to_in_degrees;
  std::vector<Operation*> sorted_ops;

  PreprocessTopoSortGraph(builder->getBlock(), &init_queue,
                          &operation_to_ouputs, &operation_to_in_degrees);
  while (!init_queue.empty()) {
    Operation* current_op = init_queue.front();
    init_queue.pop();
    sorted_ops.push_back(current_op);

    auto current_op_to_output_it = operation_to_ouputs.find(current_op);
    if (current_op_to_output_it == operation_to_ouputs.end()) {
      continue;
    }
    for (Operation* output_op : current_op_to_output_it->second) {
      auto output_op_it = operation_to_in_degrees.find(output_op);
      if (output_op_it == operation_to_in_degrees.end()) return failure();

      output_op_it->second -= 1;
      if (output_op_it->second == 0) {
        init_queue.push(output_op);
        operation_to_in_degrees.erase(output_op_it);
      }
    }
    operation_to_ouputs.erase(current_op_to_output_it);
  }

  // Before we performs the sort. We need to make sure we didn't mess the
  // ordering of original side-effect operations.
  // It's possible those side-effect operations have no topological relations
  // at all!
  std::vector<Operation*> original_side_effect_ops;
  std::vector<Operation*> after_sort_side_effect_ops;
  for (auto& op : *builder->getBlock()) {
    if (IsSideEffectOp(&op) && (&op != builder->getBlock()->getTerminator()))
      original_side_effect_ops.push_back(&op);
  }
  for (auto* op : sorted_ops) {
    if (IsSideEffectOp(op)) after_sort_side_effect_ops.push_back(op);
  }
  if (original_side_effect_ops.size() != after_sort_side_effect_ops.size())
    return failure();
  for (int i = 0; i < original_side_effect_ops.size(); ++i) {
    if (original_side_effect_ops[i] != after_sort_side_effect_ops[i])
      return failure();
  }

  // Performs the sort.
  // Ideally it would be nice to just clear the block then write the sorted ops.
  // But unfortunately that's hard to do.
  for (int i = sorted_ops.size() - 1; i > 0; --i) {
    Operation* current_op = sorted_ops[i];
    for (int j = i - 1; j >= 0; --j) {
      Operation* prev_op = sorted_ops[j];
      prev_op->moveBefore(current_op);
    }
  }

  return success();
}

Operation* BuildFusedFuncOp(StringRef func_name, StringRef fused_func_type,
                            Operation* insert_before_op,
                            const std::map<int, Value*>& inputs,
                            const std::map<int, Type>& output_types,
                            OpBuilder* builder, ModuleOp* module_op) {
  SmallVector<Type, 4> input_types;
  SmallVector<Value*, 4> input_values;
  SmallVector<int, 4> input_indexes;
  for (const auto& kv : inputs) {
    Value* input = kv.second;
    input_types.push_back(input->getType());
    input_values.push_back(input);
    input_indexes.push_back(kv.first);
  }

  SmallVector<Type, 4> func_output_types;
  for (const auto& kv : output_types) {
    func_output_types.push_back(kv.second);
  }

  FunctionType function_type =
      builder->getFunctionType(/*inputs=*/input_types,
                               /*results=*/func_output_types);

  SmallVector<NamedAttribute, 4> attrs;
  attrs.push_back(builder->getNamedAttr(
      kTfLiteFunctionName, builder->getStringAttr(fused_func_type)));
  attrs.push_back(builder->getNamedAttr(
      kTfLiteFunctionInputIndex, builder->getI32ArrayAttr(input_indexes)));
  FuncOp func_op = FuncOp::create(insert_before_op->getLoc(), func_name,
                                  function_type, llvm::makeArrayRef(attrs));
  module_op->push_back(func_op);
  builder->setInsertionPoint(insert_before_op);
  return builder->create<CallOp>(insert_before_op->getLoc(), func_op,
                                 input_values);
}

llvm::StringMap<OphintCompositeOp> FindAllOphintNodes(Block* bb) {
  llvm::StringMap<OphintCompositeOp> ophint_composite_ops;
  for (auto& op : *bb) {
    auto nameAttr = op.getAttrOfType<StringAttr>(kTfLiteFunctionName);
    if (!nameAttr) continue;
    StringRef function_name = nameAttr.getValue();
    auto uuidAttr = op.getAttrOfType<StringAttr>(kTfLiteFunctionUUID);
    if (!uuidAttr) continue;
    StringRef uuid = uuidAttr.getValue();
    auto it = ophint_composite_ops.find(uuid);
    if (it == ophint_composite_ops.end()) {
      OphintCompositeOp ophint_composite_op(uuid, function_name);
      it = ophint_composite_ops.insert({uuid, ophint_composite_op}).first;
    }

    // The default aggregation strategy is "NONE".
    StringRef aggregation = kStrategyNone;
    auto aggregationAttr =
        op.getAttrOfType<StringAttr>(kTfLiteFunctionAggregate);
    if (aggregationAttr != nullptr) aggregation = aggregationAttr.getValue();

    // The default sort index is 0.
    int sortIndex = 0;
    auto sortIndexAttr =
        op.getAttrOfType<IntegerAttr>(kTfLiteFunctionSortIndex);
    if (sortIndexAttr != nullptr) sortIndex = sortIndexAttr.getInt();

    auto inputIndexAttr =
        op.getAttrOfType<IntegerAttr>(kTfLiteFunctionInputIndex);
    if (inputIndexAttr != nullptr) {
      it->second.AddInput(inputIndexAttr.getInt(), &op, aggregation, sortIndex);
    } else {
      auto outputIndexAttr =
          op.getAttrOfType<IntegerAttr>(kTfLiteFunctionOutputIndex);
      it->second.AddOutput(outputIndexAttr.getInt(), &op, aggregation,
                           sortIndex);
    }
  }

  return ophint_composite_ops;
}

llvm::DenseSet<Operation*> BfsForReachableOps(ArrayRef<Operation*> input_ops) {
  llvm::DenseSet<Operation*> reachable_ops;
  std::queue<Operation*> ops_queue;
  for (auto& input_op : input_ops) {
    for (Value* value : input_op->getOperands()) {
      Operation* op = value->getDefiningOp();
      if (op != nullptr) ops_queue.push(op);
    }
  }

  while (!ops_queue.empty()) {
    Operation* current_op = ops_queue.front();
    ops_queue.pop();
    reachable_ops.insert(current_op);
    for (Value* value : current_op->getOperands()) {
      Operation* upstream_op = value->getDefiningOp();
      // Not visited, put it into the queue.
      if (upstream_op != nullptr &&
          !llvm::is_contained(reachable_ops, upstream_op)) {
        ops_queue.emplace(upstream_op);
      }
    }
  }

  return reachable_ops;
}

// Convert ophint to stub will remove all ops within the ophint region and
// place a new fused op right before the first op.
LogicalResult ConvertOphintToStub(StringRef stub_name,
                                  OphintCompositeOp ophint_composite_op,
                                  OpBuilder* builder, ModuleOp* module_op) {
  // Step 1, find all ops reachable by inputs.
  const llvm::DenseSet<Operation*>& reachable_by_inputs =
      BfsForReachableOps(ophint_composite_op.GetAllInputOps());

  // Step 2, find all ops reachable by outputs.
  const llvm::DenseSet<Operation*>& reachable_by_outputs =
      BfsForReachableOps(ophint_composite_op.GetAllOutputOps());

  // Step 3, deal with inputs aggregation strategies.
  const std::map<int, Value*>& aggregated_inputs =
      ophint_composite_op.GetAggregatedInputs(builder);

  // Step 4, get aggregated output types.
  const std::map<int, Type>& aggregated_output_types =
      ophint_composite_op.GetAggregatedOuputTypes(builder);

  // Step 5, create & place the fused op and rewire the inputs.
  // Here we use a funcOp to represent the fused op. This "funcOp" will be
  // coonverted to other ops (like UnidirectionalSequenceRNNOp) in the
  // legalization phase.
  Operation* inserted_before_op = ophint_composite_op.GetFirstOutputOp();
  Operation* fused_op = BuildFusedFuncOp(
      stub_name, ophint_composite_op.function_name, inserted_before_op,
      aggregated_inputs, aggregated_output_types, builder, module_op);

  for (const auto& kv : aggregated_inputs) {
    Operation* op = kv.second->getDefiningOp();
    if (op == nullptr) return failure();
    op->moveBefore(fused_op);
  }

  // Step 6, deal with outputs aggregation strategies and rewire the outputs.
  ophint_composite_op.AggregateAndRewireOutputs(builder, fused_op);

  // Step 7, remove all the removable ops where
  // (reachable_by_outputs - reachable_by_inputs) as removable and the rest
  // ops are not removable.
  // We also need to make sure all the output identity nodes are there.
  llvm::DenseSet<Operation*> ophinted_identity_nodes;
  for (auto* output : ophint_composite_op.GetAllInUseOutputOps()) {
    ophinted_identity_nodes.insert(output);
  }

  auto removeRemovableOps = [&](Operation* op) {
    if (reachable_by_inputs.count(op) == 0 &&
        reachable_by_outputs.count(op) != 0 &&
        ophinted_identity_nodes.count(op) == 0) {
      op->dropAllDefinedValueUses();
      op->dropAllReferences();
      op->erase();
    }
  };

  builder->getBlock()->walk(removeRemovableOps);

  // Step 8: Topo sort to fix any invalid temporary IRs.
  if (failed(TopoSortOperations(builder))) return failure();

  return success();
}

struct ExtractOphintPass : public ModulePass<ExtractOphintPass> {
  void runOnModule() override;
  void Verify();

 private:
  int ophint_composite_ops_count = 0;
};

// TODO(renjieliu): Current ophint extraction does not support inputs/outputs
// cross functions, we need to do that.
void ExtractOphintPass::runOnModule() {
  ModuleOp module = getModule();
  for (auto function : module.getOps<FuncOp>()) {
    // Process block by block.
    for (auto& bb : function.getBody()) {
      // Find ophints.
      const llvm::StringMap<OphintCompositeOp>& ophint_composite_ops =
          FindAllOphintNodes(&bb);
      if (ophint_composite_ops.empty()) continue;

      // Verify: Make sure all ophint_composite_ops are valid.
      for (const auto& kv : ophint_composite_ops) {
        if (failed(kv.getValue().VerifyOphint())) {
          module.emitError()
              << "Found malformed ophint regions: missing inputs or outputs.";
          return signalPassFailure();
        }
      }

      ophint_composite_ops_count = ophint_composite_ops.size();

      // Convert.
      OpBuilder builder(&bb);
      for (const auto& kv : ophint_composite_ops) {
        if (failed(ConvertOphintToStub(kv.getKey(), kv.getValue(), &builder,
                                       &module))) {
          module.emitError()
              << "Convert ophint failed, malformed inputs or outputs.";
          return signalPassFailure();
        }
      }
    }
  }
}

void ExtractOphintPass::Verify() {
  ModuleOp module = getModule();
  int ophint_func_op_count = 0;
  for (FuncOp func : getModule().getOps<FuncOp>()) {
    for (const NamedAttribute attr : func.getAttrs()) {
      if (attr.first == kTfLiteFunctionName) {
        ophint_func_op_count++;
        if (func.getNumArguments() == 0) {
          module.emitError() << "Ophint function has no inputs.";
          return signalPassFailure();
        }
        if (func.getType().getNumResults() == 0) {
          module.emitError() << "Ophint function has no outputs.";
          return signalPassFailure();
        }
      }
    }
  }
  if (ophint_func_op_count != ophint_composite_ops_count) {
    module.emitError()
        << "Ophint converted functions do not match ophint regions founded.";
    return signalPassFailure();
  }
}

}  // namespace

/// Creates an instance of the TensorFlow Lite dialect ExtractOphintPass
/// pass.
std::unique_ptr<OpPassBase<ModuleOp>> CreateExtractOphintPass() {
  return std::make_unique<ExtractOphintPass>();
}

static PassRegistration<ExtractOphintPass> pass(
    "tfl-extract-ophint", "Extract Ophint for TfLite dialect.");

}  // namespace TFL
}  // namespace mlir
