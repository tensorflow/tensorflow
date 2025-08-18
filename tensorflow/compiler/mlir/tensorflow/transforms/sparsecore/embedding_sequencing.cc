/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// This pass separates SparseCore, TensorCore, and non-TPU operations into
// separate functions for proper sequencing of TF2 TPU Embedding (see
// tpu_embedding_v3.py). This pass is a precursor for pipelining (see
// embedding_pipelining.cc) and DOES NOT permit parallel execution across SC and
// TC. This pass is a temporary fallback to use while developing full pipelining
// capabilities.
//
// Ops are broken up into:
//   1. SC forward pass
//   2. TC forward/backward pass
//   3. SC backward pass
//   4. non-TPU loop counter updates

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

#define GEN_PASS_DEF_EMBEDDINGSEQUENCINGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/sparsecore/sparsecore_passes.h.inc"

static constexpr char kEmbeddingPipelining[] = "_embedding_pipelining";
static constexpr char kEmbeddingForward[] = "forward";
static constexpr char kEmbeddingBackward[] = "backward";
static constexpr char kEmbeddingForwardSequential[] = "forward_sequential";
static constexpr char kEmbeddingBackwardSequential[] = "backward_sequential";
static constexpr char kDevice[] = "device";
static constexpr llvm::StringRef kTpuCompilationStatus =
    "_tpu_compilation_status";

namespace mlir {
namespace TFDevice {
namespace {

struct EmbeddingSequencingPass
    : public ::impl::EmbeddingSequencingPassBase<EmbeddingSequencingPass> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;
};

template <typename InputContainer>
std::vector<Type> GetValueTypes(const InputContainer& input) {
  // Convert a list of mlir::Value's into a list of mlir::Type's
  std::vector<Type> types;
  types.reserve(input.size());
  for (auto val : input) types.push_back(val.getType());
  return types;
}

bool IsResourceType(Type val_type) {
  if (auto tensor_type = mlir::dyn_cast<mlir::TensorType>(val_type)) {
    if (mlir::isa<TF::ResourceType>(tensor_type.getElementType())) {
      return true;
    }
  }
  return false;
}

bool IsTPUOp(mlir::Operation* op) {
  return op->hasAttr(TF::kReplicationInfoAttr);
}

StringAttr GetReplicationAttr(mlir::Operation* op) {
  return op->getAttrOfType<StringAttr>(TF::kReplicationInfoAttr);
}

StringAttr GetReplicationAttr(TF::TPUCompilationResultOp op) {
  // Special case for getting the replication region for
  // TPUCompilationResultsOp.
  return op->getAttrOfType<StringAttr>(kTpuCompilationStatus);
}

int64_t GetNumOps(func::FuncOp func) {
  int64_t num_ops = 0;
  for (auto it = func.begin(); it != func.end(); ++it) ++num_ops;
  return num_ops;
}

void GatherOpsForExtraction(mlir::SetVector<Operation*>* operations,
                            const mlir::SetVector<Operation*>& ops_to_avoid,
                            bool predecessors, bool successors) {
  // Walk the input and output dependencies of the Ops in `operations` to form
  // the closer of Ops needed to evaluate 'operations'. Input dependencies are
  // walked if 'predecessors' is true and output dependencies are walked if
  // 'successors' is true. In either case, if a discoverd Op is in the
  // 'ops_to_avoid' set, then the dependency walking is terminated.
  llvm::SetVector<Operation*> ops_to_process(*operations);
  llvm::SetVector<Operation*> new_ops;

  while (!ops_to_process.empty()) {
    for (Operation* op : ops_to_process) {
      if (predecessors) {
        for (Value operand : op->getOperands()) {
          // Stop at the block boundary.
          if (mlir::isa<BlockArgument>(operand)) continue;

          Operation* predecessor = operand.getDefiningOp();
          if (!operations->contains(predecessor) &&
              !ops_to_avoid.contains(predecessor)) {
            new_ops.insert(operand.getDefiningOp());
            operations->insert(operand.getDefiningOp());
          }
        }
      }
      if (successors) {
        for (mlir::Operation* successor : op->getUsers()) {
          // Don't include the return op
          if (llvm::isa<func::ReturnOp>(successor)) continue;

          if (!operations->contains(successor) &&
              !ops_to_avoid.contains(successor)) {
            new_ops.insert(successor);
            operations->insert(successor);
          }
        }
      }
    }
    ops_to_process.swap(new_ops);
    new_ops.clear();
  }
}

TF::StatefulPartitionedCallOp MakeFuncCaller(
    mlir::OpBuilder& builder, const Location& loc, func::FuncOp func,
    const llvm::SetVector<Value>& operands) {
  // Constructs a tf.StatefulPartitionedCall to the function provided in 'func'
  // using the operands in 'operands'. Assumes the insertion point on builder is
  // already set.
  auto symbol =
      mlir::SymbolRefAttr::get(builder.getContext(), func.getSymName());
  auto result_types = func.getResultTypes();
  auto caller = TF::StatefulPartitionedCallOp::create(
      builder, loc, result_types, operands.getArrayRef(),
      /*arg_attrs=*/nullptr,
      /*res_attrs=*/nullptr, symbol,
      /*config=*/builder.getStringAttr(""),
      /*config_proto=*/builder.getStringAttr(""),
      /*executor_type=*/builder.getStringAttr(""));
  caller.setFAttr(symbol);
  return caller;
}

func::FuncOp CreateFnWithSignature(ModuleOp module,
                                   const llvm::SetVector<Value>& inputs,
                                   const llvm::SetVector<Value>& outputs,
                                   const std::string& name) {
  // Creates an empty func.FuncOp with a signature compatible with 'inputs'
  // (operands) and 'outputs' (results).
  OpBuilder builder(module);

  std::vector<Type> input_types = GetValueTypes(inputs);
  std::vector<Type> output_types = GetValueTypes(outputs);
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());
  func::FuncOp func_op =
      func::FuncOp::create(builder, module.getLoc(), name,
                           builder.getFunctionType(input_types, output_types));
  func_op.setPrivate();

  return func_op;
}

TF::StatefulPartitionedCallOp EncapsulateOpsInFunc(
    OpBuilder& builder, const llvm::SetVector<Operation*>& ops,
    const llvm::SetVector<Value>& inputs, const llvm::SetVector<Value>& outputs,
    func::FuncOp parent_func, ModuleOp module, const std::string& name) {
  // Moves all of the Operations in 'ops' into a newly created func.FuncOp
  // function named 'name' and replaces the original ops with a call to the
  // newly created function using a tf.StatefulPartitionedCall. Here,
  // 'parent_func' is the function that holds the original set of ops.
  // Note, 'inputs' and 'outputs' are the predetermined set of values that
  // should become the operands and return values, respectively.
  auto insertion_point = builder.saveInsertionPoint();
  func::FuncOp new_func = CreateFnWithSignature(module, inputs, outputs,
                                                absl::StrCat("_func_", name));

  // This preserves the order of the ops that was in the original parent
  // funtion. This is critical for preserving correctness in the presence of
  // resource variables and stateful functions.
  std::vector<Operation*> topological_order;
  for (Operation& op : parent_func.getOps())
    if (ops.contains(&op)) topological_order.push_back(&op);

  // Create the partitioned call
  builder.restoreInsertionPoint(insertion_point);
  auto caller = MakeFuncCaller(builder, module.getLoc(), new_func, inputs);

  Block* block = new_func.addEntryBlock();

  for (Operation* op : topological_order) op->moveBefore(block, block->end());

  // Replace the 'inputs' values with the new function's arguments.
  for (auto p : llvm::zip(inputs, new_func.getArguments()))
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p),
                               new_func.getBody());

  builder.setInsertionPointToEnd(block);
  func::ReturnOp::create(builder, parent_func.getLoc(), outputs.getArrayRef());

  // Replace the original 'outputs' values with the result of the call to the
  // new function.
  for (auto p : llvm::zip(outputs, caller->getResults()))
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p),
                               parent_func.getBody());

  return caller;
}

void UpdateAndInsertTPUOps(TF::StatefulPartitionedCallOp caller,
                           TF::TPUReplicateMetadataOp metadata_op,
                           TF::TPUCompilationResultOp compilation_op,
                           StringAttr old_group) {
  // Adds the TPUReplicateMetatdataOp and TPUCompilationResultOp ops to the
  // function called by the provided 'caller'.
  mlir::CallInterfaceCallable callable = caller.getCallableForCallee();
  mlir::SymbolRefAttr sym = callable.dyn_cast<mlir::SymbolRefAttr>();
  auto func = llvm::dyn_cast<mlir::func::FuncOp>(
      mlir::SymbolTable::lookupNearestSymbolFrom(caller, sym));
  OpBuilder builder(func.getBody());

  StringAttr new_group = builder.getStringAttr(
      absl::StrCat(old_group.getValue().str(), caller.getF().str()));

  builder.insert(metadata_op.clone());
  for (Operation& op : func.getOps()) {
    if (!IsTPUOp(&op)) continue;
    op.setAttr(TF::kReplicationInfoAttr, new_group);
  }
  TF::TPUCompilationResultOp new_result = compilation_op.clone();
  new_result->setAttr(kTpuCompilationStatus, new_group);
  builder.insert(new_result);
}

template <typename OpType>
LogicalResult FindAndExcludeOp(func::FuncOp func,
                               const StringAttr& replication_attr,
                               llvm::SetVector<Operation*>& merged_set,
                               OpType& found_op) {
  // Find the TPUReplicationMetadata or TPUCompilationResult ops which will be
  // cloned/inserted into each region. We add them to the merged_set so that
  // they're ignored when extracting the four main functions.
  found_op = nullptr;
  for (OpType op : func.getOps<OpType>()) {
    if (found_op != nullptr) {
      func.emitOpError() << "number of " << found_op.getOperationName()
                         << " in loop body is not 1";
      return LogicalResult::failure();
    }
    if (GetReplicationAttr(op) != replication_attr) {
      op.emitOpError() << "is not part of the replication region "
                       << replication_attr << " vs " << GetReplicationAttr(op);
      return LogicalResult::failure();
    }
    found_op = op;
    merged_set.insert(found_op);
  }
  return LogicalResult::success();
}

LogicalResult FindOwningWhileOp(func::FuncOp body_func, ModuleOp module,
                                TF::WhileOp* while_op) {
  // Given a while loop body function 'body_func', find the tf.While Op that
  // uses it.
  auto uses_optional = body_func.getSymbolUses(module);
  if (!uses_optional.has_value()) {
    body_func.emitOpError() << "no use of while loop body";
    return LogicalResult::failure();
  }
  *while_op = nullptr;
  for (auto& use : uses_optional.value()) {
    if (llvm::isa<TF::WhileOp>(use.getUser())) {
      if (*while_op != nullptr) {
        use.getUser()->emitOpError() << "multiple users of function.";
        return LogicalResult::failure();
      } else {
        *while_op = llvm::cast<TF::WhileOp>(use.getUser());
      }
    } else {
      use.getUser()->emitOpError() << "non while use of function.";
      return LogicalResult::failure();
    }
  }
  // TODO(bfontain): If the while op is not present we could just split things
  // or we wait until the compiler supports multiple regions?
  if (while_op == nullptr) {
    body_func.emitOpError() << "unable to find while body user.";
    return LogicalResult::failure();
  }
  return LogicalResult::success();
}

LogicalResult FindForwardPassOps(OpBuilder& builder,
                                 llvm::SetVector<Operation*>& forward_pass_ops,
                                 llvm::SetVector<Operation*>& backward_pass_ops,
                                 llvm::SetVector<Operation*>& merged_set,
                                 func::FuncOp loop_body_func,
                                 const int num_replicas) {
  // Find all the ops that are to be included in the 'sc_forward' function which
  // will be executed on the SparseCore. Note, 'forward_pass_ops' is initially
  // seeded with ops from the input MLIR graph that have the
  // _embedding_pipelining="forward" attribute which is set by the TF2 Embedding
  // API.
  //
  // When outputs of the forward pass function are used outside of it, we'll
  // need to insert a TPUReplicatedOutput Op and include that in the
  // forward_pass_ops. And if that usage is also on the TPU (either TensorCore
  // or SparseCore) we'll need to insert a matching TPUReplicatedInput. We do
  // this before the Ops are removed from the original function/graph so that
  // function operands and return values are handled automatically.

  // First, walk the op dependencies.
  GatherOpsForExtraction(&forward_pass_ops, merged_set, /*predecessors=*/true,
                         /*successors=*/false);

  // Locate which variable inputs are part of the forwards pass. These will
  // also be used in the backwards pass. We need to create a 'private' copy
  // of the TpuReplicatedInput for for the fowards pass if there are users
  // outside the pass. Note that in the case of the backwards pass existing
  // this will be the case.
  // This means that when we have put all out sections together some resource
  // inputs will have multiple TPUReplicateInput nodes, so we will need a final
  // pass to merge these together into the earliest copy.
  llvm::SetVector<int64_t> forward_variable_inputs;

  // Validate that the only resource inputs that are read by ops in
  // forward_pass_ops are dataset and variable ops.
  int64_t resource_count = 0;
  for (auto argument : loop_body_func.getArguments()) {
    // Check that all resource arguments are either fed to iterator get next
    // or a TPUReplicatedInput with is_packed.

    if (IsResourceType(argument.getType())) {
      resource_count++;
      bool is_variable = false;
      bool is_non_variable = false;
      bool use_in_forward = false;
      bool use_in_not_forward = false;
      for (auto user : argument.getUsers()) {
        if (llvm::isa<func::ReturnOp>(user)) continue;
        if (!forward_pass_ops.contains(user)) {
          use_in_not_forward = true;
        } else {
          use_in_forward = true;
        }
        if (TF::TPUReplicatedInputOp input =
                llvm::dyn_cast<TF::TPUReplicatedInputOp>(user)) {
          if (!input.getIsPacked()) {
            input.emitOpError() << "unexpected variable input, not packed";
            return LogicalResult::failure();
          }

          if (is_variable) {
            input.emitOpError() << "unexpected multiple TPUReplicatedInputOp "
                                << "for single argument";
            return LogicalResult::failure();
          }
          is_variable = true;
        } else {
          is_non_variable = true;
        }
      }
      if (use_in_forward && use_in_not_forward) {
        loop_body_func.emitOpError()
            << "resource input " << argument.getArgNumber()
            << " is used both in the forwards and "
            << "not forward passes dataset";
        return LogicalResult::failure();
      }
      if (is_non_variable && is_variable) {
        loop_body_func.emitOpError()
            << "resource input " << argument.getArgNumber()
            << " is used both as a varible and not a variable";
        return LogicalResult::failure();
      }
      if (is_variable && use_in_forward)
        forward_variable_inputs.insert(argument.getArgNumber());
    }
  }

  VLOG(3) << "Found " << forward_variable_inputs.size()
          << " variables used in forward pass of " << resource_count
          << " total resource inputs";

  // Clone the TPUReplicatedInputs.
  int64_t cloned_inputs = 0;
  for (int64_t index : forward_variable_inputs) {
    Value argument = loop_body_func.getArgument(index);
    // Uses of this argument should only be the return and the
    // TPUReplicateInputOp. This is checked by the loop above.
    Operation* input_ptr = nullptr;
    for (Operation* user : argument.getUsers()) {
      if (llvm::isa<TF::TPUReplicatedInputOp>(user)) {
        input_ptr = user;
        break;
      }
    }
    TF::TPUReplicatedInputOp input =
        llvm::cast<TF::TPUReplicatedInputOp>(input_ptr);

    // Validate that all users of the TPUReplicatedInput are ReadVariable
    // or AssignVariable ops and check if any are outside the forwards pass.
    bool duplicate_needed = false;
    for (Operation* next_user : input.getOutput().getUsers()) {
      if (!llvm::isa<TF::ReadVariableOp>(next_user) &&
          !llvm::isa<TF::AssignVariableOp>(next_user)) {
        next_user->emitOpError()
            << "unexpected user of output of TPUReplicatedInputOp";
        return LogicalResult::failure();
      }
      if (!forward_pass_ops.contains(next_user)) duplicate_needed = true;
    }
    if (!duplicate_needed) continue;

    cloned_inputs++;
    builder.setInsertionPointAfter(input);
    forward_pass_ops.remove(input);

    TF::TPUReplicatedInputOp private_input = input.clone();
    builder.insert(private_input);
    forward_pass_ops.insert(private_input);
    for (OpOperand& next_use : input.getOutput().getUses()) {
      if (!forward_pass_ops.contains(next_use.getOwner())) continue;
      next_use.getOwner()->setOperand(next_use.getOperandNumber(),
                                      private_input.getOutput());
    }
  }

  VLOG(2) << "Cloned " << cloned_inputs << " TPUReplicatedInputOps";

  // Add TPUReplicatedInput/TPUReplicatedOutput pairs along each edge.
  llvm::SetVector<Operation*> new_forward_ops;
  for (Operation* op : forward_pass_ops) {
    // TODO(bfontain): Should validate that all the TPU ops are in the same
    // replication region.
    if (!IsTPUOp(op)) continue;
    for (Value result : op->getResults()) {
      std::vector<std::pair<Operation*, int64_t>> out_of_region_use;
      for (OpOperand& use : result.getUses()) {
        auto use_owner = use.getOwner();
        // TODO(bfontain): Error check here, if the use.getOwner() is not a TPU
        // then this op must be a TPUReplicatedOutputOp.
        if (IsTPUOp(use_owner) && !forward_pass_ops.contains(use_owner))
          out_of_region_use.push_back(
              std::make_pair(use_owner, use.getOperandNumber()));
      }
      if (out_of_region_use.empty()) continue;
      builder.setInsertionPointAfter(op);
      std::vector<Type> types(num_replicas, result.getType());
      TF::TPUReplicatedOutputOp replicated_output =
          TF::TPUReplicatedOutputOp::create(builder, op->getLoc(),
                                            TypeRange(types), result);
      new_forward_ops.insert(replicated_output);
      // TODO(bfontain): Check for other attributes.
      replicated_output->setAttr(kDevice, builder.getStringAttr(""));
      TF::TPUReplicatedInputOp input = TF::TPUReplicatedInputOp::create(
          builder, op->getLoc(), result.getType(),
          replicated_output.getResults());
      input->setAttr(kDevice, builder.getStringAttr(""));
      mlir::Value new_value = input.getOutput();

      if (mlir::isa<TF::TPUAnnotateTensorsWithDynamicShapeOp>(
              result.getDefiningOp())) {
        TF::TPUAnnotateTensorsWithDynamicShapeOp annotate_op =
            TF::TPUAnnotateTensorsWithDynamicShapeOp::create(
                builder, op->getLoc(), result.getType(), new_value,
                result.getDefiningOp()->getAttrs());
        for (auto [operation, index] : out_of_region_use) {
          if (!backward_pass_ops.contains(operation)) {
            operation->emitOpError()
                << "expect all dynamic inputs consumed by backwards pass.";
            return LogicalResult::failure();
          }
        }

        backward_pass_ops.insert(annotate_op);
        new_value = annotate_op->getResult(0);
      }
      for (auto [operation, index] : out_of_region_use)
        operation->setOperand(index, new_value);
    }
  }

  VLOG(2) << "inserted " << new_forward_ops.size() << " TPU Input/Output ops";
  forward_pass_ops.insert(new_forward_ops.begin(), new_forward_ops.end());
  return LogicalResult::success();
}

LogicalResult FindBackwardPassOps(
    OpBuilder& builder, llvm::SetVector<Operation*>& backward_pass_ops,
    llvm::SetVector<Operation*>& merged_set, const int num_replicas) {
  // Find all the ops that are to be included in the 'sc_backward' function
  // which will be executed on the SparseCore. Note, 'backward_pass_ops' is
  // initially seeded with ops from the input MLIR graph that have the
  // _embedding_pipelining="backward" attribute which is set by the TF2
  // Embedding API.
  //
  // Since we're inserting a replication boundary around the backward pass
  // function, we'll also need to make sure TPUReplicatedInputOp and
  // TPUReplicatedOutputOp ops are inserted as necessary.

  // First, walk the Ops dependencies.
  GatherOpsForExtraction(&backward_pass_ops, merged_set, /*predecessors=*/false,
                         /*successors=*/true);

  VLOG(3) << "found " << backward_pass_ops.size() << " backwards pass ops";

  // If any inputs are to the backward_pass_ops region are direct
  // TPUReplicatedInput ops, then include (if this is the only use) or
  // clone the op. This will be the case for all Read/Assign variable ops.

  llvm::SetVector<TF::TPUReplicatedInputOp> to_clone;
  llvm::SetVector<TF::TPUReplicatedInputOp> to_insert;

  for (Operation* op : backward_pass_ops) {
    for (OpOperand& input_value : op->getOpOperands()) {
      Operation* predecessor_op = input_value.get().getDefiningOp();
      if (TF::TPUReplicatedInputOp input =
              llvm::dyn_cast<TF::TPUReplicatedInputOp>(predecessor_op)) {
        if (to_clone.contains(input) || to_insert.contains(input)) continue;
        // Check if all uses in backwards pass.
        bool all_in_backwards = true;
        for (Operation* user : input->getUsers())
          if (!backward_pass_ops.contains(user)) all_in_backwards = false;
        if (all_in_backwards)
          to_insert.insert(input);
        else
          to_clone.insert(input);
      }
    }
  }
  backward_pass_ops.insert(to_insert.begin(), to_insert.end());
  for (TF::TPUReplicatedInputOp input : to_clone) {
    builder.setInsertionPointAfter(input);
    TF::TPUReplicatedInputOp private_input = input.clone();
    builder.insert(private_input);
    backward_pass_ops.insert(private_input);
    for (OpOperand& next_use : input.getOutput().getUses()) {
      if (!backward_pass_ops.contains(next_use.getOwner())) continue;
      next_use.getOwner()->setOperand(next_use.getOperandNumber(),
                                      private_input.getOutput());
    }
  }

  VLOG(2) << " cloned " << to_clone.size() << " and inserted "
          << to_insert.size() << " TPUReplicatedInput ops";

  // For all other inputs that go from TPU op to TPU op, insert the
  // TPUOutput/Input pair.

  // Add TPUReplicatedInput/TPUReplicatedOutput pairs along each edge.
  // TODO(bfontain): Should be merged with the above loop.
  llvm::SetVector<Value> values_to_add_nodes;

  for (Operation* op : backward_pass_ops) {
    // TODO(bfontain): Should validate that all the TPU ops are in the same
    // replication region.
    // If the op is already a replicated input, no need to to anything.
    if (!IsTPUOp(op) || llvm::isa<TF::TPUReplicatedInputOp>(op)) continue;
    for (OpOperand& input_value : op->getOpOperands())
      // TODO(bfontain): Error check here, this line should never be false,
      // since we skip the TF::TPUReplicatedInputOp case.
      if (IsTPUOp(input_value.get().getDefiningOp()) &&
          !backward_pass_ops.contains(input_value.get().getDefiningOp()))
        values_to_add_nodes.insert(input_value.get());
  }

  for (Value value : values_to_add_nodes) {
    builder.setInsertionPointAfter(value.getDefiningOp());
    std::vector<Type> types(num_replicas, value.getType());
    Location loc = value.getDefiningOp()->getLoc();
    TF::TPUReplicatedOutputOp output = TF::TPUReplicatedOutputOp::create(
        builder, loc, TypeRange(types), value);
    // TODO(bfontain): Check for other attributes.
    output->setAttr(kDevice, builder.getStringAttr(""));
    TF::TPUReplicatedInputOp input = TF::TPUReplicatedInputOp::create(
        builder, loc, value.getType(), output.getResults());
    input->setAttr(kDevice, builder.getStringAttr(""));
    for (OpOperand& use : value.getUses())
      if (backward_pass_ops.contains(use.getOwner()))
        use.getOwner()->setOperand(use.getOperandNumber(), input.getOutput());
    backward_pass_ops.insert(input);
  }

  VLOG(2) << " inserted " << values_to_add_nodes.size()
          << " TPUReplicatedInput/Output pairs";
  return LogicalResult::success();
}

LogicalResult FindCoreTPUOps(
    llvm::SetVector<Operation*>& core_tpu_ops,
    const llvm::SetVector<Operation*>& forward_pass_ops,
    const llvm::SetVector<Operation*>& backward_pass_ops,
    const llvm::SetVector<Operation*>& merged_set,
    func::FuncOp loop_body_func) {
  // Find all of the Ops that are part of the forward/backward pass but aren't
  // targeting the SparseCore. Note that we need to include some non-TPU ops
  // that flow out of the forward pass function. Otherwise, they would get
  // absorbed into the non_tpu function which breaks the pipelining
  // decomposition strategy.
  //
  // Find all the outputs of the forward pass that aren't fed into the backward
  // pass.
  for (Operation* op : forward_pass_ops) {
    for (Value res : op->getResults()) {
      for (auto user : res.getUsers()) {
        if (!forward_pass_ops.contains(user) &&
            !backward_pass_ops.contains(user)) {
          core_tpu_ops.insert(user);
        }
      }
    }
  }

  // Gather all TPU ops marked for compilation in this while loop body that also
  // are not in one of the two other sets.
  for (Operation& op : loop_body_func.getOps()) {
    // Find all TPU ops that don't belong to the forward or backward pass.
    if (merged_set.contains(&op) || llvm::isa<func::ReturnOp>(op) ||
        !IsTPUOp(&op) || op.hasAttr(kEmbeddingPipelining))
      continue;
    // TODO(bfontain): only collect those ops in a fixed TPUReplica.
    core_tpu_ops.insert(&op);
  }

  GatherOpsForExtraction(&core_tpu_ops, merged_set, /*predecessors=*/true,
                         /*successors=*/true);

  // TODO(patn): Verify that all the ops here fall between the forward pass
  // and backward pass ops (i.e., not before the forward pass or after the
  // backward pass).
  return LogicalResult::success();
}

LogicalResult FindNonTPUOps(llvm::SetVector<Operation*>& non_tpu_ops,
                            const llvm::SetVector<Operation*>& merged_set,
                            func::FuncOp loop_body_func) {
  // Find all of the left over Ops after the sc_forward, sc_backward and
  // core_tpu ops have been identified. What's left are just the ops necessary
  // for updating loop counters etc.
  llvm::SetVector<int64_t> non_tpu_args;
  for (Operation& op : loop_body_func.getOps()) {
    if (merged_set.contains(&op) || llvm::isa<func::ReturnOp>(op) ||
        op.hasAttr(kEmbeddingPipelining))
      continue;
    // Note, there should be no TPU ops left at this point. If this trips,
    // there's likely a bug in this pass.
    if (IsTPUOp(&op)) {
      loop_body_func.emitOpError()
          << "Unexpcted TPU op found while identifying non-TPU ops.";
      return LogicalResult::failure();
    }
    non_tpu_ops.insert(&op);
  }

  // Validate that remainder_ops takes and returns a subset of the loop carried
  // args. This will basically be our set increment fn.
  for (Operation* op : non_tpu_ops)
    for (Value input : op->getOperands())
      if (BlockArgument arg = llvm::dyn_cast<BlockArgument>(input))
        // TODO(bfontain): Check that this is actually an argument to the loop
        // body.
        non_tpu_args.insert(arg.getArgNumber());

  // All funcs have a return op so this should be safe.
  func::ReturnOp return_op = *loop_body_func.getOps<func::ReturnOp>().begin();

  for (OpOperand& operand : return_op->getOpOperands()) {
    if (non_tpu_args.contains(operand.getOperandNumber())) {
      if (BlockArgument argument =
              llvm::dyn_cast<BlockArgument>(operand.get())) {
        if (argument.getArgNumber() != operand.getOperandNumber()) {
          return_op.emitOpError()
              << "non TPU ops do not divide state into two pieces.";
          return LogicalResult::failure();
        }
      } else if (!non_tpu_ops.contains(operand.get().getDefiningOp())) {
        return_op.emitOpError()
            << "non TPU ops do not divide state into two pieces.";
        return LogicalResult::failure();
      }
    }
  }
  return LogicalResult::success();
}

LogicalResult ExtractOpsAsFunc(
    OpBuilder& builder, ModuleOp module, llvm::SetVector<Operation*>& ops,
    StringAttr replication_attr, TF::TPUReplicateMetadataOp metadata_op,
    TF::TPUCompilationResultOp compilation_op, func::FuncOp parent_func,
    const std::string& func_name, Operation** caller) {
  // Move the given set of 'ops' into it's own function and replace them with a
  // call to that function ('caller'). if 'metadata_op' and 'compilation_op' are
  // non-null, also insert those (i.e., target the resulting function to the
  // TPU). Here, 'parent_func' is the func.FuncOp that owns the ops in 'ops'.
  //
  // Returns in 'caller' a tf.StatefulPartitionedCallOp that calls the function
  // that was extracted..

  // Find the input edges to form the set of operands to the new function call.
  llvm::SetVector<Value> inputs;
  for (Operation* op : ops) {
    for (Value operand : op->getOperands()) {
      Operation* defining_op = operand.getDefiningOp();
      if (!ops.contains(defining_op)) inputs.insert(operand);
    }
  }
  // Find the output edges to form the set of resutls of the new function call.
  llvm::SetVector<OpResult> results;
  for (Operation* op : ops) {
    for (auto result : op->getResults()) {
      for (const OpOperand& operand : result.getUsers()) {
        if (!ops.contains(operand.getOwner())) {
          results.insert(result);
          break;
        }
      }
    }
  }
  llvm::SetVector<Value> outputs;
  for (auto output : results) outputs.insert(output);
  auto tf_caller = EncapsulateOpsInFunc(builder, ops, inputs, outputs,
                                        parent_func, module, func_name);
  if (!ops.empty() && metadata_op != nullptr && compilation_op != nullptr)
    UpdateAndInsertTPUOps(tf_caller, metadata_op, compilation_op,
                          replication_attr);
  *caller = tf_caller;
  return LogicalResult::success();
}

void EmbeddingSequencingPass::runOnOperation() {
  LOG(INFO) << "EmbeddingSequencingPass::runOnOperation()";
  ModuleOp module = getOperation();

  llvm::SetVector<Operation*> forward_pass_ops;
  llvm::SetVector<Operation*> backward_pass_ops;

  // Find all ops that we know compose the embedding forward and backward pass.
  // These ops are only tagged if one enables the
  // `pipeline_execution_with_tensor_core` flag in the mid-level API.
  WalkResult walk_result = module.walk([&](Operation* op) -> WalkResult {
    if (op->hasAttr(kEmbeddingPipelining)) {
      const std::string region =
          op->getAttrOfType<StringAttr>(kEmbeddingPipelining).getValue().str();
      if (region == kEmbeddingForward ||
          region == kEmbeddingForwardSequential) {
        forward_pass_ops.insert(op);
      } else if (region == kEmbeddingBackward ||
                 region == kEmbeddingBackwardSequential) {
        backward_pass_ops.insert(op);
      } else {
        return op->emitOpError()
               << "embedding op has unknown " << kEmbeddingPipelining
               << " attribute value " << region << ".";
      }
      op->removeAttr(kEmbeddingPipelining);
    }
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) return signalPassFailure();

  // If there are no forward pass ops, there is no SC, so we end early.
  if (forward_pass_ops.empty()) {
    if (backward_pass_ops.empty()) {
      LOG(INFO) << "No unprocessed embedding ops found - skipping embedding "
                << "sequencing rewrite.";
      return;
    } else {
      (*backward_pass_ops.begin())->emitOpError()
          << "embedding backwards pass op with no forwards pass ops.";
      return signalPassFailure();
    }
  }
  LOG(INFO) << "Embedding sequencing rewrite enabled.";

  // Ensure that all ops are in the same region, and have the same replication
  // info.
  // TODO(bfontain): Allow for multiple regions/loops in one module.
  // TODO(patn): move this pass after cluster formation to remove the complexity
  // with replication info and metadata, cluster checking and generalizing to
  // multiple TPU clusters.
  Region* region = (*forward_pass_ops.begin())->getParentRegion();
  StringAttr replication_attr = GetReplicationAttr(*forward_pass_ops.begin());
  llvm::SmallVector<Operation*> checkset(forward_pass_ops.getArrayRef());
  checkset.append(backward_pass_ops.begin(), backward_pass_ops.end());
  for (Operation* op : checkset) {
    if (op->getParentRegion() != region) {
      op->emitOpError() << "embedding ops in two different regions";
      return signalPassFailure();
    }
    if (GetReplicationAttr(op) != replication_attr) {
      op->emitOpError() << "embedding ops with different replication info "
                        << replication_attr << " vs " << GetReplicationAttr(op);
      return signalPassFailure();
    }
  }

  // TODO(bfontain): Check that the region here is the region
  // of the loop body func.
  // Find the FuncOp for the surrounding while loop body.
  func::FuncOp loop_body_func =
      (*forward_pass_ops.begin())->getParentOfType<func::FuncOp>();

  // merged_set will keep track of which ops are to be avoided when gather ops
  // for inclusion into the four extracted functions.
  llvm::SetVector<Operation*> merged_set;

  // Find the TPUReplicationMetadata and TPUCompilationResult ops and delete
  // them. These will be cloned/inserted into each region.
  TF::TPUReplicateMetadataOp metadata_op;
  auto result = FindAndExcludeOp(loop_body_func, replication_attr, merged_set,
                                 metadata_op);
  if (failed(result)) return signalPassFailure();
  const int num_replicas = metadata_op.getNumReplicas();

  TF::TPUCompilationResultOp compilation_op;
  result = FindAndExcludeOp<TF::TPUCompilationResultOp>(
      loop_body_func, replication_attr, merged_set, compilation_op);
  if (failed(result)) return signalPassFailure();

  TF::WhileOp while_op = nullptr;
  result = FindOwningWhileOp(loop_body_func, module, &while_op);
  if (failed(result)) {
    LOG(INFO) << "WhileOp not found: assuming external loop.";
  } else {
    // Override the WhileOp parallel_iterations if requested by flag.
    int parallel_iterations_flag = tensorflow::GetBuildXlaOpsPassFlags()
                                       ->tf_xla_embedding_parallel_iterations;
    if (parallel_iterations_flag > 0) {
      LOG(INFO) << "Setting WhileOp parallel_iterations_flag to "
                << parallel_iterations_flag;
      while_op.setParallelIterations(parallel_iterations_flag);
    } else {
      LOG(INFO) << "Using original WhileOp parallel_iteration";
    }
  }

  OpBuilder builder(module);

  result = FindForwardPassOps(builder, forward_pass_ops, backward_pass_ops,
                              merged_set, loop_body_func, num_replicas);
  if (failed(result)) return signalPassFailure();
  merged_set.insert(forward_pass_ops.begin(), forward_pass_ops.end());

  result =
      FindBackwardPassOps(builder, backward_pass_ops, merged_set, num_replicas);
  if (failed(result)) return signalPassFailure();
  merged_set.insert(backward_pass_ops.begin(), backward_pass_ops.end());

  llvm::SetVector<Operation*> core_tpu_ops;
  result = FindCoreTPUOps(core_tpu_ops, forward_pass_ops, backward_pass_ops,
                          merged_set, loop_body_func);
  if (failed(result)) return signalPassFailure();
  merged_set.insert(core_tpu_ops.begin(), core_tpu_ops.end());

  llvm::SetVector<Operation*> non_tpu_ops;
  result = FindNonTPUOps(non_tpu_ops, merged_set, loop_body_func);
  if (failed(result)) return signalPassFailure();
  merged_set.insert(non_tpu_ops.begin(), non_tpu_ops.end());

  LOG(INFO) << "Forwards pass " << forward_pass_ops.size()
            << " ops, backwards pass " << backward_pass_ops.size()
            << " ops, core " << core_tpu_ops.size()
            << " ops. Total = " << merged_set.size() << " of "
            << GetNumOps(loop_body_func) << ".\n";

  builder.setInsertionPointAfter(*non_tpu_ops.begin());
  Operation* non_tpu_caller = nullptr;
  result =
      ExtractOpsAsFunc(builder, module, non_tpu_ops, replication_attr, nullptr,
                       nullptr, loop_body_func, "non_tpu", &non_tpu_caller);
  if (failed(result)) return signalPassFailure();

  builder.setInsertionPointAfter(non_tpu_caller);
  Operation* forward_caller = nullptr;
  result = ExtractOpsAsFunc(builder, module, forward_pass_ops, replication_attr,
                            metadata_op, compilation_op, loop_body_func,
                            "sc_forward", &forward_caller);
  if (failed(result)) return signalPassFailure();

  // Create tpu_core function
  builder.setInsertionPointAfter(forward_caller);
  Operation* core_tpu_caller = nullptr;
  result = ExtractOpsAsFunc(builder, module, core_tpu_ops, replication_attr,
                            metadata_op, compilation_op, loop_body_func,
                            "core_tpu", &core_tpu_caller);
  if (failed(result)) return signalPassFailure();

  builder.setInsertionPointAfter(core_tpu_caller);
  Operation* backwards_pass_caller = nullptr;
  result = ExtractOpsAsFunc(
      builder, module, backward_pass_ops, replication_attr, metadata_op,
      compilation_op, loop_body_func, "sc_backward", &backwards_pass_caller);
  if (failed(result)) return signalPassFailure();

  metadata_op->erase();
  compilation_op->erase();

  LOG(INFO) << "EmbeddingSequencingPass::runOnOperation done.";
}

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> CreateEmbeddingSequencingPass() {
  return std::make_unique<EmbeddingSequencingPass>();
}

}  // namespace TFDevice
}  // namespace mlir
