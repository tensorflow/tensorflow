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

/******************************************************************************
This pass implements automated pipelining for TPU embeddings defined using
the TF2 Embedding API. This is designed for applications that have an
embedding lookup on the SparseCore, followed by one or more dense layers on
TensorCores, optionally followed by a backward pass (training update) with
more ops on the SparseCore. Ops are broken up into:
  1. SC forward pass
  2. TC forward/backward pass
  3. SC backward pass
  4. non-TPU loop counter updates
These 4 functions are then staggered so as to enable parallel execution.

In pseudocode, the algorithm is as follows:

// Start step 0
C_0 = cond(args_0)
N_0 = non_tpu(args_0)
if (C_0) {
   F_0 = forward(args_0, N_0)
   T_0 = core_tpu(args_0, N_0, F_0)
   // B_0 = backward() is not evaluated here.
}

args_1 = update_args(args_0, N_0, T_0)

// Start step 1
C_1 = cond(args_1)
N_1 = non_tpu(args_1)
if (C_1) {
   F_1 = forward(args_1, N_1)
   // T_1 = core_tpu() is not evaluated here.
   // B_1 = backward() is not evaluated here.
}

// Partial update of args. We expect this to be sufficient
// for evaluating cond().
args_2a = update_args(args_1, N_1)  // NO T_1 here

// Conditional for step 2
C_2 = cond(args_2)

new_while_body (new_args) {  // starts at i==2
   // Finish step i-2
   B_im2 = backward(args_im2, N_im2, F_im2, T_im2)

   // Advance step i-1
   T_im1 = core_tpu(args_im1, N_im1, F_im1)

   // Finish the update of args_2
   args_i = args_2b = update_args(args_2a, T_im1)

   // Start step i
   N_i = non_tpu(args_i)
   F_i = forward(args_i, N_i)

   // Conditional update
   args_ip1 = update_args(args_i, N_i)  // T_i is lagged.
   C_ip1 = cond(args_ip1)

   return (...)
}
// Note: the tf.while conditional is based on Ci which is initially C2. The
// tf.while op returns the inputs unmodified if the initial conditional is
// false. Thus, the following special cases hold for N <= 2:
//                   N==0  | N==1  | N==2 | N==3
//                  -----------------------------
//   C_nm2 == C_0 -> false | true  | true  | true
//   C_nm1 == C_1 -> false | false | true  | true

// Finish step N-2
if (C_nm2) {
   backward(args_nm2, N_nm2, F_nm2, T_nm2)
}

// Finish step N-1
if (C_nm1) {
   T_nm1 = core_tpu(args_nm1, N_nm1, F_nm1)
   backward(args_nm1, N_nm1, F_nm1, T_nm1)
}

// To match the original, un-pipelined while loop, we need to return the
// correct results from the pipelined version. Nominally, we'd like to do
// this:
// if ( NOT(C_nm2) ) {
//   return args_nm2
// } else if (NOT(C_nm1)) {
//   return args_nm1
// } else {
//   return args_n
// }
// but we don't have if/else-if operators. We can convert this to a CaseOp.
// Note, if C_nm1==true and C_nm2 must also be true.
branch_index = int(C_nm2) + int(C_nm1)
selected_results = switch(branch_index) {
  case 0: return args_nm2
  case 1: return args_nm1
  case 2: return args_n
}
return selected_results
******************************************************************************/

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// #include "smartass/brain/ops/flogs_ops.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
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
#include "mlir/Transforms/Inliner.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

#define GEN_PASS_DEF_EMBEDDINGPIPELININGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/sparsecore/sparsecore_passes.h.inc"

static constexpr char kEmbeddingPipelining[] = "_embedding_pipelining";
static constexpr char kEmbeddingPipeliningInlineAttr[] =
    "_embedding_pipelining_inline";
static constexpr char kEmbeddingForward[] = "forward";
static constexpr char kEmbeddingBackward[] = "backward";
static constexpr char kEmbeddingForwardSequential[] = "forward_sequential";
static constexpr char kEmbeddingBackwardSequential[] = "backward_sequential";
static constexpr char kDevice[] = "device";
static constexpr char kLower[] = "_lower_using_switch_merge";
static constexpr llvm::StringRef kTpuCompilationStatus =
    "_tpu_compilation_status";

namespace mlir {
namespace TFDevice {
namespace {

bool IsResourceType(Type val_type) {
  if (auto tensor_type = mlir::dyn_cast<mlir::TensorType>(val_type)) {
    if (mlir::isa<TF::ResourceType>(tensor_type.getElementType())) {
      return true;
    }
  }
  return false;
}

struct EmbeddingPipeliningPass
    : public ::impl::EmbeddingPipeliningPassBase<EmbeddingPipeliningPass> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;
};

bool UseEmbeddingPipelining(ModuleOp& module) {
  // Enable automated pipelining pass unless:
  // 1. The user disables it via flag, or
  // 2. The graph contains TF.Summary ops. Graphs like this typically only run
  //    for a single step which doesn't work in pipelining.

  if (tensorflow::GetBuildXlaOpsPassFlags()
          ->tf_xla_disable_full_embedding_pipelining) {
    LOG(INFO) << "Embedding pipelining disabled via flag.";
    return false;
  }

  if (tensorflow::GetBuildXlaOpsPassFlags()
          ->tf_xla_disable_full_embedding_pipelining_with_summaries) {
    // Detect summaries by looking for key Ops in the graph. It would be better
    // to do this via operator attributes rather than looking for a specific op.
    WalkResult walk_result =
        module.walk([&](TF::WriteSummaryOp op) -> WalkResult {
          return WalkResult::interrupt();
        });
    if (walk_result.wasInterrupted()) {
      LOG(WARNING) << "TF summaries detected - disabling embedding pipelining.";
      return false;
    }
  }
  LOG(INFO) << "Embedding pipelining rewrite enabled.";
  return true;
}

StringAttr GetReplicationAttr(mlir::Operation* op) {
  return op->getAttrOfType<StringAttr>(TF::kReplicationInfoAttr);
}

StringAttr GetReplicationAttr(TF::TPUCompilationResultOp op) {
  // Special case for getting the replication region for
  // TPUCompilationResultsOp.
  return op->getAttrOfType<StringAttr>(kTpuCompilationStatus);
}

// Replaces the replication region attribute if it already exists.
void UpdateReplicationAttr(Operation* op, StringAttr attr) {
  if (op->hasAttr(TF::kReplicationInfoAttr)) {
    op->setAttr(TF::kReplicationInfoAttr, attr);
  }
}

// Replaces the replication region attribute if it already exists.
void UpdateReplicationAttr(TF::TPUCompilationResultOp& op, StringAttr attr) {
  // Special case for getting the replication region for
  // TPUCompilationResultsOp.
  if (op->hasAttr(kTpuCompilationStatus)) {
    op->setAttr(kTpuCompilationStatus, attr);
  }
}

// A helper class to inline TF::StatefulPartitionedCall ops
struct Inliner : public InlinerInterface {
  Inliner(OpBuilder& builder, SymbolTable& symbol_table)
      : InlinerInterface(builder.getContext()),
        builder(builder),
        symbol_table(symbol_table) {}

  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const override {
    return true;
  }
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       IRMapping& valueMapping) const override {
    return true;
  }
  bool isLegalToInline(Operation* op, Region* dest, bool wouldBeCloned,
                       IRMapping& valueMapping) const override {
    return true;
  }

  // Don't recursively analyze operations, because they can all be "inlined".
  bool shouldAnalyzeRecursively(Operation* op) const override { return true; }

  LogicalResult UnifyReplicationInfo(func::FuncOp func) {
    auto new_repl_info =
        builder.getStringAttr(func.getSymName().str() + "_repl_info");
    for (auto& op : func.getRegion().getOps()) {
      if (auto compile_op = llvm::dyn_cast<TF::TPUCompilationResultOp>(op)) {
        UpdateReplicationAttr(compile_op, new_repl_info);
      } else {
        UpdateReplicationAttr(&op, new_repl_info);
      }
    }
    return LogicalResult::success();
  }

  // After inlining, there will likely be some instances where a
  // TPUReplicatedInput feeds directly into a TPUReplicatedOutput. Find such
  // pairs and remove them.
  LogicalResult RemoveOutputInputPairs(func::FuncOp func) {
    llvm::SetVector<Operation*> ops_to_erase;
    // Inlining can result in multiple TPUCompilationResultOp and
    // TPUReplicateMetadataOp ops. Only keep one, the first will do fine.
    TF::TPUCompilationResultOp compile_op = nullptr;
    for (auto op : func.getRegion().getOps<TF::TPUCompilationResultOp>()) {
      if (compile_op == nullptr) {
        compile_op = op;
      } else {
        ops_to_erase.insert(op);
      }
    }
    // If there's no outside compilation, we can exit early because this isn't
    // a TPU function.
    if (compile_op == nullptr) {
      return LogicalResult::success();
    }

    TF::TPUReplicateMetadataOp metadata_op = nullptr;
    for (auto op : func.getRegion().getOps<TF::TPUReplicateMetadataOp>()) {
      if (metadata_op == nullptr)
        metadata_op = op;
      else
        ops_to_erase.insert(op);
    }
    if (metadata_op == nullptr) {
      func->emitError(
          "Expected to find TPUReplicateMetadataOps but found none.");
      return LogicalResult::failure();
    }

    for (auto output_op :
         func.getRegion().getOps<TF::TPUReplicatedOutputOp>()) {
      bool outputs_are_returned = false;
      TF::TPUReplicatedInputOp input_op = nullptr;
      // Only visit each user of the results once.
      llvm::SetVector<Operation*> seen_users;
      for (auto user : output_op->getUsers()) {
        if (!seen_users.insert(user)) continue;
        if (llvm::isa<TF::TPUReplicatedInputOp>(user)) {
          if (input_op != nullptr) {
            func->emitError(
                "Found multiple TPUReplicatedInput ops but only expected 1.");
            return LogicalResult::failure();
          }
          input_op = llvm::dyn_cast<TF::TPUReplicatedInputOp>(user);
        }
        if (llvm::isa<func::ReturnOp>(user)) {
          outputs_are_returned = true;
        }
      }
      if (input_op == nullptr) continue;

      // If we found matching input ops, we can remove the TPUReplicatedInput
      // ops and replace their result values with the inputs to the matching
      // TPUReplicatedOutput op.
      replaceAllUsesInRegionWith(input_op.getResult(), output_op.getOperand(),
                                 func.getRegion());
      ops_to_erase.insert(input_op);

      // If the outputs aren't also returned from this function, then we can
      // remove the TPUReplicatedOutput op as well. In some cases we'll
      // still need these ops.
      if (!outputs_are_returned) ops_to_erase.insert(output_op);
    }
    for (auto op : ops_to_erase) op->erase();

    return LogicalResult::success();
  }

  LogicalResult RemoveDuplicateReplication(func::FuncOp func) {
    llvm::SetVector<Operation*> ops_to_erase;
    llvm::MapVector<BlockArgument, TF::TPUReplicatedInputOp> cache;
    for (auto input_op : func.getRegion().getOps<TF::TPUReplicatedInputOp>()) {
      // We're only expecting a single input argument to be replicated.
      if (input_op->getNumOperands() > 1) continue;
      Value operand = input_op->getOperand(0);
      if (!llvm::isa<BlockArgument>(operand)) continue;
      BlockArgument arg = llvm::dyn_cast<BlockArgument>(operand);

      // See if we've run across this TPUReplicatedInputOp before.
      if (!cache.insert({arg, input_op}).second) {
        // We've seen this before. Replace this instance with the cached op.
        for (auto p :
             llvm::zip(input_op->getResults(), cache[arg]->getResults())) {
          replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p),
                                     func.getRegion());
        }
        ops_to_erase.insert(input_op);
      }
    }
    for (auto op : ops_to_erase) op->erase();
    return LogicalResult::success();
  }

  LogicalResult PatchCollectiveGatherInstanceKey(func::FuncOp func) {
    // We're expecting the original model to have a single CollectiveGatherV2Op
    // that gets split into 3 copies in the start_step_0, start_step_1 and
    // new_while_body functions. We use global iter id to set the instance key.
    // Therefore, we add an offset to the output of the global_iter_id_op to
    // make sure the instance keys are unique among them we replace the original
    // instance key as:
    //   global_iter_id = c + original_global_iter_id
    // where c = -2, -1 or 0 depending on which function it's being replaced in.
    static int64_t offset_value = -2;
    for (auto global_iter_id_op :
         func.getRegion().getOps<TF::GlobalIterIdOp>()) {
      auto loc = global_iter_id_op->getLoc();
      builder.setInsertionPointAfter(global_iter_id_op);
      auto offset = builder.create<TF::ConstOp>(
          loc, builder.getI64IntegerAttr(offset_value));
      auto new_global_iter_id = builder.create<TF::AddV2Op>(
          loc, global_iter_id_op->getResultTypes(),
          global_iter_id_op->getResult(0), offset->getResult(0));
      global_iter_id_op->getResult(0).replaceAllUsesExcept(
          new_global_iter_id->getResult(0), new_global_iter_id);
      std::vector<std::string> attr_names = {
          TF::kReplicationInfoAttr.str(), "_xla_compile_device_type",
          kEmbeddingPipelining, "_xla_outside_compilation", "device"};
      for (const auto& attr_name : attr_names) {
        if (!global_iter_id_op->hasAttr(attr_name)) continue;
        offset->setAttr(attr_name, global_iter_id_op->getAttr(attr_name));
        new_global_iter_id->setAttr(attr_name,
                                    global_iter_id_op->getAttr(attr_name));
      }
      // Make the next function to get inlined use a different offset.
      ++offset_value;
    }
    return LogicalResult::success();
  }

  // Find any StatefulPartitionedCalls and inline their contents in this func.
  LogicalResult InlineCallsInFunc(func::FuncOp func,
                                  bool inline_all_funcs = false) {
    llvm::SetVector<Operation*> ops_to_erase;
    InlinerConfig config;
    for (auto caller :
         func.getRegion().getOps<TF::StatefulPartitionedCallOp>()) {
      if (!inline_all_funcs &&
          !caller->hasAttr(kEmbeddingPipeliningInlineAttr)) {
        continue;
      }
      Operation* symbol = symbol_table.lookup(caller.getF());
      if (symbol == nullptr) {
        func.emitError() << "Symbol not found in SymbolTable: "
                         << caller.getF();
        return LogicalResult::failure();
      }
      if (!llvm::isa<func::FuncOp>(symbol)) {
        func.emitError() << "Invalid callee: " << caller.getF();
        return LogicalResult::failure();
      }
      auto callee =
          llvm::dyn_cast<func::FuncOp>(symbol_table.lookup(caller.getF()));
      auto& src_region = callee.getRegion();
      auto result = inlineCall(*this, config.getCloneCallback(), caller, callee,
                               &src_region, true);
      if (failed(result)) {
        func.emitError("Inliner failed");
        return result;
      }
      ops_to_erase.insert(caller);
    }
    for (auto op : ops_to_erase) op->erase();

    auto result = PatchCollectiveGatherInstanceKey(func);
    if (failed(result)) return result;

    result = UnifyReplicationInfo(func);
    if (failed(result)) return result;

    result = RemoveOutputInputPairs(func);
    if (failed(result)) return result;

    result = RemoveDuplicateReplication(func);
    if (failed(result)) return result;

    return LogicalResult::success();
  }

 private:
  OpBuilder& builder;
  SymbolTable& symbol_table;
};

LogicalResult EliminateResourceLoops(OpBuilder& builder,
                                     SymbolTable& symbol_table,
                                     func::FuncOp func) {
  // Examine all StatefulPartitionedCall ops that have resources as return
  // types. If the returned resource traces back to an input argument for the
  // SPC, then replace uses of the returned copy with the original input.
  //
  // Note: This does not descend through nested SCPs.
  auto ComesFromBlockArgNumber = [](Value val) -> int {
    while (true) {
      if (auto block_arg = llvm::dyn_cast<BlockArgument>(val)) {
        return block_arg.getArgNumber();
      }
      if (auto identity_op =
              llvm::dyn_cast<TF::IdentityOp>(val.getDefiningOp())) {
        val = identity_op.getOperand();
      } else {
        return -1;
      }
    }
  };

  for (auto call_op :
       func.getRegion().getOps<TF::StatefulPartitionedCallOp>()) {
    for (int i = 0; i < call_op->getNumResults(); ++i) {
      if (IsResourceType(call_op->getResult(i).getType())) {
        Operation* symbol = symbol_table.lookup(call_op.getF());
        if (symbol == nullptr) {
          func.emitError() << "Symbol not found in SymbolTable: "
                           << call_op.getF();
          return LogicalResult::failure();
        }
        if (!llvm::isa<func::FuncOp>(symbol)) {
          func.emitError() << "Invalid callee: " << call_op.getF();
          return LogicalResult::failure();
        }
        auto callee =
            llvm::dyn_cast<func::FuncOp>(symbol_table.lookup(call_op.getF()));
        func::ReturnOp return_op = *callee.getOps<func::ReturnOp>().begin();
        auto val = return_op.getOperand(i);
        auto block_arg_number = ComesFromBlockArgNumber(val);
        if (block_arg_number >= 0) {
          replaceAllUsesInRegionWith(call_op->getResult(i),
                                     call_op->getOperand(block_arg_number),
                                     func.getRegion());
        }
      }
    }
  }
  return LogicalResult::success();
}

struct Callers {
  TF::StatefulPartitionedCallOp forward;
  TF::StatefulPartitionedCallOp core_tpu;
  TF::StatefulPartitionedCallOp backward;
  TF::StatefulPartitionedCallOp non_tpu;
};

template <typename InputContainer>
std::vector<Type> GetValueTypes(const InputContainer& input) {
  // Convert a list of mlir::Value's into a list of mlir::Type's
  std::vector<Type> types;
  types.reserve(input.size());
  for (auto val : input) types.push_back(val.getType());
  return types;
}

bool IsTPUOp(mlir::Operation* op) {
  return op->hasAttr(TF::kReplicationInfoAttr);
}

template <typename Vector, typename Container>
void Append(Vector& a, const Container& b) {
  a.insert(a.end(), b.begin(), b.end());
}

template <typename Vector>
void Append(Vector& a, const Vector& b) {
  a.insert(a.end(), b.begin(), b.end());
}

int64_t GetNumOps(func::FuncOp func) {
  int64_t num_ops = 0;
  for (auto it = func.begin(); it != func.end(); ++it) ++num_ops;
  return num_ops;
}

std::vector<Value> ResultsAsVector(Operation* op) {
  std::vector<Value> vec;
  vec.reserve(op->getNumResults());
  for (auto res : op->getResults()) vec.push_back(res);
  return vec;
}

void SetBasicBlockAttributes(OpBuilder& builder, Operation* op) {
  op->setAttr(kDevice, builder.getStringAttr(""));
  op->setAttr(kLower, builder.getBoolAttr(true));
}

std::vector<Value> ResultsAsVector(Operation* op, int begin, int num) {
  int end = begin + num;
  std::vector<Value> vec;
  vec.reserve(end - begin);
  for (int i = begin; i < end; ++i) vec.push_back(op->getResult(i));
  return vec;
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

TF::StatefulPartitionedCallOp MakeFuncCaller(mlir::OpBuilder& builder,
                                             const Location& loc,
                                             func::FuncOp func,
                                             const ArrayRef<Value>& operands,
                                             bool flag_for_inlining) {
  // Constructs a tf.StatefulPartitionedCall to the function provided in 'func'
  // using the operands in 'operands'. Assumes the insertion point on builder is
  // already set.
  auto symbol =
      mlir::SymbolRefAttr::get(builder.getContext(), func.getSymName());
  auto result_types = func.getResultTypes();
  auto caller = builder.create<TF::StatefulPartitionedCallOp>(
      loc, result_types, operands, /*args_attrs=*/nullptr,
      /*res_attrs=*/nullptr, symbol,
      /*config=*/builder.getStringAttr(""),
      /*config_proto=*/builder.getStringAttr(""),
      /*executor_type=*/builder.getStringAttr(""));
  caller.setFAttr(symbol);

  // Set an attribute that our inliner will look for when choosing which
  // TF::StatefulPartitionedCallOps to inline.
  if (flag_for_inlining)
    caller->setAttr(kEmbeddingPipeliningInlineAttr, builder.getBoolAttr(true));
  return caller;
}

func::FuncOp CreateFnWithSignature(ModuleOp module, SymbolTable& symbol_table,
                                   const llvm::SetVector<Value>& inputs,
                                   const llvm::SetVector<Value>& outputs,
                                   const std::string& name) {
  // Creates an empty func.FuncOp with a signature compatible with 'inputs'
  // (operands) and 'outputs' (results).
  OpBuilder builder(module);
  auto in_types = GetValueTypes(inputs);
  auto out_types = GetValueTypes(outputs);
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());
  auto func_op = builder.create<func::FuncOp>(
      module.getLoc(), name, builder.getFunctionType(in_types, out_types));
  func_op.setPrivate();
  symbol_table.insert(func_op);
  return func_op;
}

TF::StatefulPartitionedCallOp EncapsulateOpsInFunc(
    OpBuilder& builder, SymbolTable& symbol_table,
    const llvm::SetVector<Operation*>& ops,
    const llvm::SetVector<Value>& inputs, const llvm::SetVector<Value>& outputs,
    func::FuncOp parent_func, ModuleOp module, const std::string& name,
    bool flag_for_inlining) {
  // Moves all of the Operations in 'ops' into a newly created func.FuncOp
  // function named 'name' and replaces the original ops with a call to the
  // newly created function using a tf.StatefulPartitionedCall. Here,
  // 'parent_func' is the function that holds the original set of ops.
  // Note, 'inputs' and 'outputs' are the predetermined set of values that
  // should become the operands and return values, respectively.
  auto saved_insertion_point = builder.saveInsertionPoint();
  func::FuncOp new_func =
      CreateFnWithSignature(module, symbol_table, inputs, outputs, name);

  // This preserves the order of the ops that was in the original parent
  // function. This is critical for preserving correctness in the presence of
  // resource variables and stateful functions.
  std::vector<Operation*> topological_order;
  for (Operation& op : parent_func.getOps())
    if (ops.contains(&op)) topological_order.push_back(&op);

  // Create the partitioned call
  builder.restoreInsertionPoint(saved_insertion_point);
  auto caller = MakeFuncCaller(builder, module.getLoc(), new_func,
                               inputs.getArrayRef(), flag_for_inlining);

  Block* block = new_func.addEntryBlock();
  for (Operation* op : topological_order) op->moveBefore(block, block->end());

  // Replace the 'inputs' values with the new function's arguments.
  for (auto p : llvm::zip(inputs, new_func.getArguments()))
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p),
                               new_func.getBody());

  builder.setInsertionPointToEnd(block);
  builder.create<func::ReturnOp>(parent_func.getLoc(), outputs.getArrayRef());

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
                                TF::WhileOp& while_op) {
  // Given a while loop body function 'body_func', find the tf.While Op that
  // uses it.
  auto uses_optional = body_func.getSymbolUses(module);
  if (!uses_optional.has_value()) {
    body_func.emitOpError() << "no use of while loop body";
    return LogicalResult::failure();
  }
  while_op = nullptr;
  for (auto& use : uses_optional.value()) {
    if (llvm::isa<TF::WhileOp>(use.getUser())) {
      if (while_op != nullptr) {
        use.getUser()->emitOpError() << "multiple users of function.";
        return LogicalResult::failure();
      } else {
        while_op = llvm::cast<TF::WhileOp>(use.getUser());
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
            << " is used both in the forwards and not forward passes dataset";
        return LogicalResult::failure();
      }
      if (is_non_variable && is_variable) {
        loop_body_func.emitOpError()
            << "resource input " << argument.getArgNumber()
            << " is used both as a variable and not a variable";
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

    input.getOutput().replaceUsesWithIf(
        private_input.getOutput(), [&](OpOperand& use) {
          return forward_pass_ops.contains(use.getOwner());
        });
  }

  VLOG(3) << "Cloned " << cloned_inputs << " TPUReplicatedInputOps";

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
          builder.create<TF::TPUReplicatedOutputOp>(op->getLoc(),
                                                    TypeRange(types), result);
      new_forward_ops.insert(replicated_output);
      // TODO(bfontain): Check for other attributes.
      replicated_output->setAttr(kDevice, builder.getStringAttr(""));
      TF::TPUReplicatedInputOp input = builder.create<TF::TPUReplicatedInputOp>(
          op->getLoc(), result.getType(), replicated_output.getResults());
      input->setAttr(kDevice, builder.getStringAttr(""));
      mlir::Value new_value = input.getOutput();

      if (mlir::isa<TF::TPUAnnotateTensorsWithDynamicShapeOp>(
              result.getDefiningOp())) {
        TF::TPUAnnotateTensorsWithDynamicShapeOp annotate_op =
            builder.create<TF::TPUAnnotateTensorsWithDynamicShapeOp>(
                op->getLoc(), result.getType(), new_value,
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

  VLOG(3) << "Inserted " << new_forward_ops.size() << " TPU Input/Output ops.";
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

  VLOG(3) << "Found " << backward_pass_ops.size() << " backwards pass ops.";

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
    input.getOutput().replaceUsesWithIf(
        private_input.getOutput(), [&](OpOperand& use) {
          return backward_pass_ops.contains(use.getOwner());
        });
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
    TF::TPUReplicatedOutputOp output =
        builder.create<TF::TPUReplicatedOutputOp>(loc, TypeRange(types), value);
    // TODO(bfontain): Check for other attributes.
    output->setAttr(kDevice, builder.getStringAttr(""));
    TF::TPUReplicatedInputOp input = builder.create<TF::TPUReplicatedInputOp>(
        loc, value.getType(), output.getResults());
    input->setAttr(kDevice, builder.getStringAttr(""));
    value.replaceUsesWithIf(input.getOutput(), [&](OpOperand& use) {
      return backward_pass_ops.contains(use.getOwner());
    });
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
    OpBuilder& builder, ModuleOp module, SymbolTable& symbol_table,
    llvm::SetVector<Operation*>& ops, StringAttr replication_attr,
    TF::TPUReplicateMetadataOp metadata_op,
    TF::TPUCompilationResultOp compilation_op, func::FuncOp parent_func,
    const std::string& func_name, TF::StatefulPartitionedCallOp* caller,
    bool flag_for_inlining) {
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
  auto tf_caller =
      EncapsulateOpsInFunc(builder, symbol_table, ops, inputs, outputs,
                           parent_func, module, func_name, flag_for_inlining);
  if (!ops.empty() && metadata_op != nullptr && compilation_op != nullptr)
    UpdateAndInsertTPUOps(tf_caller, metadata_op, compilation_op,
                          replication_attr);
  *caller = tf_caller;
  return LogicalResult::success();
}

LogicalResult FindSourceTPUReplicatedOutput(
    Value val, TF::TPUReplicatedOutputOp& rep_out) {
  Operation* op = val.getDefiningOp();
  if (auto src = llvm::dyn_cast<TF::TPUReplicatedOutputOp>(op)) {
    rep_out = src;
    return LogicalResult::success();
  }
  if (auto src = llvm::dyn_cast<TF::IdentityOp>(op)) {
    return FindSourceTPUReplicatedOutput(src->getOperand(0), rep_out);
  }
  op->emitOpError() << "Value did not come from a TPUReplicatedOutput op: "
                    << val;
  return LogicalResult::failure();
}

int FindReturnIndex(Value val) {
  const int not_found = -1;
  for (auto user : val.getUsers()) {
    if (auto ret_op = llvm::dyn_cast<func::ReturnOp>(user)) {
      for (auto index = 0; index < ret_op->getNumOperands(); ++index) {
        if (val == ret_op->getOperand(index)) {
          return index;
        }
      }
    }
    if (auto ident_op = llvm::dyn_cast<TF::IdentityOp>(user)) {
      auto index = FindReturnIndex(ident_op->getResult(0));
      if (index != not_found) return index;
    }
  }
  return not_found;
}

// Skip the assertions because they currently create problematic dependencies.
constexpr bool kDoAssertions = true;

void AddAssertion(OpBuilder& builder, Location& loc, Value cond,
                  const std::string& message) {
  if (!kDoAssertions) return;
  auto shape_type =
      RankedTensorType::get({1}, builder.getType<TF::StringType>());
  auto msg = builder.create<TF::ConstOp>(
      loc, DenseStringElementsAttr::get(shape_type,
                                        llvm::ArrayRef<StringRef>{message}));
  builder.create<TF::AssertOp>(loc, cond, msg.getResult());
}

LogicalResult StartStep0(OpBuilder& builder, Location& loc,
                         SymbolTable& symbol_table,
                         TF::TPUReplicateMetadataOp& metadata_op,
                         TF::TPUCompilationResultOp& compilation_op,
                         Value& cond_value, Callers& callers,
                         const std::vector<Value>& loop_operands_nm0,
                         TF::StatefulPartitionedCallOp& caller) {
  const std::string name = "start_step_0";

  AddAssertion(builder, loc, cond_value,
               "[StartStep0] Auto-pipelining requires at least two steps.");
  auto insertion_point = builder.saveInsertionPoint();

  func::FuncOp orig_parent_func =
      callers.backward->getParentOfType<func::FuncOp>();

  const std::vector<Value>& operands = loop_operands_nm0;

  // Input types will be the same as the original loop body.
  std::vector<Type> input_types = GetValueTypes(operands);

  // Determine the results types.
  // Return ALL outputs, respecting the provided order of the Operations. This
  // makes it straightforward for users of this function to map the return
  // values.
  llvm::SetVector<Operation*> ops;
  ops.insert(callers.forward);
  ops.insert(callers.core_tpu);
  std::vector<int> result_map;
  result_map.reserve(callers.forward->getNumResults() +
                     callers.core_tpu->getNumResults());
  int result_pos = 0;
  for (auto res : callers.forward->getResults()) {
    bool is_output = false;
    for (auto user : res.getUsers()) {
      if (!ops.contains(user)) {
        is_output = true;
        break;
      }
    }
    result_map.push_back(is_output ? result_pos++ : -1);
  }
  std::vector<Type> result_types;
  Append(result_types, callers.forward->getResultTypes());
  Append(result_types, callers.core_tpu->getResultTypes());

  // Create the function based on input and result types and values.
  auto func_type =
      mlir::FunctionType::get(builder.getContext(), input_types, result_types);
  func::FuncOp then_func = func::FuncOp::create(loc, name, func_type);
  then_func.setPrivate();
  symbol_table.insert(then_func);
  mlir::OpBuilder func_builder =
      mlir::OpBuilder::atBlockBegin(then_func.addEntryBlock());

  // This must match the concatenation order in 'operands' above.
  IRMapping ir_map;
  int pos = 0;
  for (auto orig : orig_parent_func.getArguments())
    ir_map.map(orig, then_func.getArgument(pos++));

  // Clone the specified ops into the new function.
  auto new_forward = func_builder.insert(callers.forward->clone(ir_map));
  for (auto p :
       llvm::zip(callers.core_tpu->getResults(), new_forward->getResults()))
    ir_map.map(std::get<0>(p), std::get<1>(p));
  auto new_core_tpu = func_builder.insert(callers.core_tpu->clone(ir_map));

  // Add the function return;
  std::vector<Value> results;
  Append(results, new_forward->getResults());
  Append(results, new_core_tpu->getResults());
  func_builder.create<func::ReturnOp>(loc, results);

  // Inline any StatefulPartitionCall Ops.
  auto result = Inliner(builder, symbol_table).InlineCallsInFunc(then_func);
  if (failed(result)) return result;

  builder.restoreInsertionPoint(insertion_point);
  caller = MakeFuncCaller(builder, loc, then_func, operands,
                          /*flag_for_inlining=*/false);
  return LogicalResult::success();
}

LogicalResult StartStep1(OpBuilder& builder, Location& loc,
                         SymbolTable& symbol_table,
                         TF::TPUReplicateMetadataOp& metadata_op,
                         TF::TPUCompilationResultOp& compilation_op,
                         Value& cond_value, Callers& callers,
                         const std::vector<Value>& loop_operands_1,
                         TF::StatefulPartitionedCallOp& caller) {
  const std::string name = "start_step_1";

  AddAssertion(builder, loc, cond_value,
               "[StartStep1] Auto-pipelining requires at least two steps.");

  auto insertion_point = builder.saveInsertionPoint();
  func::FuncOp orig_parent_func =
      callers.backward->getParentOfType<func::FuncOp>();

  const std::vector<Value>& operands = loop_operands_1;

  // Input types will be the same as the original loop body.
  std::vector<Type> input_types = GetValueTypes(operands);

  // Determine the results types.
  // Return ALL outputs, respecting the provided order of the Operations. This
  // makes it straightforward for users of this function to map the return
  // values.
  auto result_types = callers.forward->getResultTypes();

  // Create the function based on input and result types and values.
  auto func_type =
      mlir::FunctionType::get(builder.getContext(), input_types, result_types);
  func::FuncOp then_func = func::FuncOp::create(loc, name, func_type);
  then_func.setPrivate();
  symbol_table.insert(then_func);
  mlir::OpBuilder func_builder =
      mlir::OpBuilder::atBlockBegin(then_func.addEntryBlock());

  // This must match the concatenation order in 'operands' above.
  IRMapping ir_map;
  int pos = 0;
  for (auto orig : orig_parent_func.getArguments())
    ir_map.map(orig, then_func.getArgument(pos++));

  // Clone the specified ops into the new function.
  auto new_forward = func_builder.insert(callers.forward->clone(ir_map));

  // Add the function return;
  func_builder.create<func::ReturnOp>(loc, new_forward->getResults());

  // Inline any StatefulPartitionCall Ops.
  auto result = Inliner(builder, symbol_table).InlineCallsInFunc(then_func);
  if (failed(result)) return result;

  builder.restoreInsertionPoint(insertion_point);
  caller = MakeFuncCaller(builder, loc, then_func, operands,
                          /*flag_for_inlining=*/false);
  return LogicalResult::success();
}

LogicalResult FinishStepNm2(OpBuilder& builder, Location& loc,
                            SymbolTable& symbol_table,
                            TF::TPUReplicateMetadataOp& metadata_op,
                            TF::TPUCompilationResultOp& compilation_op,
                            Value& cond_value, Callers& callers,
                            const std::vector<Value>& loop_operands_nm2,
                            const std::vector<Value>& forward_res_nm2,
                            const std::vector<Value>& core_tpu_res_nm2,
                            TF::StatefulPartitionedCallOp& caller) {
  const std::string name = "finish_step_nm2";

  AddAssertion(builder, loc, cond_value,
               "[FinishStepNm2] Auto-pipelining requires at least two steps.");

  auto insertion_point = builder.saveInsertionPoint();
  func::FuncOp orig_parent_func =
      callers.backward->getParentOfType<func::FuncOp>();

  std::vector<Value> operands = loop_operands_nm2;
  Append(operands, forward_res_nm2);
  Append(operands, core_tpu_res_nm2);

  // Input types will be the same as the original loop body.
  std::vector<Type> input_types = GetValueTypes(operands);

  // Determine the results types.
  // Return ALL outputs, respecting the provided order of the Operations. This
  // makes it straightforward for users of this function to map the return
  // values.
  auto result_types = callers.backward->getResultTypes();

  // Create the function based on input and result types and values.
  auto func_type =
      mlir::FunctionType::get(builder.getContext(), input_types, result_types);
  func::FuncOp then_func = func::FuncOp::create(loc, name, func_type);
  then_func.setPrivate();
  symbol_table.insert(then_func);
  mlir::OpBuilder func_builder =
      mlir::OpBuilder::atBlockBegin(then_func.addEntryBlock());

  // This must match the concatenation order in 'operands' above.
  IRMapping ir_map;
  int pos = 0;
  for (auto orig : orig_parent_func.getArguments())
    ir_map.map(orig, then_func.getArgument(pos++));
  for (auto orig : callers.forward->getResults())
    ir_map.map(orig, then_func.getArgument(pos++));
  for (auto orig : callers.core_tpu->getResults())
    ir_map.map(orig, then_func.getArgument(pos++));

  // Clone the specified ops into the new function.
  auto new_backward = func_builder.insert(callers.backward->clone(ir_map));

  // Add the function return;
  func_builder.setInsertionPointAfter(new_backward);
  func_builder.create<func::ReturnOp>(loc, new_backward->getResults());

  // Inline any StatefulPartitionCall Ops.
  auto result = Inliner(builder, symbol_table).InlineCallsInFunc(then_func);
  if (failed(result)) return result;

  builder.restoreInsertionPoint(insertion_point);
  caller = MakeFuncCaller(builder, loc, then_func, operands,
                          /*flag_for_inlining=*/false);
  return LogicalResult::success();
}

LogicalResult FinishStepNm1(OpBuilder& builder, Location& loc,
                            SymbolTable& symbol_table,
                            TF::TPUReplicateMetadataOp& metadata_op,
                            TF::TPUCompilationResultOp& compilation_op,
                            Value& cond_value, Callers& callers,
                            const std::vector<Value>& loop_operands_nm1,
                            const std::vector<Value>& forward_res_nm1,
                            TF::StatefulPartitionedCallOp& caller) {
  const std::string name = "finish_step_nm1";

  AddAssertion(builder, loc, cond_value,
               "Auto-pipelining requires at least two steps.");

  auto insertion_point = builder.saveInsertionPoint();
  func::FuncOp orig_parent_func =
      callers.backward->getParentOfType<func::FuncOp>();

  std::vector<Value> operands = loop_operands_nm1;
  Append(operands, forward_res_nm1);

  // Input types will be the same as the original loop body.
  std::vector<Type> input_types = GetValueTypes(operands);

  // Determine the results types.
  // Return ALL outputs, respecting the provided order of the Operations. This
  // makes it straightforward for users of this function to map the return
  // values.
  std::vector<Type> result_types;
  Append(result_types, callers.core_tpu->getResultTypes());
  Append(result_types, callers.backward->getResultTypes());

  // Create the function based on input and result types and values.
  auto func_type =
      mlir::FunctionType::get(builder.getContext(), input_types, result_types);
  func::FuncOp then_func = func::FuncOp::create(loc, name, func_type);
  then_func.setPrivate();
  symbol_table.insert(then_func);
  mlir::OpBuilder func_builder =
      mlir::OpBuilder::atBlockBegin(then_func.addEntryBlock());

  // This must match the concatenation order in 'operands' above.
  IRMapping ir_map;
  int pos = 0;
  for (auto orig : orig_parent_func.getArguments())
    ir_map.map(orig, then_func.getArgument(pos++));
  for (auto orig : callers.forward->getResults())
    ir_map.map(orig, then_func.getArgument(pos++));

  // Clone the specified ops into the new function.
  auto new_core_tpu = func_builder.insert(callers.core_tpu->clone(ir_map));
  for (auto p :
       llvm::zip(callers.core_tpu->getResults(), new_core_tpu->getResults()))
    ir_map.map(std::get<0>(p), std::get<1>(p));
  auto new_backward = func_builder.insert(callers.backward->clone(ir_map));
  // Add the function return;
  std::vector<Value> results;
  Append(results, new_core_tpu->getResults());
  Append(results, new_backward->getResults());
  func_builder.create<func::ReturnOp>(loc, results);

  // Inline any StatefulPartitionCall Ops.
  auto result = Inliner(builder, symbol_table).InlineCallsInFunc(then_func);
  if (failed(result)) return result;

  builder.restoreInsertionPoint(insertion_point);
  caller = MakeFuncCaller(builder, loc, then_func, operands,
                          /*flag_for_inlining=*/false);
  return LogicalResult::success();
}

LogicalResult MakeForwardOperands(Operation* forward_caller,
                                  Operation* non_tpu_caller,
                                  const std::vector<Value>& loop_operands,
                                  const std::vector<Value>& non_tpu_res,
                                  std::vector<Value>& f_operands) {
  f_operands.clear();
  f_operands.reserve(forward_caller->getNumOperands());
  for (auto operand : forward_caller->getOperands()) {
    if (llvm::isa<BlockArgument>(operand)) {
      // Pull this from the original operands to the original while op.
      auto arg = llvm::cast<BlockArgument>(operand);
      f_operands.push_back(loop_operands[arg.getArgNumber()]);
      continue;
    }
    auto src = operand.getDefiningOp();
    auto res = llvm::cast<OpResult>(operand);
    if (src == non_tpu_caller) {
      f_operands.push_back(non_tpu_res[res.getResultNumber()]);
    } else {
      forward_caller->emitOpError()
          << "Unknown op source for operand " << operand;
      return LogicalResult::failure();
    }
  }
  return LogicalResult::success();
}

LogicalResult MakeCoreTPUOperands(Operation* core_tpu_caller,
                                  Operation* non_tpu_caller,
                                  Operation* forward_caller,
                                  const std::vector<Value>& loop_operands,
                                  const std::vector<Value>& non_tpu_res,
                                  const std::vector<Value>& forward_res,
                                  std::vector<Value>& t_operands) {
  t_operands.clear();
  t_operands.reserve(core_tpu_caller->getNumOperands());
  for (auto operand : core_tpu_caller->getOperands()) {
    if (llvm::isa<BlockArgument>(operand)) {
      // Pull this from the original operands to the original while op.
      auto arg = llvm::cast<BlockArgument>(operand);
      t_operands.push_back(loop_operands[arg.getArgNumber()]);
      continue;
    }
    auto src = operand.getDefiningOp();
    auto res = llvm::cast<OpResult>(operand);
    if (src == non_tpu_caller) {
      t_operands.push_back(non_tpu_res[res.getResultNumber()]);
    } else if (src == forward_caller) {
      t_operands.push_back(forward_res[res.getResultNumber()]);
    } else {
      core_tpu_caller->emitOpError() << "Unknown op source for operand "
                                     << operand << ": " << src->getName();
      return LogicalResult::failure();
    }
  }
  return LogicalResult::success();
}

LogicalResult MakeBackwardOperands(Operation* forward_caller,
                                   Operation* core_tpu_caller,
                                   Operation* backward_caller,
                                   const std::vector<Value>& loop_operands,
                                   const std::vector<Value>& forward_res,
                                   const std::vector<Value>& core_tpu_res,
                                   std::vector<Value>& b_operands) {
  b_operands.clear();
  b_operands.reserve(backward_caller->getNumOperands());
  for (auto operand : backward_caller->getOperands()) {
    if (llvm::isa<BlockArgument>(operand)) {
      // Pull this from the original operands to the original while op.
      auto arg = llvm::cast<BlockArgument>(operand);
      b_operands.push_back(loop_operands[arg.getArgNumber()]);
      continue;
    }
    auto src = operand.getDefiningOp();
    auto res = llvm::cast<OpResult>(operand);
    if (src == forward_caller) {
      b_operands.push_back(forward_res[res.getResultNumber()]);
    } else if (src == core_tpu_caller) {
      b_operands.push_back(core_tpu_res[res.getResultNumber()]);
    } else {
      // Note: we're expecting no edges from non_tpu() to backward().
      backward_caller->emitOpError() << "Unknown op source for operand "
                                     << operand << ": " << src->getName();
      return LogicalResult::failure();
    }
  }
  return LogicalResult::success();
}

LogicalResult MakeNonTPUOperands(Operation* non_tpu_caller,
                                 const std::vector<Value>& loop_operands,
                                 std::vector<Value>& n_operands) {
  n_operands.clear();
  n_operands.reserve(non_tpu_caller->getNumOperands());
  for (auto operand : non_tpu_caller->getOperands()) {
    if (llvm::isa<BlockArgument>(operand)) {
      auto arg = llvm::cast<BlockArgument>(operand);
      n_operands.push_back(loop_operands[arg.getArgNumber()]);
      continue;
    }
    // This shouldn't happen:
    auto src = operand.getDefiningOp();
    non_tpu_caller->emitOpError() << "Unknown op source for operand " << operand
                                  << ": " << src->getName();
    return LogicalResult::failure();
  }
  return LogicalResult::success();
}

Operation* LiftNonTpuFuncCaller(mlir::OpBuilder& builder,
                                Operation* orig_non_tpu_caller,
                                const std::vector<Value>& operands) {
  // Use this to clone an op and lift it outside its parent function. The
  // original while body is unchanged. Example:
  // Original:
  //    %x = tf.while(%a, %b)
  //    ...
  //    while_body:
  //       call(f=@sc_fw, %arg0, %arg1)
  // Lifted:
  //    call(f=@sc_fw, %a, %b)
  //    %x = tf.while(%a, %b)
  //    ...
  func::FuncOp orig_parent_func =
      orig_non_tpu_caller->getParentOfType<func::FuncOp>();
  IRMapping ir_map;
  ir_map.map(orig_parent_func.getArguments(), operands);
  Operation* new_caller = builder.clone(*orig_non_tpu_caller, ir_map);
  return new_caller;
}

void EmbeddingPipeliningPass::runOnOperation() {
  LOG(INFO) << "EmbeddingPipeliningPass::runOnOperation()";
  ModuleOp module = getOperation();

  // We only use one of the EmbeddingPipelining and EmbeddingSequencing passes.
  if (!UseEmbeddingPipelining(module)) return;

  SymbolTable symbol_table(module);

  llvm::SetVector<Operation*> forward_pass_ops;
  llvm::SetVector<Operation*> backward_pass_ops;

  // Find all ops that we know compose the embedding forward and backward pass.
  // These ops are only tagged if one enables the
  // `pipeline_execution_with_tensor_core` flag in the mid-level API.
  bool sequencing_requested = false;
  WalkResult walk_result = module.walk([&](Operation* op) -> WalkResult {
    if (op->hasAttr(kEmbeddingPipelining)) {
      const std::string region =
          op->getAttrOfType<StringAttr>(kEmbeddingPipelining).getValue().str();
      if (region == kEmbeddingForward) {
        forward_pass_ops.insert(op);
        op->removeAttr(kEmbeddingPipelining);
      } else if (region == kEmbeddingBackward) {
        backward_pass_ops.insert(op);
        op->removeAttr(kEmbeddingPipelining);
      } else if (region == kEmbeddingForwardSequential ||
                 region == kEmbeddingBackwardSequential) {
        sequencing_requested = true;
      } else {
        return op->emitOpError()
               << "embedding op has unknown " << kEmbeddingPipelining
               << " attribute value " << region << ".";
      }
    }
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) return signalPassFailure();

  if (sequencing_requested) {
    LOG(INFO) << "EmbeddingSequencingPass requested, skipping "
                 "EmbeddingPipeliningPass";
    return;
  }

  // If there are no forward pass ops, there is no SC, so we end early.
  if (forward_pass_ops.empty()) {
    if (backward_pass_ops.empty()) {
      LOG(INFO) << "no pipelining ops found";
      return;
    } else {
      (*backward_pass_ops.begin())->emitOpError()
          << "embedding backwards pass op with no forwards pass ops.";
      return signalPassFailure();
    }
  }

  // Ensure that all ops are in the same region, and have the same replication
  // info.
  // TODO(bfontain): Allow for multiple regions/loops in one module.
  // TODO(patn): move this pass after cluster formation to remove the
  // complexity with replication info and metadata, cluster checking and
  // generalizing to multiple TPU clusters.
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
  auto loop_body_func =
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

  TF::WhileOp orig_while_op = nullptr;
  result = FindOwningWhileOp(loop_body_func, module, orig_while_op);
  if (failed(result)) return signalPassFailure();
  Location loc = orig_while_op->getLoc();

  OpBuilder builder(module);

  // A special fix for models that pass resources into helper functions and
  // return the same resource (after passing it through multiple identity ops).
  // Some subsequent ops use the original resource and others use the returned
  // version. Pipelining splits these uses across loop iterations resulting in
  // terrible things.
  result = EliminateResourceLoops(builder, symbol_table, loop_body_func);
  if (failed(result)) return signalPassFailure();

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
            << GetNumOps(loop_body_func);

  builder.setInsertionPointAfter(*non_tpu_ops.begin());
  TF::StatefulPartitionedCallOp non_tpu_caller = nullptr;
  result =
      ExtractOpsAsFunc(builder, module, symbol_table, non_tpu_ops,
                       replication_attr, nullptr, nullptr, loop_body_func,
                       "non_tpu", &non_tpu_caller, /*flag_for_inlining=*/false);
  if (failed(result)) return signalPassFailure();

  builder.setInsertionPointAfter(non_tpu_caller);
  TF::StatefulPartitionedCallOp forward_caller = nullptr;
  result = ExtractOpsAsFunc(builder, module, symbol_table, forward_pass_ops,
                            replication_attr, metadata_op, compilation_op,
                            loop_body_func, "sc_forward", &forward_caller,
                            /*flag_for_inlining=*/true);
  if (failed(result)) return signalPassFailure();

  // Create tpu_core function
  builder.setInsertionPointAfter(forward_caller);
  TF::StatefulPartitionedCallOp core_tpu_caller = nullptr;
  result = ExtractOpsAsFunc(builder, module, symbol_table, core_tpu_ops,
                            replication_attr, metadata_op, compilation_op,
                            loop_body_func, "core_tpu", &core_tpu_caller,
                            /*flag_for_inlining=*/true);
  if (failed(result)) return signalPassFailure();

  builder.setInsertionPointAfter(core_tpu_caller);
  TF::StatefulPartitionedCallOp backward_caller = nullptr;
  result = ExtractOpsAsFunc(builder, module, symbol_table, backward_pass_ops,
                            replication_attr, metadata_op, compilation_op,
                            loop_body_func, "sc_backward", &backward_caller,
                            /*flag_for_inlining=*/true);
  if (failed(result)) return signalPassFailure();

  Callers orig_callers;
  orig_callers.forward = forward_caller;
  orig_callers.backward = backward_caller;
  orig_callers.core_tpu = core_tpu_caller;
  orig_callers.non_tpu = non_tpu_caller;

  // The output of the original while op also serves as subsequent input to
  // the same function so input_signature == output_signature. Figure out the
  // mapping from the result of each of the four functions into the result
  // vector.
  auto orig_return_op = *loop_body_func.getOps<func::ReturnOp>().begin();
  std::map<int, int> loop_arg_update_map_non_tpu;
  std::map<int, int> loop_arg_update_map_core_tpu;
  for (int ret_pos = 0; ret_pos < orig_return_op->getNumOperands(); ++ret_pos) {
    auto operand = orig_return_op->getOperand(ret_pos);
    auto def_op = operand.getDefiningOp();
    auto result = mlir::dyn_cast<OpResult>(operand);
    if (def_op == non_tpu_caller) {
      loop_arg_update_map_non_tpu[result.getResultNumber()] = ret_pos;
    } else if (def_op == core_tpu_caller) {
      loop_arg_update_map_core_tpu[result.getResultNumber()] = ret_pos;
    } else if (def_op == forward_caller) {
      loop_body_func->emitOpError(
          "Unexpected loop carried variable dependency on sc_forward");
      return signalPassFailure();
    } else if (def_op == backward_caller) {
      loop_body_func->emitOpError(
          "Unexpected loop carried variable dependency on sc_");
      return signalPassFailure();
    } else if (llvm::isa<BlockArgument>(operand)) {
      // pass
    } else {
      // This should never happen.
      loop_body_func->emitOpError("Couldn't find mapping for return value ");
      return signalPassFailure();
    }
  }

  const int num_f_res = forward_caller->getNumResults();
  const int num_t_res = core_tpu_caller->getNumResults();

  // At this point, we have separated the main while body ops into four
  // functions:
  //   1. SC forward pass ("forward_ops")
  //   2. TC forward/backward pass ("core_tput_ops")
  //   3. SC backward pass ("backward_ops")
  //   4. Loop counter updates ("non_tpu_ops")
  //
  // Next, extract the original conditional function which we'll use to
  // kick off the pre-loop pipelining steps.
  // are just the operands passed to the original WhileOp.
  func::FuncOp orig_cond_func = orig_while_op.cond_function();

  std::vector<Value> loop_operands_0;
  const int num_orig_loop_operands = orig_while_op->getNumOperands();
  loop_operands_0.reserve(num_orig_loop_operands);
  Append(loop_operands_0, orig_while_op->getOperands());

  // Evaluate the real conditional function before the new while loop.
  builder.setInsertionPoint(orig_while_op);
  Operation* cond_caller_0 =
      MakeFuncCaller(builder, orig_while_op->getLoc(), orig_cond_func,
                     loop_operands_0, /*flag_for_inlining=*/false);
  Value C_0 = cond_caller_0->getResults().front();

  // Call the non_tpu function to update the loop counters. This is still
  // part of the i=0 loop iteration.
  builder.setInsertionPointAfter(cond_caller_0);
  Operation* non_tpu_caller_0 =
      LiftNonTpuFuncCaller(builder, non_tpu_caller, loop_operands_0);
  // Save the results for later reference.
  auto non_tpu_res_0 = ResultsAsVector(non_tpu_caller_0);

  // Start step 0.
  // Now make the sc_fw + tc_fb call in the pre-loop. We assume (and assert)
  // that we'll execute at least two steps.
  builder.setInsertionPointAfter(non_tpu_caller_0);
  TF::StatefulPartitionedCallOp start_step_0;
  result = StartStep0(builder, loc, symbol_table, metadata_op, compilation_op,
                      C_0, orig_callers, loop_operands_0, start_step_0);
  if (failed(result)) return signalPassFailure();

  // Save the results of the forward_0 and core_tpu_0 calls by slicing them
  // out of the results.
  auto forward_res_0 = ResultsAsVector(start_step_0, 0, num_f_res);
  auto core_tpu_res_0 = ResultsAsVector(start_step_0, num_f_res, num_t_res);

  // Update the loop operands with results of non_tpu() and core_tpu().
  std::vector<Value> loop_operands_1 = loop_operands_0;
  for (auto p : loop_arg_update_map_non_tpu)
    loop_operands_1[p.second] = non_tpu_res_0[p.first];
  for (auto p : loop_arg_update_map_core_tpu)
    loop_operands_1[p.second] = core_tpu_res_0[p.first];

  // The second conditional evaluation.
  builder.setInsertionPointAfter(start_step_0);
  Operation* cond_caller_1 =
      MakeFuncCaller(builder, orig_while_op->getLoc(), orig_cond_func,
                     loop_operands_1, /*flag_for_inlining=*/false);
  Value C_1 = cond_caller_1->getResults().front();

  builder.setInsertionPointAfter(cond_caller_1);
  Operation* non_tpu_caller_1 =
      LiftNonTpuFuncCaller(builder, non_tpu_caller, loop_operands_1);
  auto non_tpu_res_1 = ResultsAsVector(non_tpu_caller_1);

  // Start step 1. Again, assume.
  builder.setInsertionPointAfter(non_tpu_caller_1);
  TF::StatefulPartitionedCallOp start_step_1;
  result = StartStep1(builder, loc, symbol_table, metadata_op, compilation_op,
                      C_1, orig_callers, loop_operands_1, start_step_1);
  if (failed(result)) return signalPassFailure();

  // Save the results of the forward_1 call.
  auto forward_res_1 = ResultsAsVector(start_step_1);

  // Update the loop operands with any outputs from the non_tpu and core_tpu
  // functions. Note, core_tpu isn't called again until the middle of the loop
  // body. So, loop_operands_2 is only partially updated here. We'll finish
  // updating this after core_tpu() is called in the new while body.
  std::vector<Value> loop_operands_2 = loop_operands_1;
  for (auto p : loop_arg_update_map_non_tpu)
    loop_operands_2[p.second] = non_tpu_res_1[p.first];

  // The second conditional evaluation. The assumption here is that the
  // partially updated loop_operands_2 is sufficient for correct evaluation of
  // the cond() function.
  builder.setInsertionPointAfter(start_step_1);
  Operation* cond_caller_2 =
      MakeFuncCaller(builder, orig_while_op->getLoc(), orig_cond_func,
                     loop_operands_2, /*flag_for_inlining=*/false);
  Value C_2 = cond_caller_2->getResults().front();

  // The new while body:
  //
  // First, we need to construct the body and conditional functions. To do so,
  // we need to create the initial operand list that we'll need. This will
  // determine the type signature for the body and cond functions.
  std::vector<Value> tmp_while_operands;
  Append(tmp_while_operands, loop_operands_0);
  Append(tmp_while_operands, loop_operands_1);
  Append(tmp_while_operands, loop_operands_2);
  Append(tmp_while_operands, forward_res_0);
  Append(tmp_while_operands, forward_res_1);
  Append(tmp_while_operands, core_tpu_res_0);
  Append(tmp_while_operands, non_tpu_res_1);
  Append(tmp_while_operands, {C_0, C_1, C_2});

  // Dedupe the operands. We'll need a map to help translate.
  llvm::SetVector<Value> new_while_operands;
  llvm::MapVector<Value, int> loop_var_map;
  for (auto operand : tmp_while_operands) {
    if (new_while_operands.insert(operand)) {
      // First time seeing this operand. Let's record the final resting place
      // in the new_while_operands vector.
      loop_var_map[operand] = new_while_operands.size() - 1;
    }
  }
  // Save index mappings for canonical vectors.
  auto BuildUnpackIndexes =
      [&loop_var_map](std::vector<Value>& prototype_vals) {
        std::vector<int> indexes;
        indexes.reserve(prototype_vals.size());
        for (auto prototype_val : prototype_vals)
          indexes.push_back(loop_var_map[prototype_val]);
        return indexes;
      };
  auto loop_operands_indexes_im2 = BuildUnpackIndexes(loop_operands_0);
  auto loop_operands_indexes_im1 = BuildUnpackIndexes(loop_operands_1);
  auto loop_operands_indexes_i = BuildUnpackIndexes(loop_operands_2);
  auto forward_res_indexes_im2 = BuildUnpackIndexes(forward_res_0);
  auto forward_res_indexes_im1 = BuildUnpackIndexes(forward_res_1);
  auto core_tpu_res_indexes_im2 = BuildUnpackIndexes(core_tpu_res_0);
  auto non_tpu_res_indexes_im1 = BuildUnpackIndexes(non_tpu_res_1);
  int C_index_im2 = loop_var_map[C_0];
  int C_index_im1 = loop_var_map[C_1];
  int C_index_i = loop_var_map[C_2];

  // Get the operand types.
  std::vector<Type> new_while_operand_types = GetValueTypes(new_while_operands);

  // Make cond and body functions for the new while op.
  // Create the function based on input and result types and values.
  // Note, for a while loop body function, the operand types and result types
  // are identical.
  auto body_func_type = mlir::FunctionType::get(
      &getContext(), new_while_operand_types, new_while_operand_types);
  auto cond_func_type = mlir::FunctionType::get(
      &getContext(), new_while_operand_types, orig_cond_func.getResultTypes());
  func::FuncOp cond =
      func::FuncOp::create(loc, "new_while_cond", cond_func_type);
  func::FuncOp body =
      func::FuncOp::create(loc, "new_while_body", body_func_type);
  cond.setPrivate();
  body.setPrivate();
  symbol_table.insert(cond);
  symbol_table.insert(body);
  OpBuilder cond_builder = OpBuilder::atBlockBegin(cond.addEntryBlock());
  OpBuilder body_builder = OpBuilder::atBlockBegin(body.addEntryBlock());

  //****************************************************************************
  // Build the internals of the new tf.While op's conditional function.
  //****************************************************************************
  // Build the cond function body. All we need is a ReturnOp that returns C_i
  // which is the last argument.
  cond_builder.create<func::ReturnOp>(loc, cond.getArgument(C_index_i));

  //****************************************************************************
  // Build the internals of the new tf.While op's body function.
  //****************************************************************************
  auto body_args = body.getArguments();
  // First, let's unpack all the body arguments.
  auto UnpackArgs = [&body_args](std::vector<int>& indexes) {
    // This helper makes it easy to unpack "natural" vectors of values while
    // still respecting the impact of deduping.
    std::vector<Value> slice;
    int num = indexes.size();
    slice.reserve(num);
    for (auto i : indexes) slice.push_back(body_args[i]);
    return slice;
  };
  auto loop_operands_im2 = UnpackArgs(loop_operands_indexes_im2);
  auto loop_operands_im1 = UnpackArgs(loop_operands_indexes_im1);
  auto loop_operands_i = UnpackArgs(loop_operands_indexes_i);
  auto forward_res_im2 = UnpackArgs(forward_res_indexes_im2);
  auto forward_res_im1 = UnpackArgs(forward_res_indexes_im1);
  auto core_tpu_res_im2 = UnpackArgs(core_tpu_res_indexes_im2);
  auto non_tpu_res_im1 = UnpackArgs(non_tpu_res_indexes_im1);
  auto C_im1 = body_args[C_index_im1];
  auto C_i = body_args[C_index_i];

  // Now, construct the operand least for each op by unpacking values.

  //
  // Finish step i-2
  //
  // First, add all the inputs to sc_backward(). These all come from the block
  // arguments, sc_forward() and core_tpu() and need to be pulled from the
  // "i-2" (or "0") version of the inputs.
  std::vector<Value> b_operands;
  result = MakeBackwardOperands(forward_caller, core_tpu_caller,
                                backward_caller, loop_operands_im2,
                                forward_res_im2, core_tpu_res_im2, b_operands);
  if (failed(result)) return signalPassFailure();
  auto backward_caller_im2 = body_builder.clone(*backward_caller);
  backward_caller_im2->setOperands(b_operands);

  //
  // Finish step i-1
  //
  // Second, add all the inputs to core_tpu(). Thesse all come from the while
  // loop opernads, sc_forward() or non_tpu() and need to be pulled from the
  // "i-1" (or "1") version of the inputs.
  std::vector<Value> t_operands;
  result = MakeCoreTPUOperands(core_tpu_caller, non_tpu_caller, forward_caller,
                               loop_operands_im1, non_tpu_res_im1,
                               forward_res_im1, t_operands);
  if (failed(result)) return signalPassFailure();
  auto core_tpu_caller_im1 = body_builder.clone(*core_tpu_caller);
  core_tpu_caller_im1->setOperands(t_operands);
  auto core_tpu_res_im1 = ResultsAsVector(core_tpu_caller_im1);

  // Update the loop operands with results of core_tpu().
  for (auto p : loop_arg_update_map_core_tpu)
    loop_operands_i[p.second] = core_tpu_res_im1[p.first];

  //
  // Start step i
  //
  // Third, add all the inputs to non_tpu(). These all come from the while
  // loop operands and need to be pulled from the "i" (or "2") version of the
  // inputs.
  std::vector<Value> n_operands;
  result = MakeNonTPUOperands(non_tpu_caller, loop_operands_i, n_operands);
  if (failed(result)) return signalPassFailure();
  auto non_tpu_caller_i = body_builder.clone(*non_tpu_caller);
  non_tpu_caller_i->setOperands(n_operands);
  auto non_tpu_res_i = ResultsAsVector(non_tpu_caller_i);

  // Fourth, add all the inputs to sc_forward(). These all come from the
  // while loop operands or the non_tpu() call that's in the loop body. The
  // loop operands need to be pulled from the "i" (or "2") version of the
  // inputs. The inputs coming from non_tpu() are from the same loop iteration
  // (non_tpu_res_i).
  std::vector<Value> f_operands;
  result = MakeForwardOperands(forward_caller, non_tpu_caller, loop_operands_i,
                               non_tpu_res_i, f_operands);
  if (failed(result)) return signalPassFailure();
  auto forward_caller_i = body_builder.clone(*forward_caller);
  forward_caller_i->setOperands(f_operands);
  auto forward_res_i = ResultsAsVector(forward_caller_i);

  // Update the loop operands with results of non_tpu(). Results for
  // core_tpu() are lagged.
  std::vector<Value> loop_operands_ip1 = loop_operands_i;
  for (auto p : loop_arg_update_map_non_tpu)
    loop_operands_ip1[p.second] = non_tpu_res_i[p.first];

  // Add the conditional evaluation for the next loop iteration.
  Operation* cond_caller_ip1 =
      MakeFuncCaller(body_builder, orig_while_op->getLoc(), orig_cond_func,
                     loop_operands_ip1, /*flag_for_inlining=*/false);
  Value C_ip1 = cond_caller_ip1->getResults().front();

  // Build the ReturnOp. This mirrors the construction of the operands with
  // 'i' values incremented.
  std::vector<Value> tmp_body_results;
  Append(tmp_body_results, loop_operands_im1);
  Append(tmp_body_results, loop_operands_i);
  Append(tmp_body_results, loop_operands_ip1);
  Append(tmp_body_results, forward_res_im1);
  Append(tmp_body_results, forward_res_i);
  Append(tmp_body_results, core_tpu_res_im1);
  Append(tmp_body_results, non_tpu_res_i);
  Append(tmp_body_results, {C_im1, C_i, C_ip1});

  llvm::SetVector<Value> new_body_results;
  // This should pack the same as deduping code above.
  new_body_results.insert(tmp_body_results.begin(), tmp_body_results.end());
  auto new_body_return_types = GetValueTypes(new_body_results);

  body_builder.setInsertionPointAfter(cond_caller_ip1);
  body_builder.create<func::ReturnOp>(orig_while_op->getLoc(),
                                      new_body_results.getArrayRef());

  // Finally, create the new tf.WhileOp.
  builder.setInsertionPoint(orig_while_op);
  // Use the same parallel_iterations as the original WhileOp unless there's a
  // flag override.
  int parallel_iterations_flag = tensorflow::GetBuildXlaOpsPassFlags()
                                     ->tf_xla_embedding_parallel_iterations;
  int parallel_iterations = parallel_iterations_flag > 0
                                ? parallel_iterations_flag
                                : orig_while_op.getParallelIterations();
  LOG(INFO) << "Setting parallel_iterations_flag to "
            << parallel_iterations_flag;
  auto new_while_op = builder.create<TF::WhileOp>(
      orig_while_op->getLoc(), new_body_return_types,
      new_while_operands.getArrayRef(), cond.getSymName(), body.getSymName(),
      /*parallel_iterations=*/parallel_iterations,
      /*is_stateless=*/false,
      /*shape_invariant=*/false);
  SetBasicBlockAttributes(builder, new_while_op);

  // First, let's unpack all the body arguments.
  auto UnpackResults = [&new_while_op](std::vector<int>& indexes) {
    int num = indexes.size();
    std::vector<Value> slice;
    slice.reserve(num);
    for (auto i : indexes) slice.push_back(new_while_op->getResult(i));
    return slice;
  };
  auto loop_operands_nm2 = UnpackResults(loop_operands_indexes_im2);
  auto loop_operands_nm1 = UnpackResults(loop_operands_indexes_im1);
  auto loop_operands_n = UnpackResults(loop_operands_indexes_i);
  auto forward_res_nm2 = UnpackResults(forward_res_indexes_im2);
  auto forward_res_nm1 = UnpackResults(forward_res_indexes_im1);
  auto core_tpu_res_nm2 = UnpackResults(core_tpu_res_indexes_im2);
  auto non_tpu_res_nm1 = UnpackResults(non_tpu_res_indexes_im1);
  auto C_nm2 = new_while_op->getResult(C_index_im2);
  auto C_nm1 = new_while_op->getResult(C_index_im1);

  // Finish step n-2.
  builder.setInsertionPointAfter(new_while_op);
  TF::StatefulPartitionedCallOp finish_step_nm2;
  result = FinishStepNm2(builder, loc, symbol_table, metadata_op,
                         compilation_op, C_nm2, orig_callers, loop_operands_nm2,
                         forward_res_nm2, core_tpu_res_nm2, finish_step_nm2);
  if (failed(result)) return signalPassFailure();

  // Finish step n-1.
  builder.setInsertionPointAfter(finish_step_nm2);
  TF::StatefulPartitionedCallOp finish_step_nm1;
  result = FinishStepNm1(builder, loc, symbol_table, metadata_op,
                         compilation_op, C_nm1, orig_callers, loop_operands_nm1,
                         forward_res_nm1, finish_step_nm1);
  if (failed(result)) return signalPassFailure();

  // Save the results of the core_tpu_0 call and use it to finalize the
  // loop_operands_n array.
  auto core_tpu_res_nm1 = ResultsAsVector(finish_step_nm1, 0, num_t_res);
  for (auto p : loop_arg_update_map_core_tpu)
    loop_operands_n[p.second] = core_tpu_res_nm1[p.first];

  // Replace the return values from the original WhileOp with the output of
  // the pipelining.
  for (auto p : llvm::zip(orig_while_op->getResults(), loop_operands_n))
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p),
                               *orig_while_op->getParentRegion());

  // Inline the new while body.
  result = Inliner(builder, symbol_table).InlineCallsInFunc(body, false);
  if (failed(result)) return signalPassFailure();

  // Erase original while op and temporary functions. Note, we use the non_tpu
  // function in the output graph.
  symbol_table.lookup(orig_callers.forward.getF())->erase();
  symbol_table.lookup(orig_callers.core_tpu.getF())->erase();
  symbol_table.lookup(orig_callers.backward.getF())->erase();
  orig_while_op.body_function().erase();
  orig_while_op.erase();

  LOG(INFO) << "EmbeddingPipeliningPass::runOnOperation done.";
}
}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> CreateEmbeddingPipeliningPass() {
  return std::make_unique<EmbeddingPipeliningPass>();
}

}  // namespace TFDevice
}  // namespace mlir
