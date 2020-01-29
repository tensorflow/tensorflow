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

#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "mlir/Transforms/RegionUtils.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";
constexpr char kDefaultShardingValue[] = "";
constexpr char kMirroredVariableIndicesAttr[] = "_mirrored_variable_indices";

std::string GetRandomStateVariableName() {
  return absl::StrCat("VariablesFormatState_", tensorflow::random::New64());
}

// A pass that takes advantage of a loop to add ops that allow the execution to
// avoid repeatedly formatting variables back and forth. The desired formatting
// is determined by TPU program compilation, so this pass does not include how
// to reformat the variables, but only inserts general TPUReshardVariablesOps in
// proper places, and TPUReshardVariablesOps interpret the compilation.
//
// The core idea of this optimization is to keep track of the formatting state
// of variables, and when the next desired state does not change, it can avoid
// reformatting. We associate a set of variables on a device with a formatting
// state, and TPUReshardVariablesOps compares the current state with a desired
// state (which can be the compilation result). If they mismatch,
// TPUReshardVariablesOp reformats the variables to the desired state; if they
// match, TPUReshardVariablesOp is a no-op.
//
// A major use of this pass is weight-update sharding in data parallelism, so we
// require there is a tf_device.replicate in the loop.
//
// For example, suppose we have a training loop (for simplicity we write the
// loop body inine):
//
//  %var0 = ...
//  %var1 = ...
//  tf.while (..., %var0, %var1) {
//    tf_device.replicate ([%var0, %var1] as %rvar) {
//      %compile:2 = "tf._TPUCompileMlir"()
//      tf.TPUExecuteAndUpdateVariablesOp(%rvar, compile#1)
//    }
//  }
//
// This pass will transform it into
//
//  %var0 = ...
//  %var1 = ...
//  %state_var0 = ...
//  %state_var1 = ...
//  tf.while (..., %var0, %var1, %state_var0, %state_var1) {
//    tf_device.replicate ([%var0, %var1] as %rvar,
//                         [%state_var0, %state_var1] as %rstate) {
//      %compile:2 = "tf._TPUCompileMlir"()
//      tf.TPUReshardVariablesOp(%rvar, %compile#1, %rstate)
//      tf.TPUExecuteAndUpdateVariablesOp(%rvar, compile#1)
//    }
//  }
//  %default_format = tf.constant()
//  tf_device.replicate ([%var0, %var1] as %rvar,
//                       [%state_var0, %state_var1] as %rstate) {
//    tf.TPUReshardVariablesOp(%rvar, %default_format, %rstate)
//  }
struct TPUVariableRuntimeReformattingPass
    : public ModulePass<TPUVariableRuntimeReformattingPass> {
  void runOnModule() override;
};

// Returns the earlier value of which `v` is an identity.
Value SkipIdentity(Value v, bool allow_other_use) {
  while (auto result = v.dyn_cast<OpResult>()) {
    if (!(allow_other_use || v.hasOneUse())) break;
    auto op = result.getDefiningOp();
    if (!llvm::isa<TF::IdentityOp>(op) && !llvm::isa<TF::IdentityNOp>(op)) {
      break;
    }
    v = op->getOperand(result.getResultNumber());
  }
  return v;
}

// Finds the formattable arguments of `execute` and annotates the metadata of
// `compile` to record these arguments. In addition, it returns a mapping from
// the formattable arguments of `execute` to the corresponding arguments of
// `while_op` (which should be passed through to `execute` via `replicate`). The
// entries in the mapping are sorted in the order of operands of `execute`.
llvm::SmallVector<std::pair<int64_t, llvm::SmallVector<Value, 4>>, 4>
AnnotateCompileOpAndGetExecuteArgToWhileArgsMapping(
    TF::WhileOp while_op, tf_device::ReplicateOp replicate,
    TF::TPUExecuteAndUpdateVariablesOp execute, Operation* compile, FuncOp body,
    FuncOp cond) {
  llvm::SmallVector<std::pair<int64_t, llvm::SmallVector<Value, 4>>, 4> mapping;
  auto mirrored_variable_indices_attr =
      replicate.getAttrOfType<ArrayAttr>(kMirroredVariableIndicesAttr);
  if (!mirrored_variable_indices_attr) return mapping;

  // Finds the mapping from a replicate argument to an execute operand.
  llvm::SmallDenseMap<int64_t, int64_t, 8> replicate_arg_to_execute_arg;
  for (auto index_and_arg : llvm::enumerate(execute.args())) {
    auto arg = SkipIdentity(index_and_arg.value(), /*allow_other_use=*/false);
    if (!arg.hasOneUse() ||
        !getElementTypeOrSelf(arg.getType()).isa<TF::ResourceType>()) {
      continue;
    }
    auto block_arg = arg.dyn_cast<BlockArgument>();
    if (!block_arg || block_arg.getOwner() != &replicate.GetBody()) continue;
    assert(replicate_arg_to_execute_arg.count(block_arg.getArgNumber()) == 0 &&
           "Found duplicate use of a resource in the execute op.");
    replicate_arg_to_execute_arg[block_arg.getArgNumber()] =
        index_and_arg.index();
  }
  if (replicate_arg_to_execute_arg.empty()) return mapping;

  // Parse the original compile metadata.
  auto metadata_str = compile->getAttrOfType<StringAttr>("metadata");
  assert(metadata_str && "Missing compilation metadata");
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  metadata.ParseFromString(std::string(metadata_str.getValue()));
  int64_t num_replicas = replicate.n().getLimitedValue();
  // Find the formattable operands of `execute`, which must be mirrored
  // variables (arguments of `replicate`), and must be pass-throughs from while
  // operands.
  for (const auto& mirrored_index : mirrored_variable_indices_attr) {
    int64_t replicate_arg = mirrored_index.cast<IntegerAttr>().getInt();
    // Check if the mirrored variable is an input to `execute`.
    auto it = replicate_arg_to_execute_arg.find(replicate_arg);
    if (it == replicate_arg_to_execute_arg.end()) continue;
    // Get the data type of the resource.
    auto subtypes = getElementTypeOrSelf(execute.getOperand(it->second))
                        .cast<TF::ResourceType>()
                        .getSubtypes();
    if (subtypes.size() != 1) continue;
    auto data_type = getElementTypeOrSelf(subtypes[0]);
    // The XLA backend does not yet support formatting 64-bit data types.
    if (data_type.getIntOrFloatBitWidth() == 64) continue;

    // We have found a mirrored variable which is an input to the replicated
    // `execute`. Now set the enable_xla_sharding field in the metadata to
    // inform the compile op.
    auto metadata_arg = metadata.mutable_args(it->second);
    metadata_arg->set_enable_xla_sharding(
        ::tensorflow::tpu::TPUCompileMetadataProto_Arg::ALLOWED);

    // Now find if this mirrored variable is a pass-through of while arguments.
    llvm::SmallVector<Value, 4> while_args;
    for (int64_t i = 0; i < num_replicas; ++i) {
      auto replicate_operand =
          SkipIdentity(replicate.getOperand(num_replicas * replicate_arg + i),
                       /*allow_other_use=*/false);
      auto block_arg = replicate_operand.dyn_cast<BlockArgument>();
      // To qualify for a valid pass-through mirrored variable, it must satisfy
      //   1) it is the body's argument;
      //   2) it has no other uses than `replicate`, the skipped identitiy ops,
      //      or the return;
      //   3) the corresponding argument in the cond function has no uses.
      if (!block_arg || block_arg.getOwner() != &body.front() ||
          llvm::any_of(replicate_operand.getUsers(),
                       [&](Operation* user) {
                         return user != body.front().getTerminator() &&
                                !llvm::isa<TF::IdentityOp>(user) &&
                                user != replicate;
                       }) ||
          !cond.getArgument(block_arg.getArgNumber()).use_empty()) {
        while_args.clear();
        break;
      }
      while_args.push_back(while_op.getOperand(block_arg.getArgNumber()));
    }
    if (while_args.empty()) continue;
    mapping.emplace_back(it->second, std::move(while_args));
  }
  // Sort the mapping according to execute operand order.
  llvm::sort(mapping);
  // Populate the `retval_index_for_sharding` field of the argument metadate.
  for (auto entry : llvm::enumerate(execute.device_var_reads_indices())) {
    int64_t arg_index = entry.value().cast<IntegerAttr>().getInt();
    auto arg_metadata = metadata.mutable_args(arg_index);
    if (arg_metadata->enable_xla_sharding() ==
        ::tensorflow::tpu::TPUCompileMetadataProto_Arg::ALLOWED) {
      int64_t ret_index = execute.device_var_updates_indices()
                              .getValue()[entry.index()]
                              .cast<IntegerAttr>()
                              .getInt();
      arg_metadata->set_retval_index_for_sharding(ret_index);
    }
  }
  // Update the metadata of the compile op.
  compile->setAttr("metadata", OpBuilder(compile).getStringAttr(
                                   metadata.SerializeAsString()));
  return mapping;
}

// Adds a new replicated input to the replicate op.
tf_device::ReplicateOp AddInputsToReplicateOp(tf_device::ReplicateOp replicate,
                                              ArrayRef<Value> new_inputs,
                                              ArrayRef<StringRef> devices) {
  int64_t num_replicas = replicate.n().getLimitedValue();
  assert(new_inputs.size() == num_replicas);
  assert(devices.size() == num_replicas);
  llvm::SmallVector<std::pair<llvm::ArrayRef<Value>, Type>, 8>
      new_replicated_inputs;
  llvm::SmallVector<llvm::SmallVector<Value, 8>, 8> replicated_inputs;
  for (auto arg : llvm::enumerate(replicate.GetBody().getArguments())) {
    int64_t i = arg.index();
    replicated_inputs.emplace_back();
    for (int64_t j = i * num_replicas; j < (i + 1) * num_replicas; ++j) {
      replicated_inputs.back().push_back(replicate.getOperand(j));
    }
    new_replicated_inputs.emplace_back(replicated_inputs.back(),
                                       arg.value().getType());
  }
  new_replicated_inputs.emplace_back(new_inputs, new_inputs.front().getType());
  OpBuilder builder(replicate);
  auto new_replicate = builder.create<tf_device::ReplicateOp>(
      replicate.getLoc(), num_replicas, devices, new_replicated_inputs,
      llvm::to_vector<8>(
          replicate.GetBody().getTerminator()->getResultTypes()));
  for (auto arg : replicate.GetBody().getArguments()) {
    arg.replaceAllUsesWith(
        new_replicate.GetBody().getArgument(arg.getArgNumber()));
  }
  for (auto& op : llvm::make_early_inc_range(replicate.GetBody())) {
    op.moveBefore(&new_replicate.GetBody(), new_replicate.GetBody().end());
  }
  replicate.replaceAllUsesWith(new_replicate);
  replicate.erase();
  return new_replicate;
}

// Adds the per-device state variables to the while-loop's inputs/outputs.
TF::WhileOp AddStateVarsToWhileOp(TF::WhileOp while_op, FuncOp body,
                                  FuncOp cond,
                                  ArrayRef<TF::VarHandleOp> state_vars) {
  auto body_return = llvm::cast<ReturnOp>(body.front().back());
  auto new_body_return_vals = llvm::to_vector<4>(body_return.getOperands());
  auto new_while_operands = llvm::to_vector<4>(while_op.getOperands());
  auto append_types = [&](ArrayRef<Type> types) {
    auto new_types = llvm::to_vector<4>(types);
    for (auto state_var : state_vars) {
      new_types.push_back(state_var.resource().getType());
    }
    return new_types;
  };
  for (auto state_var : state_vars) {
    body.front().addArgument(state_var.resource().getType());
    cond.front().addArgument(state_var.resource().getType());
    auto inner_arg = body.getArgument(body.front().getNumArguments() - 1);
    new_body_return_vals.push_back(inner_arg);
    new_while_operands.push_back(state_var.resource());
  }
  OpBuilder builder(&body.front());
  // Update return values.
  builder.create<ReturnOp>(body_return.getLoc(), new_body_return_vals);
  body_return.erase();

  body.setType(FunctionType::get(append_types(body.getType().getInputs()),
                                 append_types(body.getType().getResults()),
                                 body.getContext()));
  cond.setType(FunctionType::get(append_types(cond.getType().getInputs()),
                                 cond.getType().getResults(),
                                 cond.getContext()));
  for (int64_t i = 0; i < state_vars.size(); ++i) {
    int64_t arg_index = body.getNumArguments() - state_vars.size() + i;
    TF::VarHandleOp state_var = state_vars[i];
    auto device_attr = state_var.getAttr(kDeviceAttr);
    if (device_attr) {
      body.setArgAttr(arg_index, kFuncDeviceAttr, device_attr);
      cond.setArgAttr(arg_index, kFuncDeviceAttr, device_attr);
    }
  }
  builder.setInsertionPoint(while_op);
  auto new_while_op = builder.create<TF::WhileOp>(
      while_op.getLoc(),
      append_types(llvm::to_vector<4>(while_op.getResultTypes())),
      new_while_operands, while_op.getAttrs());
  if (new_while_op.output_shapes().size() != 0) {
    auto new_output_shapes = llvm::to_vector<4>(new_while_op.output_shapes());
    // VarHandleOp is a scalar shape resource.
    tensorflow::TensorShapeProto scalar;
    scalar.set_unknown_rank(false);
    for (int64_t i = 0; i < state_vars.size(); ++i) {
      new_output_shapes.push_back(builder.getStringAttr(
          tensorflow::mangling_util::MangleShape(scalar)));
    }
    new_while_op.setAttr("output_shapes",
                         builder.getArrayAttr(new_output_shapes));
  }
  while_op.replaceAllUsesWith(
      new_while_op.getResults().take_front(while_op.getNumResults()));
  while_op.erase();
  return new_while_op;
}

// Creates the per-device variables that represent the formatting state of each
// device.
llvm::SmallVector<TF::VarHandleOp, 4> CreateStateVars(
    ArrayRef<llvm::StringRef> devices, Location loc, RankedTensorType key_type,
    OpBuilder* builder) {
  llvm::SmallVector<TF::VarHandleOp, 4> state_vars;
  // Create the state variable for each device.
  for (llvm::StringRef device : devices) {
    state_vars.push_back(builder->create<TF::VarHandleOp>(
        loc,
        llvm::ArrayRef<Type>{RankedTensorType::get(
            {}, TF::ResourceType::get(llvm::ArrayRef<TensorType>{key_type},
                                      builder->getContext()))},
        llvm::ArrayRef<Value>{},
        llvm::ArrayRef<NamedAttribute>{
            builder->getNamedAttr(kDeviceAttr, builder->getStringAttr(device)),
            builder->getNamedAttr("container", builder->getStringAttr("")),
            builder->getNamedAttr(
                "shared_name",
                builder->getStringAttr(GetRandomStateVariableName()))}));
  }
  return state_vars;
}

// Performs the transformation for a replciate op inside a while loop.
void HandleReplicateOp(TF::WhileOp while_op, tf_device::ReplicateOp replicate,
                       MLIRContext* context) {
  int64_t num_replicas = replicate.n().getLimitedValue();
  if (num_replicas == 1) return;
  TF::TPUExecuteAndUpdateVariablesOp execute;
  for (auto execute_op :
       replicate.GetBody().getOps<TF::TPUExecuteAndUpdateVariablesOp>()) {
    if (execute == nullptr) {
      execute = execute_op;
    } else {
      // We only support one execute op inside replicate.
      execute = nullptr;
      break;
    }
  }
  if (!execute) return;
  auto compile =
      SkipIdentity(execute.key(), /*allow_other_use=*/true).getDefiningOp();
  if (!compile) return;

  auto module = while_op.getParentOfType<ModuleOp>();
  auto body = llvm::cast<FuncOp>(module.lookupSymbol(while_op.body()));
  auto cond = llvm::cast<FuncOp>(module.lookupSymbol(while_op.cond()));

  // Analyze the formattable inputs.
  auto execute_arg_to_outer_args =
      AnnotateCompileOpAndGetExecuteArgToWhileArgsMapping(
          while_op, replicate, execute, compile, body, cond);
  if (execute_arg_to_outer_args.empty()) return;

  // Extract the replicated devices.
  auto devices_attr = replicate.devices();
  if (!devices_attr) return;
  llvm::SmallVector<llvm::StringRef, 4> devices;
  for (auto dev : *devices_attr) {
    devices.push_back(dev.cast<StringAttr>().getValue());
  }
  assert(num_replicas == devices.size());

  OpBuilder builder(replicate);
  builder.setInsertionPoint(while_op);
  // Create per-device variables for formatting state, and add them to the while
  // loop.
  auto key_type =
      RankedTensorType::get({2}, TF::StringType::get(builder.getContext()));
  auto state_vars =
      CreateStateVars(devices, while_op.getLoc(), key_type, &builder);
  while_op = AddStateVarsToWhileOp(while_op, body, cond, state_vars);
  // Add the new while loop inputs to the replicate op inside the body.
  int64_t new_while_operand_count = while_op.getNumOperands();
  llvm::SmallVector<Value, 4> inner_state_vars;
  for (int64_t i = new_while_operand_count - num_replicas;
       i < new_while_operand_count; ++i) {
    inner_state_vars.push_back(body.front().getArgument(i));
  }
  replicate = AddInputsToReplicateOp(replicate, inner_state_vars, devices);

  // Build the reformat according to the compilation. Build it inside
  // `replicate`.
  llvm::SmallVector<Value, 8> reformat_operands;
  for (const auto& entry : execute_arg_to_outer_args) {
    reformat_operands.push_back(execute.args()[entry.first]);
  }
  reformat_operands.push_back(compile->getResult(1));
  reformat_operands.push_back(replicate.GetBody().getArgument(
      replicate.GetBody().getNumArguments() - 1));
  builder.setInsertionPoint(execute);
  builder.create<TF::TPUReshardVariablesOp>(
      execute.getLoc(), llvm::ArrayRef<Type>{}, reformat_operands,
      llvm::ArrayRef<NamedAttribute>{});

  // Build the replicated unformat op after the loop. First prepare building the
  // replicate op.
  llvm::SmallVector<std::pair<llvm::ArrayRef<Value>, Type>, 8>
      unformat_replicate_operands;
  for (const auto& entry : execute_arg_to_outer_args) {
    unformat_replicate_operands.emplace_back(entry.second,
                                             entry.second.front().getType());
  }
  llvm::SmallVector<Value, 4> state_var_vals(state_vars.size());
  for (const auto& entry : llvm::enumerate(state_vars)) {
    state_var_vals[entry.index()] = entry.value().resource();
  }
  unformat_replicate_operands.emplace_back(state_var_vals,
                                           state_var_vals.front().getType());
  // Build a constant default key to specify that the unformatting should
  // transform the variables to the original format.
  builder.setInsertionPointAfter(while_op);
  tensorflow::Tensor default_key_tensor(tensorflow::DT_STRING, {2});
  default_key_tensor.vec<tensorflow::tstring>()(0) = kDefaultShardingValue;
  default_key_tensor.vec<tensorflow::tstring>()(1) = kDefaultShardingValue;
  auto default_state_key = builder.create<TF::ConstOp>(
      while_op.getLoc(),
      tensorflow::ConvertTensor(default_key_tensor, &builder).ValueOrDie());
  // With all replicated inputs, now build the replicate op.
  auto unformat_replicate = builder.create<tf_device::ReplicateOp>(
      while_op.getLoc(), num_replicas, devices, unformat_replicate_operands,
      ArrayRef<Type>{});
  // Then build the unformat op in the replicate op.
  builder.setInsertionPointToEnd(&unformat_replicate.GetBody());
  llvm::SmallVector<Value, 8> unformat_operands;
  for (auto arg : unformat_replicate.GetBody().getArguments()) {
    unformat_operands.push_back(arg);
  }
  // Insert the default key as the second last operand.
  unformat_operands.insert(
      unformat_operands.begin() + unformat_operands.size() - 1,
      default_state_key.getResult());
  // Unformat op.
  builder.create<TF::TPUReshardVariablesOp>(
      while_op.getLoc(), llvm::ArrayRef<Type>{}, unformat_operands,
      llvm::ArrayRef<NamedAttribute>{});
  builder.create<tf_device::ReturnOp>(while_op.getLoc(), ArrayRef<Value>{});
}

void TPUVariableRuntimeReformattingPass::runOnModule() {
  auto module = getModule();
  module.walk([&](TF::WhileOp while_op) {
    auto body = llvm::cast<FuncOp>(module.lookupSymbol(while_op.body()));
    tf_device::ReplicateOp replicate;
    body.walk([&](tf_device::ReplicateOp replicate_op) {
      if (replicate == nullptr) {
        replicate = replicate_op;
        return WalkResult::advance();
      }
      // We do not handle loops with multiple replicate ops.
      replicate = nullptr;
      return WalkResult::interrupt();
    });
    if (replicate) HandleReplicateOp(while_op, replicate, &getContext());
  });
}

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> CreateTPUVariableReformattingPass() {
  return std::make_unique<TPUVariableRuntimeReformattingPass>();
}

static PassRegistration<TPUVariableRuntimeReformattingPass> pass(
    "tf-tpu-variable-runtime-reformatting",
    "Adds device variable formatting op to allow compilation-guided variable "
    "formatting.");

}  // namespace TFTPU
}  // namespace mlir
