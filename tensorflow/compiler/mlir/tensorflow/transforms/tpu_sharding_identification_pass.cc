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

#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kShardingAttr[] = "xla_hlo.sharding";

struct TPUShardingIdentificationPass
    : public PassWrapper<TPUShardingIdentificationPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Sets `sharding_op` if `op` is XlaShardingOp or if XlaSharding op is adjacent
// to `op`. XlaSharding op may be direct user of inputs but it may also be
// followed by an Identity op and, in the case where bfloat16 type is used, Cast
// op may be added right after the input. As so, parse the users of the
// operation to access connected XlaSharding op.
//
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect sharded
// inputs.
void GetAdjacentXlaShardingOp(Operation* op,
                              llvm::Optional<TF::XlaShardingOp>* sharding_op) {
  // TODO(hongjunchoi): Detect the case when sharding configuration is ambiguous
  // for a single input (i.e. multiple different XlaSharding ops with different
  // configuration policies are connected).
  if (sharding_op->hasValue()) return;

  if (auto sharding = llvm::dyn_cast<TF::XlaShardingOp>(op)) {
    sharding_op->emplace(sharding);
    return;
  }

  if (llvm::isa<TF::IdentityOp, TF::CastOp>(op)) {
    for (auto user : op->getUsers())
      GetAdjacentXlaShardingOp(user, sharding_op);
  }
}

// Parses XlaSharding op connected to input args. If Input to
// tf_device.ClusterFunc op is of resource type, then XlaSharding op will be
// connected to following ReadVariable op.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside a Call op or
// If/While op.
llvm::Optional<llvm::StringRef> ParseInputSharding(const Value& arg) {
  llvm::Optional<TF::XlaShardingOp> parsed_sharding_op;
  for (auto user : arg.getUsers()) {
    if (parsed_sharding_op) continue;

    GetAdjacentXlaShardingOp(user, &parsed_sharding_op);
    if (parsed_sharding_op) continue;

    if (llvm::isa<TF::ReadVariableOp>(user))
      for (auto read_variable_user : user->getUsers())
        GetAdjacentXlaShardingOp(read_variable_user, &parsed_sharding_op);
  }

  if (!parsed_sharding_op) return llvm::Optional<llvm::StringRef>();
  return parsed_sharding_op.getValue()._XlaSharding();
}

// Returns the provided sharding configuration if operand of return value of
// tf_device.ClusterFunc op is directly from XlaSharding op,
llvm::Optional<StringRef> ParseReturnValueSharding(FuncOp func,
                                                   const int output_index,
                                                   const OpOperand& operand) {
  if (auto sharding_op = llvm::dyn_cast_or_null<TF::XlaShardingOp>(
          operand.get().getDefiningOp()))
    return sharding_op._XlaSharding();

  return llvm::Optional<StringRef>();
}

// Includes information on Func op and argument index of the input value. This
// is used to trace Value that is fed into function call ops.
struct FunctionAndArgumentInfo {
  FuncOp func;
  int argument_index;
};

// Adds tf.PartitionedCall op or tf.StatefulPartitionedCall op to `list`. If
// `op` is a function call op, then find the func op from provided `module` and
// add the func op with `arg_index` to `list`. `list` will later be used to
// trace mlir::Value that is fed into (potentially nested) function call ops.
void AddFunctionalOpsToList(
    const int arg_index, ModuleOp module, Operation* op,
    llvm::SmallVectorImpl<FunctionAndArgumentInfo>* list) {
  if (auto pcall_op = llvm::dyn_cast<TF::PartitionedCallOp>(op)) {
    if (!pcall_op.f().isa<FlatSymbolRefAttr>()) return;

    auto pcall_func = llvm::cast<FuncOp>(
        module.lookupSymbol(pcall_op.f().getRootReference()));
    assert(pcall_func);
    list->emplace_back(FunctionAndArgumentInfo{pcall_func, arg_index});

  } else if (auto spcall_op =
                 llvm::dyn_cast<TF::StatefulPartitionedCallOp>(op)) {
    auto sp_call_func = llvm::cast<FuncOp>(module.lookupSymbol(spcall_op.f()));
    assert(sp_call_func);
    list->emplace_back(FunctionAndArgumentInfo{sp_call_func, arg_index});
  }
}

// Walks the MLIR graph from `arg` and return a list of all function call ops to
// which the `arg` op is directly connected.
//
// For example:
//   argument0 -> PartitionedCallOp -> StatefulPartitionedCallOp -> AddOp
//
// For above case, PartitionedCall op and StatefulPartitionedCallOp will be
// returned.
llvm::SmallVector<FunctionAndArgumentInfo, 4> ExtractFunctionsConnectedToArg(
    BlockArgument arg, ModuleOp module) {
  llvm::SmallVector<FunctionAndArgumentInfo, 4> functions_connected_to_arg;
  for (auto& arg_use : arg.getUses())
    AddFunctionalOpsToList(arg_use.getOperandNumber(), module,
                           arg_use.getOwner(), &functions_connected_to_arg);

  llvm::SmallVector<FunctionAndArgumentInfo, 4> functions_to_parse{
      functions_connected_to_arg.begin(), functions_connected_to_arg.end()};

  while (!functions_to_parse.empty()) {
    llvm::SmallVector<FunctionAndArgumentInfo, 4> newly_discovered_functions;
    for (auto function_info : functions_to_parse) {
      Block& func_entry_block = function_info.func.front();
      auto argument =
          func_entry_block.getArgument(function_info.argument_index);

      for (auto& arg_use : argument.getUses())
        AddFunctionalOpsToList(arg_use.getOperandNumber(), module,
                               arg_use.getOwner(), &newly_discovered_functions);
    }

    functions_connected_to_arg.append(newly_discovered_functions.begin(),
                                      newly_discovered_functions.end());
    std::swap(functions_to_parse, newly_discovered_functions);
  }

  return functions_connected_to_arg;
}

// Walks the graph from the arguments of the `cluster_func_op` and extracts
// sharding configurations for all inputs by parsing XlaSharding op connected to
// the arguments. If argument to the `cluster_func_op` directly feeds into
// another function call op, then recursively walk the function definition to
// find the connected XlaSharding op.
void IdentifyXlaShardingForComputationInputs(
    StringRef logical_core_0_sharding, tf_device::ClusterFuncOp cluster_func_op,
    FuncOp cluster_function, Builder* builder) {
  // Look up function definition from module.
  Block& cluster_function_block = cluster_function.front();
  ModuleOp module = cluster_func_op.getParentOfType<ModuleOp>();

  llvm::SmallVector<llvm::StringRef, 8> sharding_for_args(
      cluster_function_block.getNumArguments(), logical_core_0_sharding);

  // Iterate through input arguments to the entry block of
  // tf_device.ClusterFunc. For input ops, look for following XlaSharding ops.
  // XlaSharding ops can:
  //   1) Directly follow the input argument if input argument has non-resource
  //      types.
  //   2) Follow ReadVariableOp if the input type is of resource type.
  //   3) Follow IdentityOp or CastOp after above cases (1), (2).
  //
  // Sharding configurations are added to the tf_device.ClusterFunc as an
  // attribute and the function as an argument attribute.
  for (auto& arg : cluster_function_block.getArguments()) {
    auto arg_sharding = ParseInputSharding(arg);
    const int arg_index_to_tpu_computation = arg.getArgNumber();

    if (!arg_sharding.hasValue()) {
      auto connected_functions_to_arg =
          ExtractFunctionsConnectedToArg(arg, module);
      for (auto& function_arg_info : connected_functions_to_arg) {
        if (arg_sharding.hasValue()) break;

        const int function_argument_index = function_arg_info.argument_index;
        auto& parsed_function = function_arg_info.func;
        Block& parsed_function_block = parsed_function.front();
        arg_sharding = ParseInputSharding(
            parsed_function_block.getArgument(function_argument_index));
      }
    }

    if (arg_sharding) {
      sharding_for_args[arg_index_to_tpu_computation] = arg_sharding.getValue();
      cluster_function.setArgAttr(
          arg_index_to_tpu_computation, kShardingAttr,
          builder->getStringAttr(arg_sharding.getValue()));
    } else {
      cluster_function.setArgAttr(
          arg_index_to_tpu_computation, kShardingAttr,
          builder->getStringAttr(logical_core_0_sharding));
    }
  }

  cluster_func_op.setAttr(tensorflow::kInputShardingAttr,
                          builder->getStrArrayAttr(sharding_for_args));
}

// Parses XlaSharding op directly connected from the outputs of the
// `cluster_func` and extract sharding configurations for outputs.
void IdentifyXlaShardingForComputationOutputs(
    StringRef logical_core_0_sharding, FuncOp func,
    tf_device::ClusterFuncOp cluster_func, Builder* builder) {
  // By default return values from logical core 0 is used if no sharding
  // configuration is defined.
  Block& function_block = func.front();
  Operation* terminator = function_block.getTerminator();
  llvm::SmallVector<llvm::StringRef, 8> sharding_for_rets(
      terminator->getNumOperands(), logical_core_0_sharding);

  // Iterate through operands of the terminator. If the preceding op is
  // XlaShardingOp, then the provided sharding configuration is added to the
  // tf_device.ClusterFunc as an attribute and the function as a result
  // attribute.
  for (auto& ret : terminator->getOpOperands()) {
    const int index = ret.getOperandNumber();
    auto ret_sharding = ParseReturnValueSharding(func, index, ret);

    if (ret_sharding) {
      sharding_for_rets[index] = ret_sharding.getValue();
      func.setResultAttr(index, kShardingAttr,
                         builder->getStringAttr(ret_sharding.getValue()));
    } else {
      func.setResultAttr(index, kShardingAttr,
                         builder->getStringAttr(logical_core_0_sharding));
    }
  }
  cluster_func.setAttr(tensorflow::kOutputShardingAttr,
                       builder->getStrArrayAttr(sharding_for_rets));
}

// Extracts input/output sharding configuration of `cluster_func` by parsing
// XlaSharding ops inside the `cluster_func`.
void IdentifyXlaShardingForTPUComputation(
    Builder* builder, tf_device::ClusterFuncOp cluster_func) {
  // Look up function definition from module.
  FuncOp func = cluster_func.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      cluster_func.func());

  // By default inputs/outputs have maximal sharding and are assigned to logical
  // core 0 if no sharding is defined.
  const std::string logical_core_0_sharding =
      xla::sharding_builder::AssignDevice(0).SerializeAsString();

  IdentifyXlaShardingForComputationInputs(logical_core_0_sharding, cluster_func,
                                          func, builder);

  IdentifyXlaShardingForComputationOutputs(logical_core_0_sharding, func,
                                           cluster_func, builder);
}

void TPUShardingIdentificationPass::runOnOperation() {
  Builder builder(getOperation().getContext());

  getOperation().walk([&](tf_device::ClusterFuncOp cluster_func) {
    IdentifyXlaShardingForTPUComputation(&builder, cluster_func);
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUShardingIdentificationPass() {
  return std::make_unique<TPUShardingIdentificationPass>();
}

static PassRegistration<TPUShardingIdentificationPass> pass(
    "tf-tpu-sharding-identification",
    "Identifies and handles inputs/outputs of TPU computation that is "
    "sharded across logical cores.");

}  // namespace TFTPU
}  // namespace mlir
