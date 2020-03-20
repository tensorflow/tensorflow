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
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Block.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kShardingAttr[] = "xla_hlo.sharding";

struct TPUShardingIdentificationPass
    : public ModulePass<TPUShardingIdentificationPass> {
  void runOnModule() override;
};

// XlaSharding op may be direct user of inputs but it may also be followed by
// an Identity op and, in the case where bfloat16 type is used, Cast op may be
// added right after the input. As so, parse the users of the operation to
// access connected XlaSharding op.
//
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect
// sharded inputs.
void GetAdjacentToXlaShardingOp(
    Operation* op, llvm::Optional<TF::XlaShardingOp>* sharding_op) {
  // TODO(hongjunchoi): Detect the case when sharding configuration is
  // ambiguous for a single input (i.e. multiple different XlaSharding ops
  // with different configuration policies are connected).
  if (sharding_op->hasValue()) return;

  if (auto sharding = llvm::dyn_cast<TF::XlaShardingOp>(op)) {
    sharding_op->emplace(sharding);
    return;
  }

  if (llvm::isa<TF::IdentityOp>(op) || llvm::isa<TF::CastOp>(op)) {
    for (auto user : op->getUsers())
      GetAdjacentToXlaShardingOp(user, sharding_op);
  }
}

// Parse XlaSharding op connected to input args. If Input to
// tf_device.LaunchFunc op is of resource type, then XlaSharding op
// will be connected to following ReadVariable op.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside a
// Call op or if/while op.
llvm::Optional<llvm::StringRef> ParseInputSharding(const FuncOp func,
                                                   const int arg_index,
                                                   const Value& arg) {
  llvm::Optional<TF::XlaShardingOp> parsed_sharding_op;
  for (auto user : arg.getUsers()) {
    if (parsed_sharding_op) continue;

    GetAdjacentToXlaShardingOp(user, &parsed_sharding_op);
    if (parsed_sharding_op) continue;

    if (llvm::isa<TF::ReadVariableOp>(user))
      for (auto read_variable_user : user->getUsers())
        GetAdjacentToXlaShardingOp(read_variable_user, &parsed_sharding_op);
  }

  if (!parsed_sharding_op) return llvm::Optional<llvm::StringRef>();
  return tensorflow::ParseShardingAttribute(parsed_sharding_op->getOperation());
}

// If operand of return value of tf_device.LaunchFunc op is directly from
// XlaSharding op, return the provided sharding configuration.
llvm::Optional<StringRef> ParseReturnValueSharding(FuncOp func,
                                                   const int output_index,
                                                   const OpOperand& operand) {
  if (auto sharding_op = llvm::dyn_cast_or_null<TF::XlaShardingOp>(
          operand.get().getDefiningOp())) {
    return tensorflow::ParseShardingAttribute(sharding_op.getOperation());
  }

  return llvm::Optional<StringRef>();
}

// If XlaSharding op is connected to input/output of the tf_device.LaunchFuncOp,
// then add attributes to the op specifying the sharding configurations.
void IdentifyXlaShardingForTPUComputation(Builder* builder,
                                          tf_device::LaunchFuncOp launch_func) {
  // Look up function definition from module.
  FuncOp func = launch_func.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      launch_func.func());
  Block& func_entry_block = func.getBody().getBlocks().front();

  // By default inputs have maximal sharding and inputs are assigned to
  // logical core 0 if no sharding is defined.
  const std::string logical_core_0_sharding =
      xla::sharding_builder::AssignDevice(0).SerializeAsString();
  auto logical_core_0_sharding_attr =
      builder->getStringAttr(logical_core_0_sharding);

  llvm::SmallVector<llvm::StringRef, 8> sharding_for_args(
      func_entry_block.getNumArguments(), logical_core_0_sharding);

  // Iterate through input arguments to the entry block of tf_device.LaunchFunc.
  // For input ops, look for following XlaSharding ops. XlaSharding ops can:
  //   1) Directly follow the input argument if input argument has non-resource
  //      types.
  //   2) Follow ReadVariableOp if the input type is of resource type.
  //   3) Follow IdentityOp or CastOp after above cases (1), (2).
  //
  // Sharding configurations are added to the tf_device.LaunchFunc as an
  // attribute and the function as an argument attribute.
  for (auto& arg : func_entry_block.getArguments()) {
    const int index = arg.getArgNumber();
    auto arg_sharding = ParseInputSharding(func, index, arg);

    if (arg_sharding) {
      sharding_for_args[index] = arg_sharding.getValue();
      func.setArgAttr(index, kShardingAttr,
                      builder->getStringAttr(arg_sharding.getValue()));
    } else {
      func.setArgAttr(index, kShardingAttr, logical_core_0_sharding_attr);
    }
  }
  launch_func.setAttr(tensorflow::kInputShardingAttr,
                      builder->getStrArrayAttr(sharding_for_args));

  // By default return values from logical core 0 is used if no sharding
  // configuration is defined.
  Operation* terminator = func_entry_block.getTerminator();
  llvm::SmallVector<llvm::StringRef, 8> sharding_for_rets(
      terminator->getNumOperands(), logical_core_0_sharding);

  // Iterate through operands of the terminator. If the preceding op is
  // XlaShardingOp, then the provided sharding configuration is added to the
  // tf_device.LaunchFunc as an attribute and the function as a result
  // attribute.
  for (auto& ret : terminator->getOpOperands()) {
    const int index = ret.getOperandNumber();
    auto ret_sharding = ParseReturnValueSharding(func, index, ret);

    if (ret_sharding) {
      sharding_for_rets[index] = ret_sharding.getValue();
      func.setResultAttr(index, kShardingAttr,
                         builder->getStringAttr(ret_sharding.getValue()));
    } else {
      func.setResultAttr(index, kShardingAttr, logical_core_0_sharding_attr);
    }
  }
  launch_func.setAttr(tensorflow::kOutputShardingAttr,
                      builder->getStrArrayAttr(sharding_for_rets));
}

void TPUShardingIdentificationPass::runOnModule() {
  Builder builder(getModule().getContext());
  getModule().walk([&](tf_device::LaunchFuncOp launch_func) {
    IdentifyXlaShardingForTPUComputation(&builder, launch_func);
  });
}

}  // anonymous namespace

std::unique_ptr<OpPassBase<ModuleOp>> CreateTPUShardingIdentificationPass() {
  return std::make_unique<TPUShardingIdentificationPass>();
}

static PassRegistration<TPUShardingIdentificationPass> pass(
    "tf-tpu-sharding-identification",
    "Identifies and handles inputs/outputs of TPU computation that is "
    "sharded across logical cores.");

}  // namespace TFTPU
}  // namespace mlir
