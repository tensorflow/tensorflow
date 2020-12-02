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
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kShardingAttr[] = "mhlo.sharding";

struct TPUShardingIdentificationPass
    : public PassWrapper<TPUShardingIdentificationPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Finds XlaSharding op connected to an argument value. If value is a resource
// type then XlaSharding op will be connected to a ReadVariable op. XlaSharding
// op may be direct user of inputs but it may also be followed by an Identity op
// and, in the case where bfloat16 type is used, Cast op may be added right
// after the input.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside control flow (If,
// Case, While) ops and Caller return values.
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect sharded
// inputs.
llvm::Optional<llvm::StringRef> GetXlaShardingFromArg(const Value& value) {
  llvm::SmallPtrSet<Value, 4> visited_values;
  llvm::SmallVector<Value, 4> values_to_visit{value};
  while (!values_to_visit.empty()) {
    llvm::SmallVector<Value, 4> next_values_to_visit;
    for (Value value_to_visit : values_to_visit) {
      if (!visited_values.insert(value_to_visit).second) continue;

      for (auto& use : value_to_visit.getUses()) {
        Operation* owner = use.getOwner();
        if (auto sharding = llvm::dyn_cast<TF::XlaShardingOp>(owner))
          return sharding._XlaSharding();

        if (llvm::isa<TF::IdentityOp, TF::CastOp, TF::ReadVariableOp>(owner)) {
          next_values_to_visit.push_back(use.getOwner()->getResult(0));
          continue;
        }

        if (auto call_op = llvm::dyn_cast<CallOpInterface>(owner)) {
          FuncOp func = llvm::dyn_cast<FuncOp>(call_op.resolveCallable());
          if (!func) continue;
          next_values_to_visit.push_back(
              func.getArgument(use.getOperandNumber()));
        }
      }
    }

    values_to_visit.swap(next_values_to_visit);
  }

  return llvm::None;
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
    auto arg_sharding = GetXlaShardingFromArg(arg);
    const int index = arg.getArgNumber();

    if (arg_sharding) {
      sharding_for_args[index] = arg_sharding.getValue();
      cluster_function.setArgAttr(
          index, kShardingAttr,
          builder->getStringAttr(arg_sharding.getValue()));
    } else {
      cluster_function.setArgAttr(
          index, kShardingAttr,
          builder->getStringAttr(logical_core_0_sharding));
    }
  }

  cluster_func_op.setAttr(tensorflow::kInputShardingAttr,
                          builder->getStrArrayAttr(sharding_for_args));
}

// Finds XlaSharding op connected to a result value. XlaSharding op may be
// direct user of inputs but it may also be followed by an Identity op and, in
// the case where bfloat16 type is used, Cast op may be added right after the
// input.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside control flow (If,
// Case, While) ops and Caller argument values.
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect sharded
// inputs.
llvm::Optional<StringRef> GetXlaShardingFromRetval(const Value& value) {
  llvm::SmallPtrSet<Value, 4> visited_values;
  Value value_to_visit = value;
  while (value_to_visit) {
    if (!visited_values.insert(value_to_visit).second) return llvm::None;

    Operation* def = value_to_visit.getDefiningOp();
    if (auto sharding = llvm::dyn_cast_or_null<TF::XlaShardingOp>(def))
      return sharding._XlaSharding();

    if (llvm::isa_and_nonnull<TF::IdentityOp, TF::CastOp>(def)) {
      value_to_visit = def->getOperand(0);
      continue;
    }

    if (auto call_op = llvm::dyn_cast_or_null<CallOpInterface>(def)) {
      FuncOp func = llvm::dyn_cast<FuncOp>(call_op.resolveCallable());
      if (!func) continue;
      value_to_visit = func.front().getTerminator()->getOperand(
          value_to_visit.cast<OpResult>().getResultNumber());
      continue;
    }

    break;
  }

  return llvm::None;
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
    auto ret_sharding = GetXlaShardingFromRetval(ret.get());
    const int index = ret.getOperandNumber();

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
