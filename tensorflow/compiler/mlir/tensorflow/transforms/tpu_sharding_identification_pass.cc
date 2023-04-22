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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
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
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kReplicateSharding[] = "";
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kUseSpmdAttr[] = "use_spmd_for_xla_partitioning";

struct TPUShardingIdentificationPass
    : public PassWrapper<TPUShardingIdentificationPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const final {
    return "tf-tpu-sharding-identification";
  }

  StringRef getDescription() const final {
    return "Identifies and handles inputs/outputs of TPU computation that is "
           "sharded across logical cores.";
  }

  void runOnOperation() override;
};

// Returns XLA sharding from TPUPartitionedInput op connected to a
// `tf_device.cluster_func` operand value. If value is a resource type then
// TPUPartitionedInput op will be connected to a ReadVariable op that feeds into
// a `tf_device.cluster_func`.
llvm::Optional<llvm::StringRef> GetXlaShardingFromOperand(Value value) {
  Value value_to_visit = value;
  if (auto read_var = llvm::dyn_cast_or_null<TF::ReadVariableOp>(
          value_to_visit.getDefiningOp()))
    value_to_visit = read_var.resource();

  if (auto partitioned_input =
          llvm::dyn_cast_or_null<TF::TPUPartitionedInputOp>(
              value_to_visit.getDefiningOp()))
    return partitioned_input._XlaSharding();

  return llvm::None;
}

// Returns XLA sharding from a XlaSharding op connected to an argument value. If
// value is a resource type then XlaSharding op will be connected to a
// ReadVariable op. XlaSharding op may be direct user of inputs but it may also
// be followed by an Identity op and, in the case where bfloat16 type is used,
// Cast op may be added right after the input.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside control flow (If,
// Case, While) ops and Caller return values.
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect sharded
// inputs.
llvm::Optional<llvm::StringRef> GetXlaShardingFromArg(Value value) {
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

// Extracts sharding configurations for all inputs by parsing XlaSharding/
// TPUPartitionedInput op connected to the operands/arguments. If argument to
// the `cluster_func` directly feeds into another function call op, then
// recursively walk the function definition to find the connected XlaSharding
// op.
void IdentifyXlaShardingForComputationInputs(
    StringRef logical_core_0_sharding, bool use_spmd,
    tf_device::ClusterFuncOp cluster_func, FuncOp func, Builder* builder,
    llvm::SmallVectorImpl<llvm::StringRef>& sharding_for_args) {
  // Look up function definition from module.
  Block& function_block = func.front();

  sharding_for_args.reserve(function_block.getNumArguments());

  // Iterate through operands of `cluster_func`.
  // The computation operand can either be:
  //   1) a TPUPartitionedInput Op if the input has a non-resource type;
  //   2) a ReadVariableOp else.
  //
  // Replicate sharding is used if `use_spmd` is set.
  //
  // Iterate through input arguments to the entry block of
  // tf_device.ClusterFunc. For input ops, look for XlaSharding ops.
  // XlaSharding ops can:
  //   1) Directly follow the input argument if input argument has non-resource
  //      types.
  //   2) Follow ReadVariableOp if the input type is of resource type.
  //   3) Follow IdentityOp or CastOp after above cases (1), (2).
  //
  // Sharding configurations are added to the tf_device.ClusterFunc as an
  // attribute and the function as an argument attribute.
  for (auto operand_and_arg :
       llvm::zip(cluster_func.operands(), function_block.getArguments())) {
    Value operand = std::get<0>(operand_and_arg);
    BlockArgument arg = std::get<1>(operand_and_arg);

    if (auto operand_sharding = GetXlaShardingFromOperand(operand)) {
      sharding_for_args.push_back(operand_sharding.getValue());
      continue;
    }

    if (use_spmd) {
      // If XLA SPMD is enabled, host variables or non-variable per-replica
      // inputs should take on replicate sharding, unless another sharding is
      // set via a TPUPartitionedInput op.
      sharding_for_args.push_back(kReplicateSharding);
      continue;
    }

    auto arg_sharding = GetXlaShardingFromArg(arg);
    if (arg_sharding) {
      sharding_for_args.push_back(arg_sharding.getValue());
      continue;
    }

    // Default to maximal sharding core 0 if no sharding is present.
    sharding_for_args.push_back(logical_core_0_sharding);
  }
}

// Returns XLA sharding from TPUPartitionedOutput or TPUPartitionedInput (via
// AssignVariableOp/resource write) op connected to a `tf_device.cluster_func`
// result value.
llvm::Optional<llvm::StringRef> GetXlaShardingFromResult(Value value) {
  if (!value.hasOneUse()) return llvm::None;

  Operation* user = *value.getUsers().begin();
  if (auto partitioned_output =
          llvm::dyn_cast<TF::TPUPartitionedOutputOp>(user))
    return partitioned_output._XlaSharding();

  if (auto assign_var = llvm::dyn_cast<TF::AssignVariableOp>(user))
    if (auto partitioned_input =
            llvm::dyn_cast_or_null<TF::TPUPartitionedInputOp>(
                assign_var.resource().getDefiningOp()))
      return partitioned_input._XlaSharding();

  return llvm::None;
}

// Returns XLA sharding from XlaSharding op connected to a result value.
// XlaSharding op may be direct user of inputs but it may also be followed by an
// Identity op and, in the case where bfloat16 type is used, Cast op may be
// added right after the input.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside control flow (If,
// Case, While) ops and Caller argument values.
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect sharded
// inputs.
llvm::Optional<StringRef> GetXlaShardingFromRetval(Value value) {
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

// Extracts sharding configurations for all outputs by parsing XlaSharding/
// TPUPartitionedOutput op connected to the retvals/results.
void IdentifyXlaShardingForComputationOutputs(
    StringRef logical_core_0_sharding, bool use_spmd,
    tf_device::ClusterFuncOp cluster_func, FuncOp func, Builder* builder,
    llvm::SmallVectorImpl<llvm::StringRef>& sharding_for_rets) {
  Block& function_block = func.front();
  Operation* terminator = function_block.getTerminator();
  sharding_for_rets.reserve(terminator->getNumOperands());

  // Iterate through results of `cluster_func`. For output ops, look for
  // TPUPartitionedOutput ops.
  //
  // Replicate sharding is used if `use_spmd` is set.
  //
  // Iterate through operands of the terminator. If the preceding op is
  // XlaShardingOp, then the provided sharding configuration is added to the
  // tf_device.ClusterFunc as an attribute and the function as a result
  // attribute.
  for (auto result_and_retval :
       llvm::zip(cluster_func.results(), terminator->getOpOperands())) {
    Value result = std::get<0>(result_and_retval);
    OpOperand& retval = std::get<1>(result_and_retval);

    if (auto result_sharding = GetXlaShardingFromResult(result)) {
      sharding_for_rets.push_back(result_sharding.getValue());
      continue;
    }

    if (use_spmd) {
      // If XLA SPMD is enabled, outputs all should have replicate sharding,
      // unless another sharding is set via a TPUPartitionedOutput op.
      sharding_for_rets.push_back(kReplicateSharding);
      continue;
    }

    if (auto retval_sharding = GetXlaShardingFromRetval(retval.get())) {
      sharding_for_rets.push_back(retval_sharding.getValue());
      continue;
    }

    // Default to maximal sharding core 0 if no sharding is present.
    sharding_for_rets.push_back(logical_core_0_sharding);
  }
}

// Extracts input/output sharding configuration of `cluster_func` by parsing
// XlaSharding ops inside the `cluster_func`.
void IdentifyXlaShardingForTPUComputation(
    Builder* builder, tf_device::ClusterFuncOp cluster_func) {
  // Look up function definition from module.
  FuncOp func = cluster_func->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      cluster_func.func());

  // By default inputs/outputs have maximal sharding and are assigned to logical
  // core 0 if no sharding is defined.
  const std::string logical_core_0_sharding =
      xla::sharding_builder::AssignDevice(0).SerializeAsString();

  bool use_spmd = false;
  if (auto use_spmd_attr = cluster_func->getAttrOfType<BoolAttr>(kUseSpmdAttr))
    use_spmd = use_spmd_attr.getValue();

  llvm::SmallVector<llvm::StringRef, 8> sharding_for_args;
  IdentifyXlaShardingForComputationInputs(logical_core_0_sharding, use_spmd,
                                          cluster_func, func, builder,
                                          sharding_for_args);

  llvm::SmallVector<llvm::StringRef, 8> sharding_for_rets;
  IdentifyXlaShardingForComputationOutputs(logical_core_0_sharding, use_spmd,
                                           cluster_func, func, builder,
                                           sharding_for_rets);

  auto has_maximal_sharding = [](llvm::StringRef sharding_string) -> bool {
    xla::OpSharding sharding;
    sharding.ParseFromString(sharding_string.str());
    return sharding.type() == xla::OpSharding::MAXIMAL;
  };

  // XLA SPMD only supports cases where all inputs/outputs exist on every
  // partition (sharded or replicated). If any of the inputs/outputs have
  // maximal sharding, then fallback to MPMD.
  if (use_spmd && (absl::c_any_of(sharding_for_args, has_maximal_sharding) ||
                   absl::c_any_of(sharding_for_rets, has_maximal_sharding))) {
    LOG(WARNING) << "XLA SPMD only supports cases where all inputs/outputs "
                    "exist on every partition (sharded or replicated). If any "
                    "of the inputs/outputs have maximal sharding, then "
                    "fallback to MPMD.";
    sharding_for_args.clear();
    sharding_for_rets.clear();
    cluster_func->setAttr(kUseSpmdAttr, builder->getBoolAttr(false));
    IdentifyXlaShardingForComputationInputs(logical_core_0_sharding,
                                            /*use_spmd=*/false, cluster_func,
                                            func, builder, sharding_for_args);
    IdentifyXlaShardingForComputationOutputs(logical_core_0_sharding,
                                             /*use_spmd=*/false, cluster_func,
                                             func, builder, sharding_for_rets);
  }

  // Update sharding on function arguments and returns.
  Block& function_block = func.front();
  for (auto sharding_and_arg :
       llvm::zip(sharding_for_args, function_block.getArguments())) {
    StringRef sharding = std::get<0>(sharding_and_arg);
    BlockArgument arg = std::get<1>(sharding_and_arg);
    func.setArgAttr(arg.getArgNumber(), kShardingAttr,
                    builder->getStringAttr(sharding));
  }

  Operation* terminator = function_block.getTerminator();
  for (auto sharding_and_retval :
       llvm::zip(sharding_for_rets, terminator->getOpOperands())) {
    StringRef sharding = std::get<0>(sharding_and_retval);
    OpOperand& retval = std::get<1>(sharding_and_retval);
    func.setResultAttr(retval.getOperandNumber(), kShardingAttr,
                       builder->getStringAttr(sharding));
  }

  // Update input/output sharding attributes on tf_device.cluster_func op.
  cluster_func->setAttr(tensorflow::kInputShardingAttr,
                        builder->getStrArrayAttr(sharding_for_args));
  cluster_func->setAttr(tensorflow::kOutputShardingAttr,
                        builder->getStrArrayAttr(sharding_for_rets));
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

static PassRegistration<TPUShardingIdentificationPass> pass;

}  // namespace TFTPU
}  // namespace mlir
