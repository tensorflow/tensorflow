/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
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

#define GEN_PASS_DEF_TPUVARIABLERUNTIMEREFORMATTINGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/runtime_passes.h.inc"

struct TPUVariableRuntimeReformattingPass
    : public impl::TPUVariableRuntimeReformattingPassBase<
          TPUVariableRuntimeReformattingPass> {
  void runOnOperation() final;
};

// Returns the earlier value of which `v` is an identity. If `skipped` is
// provided, it will be used to store the identity nodes skipped.
Value SkipIdentity(Value v, bool allow_other_use,
                   llvm::SmallPtrSet<Operation*, 4>* skipped = nullptr) {
  while (auto result = mlir::dyn_cast<OpResult>(v)) {
    if (!(allow_other_use || v.hasOneUse())) break;
    auto op = result.getDefiningOp();
    if (!llvm::isa<TF::IdentityOp, TF::IdentityNOp>(op)) {
      break;
    }
    v = op->getOperand(result.getResultNumber());
    if (skipped) skipped->insert(op);
  }
  return v;
}

// Finds the formattable arguments of `execute` and annotates the metadata of
// `compile` to record these arguments. In addition, it returns a mapping from
// the formattable arguments of `execute` to the corresponding operand of
// `replicate`. The
// entries in the mapping are sorted in the order of operands of `execute`.
llvm::SmallVector<std::pair<int64_t, llvm::SmallVector<Value, 4>>, 4>
AnnotateCompileOpAndGetExecuteArgToWhileArgsMapping(
    TF::WhileRegionOp while_op, tf_device::ReplicateOp replicate,
    TF::TPUExecuteAndUpdateVariablesOp execute,
    tf_device::LaunchOp compile_launch) {
  Region& body = while_op.getBody();
  Region& cond = while_op.getCond();

  llvm::SmallVector<std::pair<int64_t, llvm::SmallVector<Value, 4>>, 4> mapping;
  auto mirrored_variable_indices_attr =
      replicate->getAttrOfType<ArrayAttr>(kMirroredVariableIndicesAttr);
  if (!mirrored_variable_indices_attr) return mapping;

  // Finds the mapping from a replicate argument to an execute operand.
  llvm::SmallDenseMap<int64_t, int64_t, 8> replicate_arg_to_execute_arg;
  for (auto index_and_arg : llvm::enumerate(execute.getArgs())) {
    auto arg = SkipIdentity(index_and_arg.value(), /*allow_other_use=*/false);
    if (!arg.hasOneUse() ||
        !mlir::isa<TF::ResourceType>(getElementTypeOrSelf(arg.getType()))) {
      continue;
    }
    auto block_arg = mlir::dyn_cast<BlockArgument>(arg);
    if (!block_arg || block_arg.getOwner() != &replicate.GetBody()) continue;
    assert(replicate_arg_to_execute_arg.count(block_arg.getArgNumber()) == 0 &&
           "Found duplicate use of a resource in the execute op.");
    replicate_arg_to_execute_arg[block_arg.getArgNumber()] =
        index_and_arg.index();
  }
  if (replicate_arg_to_execute_arg.empty()) return mapping;

  // Parse the original compile metadata.
  Operation& compile = compile_launch.GetBody().front();
  auto metadata_str = compile.getAttrOfType<StringAttr>("metadata");
  assert(metadata_str && "Missing compilation metadata");
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  metadata.ParseFromString(std::string(metadata_str.getValue()));
  int64_t num_replicas = replicate.getN();
  // Find the formattable operands of `execute`, which must be mirrored
  // variables (arguments of `replicate`), and must be pass-throughs from while
  // operands.
  for (const auto& mirrored_index : mirrored_variable_indices_attr) {
    int64_t replicate_arg = mlir::cast<IntegerAttr>(mirrored_index).getInt();
    // Check if the mirrored variable is an input to `execute`.
    auto it = replicate_arg_to_execute_arg.find(replicate_arg);
    if (it == replicate_arg_to_execute_arg.end()) continue;
    // Get the data type of the resource.
    auto subtypes = mlir::cast<TF::ResourceType>(
                        getElementTypeOrSelf(execute.getOperand(it->second)))
                        .getSubtypes();
    if (subtypes.size() != 1) continue;
    auto data_type = getElementTypeOrSelf(subtypes[0]);
    // The XLA backend does not yet support formatting 64-bit data types.
    if (data_type.getIntOrFloatBitWidth() == 64) continue;

    const auto& block_arg = replicate.GetBody().getArgument(replicate_arg);

    int64_t num_inputs = 0;
    if (replicate.IsReplicatedBlockArgument(block_arg)) {
      num_inputs = num_replicas;
    } else {
      num_inputs = 1;
    }

    // We have found a mirrored variable which is an input to the replicated
    // `execute`. Now find if this mirrored variable is a pass-through of while
    // arguments.
    llvm::SmallVector<Value, 4> replicate_args;
    for (int64_t i = 0; i < num_inputs; ++i) {
      llvm::SmallPtrSet<Operation*, 4> skipped_identities;

      auto replicate_operand = SkipIdentity(
          replicate.GetReplicaOperandForBlockArgument(block_arg, i),
          /*allow_other_use=*/false, &skipped_identities);
      // For region based control flow, the resource operand for the replicate
      // should be a region capture. If this has any use other than the
      // replicate op (within the body of the while) or the skipped identities,
      // then do not apply the transformation to this variable.
      bool is_region_capture =
          replicate_operand.getParentRegion()->isProperAncestor(&body);
      bool has_other_use_in_body =
          llvm::any_of(replicate_operand.getUsers(), [&](Operation* user) {
            // Ignore uses that are not in the while body or condition.
            if (!body.isAncestor(user->getParentRegion()) &&
                !cond.isAncestor(user->getParentRegion()))
              return false;
            // Within the body or cond, only uses in replicate and the skipped
            // identities is allowed.
            return user != replicate && skipped_identities.count(user) == 0;
          });

      if (!is_region_capture || has_other_use_in_body) {
        replicate_args.clear();
        break;
      }
      replicate_args.push_back(replicate_operand);
    }
    if (replicate_args.empty()) continue;
    // Now set the enable_xla_sharding field in the metadata to inform the
    // compile op.
    auto metadata_arg = metadata.mutable_args(it->second);
    metadata_arg->set_enable_xla_sharding(
        ::tensorflow::tpu::TPUCompileMetadataProto_Arg::ALLOWED);
    mapping.emplace_back(it->second, std::move(replicate_args));
  }
  // Sort the mapping according to execute operand order.
  llvm::sort(mapping, llvm::less_first());
  // Populate the `retval_index_for_sharding` field of the argument metadate.
  for (auto entry : llvm::enumerate(execute.getDeviceVarReadsIndices())) {
    int64_t arg_index = mlir::cast<IntegerAttr>(entry.value()).getInt();
    auto arg_metadata = metadata.mutable_args(arg_index);
    if (arg_metadata->enable_xla_sharding() ==
        ::tensorflow::tpu::TPUCompileMetadataProto_Arg::ALLOWED) {
      int64_t ret_index =
          mlir::cast<IntegerAttr>(
              execute.getDeviceVarUpdatesIndices().getValue()[entry.index()])
              .getInt();
      arg_metadata->set_retval_index_for_sharding(ret_index);
    }
  }
  // Update the metadata of the compile op.
  compile.setAttr("metadata", StringAttr::get(compile.getContext(),
                                              metadata.SerializeAsString()));
  return mapping;
}

// Adds a new replicated input to the replicate op.
tf_device::ReplicateOp AddInputsToReplicateOp(
    tf_device::ReplicateOp replicate,
    MutableArrayRef<TF::VarHandleOp> new_inputs,
    const llvm::SmallDenseMap<llvm::StringRef, llvm::SmallVector<StringRef, 4>>&
        devices) {
  int64_t num_replicas = replicate.getN();
  assert(new_inputs.size() == num_replicas);

  // As model parallelism is not yet supported, we assume that all ops are
  // placed in logical core 0.
  // TODO(b/148913020): Remove this constraint once model parallelism is
  // supported.
  assert(devices.find(tensorflow::GetDeviceAliasForLogicalCore(0))
             ->getSecond()
             .size() == num_replicas);

  llvm::SmallVector<std::pair<ValueRange, Type>, 8> new_replicated_inputs;
  llvm::SmallVector<Value, 8> new_packed_inputs;
  llvm::SmallVector<llvm::SmallVector<Value, 8>, 8> replicated_inputs;
  replicated_inputs.reserve(replicate.GetNumReplicatedBlockArguments());
  new_packed_inputs.reserve(replicate.GetNumPackedBlockArguments());
  for (const auto& arg : replicate.GetReplicatedBlockArguments()) {
    replicated_inputs.emplace_back();
    for (int64_t i = 0; i < num_replicas; ++i) {
      replicated_inputs.back().push_back(
          replicate.GetReplicaOperandForBlockArgument(arg, i));
    }
    new_replicated_inputs.emplace_back(replicated_inputs.back(), arg.getType());
  }
  for (const auto& arg : replicate.GetPackedBlockArguments()) {
    new_packed_inputs.emplace_back(
        replicate.GetReplicaOperandForBlockArgument(arg, /*replica=*/0));
  }
  SmallVector<Value, 4> new_input_values;
  new_input_values.reserve(new_inputs.size());
  for (auto var : new_inputs) new_input_values.push_back(var.getResource());
  new_replicated_inputs.emplace_back(new_input_values,
                                     new_input_values.front().getType());
  OpBuilder builder(replicate);
  auto new_replicate = builder.create<tf_device::ReplicateOp>(
      replicate.getLoc(), num_replicas, devices, new_replicated_inputs,
      new_packed_inputs,
      replicate.GetBody().getTerminator()->getOperandTypes());
  for (auto arg : replicate.GetBody().getArguments()) {
    if (replicate.IsReplicatedBlockArgument(arg)) {
      arg.replaceAllUsesWith(
          new_replicate.GetBody().getArgument(arg.getArgNumber()));
    } else {
      // There is a new added replicated state variable between replicated args
      // and packed args.
      arg.replaceAllUsesWith(
          new_replicate.GetBody().getArgument(arg.getArgNumber() + 1));
    }
  }
  for (auto& op : llvm::make_early_inc_range(replicate.GetBody())) {
    op.moveBefore(&new_replicate.GetBody(), new_replicate.GetBody().end());
  }
  replicate.replaceAllUsesWith(new_replicate);
  replicate.erase();
  return new_replicate;
}

// Creates the per-device variables that represent the formatting state of each
// device.
llvm::SmallVector<TF::VarHandleOp, 4> CreateStateVars(
    const llvm::SmallDenseMap<llvm::StringRef, llvm::SmallVector<StringRef, 4>>&
        devices,
    Location loc, RankedTensorType key_type, OpBuilder* builder) {
  llvm::SmallVector<TF::VarHandleOp, 4> state_vars;

  // TODO(b/148913020): Remove this constraint once model parallelism is
  // supported.
  const auto& device_list =
      devices.find(tensorflow::GetDeviceAliasForLogicalCore(0))->getSecond();

  // Create the state variable for each device.
  for (llvm::StringRef device : device_list) {
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

// Wraps single op in `tf_device.launch` for explicit device assignment.
void WrapOpInLaunch(OpBuilder* builder, Location loc, Operation* op,
                    llvm::StringRef device) {
  OpBuilder::InsertPoint insert_point = builder->saveInsertionPoint();

  auto launch = builder->create<tf_device::LaunchOp>(
      loc, builder->getStringAttr(device), op->getResultTypes());
  launch.getBody().push_back(new Block);

  builder->setInsertionPointToEnd(&launch.GetBody());
  builder->create<tf_device::ReturnOp>(loc, op->getResults());

  // Move op inside launch.
  op->moveBefore(launch.GetBody().getTerminator());

  builder->restoreInsertionPoint(insert_point);
}

// Performs the transformation for a replicate op inside a while loop. Returns
// true when any change was made by this function.
bool HandleReplicateOp(TF::WhileRegionOp while_op,
                       tf_device::ReplicateOp replicate) {
  int64_t num_replicas = replicate.getN();
  if (num_replicas == 1) return false;

  // Set execute_launch when there is exactly one
  // TPUExecuteAndUpdateVariablesOp. More than one means there is model
  // parallelism, which is not supported with TPUReshardVariables. None
  // means there is no TPU computation.
  tf_device::LaunchOp execute_launch;
  TF::TPUExecuteAndUpdateVariablesOp execute;
  replicate.walk([&](TF::TPUExecuteAndUpdateVariablesOp execute_op) {
    execute_launch =
        llvm::dyn_cast<tf_device::LaunchOp>(execute_op->getParentOp());
    if (execute_launch == nullptr) {
      // This pass requires execute_op to be wrapped in a launch.
      return WalkResult::interrupt();
    }
    if (execute == nullptr) {
      execute = execute_op;
      return WalkResult::advance();
    }
    execute = nullptr;
    return WalkResult::interrupt();
  });
  if (!execute) return false;
  auto compile =
      SkipIdentity(execute.getKey(), /*allow_other_use=*/true).getDefiningOp();
  if (!compile) return false;
  auto compile_launch = llvm::dyn_cast<tf_device::LaunchOp>(compile);
  if (!compile_launch || !compile_launch.WrapsSingleOp() ||
      !llvm::isa<TF::_TPUCompileMlirOp>(compile_launch.GetBody().front()))
    return false;

  // Analyze the formattable inputs.
  auto execute_arg_to_outer_args =
      AnnotateCompileOpAndGetExecuteArgToWhileArgsMapping(
          while_op, replicate, execute, compile_launch);
  if (execute_arg_to_outer_args.empty()) return false;

  // Extract the replicated devices.
  auto devices_attr = replicate.getDevices();
  if (!devices_attr) return false;

  auto device_map = devices_attr.value();
  llvm::SmallDenseMap<llvm::StringRef, llvm::SmallVector<StringRef, 4>> devices;
  devices.reserve(device_map.size());

  for (auto it : device_map) {
    auto device_alias = it.getName().strref();
    auto device_list = mlir::cast<ArrayAttr>(it.getValue());
    llvm::SmallVector<StringRef, 4> device_list_for_alias;
    device_list_for_alias.reserve(device_list.size());

    for (auto device : device_list)
      device_list_for_alias.emplace_back(
          mlir::cast<StringAttr>(device).getValue());

    devices.insert({device_alias, device_list_for_alias});
  }

  OpBuilder builder(replicate);
  builder.setInsertionPoint(while_op);
  // Create per-device variables for formatting state, and add them to the while
  // loop.
  auto key_type =
      RankedTensorType::get({2}, TF::StringType::get(builder.getContext()));
  auto state_vars =
      CreateStateVars(devices, while_op.getLoc(), key_type, &builder);
  replicate = AddInputsToReplicateOp(replicate, state_vars, devices);
  // Build the reformat according to the compilation. Build it inside
  // `replicate`.
  llvm::SmallVector<Value, 8> reformat_operands;
  for (const auto& entry : execute_arg_to_outer_args) {
    reformat_operands.push_back(execute.getArgs()[entry.first]);
  }
  reformat_operands.push_back(compile_launch.getResult(1));
  reformat_operands.push_back(replicate.GetBody().getArgument(
      replicate.GetNumReplicatedBlockArguments() - 1));
  builder.setInsertionPoint(execute_launch);
  auto reformat_op = builder.create<TF::TPUReshardVariablesOp>(
      execute_launch.getLoc(), llvm::ArrayRef<Type>{}, reformat_operands);
  WrapOpInLaunch(&builder, execute_launch.getLoc(), reformat_op,
                 execute_launch.getDevice());

  // Build the replicated unformat op after the loop. First prepare building the
  // replicate op.
  llvm::SmallVector<std::pair<ValueRange, Type>, 8> unformat_replicate_operands;
  llvm::SmallVector<Value, 8> unformat_packed_operands;
  for (const auto& entry : execute_arg_to_outer_args) {
    if (entry.second.size() > 1) {
      unformat_replicate_operands.emplace_back(entry.second,
                                               entry.second.front().getType());
    } else {
      unformat_packed_operands.emplace_back(entry.second.front());
    }
  }
  llvm::SmallVector<Value, 4> state_var_vals(state_vars.size());
  for (const auto& entry : llvm::enumerate(state_vars)) {
    state_var_vals[entry.index()] = entry.value().getResource();
  }
  // Add the replicated state var to the end of the replicate operands.
  unformat_replicate_operands.emplace_back(state_var_vals,
                                           state_var_vals.front().getType());
  // Build a constant default key to specify that the unformatting should
  // transform the variables to the original format.
  builder.setInsertionPointAfter(while_op);
  tensorflow::Tensor default_key_tensor(tensorflow::DT_STRING, {3});
  default_key_tensor.vec<tensorflow::tstring>()(0) = kDefaultShardingValue;
  default_key_tensor.vec<tensorflow::tstring>()(1) = kDefaultShardingValue;
  default_key_tensor.vec<tensorflow::tstring>()(2) = kDefaultShardingValue;
  auto default_state_key = builder.create<TF::ConstOp>(
      while_op.getLoc(),
      tensorflow::ConvertTensor(default_key_tensor, &builder).value());
  // With all replicated inputs, now build the replicate op.
  auto unformat_replicate = builder.create<tf_device::ReplicateOp>(
      while_op.getLoc(), num_replicas, devices, unformat_replicate_operands,
      unformat_packed_operands, TypeRange{});
  // Then build the unformat op in the replicate op.
  builder.setInsertionPointToEnd(&unformat_replicate.GetBody());
  llvm::SmallVector<Value, 8> unformat_operands;
  // Add the replicated state var (the last replicated operand of the
  // ReplicateOp) as the last operand of TPUReshardVariablesOp.
  BlockArgument state = unformat_replicate.GetReplicatedBlockArguments().back();
  auto replicated_block_args =
      unformat_replicate.GetReplicatedBlockArguments().drop_back(1);
  auto packed_block_args = unformat_replicate.GetPackedBlockArguments();
  unformat_operands.append(replicated_block_args.begin(),
                           replicated_block_args.end());
  unformat_operands.append(packed_block_args.begin(), packed_block_args.end());
  unformat_operands.push_back(state);

  // Insert the default key as the second last operand.
  unformat_operands.insert(
      unformat_operands.begin() + unformat_operands.size() - 1,
      default_state_key.getResult());
  // Unformat op.
  auto unformat_op = builder.create<TF::TPUReshardVariablesOp>(
      while_op.getLoc(), llvm::ArrayRef<Type>{}, unformat_operands);
  WrapOpInLaunch(&builder, execute_launch.getLoc(), unformat_op,
                 execute_launch.getDevice());
  builder.create<tf_device::ReturnOp>(while_op.getLoc(), ArrayRef<Value>{});

  return true;
}

void TPUVariableRuntimeReformattingPass::runOnOperation() {
  auto module = getOperation();
  bool reshard_was_inserted = false;
  module.walk([&](TF::WhileRegionOp while_op) {
    tf_device::ReplicateOp replicate;
    while_op.getBody().walk([&](tf_device::ReplicateOp replicate_op) {
      if (replicate == nullptr) {
        replicate = replicate_op;
        return WalkResult::advance();
      }
      // We do not handle loops with multiple replicate ops.
      replicate = nullptr;
      return WalkResult::interrupt();
    });
    if (replicate)
      reshard_was_inserted |= HandleReplicateOp(while_op, replicate);
  });
  if (reshard_was_inserted)
    VLOG(1) << "tf-tpu-variable-runtime-reformatting inserted at least one "
               "TPUReshardVariables";
  else
    VLOG(1) << "tf-tpu-variable-runtime-reformatting inserted no "
               "TPUReshardVariables";
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUVariableRuntimeReformattingPass() {
  return std::make_unique<TPUVariableRuntimeReformattingPass>();
}

}  // namespace TFTPU
}  // namespace mlir
