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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "xla/hlo/builder/sharding_builder.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {
namespace {

using OpShardingVariant = std::variant<mlir::Operation*, llvm::StringRef>;
using OpShardingVector = llvm::SmallVector<OpShardingVariant, 8>;
using OptionalOpShardingVector =
    llvm::SmallVector<std::optional<OpShardingVariant>, 8>;
using llvm::StringRef;
using mlir::Block;
using mlir::BlockArgument;
using mlir::BoolAttr;
using mlir::Builder;
using mlir::IntegerAttr;
using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OpOperand;
using mlir::OpResult;
using mlir::RankedTensorType;
using mlir::StringAttr;
using mlir::Value;
using mlir::WalkResult;

constexpr char kReplicateSharding[] = "";
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kUseSpmdAttr[] = "use_spmd_for_xla_partitioning";
constexpr char kAliasingAttr[] = "tf.aliasing_output";
constexpr char kNumCoresPerReplicaAttr[] = "num_cores_per_replica";
const char kShardingAttribute[] = "_XlaSharding";
const char kShardingAttributeV2[] = "_XlaShardingV2";

#define GEN_PASS_DEF_TPUSHARDINGIDENTIFICATIONPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

struct TPUShardingIdentificationPass
    : public impl::TPUShardingIdentificationPassBase<
          TPUShardingIdentificationPass> {
  void runOnOperation() final;
};

std::string CreateMissingAttributeMsg(llvm::StringRef attribute) {
  return llvm::formatv("requires attribute '{0}'", attribute).str();
}

// Returns nullptr if the op does not have a sharding attribute.
template <typename PartitionedOp>
mlir::Operation* NullUnlessSharded(PartitionedOp op) {
  return op.get_XlaSharding() ? op : nullptr;
}

// Returns true if unitary op has one of the traits that meets the requirements
// for sharding, otherwise returns false.
bool UnaryOpHasTraitsForSharding(Operation* op) {
  // Trait "SameOperandsAndResultTypeResolveRef" for Cast, real/imag, etc.
  if (op->hasTrait<mlir::OpTrait::TF::SameOperandsAndResultTypeResolveRef>())
    return true;
  // Trait "SameOperandsAndResultType" for Exp, ceil, etc.
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultType>()) return true;
  // Trait "OperandsSameAsResultsTypeOrRef" for Identity.
  if (op->hasTrait<mlir::OpTrait::TF::OperandsSameAsResultsTypeOrRef>())
    return true;
  return false;
}

// Returns true if binary op has one of the traits that meets the requirements
// for sharding, otherwise returns false.
bool BinaryOpHasTraitsForSharding(Operation* op) {
  // Trait "CwiseBinary" and "SameOperandsAndResultElementTypeResolveRef" for
  // AddV2, Sub, etc.
  if (op->hasTrait<
          mlir::OpTrait::TF::SameOperandsAndResultElementTypeResolveRef>() &&
      op->hasTrait<mlir::OpTrait::TF::CwiseBinary>())
    return true;
  return false;
}

bool DoTypesHavePartialSameShape(Value value_0, Value value_1) {
  auto shape_0 =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(value_0.getType());
  auto shape_1 =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(value_1.getType());
  if (shape_0 && shape_1) {
    if (shape_0.hasStaticShape() && shape_1.hasStaticShape())
      return shape_0.getShape() == shape_1.getShape();
    int i = 0, j = 0;
    while (i < shape_0.getShape().size() && j < shape_1.getShape().size()) {
      if (shape_0.getShape()[i] != shape_1.getShape()[j] &&
          !shape_0.isDynamicDim(i) && !shape_1.isDynamicDim(j)) {
        return false;
      }
      if (shape_0.getShape()[i] == shape_1.getShape()[j]) {
        i++;
        j++;
      } else {
        if (shape_0.isDynamicDim(i)) {
          i++;
        }
        if (shape_1.isDynamicDim(j)) {
          j++;
        }
      }
    }
    return i == shape_0.getShape().size() && j == shape_1.getShape().size();
  }
  return false;
}
// Returns a TPUPartitionedInput op connected to a `tf_device.cluster_func`
// operand value if it has an XLA sharding. If value is a resource type then
// TPUPartitionedInput op will be connected to a ReadVariable op that feeds into
// a `tf_device.cluster_func`.
mlir::Operation* GetXlaShardingFromOperand(Value value) {
  Value value_to_visit = value;
  if (auto read_var = value_to_visit.getDefiningOp<mlir::TF::ReadVariableOp>())
    value_to_visit = read_var.getResource();

  if (auto partitioned_input =
          value_to_visit.getDefiningOp<mlir::TF::TPUPartitionedInputV2Op>()) {
    return NullUnlessSharded(partitioned_input);
  }

  return nullptr;
}

// Returns the op sharding attribute from a partitioned operator.
std::optional<StringRef> GetXlaShardingFromOperator(mlir::Operation* op) {
  if (auto partitioned_output =
          llvm::dyn_cast<mlir::TF::TPUPartitionedOutputV2Op>(op)) {
    if (partitioned_output.get_XlaShardingV2().has_value()) {
      return partitioned_output.get_XlaShardingV2();
    }
    return partitioned_output.get_XlaSharding();
  } else if (auto partitioned_input =
                 llvm::dyn_cast<mlir::TF::TPUPartitionedInputV2Op>(op)) {
    if (partitioned_input.get_XlaShardingV2().has_value()) {
      return partitioned_input.get_XlaShardingV2();
    }
    return partitioned_input.get_XlaSharding();
  }
  return std::nullopt;
}

// Returns the sharding string from a op-sharding variant if it is available.
std::optional<StringRef> GetShardingStringFromVariant(
    const OpShardingVariant& sharding_or_op) {
  return std::visit(
      [](auto&& sharding_or_op) -> std::optional<StringRef> {
        using T = std::decay_t<decltype(sharding_or_op)>;
        if constexpr (std::is_same_v<T, StringRef>) {
          return sharding_or_op;
        } else {
          return GetXlaShardingFromOperator(sharding_or_op);
        }
      },
      sharding_or_op);
}

// Returns the sharding from a op-sharding variant if it is available and valid.
std::optional<xla::OpSharding> GetShardingFromVariant(
    const OpShardingVariant& sharding_or_op) {
  xla::OpSharding sharding;
  const auto sharding_string = GetShardingStringFromVariant(sharding_or_op);
  if (!sharding_string) return std::nullopt;
  if (tensorflow::DecodeShardingAttribute(sharding_string->str(), sharding,
                                          false)
          .failed()) {
    return std::nullopt;
  }
  return sharding;
}

// Converts an op-sharding vector into a string attr using the builder.
mlir::ArrayAttr GetStrArrayAttr(Builder* builder,
                                const OpShardingVector& vect) {
  llvm::SmallVector<mlir::Attribute, 8> strings;
  for (const auto& sharding_or_op : vect) {
    if (const auto sharding = GetShardingStringFromVariant(sharding_or_op)) {
      strings.emplace_back(builder->getStringAttr(*sharding));
    }
  }
  return builder->getArrayAttr(strings);
}

// Verify whether the given sharding can be applied to the given (tensor) type.
// (A bad sharding might mean failing tf.Split ops if the graph later executes
//  on CPU)
// If the sharding is incorrect, return failure. If it's good, or if we can't
// verify it, return success.
LogicalResult VerifySharding(mlir::Type type,
                             const OpShardingVariant& sharding_or_op) {
  auto* partitioned_op =
      std::holds_alternative<mlir::Operation*>(sharding_or_op)
          ? std::get<mlir::Operation*>(sharding_or_op)
          : nullptr;
  const auto sharding = GetShardingFromVariant(sharding_or_op);
  if (!sharding || sharding->type() != xla::OpSharding::OTHER) {
    // Some test cases use \01\02\03 as sharding, to test propagation. Treat
    // a non-proto sharding as valid, and don't verify further. We also only
    // verify shardings that actually break a tensor apart.
    return mlir::success();
  }
  if (RankedTensorType ranked_type = mlir::dyn_cast<RankedTensorType>(type)) {
    const int64_t tensor_rank = ranked_type.getRank();
    int tile_assignment_rank = sharding->tile_assignment_dimensions_size();

    // When a tensor is partial or subgroup tiled, its tile assignment will
    // have one or more dimension(s) than its rank; so, we subtract them to
    // determine which rank the sharding is compatible with.
    tile_assignment_rank -= (int)sharding->replicate_on_last_tile_dim();
    tile_assignment_rank -= sharding->last_tile_dims_size();

    if (tensor_rank < tile_assignment_rank) {
      if (partitioned_op) {
        partitioned_op->emitError()
            << "tensor of type " << ranked_type << " (rank=" << tensor_rank
            << ") sharded in " << (tile_assignment_rank - tensor_rank)
            << " extra dimension(s) by: " << sharding->DebugString();
      }

      return mlir::failure();
    }
  }
  return mlir::success();
}

// Verify sharding for all arguments and return values.
LogicalResult VerifyShardings(mlir::func::FuncOp func,
                              const OpShardingVector& sharding_for_args,
                              const OpShardingVector& sharding_for_rets) {
  Block& function_block = func.front();
  for (auto sharding_and_arg :
       llvm::zip(sharding_for_args, function_block.getArguments())) {
    const auto& sharding = std::get<0>(sharding_and_arg);
    BlockArgument arg = std::get<1>(sharding_and_arg);
    if (failed(VerifySharding(arg.getType(), sharding))) return mlir::failure();
  }
  Operation* terminator = function_block.getTerminator();
  for (auto sharding_and_retval :
       llvm::zip(sharding_for_rets, terminator->getOpOperands())) {
    const auto& sharding = std::get<0>(sharding_and_retval);
    OpOperand& retval = std::get<1>(sharding_and_retval);
    if (failed(VerifySharding(retval.get().getType(), sharding)))
      return mlir::failure();
  }
  return mlir::success();
}

// Assign the logical device if an op has an attribute `TPU_REPLICATED_CORE:n`,
// the corresponding input sharding arg will be associated with
// logical device `n`.
std::optional<llvm::StringRef> AssignLogicalDeviceFromTPUReplicatedCoreAttr(
    Operation* op, const llvm::SmallVector<std::string>& logical_device_vec) {
  if (auto device = op->getAttrOfType<StringAttr>("device")) {
    if (!device.getValue().empty() && !device.getValue().str().empty()) {
      tensorflow::DeviceNameUtils::ParsedName name;
      if (tensorflow::DeviceNameUtils::ParseFullName(device.str(), &name)) {
        if (name.type == "TPU_REPLICATED_CORE") {
          // TODO(hanxiongwang): Add check for out of bound of name.id
          return llvm::StringRef(logical_device_vec[name.id]);
        }
      }
    }
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<llvm::StringRef>> GetXlaShardingFromShardingOp(
    mlir::TF::XlaShardingOp sharding) {
  TF_ASSIGN_OR_RETURN(auto attr, GetXlaShardingAttrFromShardingOp(sharding));
  return attr ? ::std::optional<::llvm::StringRef>(attr.getValue())
              : (::std::nullopt);
}

// Returns XLA sharding from a XlaSharding op connected to an argument value. If
// value is a resource type then XlaSharding op will be connected to a
// ReadVariable op. XlaSharding op may be direct user of inputs but it may also
// be followed by an Identity op and, in the case where bfloat16 type is used,
// Cast op may be added right after the input.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside control flow (If,
// Case) ops and Caller return values.
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect sharded
// inputs.
absl::StatusOr<std::optional<llvm::StringRef>> GetXlaShardingFromArg(
    Value value, const llvm::SmallVector<std::string>& logical_device_vec) {
  llvm::SmallPtrSet<Value, 4> visited_values;
  llvm::SmallVector<Value, 4> values_to_visit{value};
  while (!values_to_visit.empty()) {
    llvm::SmallVector<Value, 4> next_values_to_visit;
    for (Value value_to_visit : values_to_visit) {
      if (!visited_values.insert(value_to_visit).second) continue;

      for (auto& use : value_to_visit.getUses()) {
        Operation* owner = use.getOwner();
        if (auto sharding = llvm::dyn_cast<mlir::TF::XlaShardingOp>(owner))
          return GetXlaShardingFromShardingOp(sharding);

        if (auto logical_device = AssignLogicalDeviceFromTPUReplicatedCoreAttr(
                owner, logical_device_vec)) {
          return logical_device;
        }

        if (auto while_op = llvm::dyn_cast<mlir::TF::WhileRegionOp>(owner)) {
          const int operand_number = use.getOperandNumber();
          next_values_to_visit.push_back(
              while_op.getCond().front().getArgument(operand_number));
          next_values_to_visit.push_back(
              while_op.getBody().front().getArgument(operand_number));
          continue;
        }

        if (UnaryOpHasTraitsForSharding(owner)) {
          next_values_to_visit.push_back(use.getOwner()->getResult(0));
          continue;
        }

        if (BinaryOpHasTraitsForSharding(owner)) {
          if (DoTypesHavePartialSameShape(value_to_visit,
                                          owner->getResult(0))) {
            next_values_to_visit.push_back(use.getOwner()->getResult(0));
            continue;
          }
        }

        if (llvm::isa<mlir::TF::CastOp, mlir::TF::XlaAllReduceOp,
                      mlir::TF::ReadVariableOp>(owner)) {
          next_values_to_visit.push_back(use.getOwner()->getResult(0));
          continue;
        }

        if (auto call_op = llvm::dyn_cast<mlir::CallOpInterface>(owner)) {
          mlir::func::FuncOp func =
              llvm::dyn_cast<mlir::func::FuncOp>(call_op.resolveCallable());
          if (!func) continue;
          next_values_to_visit.push_back(
              func.getArgument(use.getOperandNumber()));
        }
      }
    }

    values_to_visit.swap(next_values_to_visit);
  }

  return std::nullopt;
}

// Tries to extract sharding configurations for all inputs by parsing
// XlaSharding/ TPUPartitionedInput op connected to the operands/arguments. If
// argument to the `cluster_func` directly feeds into another function call op,
// then recursively walk the function definition to find the connected
// XlaSharding op.
absl::Status IdentifyXlaShardingForComputationInputs(
    const llvm::SmallVector<std::string>& logical_device_vec,
    bool infer_from_computation, mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::func::FuncOp func, Builder* builder,
    OptionalOpShardingVector& sharding_for_args) {
  // Look up function definition from module.
  Block& function_block = func.front();

  sharding_for_args.reserve(function_block.getNumArguments());

  // Iterate through operands of `cluster_func`.
  // The computation operand can either be:
  //   1) a TPUPartitionedInput Op if the input has a non-resource type;
  //   2) a ReadVariableOp else.
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
       llvm::zip(cluster_func.getOperands(), function_block.getArguments())) {
    Value operand = std::get<0>(operand_and_arg);
    BlockArgument arg = std::get<1>(operand_and_arg);

    if (auto operand_sharding = GetXlaShardingFromOperand(operand)) {
      sharding_for_args.push_back(operand_sharding);
      continue;
    }

    if (infer_from_computation) {
      TF_ASSIGN_OR_RETURN(auto arg_sharding,
                          GetXlaShardingFromArg(arg, logical_device_vec));
      if (arg_sharding) {
        sharding_for_args.push_back(arg_sharding.value());
        continue;
      }
    }

    sharding_for_args.push_back(std::nullopt);
  }
  return absl::OkStatus();
}

// Returns a TPUPartitionedOutput or TPUPartitionedInput op with XLA sharding
// connected to a `tf_device.cluster_func` result value (via AssignVariableOp/
// resource write).
mlir::Operation* GetXlaShardingFromResult(Value value) {
  if (!value.hasOneUse()) return nullptr;

  Operation* user = *value.getUsers().begin();
  if (auto partitioned_output =
          llvm::dyn_cast<mlir::TF::TPUPartitionedOutputV2Op>(user))
    return NullUnlessSharded(partitioned_output);

  if (auto assign_var = llvm::dyn_cast<mlir::TF::AssignVariableOp>(user))
    if (auto partitioned_input =
            assign_var.getResource()
                .getDefiningOp<mlir::TF::TPUPartitionedInputV2Op>())
      return NullUnlessSharded(partitioned_input);

  return nullptr;
}

absl::Status DetermineShardingFromAlias(
    mlir::func::FuncOp func, OptionalOpShardingVector& input_shardings,
    OptionalOpShardingVector& output_shardings) {
  for (int arg_idx = 0; arg_idx < func.getNumArguments(); ++arg_idx) {
    if (auto v =
            func.getArgAttrOfType<mlir::IntegerAttr>(arg_idx, kAliasingAttr)) {
      if (int retval_idx = v.getInt();
          retval_idx >= 0 && retval_idx < func.getNumResults()) {
        auto& input_sharding = input_shardings[arg_idx];
        auto& output_sharding = output_shardings[retval_idx];

        if (input_sharding.has_value() && output_sharding.has_value() &&
            input_sharding.value() != output_sharding.value()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "arg#", arg_idx, " is aliased to retval#", retval_idx,
              " but their sharding configurations don't match."));
        } else if (input_sharding.has_value() && !output_sharding.has_value()) {
          output_sharding = input_sharding;
        } else if (!input_sharding.has_value() && output_sharding.has_value()) {
          input_sharding = output_sharding;
        }
      }
    }
  }

  return absl::OkStatus();
}

// Returns XLA sharding from XlaSharding op connected to a result value.
// XlaSharding op may be directly connected to output but it may also be
// followed by Identity or simple arithmetic ops. In case where bfloat16 type is
// used, we might see a Cast op.
//
// TODO(hongjunchoi): Add logic to parse XlaSharding op inside control flow (If,
// Case) ops and Caller argument values.
// TODO(hongjunchoi): Consider explicitly checking op patterns to detect sharded
// inputs.
absl::StatusOr<std::optional<StringRef>> GetXlaShardingFromRetval(
    Value value, const llvm::SmallVector<std::string>& logical_device_vec) {
  llvm::SmallPtrSet<Value, 4> visited_values;
  llvm::SmallVector<Value, 4> values_to_visit;
  values_to_visit.push_back(value);

  while (!values_to_visit.empty()) {
    Value value_to_visit = values_to_visit.pop_back_val();

    if (!visited_values.insert(value_to_visit).second) {
      continue;
    }

    Operation* def = value_to_visit.getDefiningOp();
    if (!def) {
      continue;
    }

    if (auto sharding = llvm::dyn_cast_or_null<mlir::TF::XlaShardingOp>(def))
      return GetXlaShardingFromShardingOp(sharding);

    if (auto sharding = def->getAttrOfType<StringAttr>(kShardingAttributeV2)) {
      return sharding.strref();
    }

    if (auto sharding = def->getAttrOfType<StringAttr>(kShardingAttribute)) {
      return sharding.strref();
    }

    if (auto logical_device = AssignLogicalDeviceFromTPUReplicatedCoreAttr(
            def, logical_device_vec)) {
      return logical_device;
    }

    if (UnaryOpHasTraitsForSharding(def) || BinaryOpHasTraitsForSharding(def)) {
      for (auto operand : def->getOperands()) {
        values_to_visit.push_back(operand);
      }
      continue;
    }

    if (auto call_op = llvm::dyn_cast_or_null<mlir::CallOpInterface>(def)) {
      mlir::func::FuncOp func =
          llvm::dyn_cast<mlir::func::FuncOp>(call_op.resolveCallable());
      if (!func) continue;
      value_to_visit = func.front().getTerminator()->getOperand(
          mlir::cast<OpResult>(value_to_visit).getResultNumber());
      values_to_visit.push_back(value_to_visit);
      continue;
    }

    if (auto while_op = llvm::dyn_cast<mlir::TF::WhileRegionOp>(def)) {
      if (auto op_result = mlir::cast<OpResult>(value_to_visit)) {
        int result_idx = op_result.getResultNumber();
        if (auto yield_op = llvm::dyn_cast<mlir::TF::YieldOp>(
                while_op.getBody().front().getTerminator())) {
          values_to_visit.push_back(yield_op.getOperand(result_idx));
        }
      }
      continue;
    }
  }

  return std::nullopt;
}

// Tries to extract sharding configurations for all outputs by parsing
// XlaSharding/ TPUPartitionedOutput op connected to the retvals/results.
absl::Status IdentifyXlaShardingForComputationOutputs(
    const llvm::SmallVector<std::string>& logical_device_vec,
    bool infer_from_computation, mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::func::FuncOp func, Builder* builder,
    OptionalOpShardingVector& sharding_for_rets) {
  Block& function_block = func.front();
  Operation* terminator = function_block.getTerminator();
  sharding_for_rets.reserve(terminator->getNumOperands());

  // Iterate through results of `cluster_func`. For output ops, look for
  // TPUPartitionedOutput ops.
  //
  // Iterate through operands of the terminator. If the preceding op is
  // XlaShardingOp, then the provided sharding configuration is added to the
  // tf_device.ClusterFunc as an attribute and the function as a result
  // attribute.
  for (auto result_and_retval :
       llvm::zip(cluster_func.getResults(), terminator->getOpOperands())) {
    Value result = std::get<0>(result_and_retval);
    OpOperand& retval = std::get<1>(result_and_retval);

    if (auto result_sharding = GetXlaShardingFromResult(result)) {
      sharding_for_rets.push_back(result_sharding);
      continue;
    }

    if (infer_from_computation) {
      TF_ASSIGN_OR_RETURN(
          auto retval_sharding,
          GetXlaShardingFromRetval(retval.get(), logical_device_vec));
      if (retval_sharding) {
        sharding_for_rets.push_back(retval_sharding.value());
        continue;
      }
    }

    sharding_for_rets.push_back(std::nullopt);
  }
  return absl::OkStatus();
}

void SetReplicatedOrMaximalShardingIfNoShardingFound(
    const llvm::SmallVector<std::string>& logical_device_vec, bool use_spmd,
    OptionalOpShardingVector& shardings) {
  for (auto& sharding : shardings) {
    if (sharding == std::nullopt) {
      // If we haven't found sharding, default to either replicated or maximal
      // sharding depending on whether XLA SPMD is enabled.
      if (use_spmd) {
        // If XLA SPMD is enabled, host variables or non-variable per-replica
        // inputs, and outputs should take on replicate sharding, so that every
        // device gets the whole tensor(s) (and can slice them up later eg.
        // using dynamic-slice).
        sharding = kReplicateSharding;
      } else {
        // Otherwise, default to maximal sharding core 0.
        sharding = logical_device_vec[0];
      }
    }
  }
}

// Moves shardings from `optional_shardings` to `shardings`.
absl::Status MoveSharding(OptionalOpShardingVector& optional_shardings,
                          OpShardingVector& shardings) {
  shardings.clear();
  for (auto& sharding : optional_shardings) {
    if (!sharding) {
      return absl::InternalError(
          "Couldn't find/assign sharding for an input/output. All shardings "
          "should have been identified by this point.");
    }

    shardings.push_back(std::move(sharding.value()));
  }

  return absl::OkStatus();
}

// Determines XlaSharding for inputs and outputs. If there are aliased
// inputs/outputs for which no sharding was found directly, the corresponding
// output/input sharding is used (if it exists). If we still don't find sharding
// for some inputs/outputs, we default to replicated or maximal sharding
// depending on `use_spmd`.
absl::Status IdentifyXlaShardingForInputsAndOutputs(
    const llvm::SmallVector<std::string>& logical_device_vec, bool use_spmd,
    bool infer_from_computation, mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::func::FuncOp func, Builder* builder, OpShardingVector& input_sharding,
    OpShardingVector& output_sharding) {
  OptionalOpShardingVector optional_input_sharding;
  OptionalOpShardingVector optional_output_sharding;
  TF_RETURN_IF_ERROR(IdentifyXlaShardingForComputationInputs(
      logical_device_vec, infer_from_computation, cluster_func, func, builder,
      optional_input_sharding));
  TF_RETURN_IF_ERROR(IdentifyXlaShardingForComputationOutputs(
      logical_device_vec, infer_from_computation, cluster_func, func, builder,
      optional_output_sharding));
  TF_RETURN_IF_ERROR(DetermineShardingFromAlias(func, optional_input_sharding,
                                                optional_output_sharding));
  SetReplicatedOrMaximalShardingIfNoShardingFound(logical_device_vec, use_spmd,
                                                  optional_input_sharding);
  SetReplicatedOrMaximalShardingIfNoShardingFound(logical_device_vec, use_spmd,
                                                  optional_output_sharding);
  TF_RETURN_IF_ERROR(MoveSharding(optional_input_sharding, input_sharding));
  TF_RETURN_IF_ERROR(MoveSharding(optional_output_sharding, output_sharding));

  return absl::OkStatus();
}

// Extracts input/output sharding configuration of `cluster_func` by parsing
// XlaSharding ops inside the `cluster_func`.
LogicalResult IdentifyXlaShardingForTPUComputation(
    Builder* builder, mlir::tf_device::ClusterFuncOp cluster_func) {
  // Look up function definition from module.
  mlir::func::FuncOp func =
      cluster_func->getParentOfType<ModuleOp>()
          .lookupSymbol<mlir::func::FuncOp>(cluster_func.getFunc());

  bool use_spmd = false;
  if (auto use_spmd_attr = cluster_func->getAttrOfType<BoolAttr>(kUseSpmdAttr))
    use_spmd = use_spmd_attr.getValue();

  auto num_cores_per_replica_attr =
      cluster_func->getAttrOfType<IntegerAttr>(kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr)
    return cluster_func.emitOpError(
        CreateMissingAttributeMsg(kNumCoresPerReplicaAttr));

  int num_cores_per_replica = num_cores_per_replica_attr.getInt();
  llvm::SmallVector<std::string> logical_device_vec(num_cores_per_replica);

  for (int idx = 0; idx < num_cores_per_replica; idx++) {
    logical_device_vec[idx] =
        xla::sharding_builder::AssignDevice(idx).SerializeAsString();
  }

  OpShardingVector sharding_for_args;
  OpShardingVector sharding_for_rets;
  if (auto status = IdentifyXlaShardingForInputsAndOutputs(
          logical_device_vec, use_spmd,
          /*infer_from_computation=*/true, cluster_func, func, builder,
          sharding_for_args, sharding_for_rets);
      !status.ok()) {
    return cluster_func.emitOpError(status.message());
  };

  auto has_maximal_sharding =
      [](const OpShardingVariant& sharding_or_op) -> bool {
    const auto sharding = GetShardingFromVariant(sharding_or_op);
    return sharding && sharding->type() == xla::OpSharding::MAXIMAL;
  };

  // XLA SPMD only supports cases where all inputs/outputs exist on every
  // partition (sharded or replicated). If any of the inputs/outputs have
  // maximal sharding, then fallback to MPMD. Also fall back if any of the
  // shardings aren't compatible with the rank of their tensor.
  if ((use_spmd && (absl::c_any_of(sharding_for_args, has_maximal_sharding) ||
                    absl::c_any_of(sharding_for_rets, has_maximal_sharding))) ||
      failed(VerifyShardings(func, sharding_for_args, sharding_for_rets))) {
    LOG(WARNING) << "XLA SPMD only supports cases where all inputs/outputs "
                    "exist on every partition (sharded or replicated). If any "
                    "of the inputs/outputs have maximal sharding, then "
                    "fallback to MPMD.";
    sharding_for_args.clear();
    sharding_for_rets.clear();
    cluster_func->setAttr(kUseSpmdAttr, builder->getBoolAttr(false));

    if (auto status = IdentifyXlaShardingForInputsAndOutputs(
            logical_device_vec, /*use_spmd=*/false,
            /*infer_from_computation=*/false, cluster_func, func, builder,
            sharding_for_args, sharding_for_rets);
        !status.ok()) {
      return cluster_func.emitOpError(status.message());
    }
  }

  // Update sharding on function arguments and returns.
  Block& function_block = func.front();
  for (auto sharding_and_arg :
       llvm::zip(sharding_for_args, function_block.getArguments())) {
    BlockArgument arg = std::get<1>(sharding_and_arg);
    const auto& sharding_or_op = std::get<0>(sharding_and_arg);
    if (auto sharding = GetShardingStringFromVariant(sharding_or_op)) {
      func.setArgAttr(arg.getArgNumber(), kShardingAttr,
                      builder->getStringAttr(*sharding));
    }
  }

  Operation* terminator = function_block.getTerminator();
  for (auto sharding_and_retval :
       llvm::zip(sharding_for_rets, terminator->getOpOperands())) {
    OpOperand& retval = std::get<1>(sharding_and_retval);
    const auto& sharding_or_op = std::get<0>(sharding_and_retval);
    if (auto sharding = GetShardingStringFromVariant(sharding_or_op)) {
      func.setResultAttr(retval.getOperandNumber(), kShardingAttr,
                         builder->getStringAttr(*sharding));
    }
  }

  // Update input/output sharding attributes on tf_device.cluster_func op.
  cluster_func->setAttr(tensorflow::kInputShardingAttr,
                        GetStrArrayAttr(builder, sharding_for_args));
  cluster_func->setAttr(tensorflow::kOutputShardingAttr,
                        GetStrArrayAttr(builder, sharding_for_rets));
  return mlir::success();
}

void TPUShardingIdentificationPass::runOnOperation() {
  Builder builder(getOperation().getContext());

  auto result =
      getOperation().walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
        if (failed(
                IdentifyXlaShardingForTPUComputation(&builder, cluster_func))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<ModuleOp>>
CreateTPUShardingIdentificationPass() {
  return std::make_unique<TPUShardingIdentificationPass>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
