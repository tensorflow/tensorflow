/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/translate/hlo_to_mhlo/hlo_function_importer.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/hlo/translate/hlo_to_mhlo/async_importer.h"
#include "xla/hlo/translate/hlo_to_mhlo/attribute_importer.h"
#include "xla/hlo/translate/hlo_to_mhlo/custom_call_importer.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/hlo/translate/hlo_to_mhlo/location_importer.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/primitive_util.h"
#include "xla/protobuf_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

using llvm::APInt;
using llvm::ArrayRef;
using mlir::DenseIntElementsAttr;
using mlir::NamedAttribute;
using mlir::Operation;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::Value;
using mlir::func::FuncOp;

#define DEBUG_TYPE "xla-translate"

namespace xla {

namespace {

constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kParameterReplicationAttr[] = "mhlo.parameter_replication";

// Note: This sanitization function causes an irreversible many-to-one mapping
// and any solution to mitigate this would cause issues with the reverse
// direction. Long-term solution is to add a function attribute to maintain the
// original HLO naming.
std::string SanitizeFunctionName(llvm::StringRef name) {
  std::string output(name);
  llvm::for_each(output, [](char& x) { x = x == '-' ? '_' : x; });
  return output;
}

// Returns whether the instruction is a default dot operation.
// Supports vector.vector, vector.matrix, matrix.vector, and matrix.matrix.
// Default operations have lhs_contracting dimension is 1 (or zero for vector)
// and the rhs_contracting dimension is zero, and there are no batch dimensions.
bool DotIsDefault(const HloInstruction* instruction) {
  // If LHS/RHS has rank greater than 2, not default dot
  const auto& operands = instruction->operands();
  if (operands[0]->shape().dimensions().size() > 2 ||
      operands[1]->shape().dimensions().size() > 2) {
    return false;
  }

  auto dnums = instruction->dot_dimension_numbers();
  DotDimensionNumbers default_dimension_numbers;
  default_dimension_numbers.add_lhs_contracting_dimensions(
      instruction->operand(0)->shape().dimensions().size() == 1 ? 0 : 1);
  default_dimension_numbers.add_rhs_contracting_dimensions(0);
  return protobuf_util::HaveSameSerialization(dnums, default_dimension_numbers);
}

ArrayRef<HloSharding> FlattenTupleSharding(const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    return sharding.tuple_elements();
  }
  return sharding;
}

// Returns true if changed.
bool FoldGetTupleElementOfTuple(mlir::stablehlo::GetTupleElementOp op) {
  if (auto tupleOp =
          op->getOperand(0).getDefiningOp<mlir::stablehlo::TupleOp>()) {
    int64_t idx = op.getIndex();
    llvm::SmallVector<Value> new_operand{tupleOp.getOperand(idx)};
    op->replaceAllUsesWith(new_operand);
    op->erase();
    return true;
  }
  return false;
}

// Clean up the GetTupleElementOp, created during the flattening of
// tuple arguments and return values, if eligible for folding. Removal of
// get-tuple-element can transitively make the defining TupleOp dead to be
// removed subsequently.
void CleanUpTupleOps(mlir::Block* block, mlir::OpBuilder* builder) {
  bool changed = true;
  llvm::SmallVector<Value> folded_results;

  while (changed) {
    changed = false;
    for (Operation& op : llvm::make_early_inc_range(block->getOperations())) {
      if (auto get_tuple_op =
              llvm::dyn_cast<mlir::stablehlo::GetTupleElementOp>(op)) {
        changed = FoldGetTupleElementOfTuple(get_tuple_op);
      } else if (llvm::isa<mlir::stablehlo::TupleOp>(op) &&
                 mlir::isOpTriviallyDead(&op)) {
        op.erase();
        changed = true;
      }
    }
  }
}

llvm::ArrayRef<int64_t> ToArrayRef(absl::Span<const int64_t> span) {
  return llvm::ArrayRef<int64_t>(span.data(), span.size());
}

Operation* CreateReturnOp(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::ValueRange operands,
                          mlir::Dialect* parent_dialect) {
  LLVM_DEBUG(llvm::dbgs() << "CreateReturnOp: "
                          << parent_dialect->getNamespace() << '\n');
  if (llvm::isa<mlir::mhlo::MhloDialect>(parent_dialect)) {
    // Potentially unused, but if future MHLO ops have bodies, will be needed.
    return builder.create<mlir::mhlo::ReturnOp>(loc, operands);
  }
  if (llvm::isa<mlir::stablehlo::StablehloDialect>(parent_dialect)) {
    return builder.create<mlir::stablehlo::ReturnOp>(loc, operands);
  }
  return builder.create<mlir::func::ReturnOp>(loc, operands);
}

Operation* WrapInTuple(mlir::OpBuilder* builder, Operation* op) {
  LLVM_DEBUG(llvm::dbgs() << "WrapInTuple: " << op->getName()
                          << op->getResultTypes() << '\n');
  return builder->create<mlir::stablehlo::TupleOp>(op->getLoc(),
                                                   op->getResults());
}

Operation* GetTupleElementOp(mlir::OpBuilder* builder, Value value,
                             int64_t index,
                             llvm::SmallVector<NamedAttribute>&& attributes) {
  attributes.push_back(
      builder->getNamedAttr("index", builder->getI32IntegerAttr(index)));
  return builder->create<mlir::stablehlo::GetTupleElementOp>(
      value.getLoc(), value, builder->getI32IntegerAttr(index));
}

// Creates an array of zeros like the given MLIR type, if type has bounded
// dynamism, the constant is padded and set to the dimenison size of the
// operand.
//
// Example ZeroLike([<=5] operand):
// %c = constant dense<0> : tensor<5xf32>
// %0 = get_dimension_size %operand
// %1 = set_dimension_size %c, %0, bounded_dim={0}
//
// Note: Currently this only supports a single bounded dimension.
absl::StatusOr<mlir::Value> createConstantZeroLike(mlir::Value operand,
                                                   Shape input_shape,
                                                   mlir::OpBuilder* builder,
                                                   mlir::Location loc) {
  TF_ASSIGN_OR_RETURN(
      mlir::RankedTensorType type,
      ConvertTensorShapeToType<mlir::RankedTensorType>(input_shape, *builder));

  LLVM_DEBUG(llvm::dbgs() << "CreateConstantZeroLike: " << operand << ", "
                          << type << '\n');
  if (type.hasStaticShape())
    return builder
        ->create<mlir::stablehlo::ConstantOp>(loc, builder->getZeroAttr(type))
        ->getResult(0);

  // Note: Currently this only supports a single bounded dimension.
  if (!mlir::hlo::hasSingleBoundedDimension(type))
    return Internal(
        "Currently HLO to MHLO only supports a single bounded dimension.");

  auto bounded_dim = std::distance(type.getShape().begin(),
                                   llvm::find_if(type.getShape(), [](auto dim) {
                                     return mlir::ShapedType::isDynamic(dim);
                                   }));

  // Create a constant with no bounded dynamism, drop tensor encoding.
  ArrayRef<int64_t> padded_dims(input_shape.dimensions().begin(),
                                input_shape.dimensions().end());
  auto padded_type =
      mlir::RankedTensorType::get(padded_dims, type.getElementType());
  auto padded_constant = builder->create<mlir::stablehlo::ConstantOp>(
      loc, builder->getZeroAttr(padded_type));

  // Get or Set the dimensions size based on the operand type.
  auto dim_size = builder->create<mlir::stablehlo::GetDimensionSizeOp>(
      loc, operand, builder->getI64IntegerAttr(bounded_dim));
  std::vector<mlir::Value> operands = {padded_constant->getResult(0), dim_size};
  std::vector<mlir::NamedAttribute> attributes{builder->getNamedAttr(
      "dimension", builder->getI64IntegerAttr(bounded_dim))};
  return builder->create<mlir::stablehlo::SetDimensionSizeOp>(
      loc, type, operands, attributes);
}

}  // namespace

void HloFunctionImporter::ReplaceBlockArgumentsWithImplicitOperands(
    mlir::Operation* op, llvm::ArrayRef<mlir::Value> implicit_operands) {
  assert((mlir::dyn_cast<mlir::stablehlo::IfOp>(*op) ||
          mlir::dyn_cast<mlir::stablehlo::CaseOp>(*op)) &&
         "Unexpected mlir op in "
         "HloFunctionImporter::ReplaceBlockArgumentsWithImplicitOperands!");

  int implicit_operand_index = 0;
  for (auto& region : op->getRegions()) {
    for (auto arg : region.getArguments()) {
      assert(implicit_operand_index < implicit_operands.size());
      arg.replaceAllUsesWith(implicit_operands[implicit_operand_index++]);
    }
    region.front().eraseArguments(0, region.getNumArguments());
  }
}

static bool IsNestedTupleInData(Type type) {
  auto tuple_type = mlir::dyn_cast<mlir::TupleType>(type);
  if (!tuple_type) return false;

  assert(llvm::isa<mlir::stablehlo::TokenType>(tuple_type.getType(1)) &&
         "Infeed: Non token type");
  auto data_type = tuple_type.getType(0);

  auto data_tuple_type = mlir::dyn_cast<mlir::TupleType>(data_type);
  if (!data_tuple_type) return false;

  for (auto child_type : data_tuple_type.getTypes()) {
    if (mlir::isa<mlir::TupleType>(child_type)) return true;
  }

  return false;
}

mlir::Attribute GetFrontendAttributes(mlir::Builder& b,
                                      const FrontendAttributes& attributes) {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.reserve(attributes.map_size());
  for (const auto& [k, v] : attributes.map()) {
    attrs.push_back(b.getNamedAttr(k, b.getStringAttr(v)));
  }
  return b.getDictionaryAttr(attrs);
}

void HloFunctionImporter::FlattenTupleType(
    Type type, llvm::SmallVectorImpl<Type>& flattened_types) {
  auto tuple_type = mlir::dyn_cast<mlir::TupleType>(type);
  if (!tuple_type) {
    flattened_types.push_back(type);
    return;
  }

  for (auto child_type : tuple_type.getTypes()) {
    FlattenTupleType(child_type, flattened_types);
  }
}

void HloFunctionImporter::FlattenTupleValue(
    mlir::OpBuilder* func_builder, mlir::Location loc, Value value,
    llvm::SmallVectorImpl<Value>& flattened_values) {
  auto tuple_type = llvm::dyn_cast<mlir::TupleType>(value.getType());
  if (!tuple_type) {
    flattened_values.push_back(value);
    return;
  }

  for (int64_t flattenIdx = 0; flattenIdx < tuple_type.size(); ++flattenIdx) {
    auto sub_value = GetTupleElementOp(func_builder, value, flattenIdx, {});
    FlattenTupleValue(func_builder, loc, sub_value->getResult(0),
                      flattened_values);
  }
}

llvm::SmallVector<Value> HloFunctionImporter::FlattenTupleValues(
    mlir::OpBuilder* func_builder, mlir::Location loc, mlir::ValueRange values,
    std::optional<int> reserve_size) {
  llvm::SmallVector<Value> flattened_values;
  if (reserve_size) {
    flattened_values.reserve(*reserve_size);
  }
  for (Value value : values) {
    FlattenTupleValue(func_builder, loc, value, flattened_values);
  }
  return flattened_values;
}

absl::StatusOr<mlir::func::FuncOp> HloFunctionImporter::ImportAsFunc(
    const HloComputation& computation, mlir::SymbolTable& symbol_table,
    std::unordered_map<const HloComputation*, FuncOp>* function_map,
    mlir::Builder* builder, bool is_main,
    bool flatten_computation_args_result) {
  HloFunctionImporter importer(symbol_table, function_map, builder,
                               flatten_computation_args_result);
  return importer.ImportAsFunc(computation, is_main);
}

absl::Status HloFunctionImporter::ImportAsRegion(
    const HloComputation& computation, mlir::SymbolTable& symbol_table,
    mlir::Region* region, mlir::Builder* builder,
    bool flatten_computation_args_result) {
  HloFunctionImporter importer(symbol_table, {}, builder,
                               flatten_computation_args_result);
  return importer.ImportAsRegion(computation, region);
}

absl::StatusOr<FuncOp> HloFunctionImporter::ImportAsFunc(
    const HloComputation& computation, bool is_main) {
  std::string computation_name =
      is_main ? "main" : SanitizeFunctionName(ToStringRef(computation.name()));

  FuncOp* imported(nullptr);
  if (function_map_) {
    imported = &((*function_map_)[&computation]);
    if (*imported) {
      return *imported;
    }
  }

  llvm::SmallVector<Type, 4> args;
  TF_RETURN_IF_ERROR(GetMlirTypes(computation.parameter_instructions(), &args));
  TF_ASSIGN_OR_RETURN(Type retType,
                      ConvertShapeToType<RankedTensorType>(
                          computation.root_instruction()->shape(), *builder_));

  mlir::FunctionType func_type;
  if (flatten_computation_args_result_) {
    llvm::SmallVector<Type> flattened_args;
    for (Type type : args) {
      FlattenTupleType(type, flattened_args);
    }
    llvm::SmallVector<Type> flattened_rets;
    FlattenTupleType(retType, flattened_rets);
    func_type =
        mlir::FunctionType::get(context_, flattened_args, flattened_rets);
  } else {
    func_type = mlir::FunctionType::get(context_, args, retType);
  }

  // Construct the MLIR function and map arguments.
  llvm::ArrayRef<mlir::NamedAttribute> attrs;
  auto function = FuncOp::create(mlir::UnknownLoc::get(context_),
                                 computation_name, func_type, attrs);
  function.setVisibility(is_main ? FuncOp::Visibility::Public
                                 : FuncOp::Visibility::Private);

  int arg_index = 0;
  // Create block so that the function has arguments we can attach
  // `mlir::Location`s to.
  mlir::Block* block = function.addEntryBlock();
  for (auto instruction : computation.parameter_instructions()) {
    HloParameterInstruction* parameter =
        Cast<HloParameterInstruction>(instruction);
    mlir::Attribute frontend_attributes =
        parameter->frontend_attributes().map_size() > 0
            ? GetFrontendAttributes(*builder_, parameter->frontend_attributes())
            : mlir::Attribute();

    if (flatten_computation_args_result_) {
      int64_t leaf_count = ShapeUtil::GetLeafCount(parameter->shape());
      ArrayRef<HloSharding> flattened_shardings =
          parameter->has_sharding()
              ? FlattenTupleSharding(parameter->sharding())
              : ArrayRef<HloSharding>();
      if (!flattened_shardings.empty() &&
          leaf_count != flattened_shardings.size()) {
        return Internal("Expected %d leaf parameters but got %d",
                        flattened_shardings.size(), leaf_count);
      }

      for (int i = 0; i < leaf_count; ++i) {
        if (!flattened_shardings.empty()) {
          function.setArgAttr(
              arg_index, kShardingAttr,
              ConvertSharding(flattened_shardings[i], builder_));
        }
        if (frontend_attributes) {
          if (leaf_count > 1) {
            return InvalidArgument(
                "A tuple parameter that is being flattened shouldn't have "
                "frontend attributes");
          }
          function.setArgAttr(arg_index, kFrontendAttributesAttr,
                              frontend_attributes);
        }
        if (parameter->parameter_replicated_at_leaf_buffers() &&
            parameter->parameter_replicated_at_leaf_buffers()->at(i)) {
          function.setArgAttr(arg_index, kParameterReplicationAttr,
                              builder_->getBoolArrayAttr({true}));
        }
        // NOTE: since we are flattening args, all arguments will share the same
        // location as the tuple parameter instruction.
        function.getArgument(arg_index).setLoc(
            mlir::hlo::GenerateInstructionLocation(instruction, context_));
        ++arg_index;
      }
    } else {
      if (parameter->has_sharding()) {
        function.setArgAttr(arg_index, kShardingAttr,
                            ConvertSharding(parameter->sharding(), builder_));
      }
      if (frontend_attributes) {
        function.setArgAttr(
            arg_index, kFrontendAttributesAttr,
            GetFrontendAttributes(*builder_, parameter->frontend_attributes()));
      }
      if (parameter->parameter_replicated_at_leaf_buffers().has_value()) {
        bool nontrival = false;
        llvm::SmallVector<bool> replicated_at_leaf_buffers;
        for (auto b :
             parameter->parameter_replicated_at_leaf_buffers().value()) {
          replicated_at_leaf_buffers.push_back(b);
          nontrival = nontrival || b;
        }
        if (nontrival) {
          function.setArgAttr(
              arg_index, kParameterReplicationAttr,
              builder_->getBoolArrayAttr(replicated_at_leaf_buffers));
        }
      }
      function.getArgument(arg_index).setLoc(
          mlir::hlo::GenerateInstructionLocation(instruction, context_));
      ++arg_index;
    }
  }
  // TODO(b/260756663): Token sharding is unverified, legacy users provide
  // multiple sharding values for a single token output.
  bool is_token = computation.root_instruction()->shape().IsToken();
  if (is_token && computation.root_instruction()->has_sharding()) {
    function.setResultAttr(
        0, kShardingAttr,
        ConvertSharding(computation.root_instruction()->sharding(), builder_));
  }
  if (!is_token && computation.root_instruction()->has_sharding()) {
    ArrayRef<HloSharding> ret_shardings =
        computation.root_instruction()->sharding();
    if (flatten_computation_args_result_) {
      ret_shardings = FlattenTupleSharding(ret_shardings.front());
    }
    if (function.getNumResults() != ret_shardings.size()) {
      return Internal("Expected %d results but got %d", ret_shardings.size(),
                      function.getNumResults());
    }
    for (const auto& [ret_index, ret_sharding] :
         llvm::enumerate(ret_shardings)) {
      function.setResultAttr(ret_index, kShardingAttr,
                             ConvertSharding(ret_sharding, builder_));
    }
  }
  if (computation.execution_thread() != "main") {
    function->setAttr(
        "execution_thread",
        builder_->getStringAttr(ToStringRef(computation.execution_thread())));
  }

  symbol_table_.insert(function);

  // Add to the map right away for function calls if map is set.
  if (imported) {
    *imported = function;
  }

  TF_RETURN_IF_ERROR(ImportInstructions(computation, block));

  return function;
}

absl::Status HloFunctionImporter::ImportAsRegion(
    const HloComputation& computation, mlir::Region* region) {
  auto loc = region->getLoc();
  auto* block = new mlir::Block;
  region->push_back(block);

  llvm::SmallVector<Type, 4> args;
  TF_RETURN_IF_ERROR(GetMlirTypes(computation.parameter_instructions(), &args));

  // Flatten the tuple-typed arguments.
  if (!llvm::isa<FuncOp>(region->getParentOp()) ||
      flatten_computation_args_result_) {
    for (auto arg : args) {
      llvm::SmallVector<Type> flattened_arg_types;
      FlattenTupleType(arg, flattened_arg_types);
      block->addArguments(
          flattened_arg_types,
          mlir::SmallVector<mlir::Location>(flattened_arg_types.size(), loc));
    }
  } else {
    block->addArguments(args,
                        mlir::SmallVector<mlir::Location>(args.size(), loc));
  }

  return ImportInstructions(computation, block);
}

absl::StatusOr<Value> HloFunctionImporter::ImportInstructionsImpl(
    const HloComputation& computation,
    const llvm::SmallVectorImpl<Value>& arguments, mlir::OpBuilder* builder) {
  // Setup the input parameters.
  const int num_parameters = computation.num_parameters();

  for (int i = 0; i < num_parameters; i++) {
    auto hlo_parameter = computation.parameter_instruction(i);
    instruction_value_map_[hlo_parameter] = arguments[i];
  }

  for (auto instruction : computation.MakeInstructionPostOrder()) {
    TF_ASSIGN_OR_RETURN(auto operands, GetOperands(instruction));
    TF_ASSIGN_OR_RETURN(
        auto new_operation,
        ImportInstructionWithLayout(instruction, operands, builder));
    if (new_operation) {
      unsigned int idx =
          (instruction->opcode() == HloOpcode::kRngBitGenerator &&
           instruction->shape().IsArray())
              ? 1
              : 0;
      instruction_value_map_[instruction] = new_operation->getResult(idx);
    }
  }

  // Setup the return type (HLO only supports a single return value).
  return GetMlirValue(computation.root_instruction());
}

absl::Status HloFunctionImporter::ImportInstructions(
    const HloComputation& computation, mlir::Block* block) {
  llvm::SmallVector<Value, 4> arguments(block->args_begin(), block->args_end());
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockEnd(block);

  mlir::Location loc = builder.getUnknownLoc();

  bool is_func = llvm::isa<FuncOp>(block->getParentOp());
  bool flatten_region_arg_tuple = !is_func || flatten_computation_args_result_;
  Value result;
  if (flatten_region_arg_tuple) {
    // 'effective_arguments' stores the mhlo value corresponding to each
    // computation parameter. The value could be a BlockArgument, if the
    // corresponding computation parameter is non-tuple typed, or a TupleOp,
    // otherwise.
    llvm::SmallVector<Value> effective_arguments;

    llvm::SmallVector<Type> computation_arg_types;
    TF_RETURN_IF_ERROR(GetMlirTypes(computation.parameter_instructions(),
                                    &computation_arg_types));
    int flatten_idx = 0;
    for (Type computation_arg_type : computation_arg_types) {
      auto orig_tuple_arg_type =
          mlir::dyn_cast<mlir::TupleType>(computation_arg_type);

      // If the computation-parameter type is non-tuple, no action is needed.
      if (!orig_tuple_arg_type) {
        effective_arguments.push_back(arguments[flatten_idx]);
        flatten_idx++;
        continue;
      }

      // For each tuple-typed computation parameter, create a TupleOp
      // value in the region body, using the already flattened values in
      // 'arguments'. For example: With computation parameters: [tuple<T1>,
      // tuple<T2, T4>] We have, 'arguments' = [T1 arg1, T2 arg2, T3 arg3] and
      // we need to create two tuples tuples, one using arg1, and the other
      // using arg2 and arg3.
      llvm::SmallVector<Type> flattened_arg_type;
      FlattenTupleType(orig_tuple_arg_type, flattened_arg_type);

      mlir::ValueRange sub_args(llvm::ArrayRef<Value>(
          arguments.begin() + flatten_idx,
          arguments.begin() + flatten_idx + flattened_arg_type.size()));

      auto tupleVal =
          CreateTupleValue(&builder, loc, sub_args, orig_tuple_arg_type);
      effective_arguments.push_back(tupleVal);

      flatten_idx += flattened_arg_type.size();
    }

    TF_ASSIGN_OR_RETURN(
        result,
        ImportInstructionsImpl(computation, effective_arguments, &builder));
  } else {
    TF_ASSIGN_OR_RETURN(
        result, ImportInstructionsImpl(computation, arguments, &builder));
  }

  // Create terminator op depending on the parent op of this region.
  if (flatten_region_arg_tuple) {
    // Flatten tuples in results of this region.
    llvm::SmallVector<Value> flattened_return_operands;
    FlattenTupleValue(&builder, loc, result, flattened_return_operands);
    CreateReturnOp(builder, loc, flattened_return_operands,
                   block->getParentOp()->getDialect());
  } else {
    CreateReturnOp(builder, loc, result, block->getParentOp()->getDialect());
  }

  CleanUpTupleOps(block, &builder);

  return absl::OkStatus();
}

absl::StatusOr<Value> HloFunctionImporter::ImportInstructions(
    const HloComputation& computation,
    const llvm::SmallVectorImpl<Value>& arguments,
    mlir::SymbolTable& symbol_table, mlir::OpBuilder* builder,
    bool flatten_computation_args_result) {
  mlir::Block* block = builder->getBlock();
  if (block == nullptr)
    return InvalidArgument(
        "ImportInstructions requires a valid block in the builder");

  HloFunctionImporter importer(symbol_table, {}, builder,
                               flatten_computation_args_result);
  return importer.ImportInstructionsImpl(computation, arguments, builder);
}

absl::StatusOr<mlir::Operation*> HloFunctionImporter::ImportInstruction(
    const HloInstruction* instr,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    mlir::SymbolTable& symbol_table, mlir::OpBuilder* builder,
    bool flatten_computation_args_result, DynamicShapeHandlingMode mode) {
  mlir::Block* block = builder->getBlock();
  if (block == nullptr)
    return InvalidArgument(
        "ImportInstructions requires a valid block in the builder");

  HloFunctionImporter importer(symbol_table, {}, builder,
                               flatten_computation_args_result);
  return importer.ImportInstructionWithLayout(instr, operands, builder, mode);
}

absl::StatusOr<mlir::Operation*> HloFunctionImporter::ImportInstructionImpl(
    const HloInstruction* instruction,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    mlir::OpBuilder* func_builder, DynamicShapeHandlingMode mode) {
  const Shape& instruction_shape = instruction->shape();
  const Shape& shape = mode == DynamicShapeHandlingMode::kConvertToStatic
                           ? ShapeUtil::MakeStaticShape(instruction_shape)
                           : instruction_shape;
  TF_ASSIGN_OR_RETURN(auto result_type,
                      ConvertShapeToType<RankedTensorType>(shape, *builder_));
  mlir::Location loc = mlir::hlo::GenerateInstructionLocation(
      instruction, func_builder->getContext());

  llvm::SmallVector<NamedAttribute, 10> attributes;
  if (instruction->has_sharding()) {
    attributes.push_back(builder_->getNamedAttr(
        kShardingAttr, ConvertSharding(instruction->sharding(), builder_)));
  }

  llvm::SmallVector<NamedAttribute, 4> frontend_attributes;
  for (const auto& [k, v] : instruction->frontend_attributes().map()) {
    frontend_attributes.push_back(
        builder_->getNamedAttr(k, builder_->getStringAttr(v)));
  }

  int frontend_attributes_index = 0;
  if (!frontend_attributes.empty()) {
    frontend_attributes_index = attributes.size();
    attributes.push_back(builder_->getNamedAttr(
        kFrontendAttributesAttr,
        builder_->getDictionaryAttr(frontend_attributes)));
  }

  switch (instruction->opcode()) {
    case HloOpcode::kParameter: {
      return nullptr;
    }
    case HloOpcode::kConstant: {
      auto constant = Cast<HloConstantInstruction>(instruction);
      TF_RET_CHECK(constant->HasLiteral())
          << "HloConstantInstruction " << instruction->name()
          << " has no literal set";

      const Literal& literal = constant->literal();
      auto attr = CreateDenseElementsAttrFromLiteral(literal, *builder_);
      if (!attr.ok()) return attr.status();
      mlir::Operation* new_operation =
          func_builder->create<mlir::stablehlo::ConstantOp>(loc, attr.value());
      for (auto attr : attributes) {
        new_operation->setAttr(attr.getName(), attr.getValue());
      }
      return new_operation;
    }
    case HloOpcode::kIota: {
      return func_builder
          ->create<mlir::stablehlo::IotaOp>(
              loc, result_type,
              func_builder->getI64IntegerAttr(
                  Cast<HloIotaInstruction>(instruction)->iota_dimension()))
          .getOperation();
    }
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone: {
      auto async_op = Cast<HloAsyncInstruction>(instruction);
      auto called_computation = async_op->async_wrapped_computation();
      TF_ASSIGN_OR_RETURN(FuncOp function,
                          ImportAsFunc(*called_computation, /*is_main=*/false));
      attributes.push_back(builder_->getNamedAttr(
          "called_computation",
          mlir::FlatSymbolRefAttr::get(builder_->getContext(),
                                       function.getName())));
      auto execution_thread = ToStringRef(async_op->async_execution_thread());
      attributes.push_back(builder_->getNamedAttr(
          "execution_thread", builder_->getStringAttr(execution_thread)));
      function->setAttr("execution_thread",
                        builder_->getStringAttr(execution_thread));

      if (instruction->opcode() == HloOpcode::kAsyncStart) {
        auto bundle_result_type = mlir::mhlo::AsyncBundleType::get(
            context_, llvm::cast<mlir::TupleType>(result_type).getTypes());
        // XLA Feature -- MHLO Only
        return func_builder
            ->create<mlir::mhlo::AsyncStartOp>(loc, bundle_result_type,
                                               operands, attributes)
            .getOperation();
      } else if (instruction->opcode() == HloOpcode::kAsyncUpdate) {
        auto bundle_result_type = mlir::mhlo::AsyncBundleType::get(
            context_, llvm::cast<mlir::TupleType>(result_type).getTypes());
        // XLA Feature -- MHLO Only
        return func_builder
            ->create<mlir::mhlo::AsyncUpdateOp>(loc, bundle_result_type,
                                                operands, attributes)
            .getOperation();
      } else {
        assert(instruction->opcode() == HloOpcode::kAsyncDone);
        // XLA Feature -- MHLO Only
        return func_builder
            ->create<mlir::mhlo::AsyncDoneOp>(loc, result_type, operands,
                                              attributes)
            .getOperation();
      }
    }
    case HloOpcode::kBroadcast: {
      // Note that the HLO broadcast is more powerful than the XLA broadcast
      // op. BroadcastInDim offers a superset of the HLO op's functionality.
      attributes.push_back(builder_->getNamedAttr(
          "broadcast_dimensions",
          ConvertArray(ToArrayRef(instruction->dimensions()))));
      return func_builder
          ->create<mlir::stablehlo::BroadcastInDimOp>(loc, result_type,
                                                      operands, attributes)
          .getOperation();
    }

    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
      attributes.push_back(builder_->getNamedAttr(
          "epsilon", builder_->getF32FloatAttr(instruction->epsilon())));
      attributes.push_back(builder_->getNamedAttr(
          "feature_index",
          builder_->getI64IntegerAttr(instruction->feature_index())));
      if (instruction->opcode() == HloOpcode::kBatchNormGrad) {
        // Flatten the return type if they are tuple-typed.
        llvm::SmallVector<Type> flattened_ret_types;
        FlattenTupleType(result_type, flattened_ret_types);

        auto op = func_builder
                      ->create<mlir::stablehlo::BatchNormGradOp>(
                          loc, flattened_ret_types, operands, attributes)
                      .getOperation();

        return CreateTupleFromOpResults(func_builder, loc, op, result_type);
      } else if (instruction->opcode() == HloOpcode::kBatchNormInference) {
        return func_builder
            ->create<mlir::stablehlo::BatchNormInferenceOp>(
                loc, result_type, operands, attributes)
            .getOperation();
      } else {
        assert(instruction->opcode() == HloOpcode::kBatchNormTraining);

        // Flatten the return type if they are tuple-typed.
        llvm::SmallVector<Type> flattened_ret_types;
        FlattenTupleType(result_type, flattened_ret_types);

        auto op = func_builder
                      ->create<mlir::stablehlo::BatchNormTrainingOp>(
                          loc, flattened_ret_types, operands, attributes)
                      .getOperation();

        return CreateTupleFromOpResults(func_builder, loc, op, result_type);
      }

    case HloOpcode::kDot: {
      auto dot = Cast<HloDotInstruction>(instruction);
      if (dot->sparse_operands()) {
        attributes.push_back(builder_->getNamedAttr(
            "precision_config",
            ConvertPrecisionConfig(&instruction->precision_config(),
                                   builder_)));
        attributes.push_back(builder_->getNamedAttr(
            "dot_dimension_numbers",
            ConvertDotDimensionNumbers(instruction->dot_dimension_numbers(),
                                       builder_)));
        for (const SparsityDescriptor& descriptor : dot->sparsity()) {
          TF_ASSIGN_OR_RETURN(auto sparsity,
                              ConvertSparsityDescriptor(descriptor, builder_));
          attributes.push_back(builder_->getNamedAttr(
              descriptor.index() == 0 ? "lhs_sparsity" : "rhs_sparsity",
              sparsity));
        }
        // XLA Feature -- MHLO Only
        return func_builder
            ->create<mlir::mhlo::SparseDotOp>(loc, result_type, operands,
                                              attributes)
            .getOperation();
      }

      // Dot or DotGeneral
      attributes.push_back(builder_->getNamedAttr(
          "precision_config", stablehlo::ConvertPrecisionConfig(
                                  &instruction->precision_config(), builder_)));
      if (instruction->precision_config().algorithm() !=
          PrecisionConfig::ALG_UNSET) {
        attributes.push_back(builder_->getNamedAttr(
            "algorithm",
            stablehlo::ConvertDotAlgorithm(
                instruction->precision_config().algorithm(), builder_)));
      }
      // Consider consolidating DotOps together.
      if (DotIsDefault(instruction) && !dot->sparse_operands()) {
        return func_builder
            ->create<mlir::stablehlo::DotOp>(loc, result_type, operands,
                                             attributes)
            .getOperation();
      }

      attributes.push_back(builder_->getNamedAttr(
          "dot_dimension_numbers",
          stablehlo::ConvertDotDimensionNumbers(
              instruction->dot_dimension_numbers(), builder_)));
      return func_builder
          ->create<mlir::stablehlo::DotGeneralOp>(loc, result_type, operands,
                                                  attributes)
          .getOperation();
    }
    case HloOpcode::kRaggedAllToAll: {
      llvm::SmallVector<NamedAttribute> backendConfigAttributes;
      backendConfigAttributes.push_back(
          ConvertReplicaGroups(instruction->replica_groups(), builder_));
      if (instruction->channel_id().has_value()) {
        backendConfigAttributes.push_back(builder_->getNamedAttr(
            "channel_id",
            builder_->getI64IntegerAttr(instruction->channel_id().value())));
      }
      attributes.push_back(builder_->getNamedAttr(
          "backend_config",
          builder_->getDictionaryAttr(backendConfigAttributes)));
      attributes.push_back(builder_->getNamedAttr(
          "call_target_name", builder_->getStringAttr("ragged_all_to_all")));
      attributes.push_back(builder_->getNamedAttr(
          "api_version",
          mlir::stablehlo::CustomCallApiVersionAttr::get(
              builder_->getContext(),
              mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI)));
      return func_builder
          ->create<mlir::stablehlo::CustomCallOp>(loc, result_type, operands,
                                                  attributes)
          .getOperation();
    }
    case HloOpcode::kRaggedDot: {
      attributes.push_back(builder_->getNamedAttr(
          "precision_config",
          ConvertPrecisionConfig(&instruction->precision_config(), builder_)));
      attributes.push_back(builder_->getNamedAttr(
          "ragged_dot_dimension_numbers",
          ConvertRaggedDotDimensionNumbers(
              instruction->ragged_dot_dimension_numbers(), builder_)));
      // XLA Feature -- MHLO Only
      return func_builder
          ->create<mlir::mhlo::RaggedDotOp>(loc, result_type, operands,
                                            attributes)
          .getOperation();
    }
    case HloOpcode::kCall: {
      TF_ASSIGN_OR_RETURN(
          FuncOp function,
          ImportAsFunc(*instruction->to_apply(), /*is_main=*/false));
      mlir::Operation* new_operation;
      if (instruction->is_composite()) {
        // TODO: b/354721812 -  Support flatten_computation_args_result_ flag
        // for composite calls

        mlir::DictionaryAttr frontend_attributes_attr =
            builder_->getDictionaryAttr(frontend_attributes);
        if (frontend_attributes.empty() ||
            !frontend_attributes_attr.contains("composite.attributes") ||
            !frontend_attributes_attr.contains("composite.name") ||
            !frontend_attributes_attr.contains("composite.version")) {
          return InvalidArgument(
              "A composite call op must have frontend attributes with the "
              "following keys: composite.attributes, composite.name, "
              "composite.version");
        }

        llvm::SmallVector<NamedAttribute, 4> fe_attrs_without_composite_attrs;
        for (const auto& attr : frontend_attributes) {
          if (attr.getName() != "composite.attributes" &&
              attr.getName() != "composite.name" &&
              attr.getName() != "composite.version") {
            fe_attrs_without_composite_attrs.push_back(attr);
          }
        }

        // Frontend attributes may have been created by composite related
        // attributes. If frontend attributes is empty after removing
        // composite related attributes, it is not needed, so we remove it
        // entirely. Otherwise, we update it.
        if (fe_attrs_without_composite_attrs.empty()) {
          attributes.erase(attributes.begin() + frontend_attributes_index);
        } else {
          attributes[frontend_attributes_index] = builder_->getNamedAttr(
              kFrontendAttributesAttr,
              builder_->getDictionaryAttr(fe_attrs_without_composite_attrs));
        }

        auto frontend_attributes_map = instruction->frontend_attributes().map();
        mlir::StringAttr name = builder_->getStringAttr(
            frontend_attributes_map.find("composite.name")->second);
        mlir::Attribute composite_attributes = mlir::parseAttribute(
            frontend_attributes_map.find("composite.attributes")->second,
            builder_->getContext());
        mlir::FlatSymbolRefAttr decomposition = mlir::SymbolRefAttr::get(
            builder_->getContext(),
            ToStringRef(instruction->to_apply()->name()));
        mlir::IntegerAttr version = builder_->getIntegerAttr(
            builder_->getI32Type(),
            std::stoi(
                frontend_attributes_map.find("composite.version")->second));

        new_operation = func_builder->create<mlir::stablehlo::CompositeOp>(
            loc, result_type, operands);
        new_operation->setAttr("name", name);
        new_operation->setAttr("composite_attributes", composite_attributes);
        new_operation->setAttr("decomposition", decomposition);
        new_operation->setAttr("version", version);
        for (const auto& attr : attributes) {
          new_operation->setAttr(attr.getName(), attr.getValue());
        }
        return new_operation;
      }

      // Shardy currently requires roundtripping passes after HW specific passes
      // which introduce kCall with backend_config for host communication. If
      // we get to a point where compiler flow for sharding propagation doesn't
      // require roundtrip this can likely be removed.
      auto annotateCallOp = [&](Operation* call) {
        const std::string& raw_backend_config =
            instruction->raw_backend_config_string();
        if (!raw_backend_config.empty()) {
          llvm::SmallVector<NamedAttribute, 1> frontend_attributes;
          frontend_attributes.push_back(builder_->getNamedAttr(
              "backend_config", builder_->getStringAttr(raw_backend_config)));
          call->setAttr(kFrontendAttributesAttr,
                        builder_->getDictionaryAttr(frontend_attributes));
        }
        for (auto attr : attributes) {
          call->setAttr(attr.getName(), attr.getValue());
        }
      };

      if (!flatten_computation_args_result_) {
        auto call =
            func_builder->create<mlir::func::CallOp>(loc, function, operands);
        annotateCallOp(call);
        return call;
      }
      // Flatten the tuple-typed operands.
      llvm::SmallVector<Value> flattened_operands = FlattenTupleValues(
          func_builder, loc, operands, function.getNumArguments());
      auto call = func_builder->create<mlir::func::CallOp>(loc, function,
                                                           flattened_operands);
      annotateCallOp(call);
      // Flatten the tuple-typed results.
      mlir::ValueRange flattened_results_ref(call->getResults());
      TF_ASSIGN_OR_RETURN(auto result_type,
                          ConvertShapeToType<RankedTensorType>(
                              instruction->shape(), *builder_));
      return CreateTupleValue(func_builder, loc, flattened_results_ref,
                              result_type)
          .getDefiningOp();
    }
    case HloOpcode::kCollectiveBroadcast: {
      auto collective_broadcast = Cast<HloChannelInstruction>(instruction);
      attributes.push_back(ConvertReplicaGroups(
          collective_broadcast->replica_groups(), builder_));
      if (collective_broadcast->channel_id().has_value())
        attributes.push_back(stablehlo::ConvertChannelHandle(
            collective_broadcast->channel_id().value(), builder_));
      return func_builder
          ->create<mlir::stablehlo::CollectiveBroadcastOp>(loc, result_type,
                                                           operands, attributes)
          .getOperation();
    }

    case HloOpcode::kCollectivePermute: {
      auto collective_permute = Cast<HloChannelInstruction>(instruction);
      attributes.push_back(ConvertSourceTargetPairs(
          collective_permute->source_target_pairs(), builder_));
      if (collective_permute->channel_id().has_value())
        attributes.push_back(stablehlo::ConvertChannelHandle(
            collective_permute->channel_id().value(), builder_));
      return func_builder
          ->create<mlir::stablehlo::CollectivePermuteOp>(loc, result_type,
                                                         operands, attributes)
          .getOperation();
    }
    case HloOpcode::kCollectivePermuteStart: {
      return ImportCollectivePermuteStart(instruction, loc, operands,
                                          attributes, result_type, func_builder,
                                          symbol_table_);
    }
    case HloOpcode::kCollectivePermuteDone: {
      return ImportAsyncOpDone(instruction, loc, operands, attributes,
                               result_type, func_builder);
    }
    case HloOpcode::kCustomCall: {
      auto custom_call = Cast<HloCustomCallInstruction>(instruction);
      if (IsOpEncodedCustomCall(custom_call)) {
        return ImportCustomCallAsOp(custom_call, loc, result_type, operands,
                                    func_builder);
      }
      const auto& called_computations = custom_call->called_computations();
      if (!called_computations.empty()) {
        llvm::SmallVector<mlir::Attribute> callees;
        callees.reserve(called_computations.size());
        for (HloComputation* callee : called_computations) {
          TF_ASSIGN_OR_RETURN(FuncOp function, ImportAsFunc(*callee,
                                                            /*is_main=*/false));
          callees.push_back(mlir::FlatSymbolRefAttr::get(builder_->getContext(),
                                                         function.getName()));
        }
        attributes.push_back(builder_->getNamedAttr(
            "called_computations",
            mlir::ArrayAttr::get(builder_->getContext(), callees)));
      }
      if (custom_call->layout_constrained()) {
        TF_ASSIGN_OR_RETURN(
            mlir::ArrayAttr operand_layouts,
            ExtractLayoutsFromShapes(custom_call->operand_shapes_with_layout(),
                                     builder_));
        attributes.push_back(
            builder_->getNamedAttr("operand_layouts", operand_layouts));
        mlir::ArrayAttr result_layouts;
        if (custom_call->shape().IsTuple()) {
          TF_ASSIGN_OR_RETURN(
              result_layouts,
              ExtractLayoutsFromTuple(custom_call->shape(), builder_));
        } else {
          TF_ASSIGN_OR_RETURN(
              result_layouts,
              ExtractLayoutsFromShapes({custom_call->shape()}, builder_));
        }
        attributes.push_back(
            builder_->getNamedAttr("result_layouts", result_layouts));
      }

      attributes.push_back(builder_->getNamedAttr(
          "call_target_name",
          builder_->getStringAttr(custom_call->custom_call_target())));
      attributes.push_back(builder_->getNamedAttr(
          "has_side_effect",
          builder_->getBoolAttr(custom_call->custom_call_has_side_effect())));

      // For typed FFI API version we need to parse raw backend config string
      // into a dictionary attribute.
      auto& raw_backend_config = custom_call->raw_backend_config_string();

      if (custom_call->api_version() ==
          CustomCallApiVersion::API_VERSION_TYPED_FFI) {
        if (raw_backend_config.empty()) {
          attributes.push_back(builder_->getNamedAttr(
              "backend_config", builder_->getDictionaryAttr({})));
        } else {
          mlir::Attribute attr =
              mlir::parseAttribute(raw_backend_config, builder_->getContext());
          if (!mlir::isa<mlir::DictionaryAttr>(attr))
            return Internal(
                "Couldn't parse backend config into a dictionary attribute");

          attributes.push_back(builder_->getNamedAttr("backend_config", attr));
        }
      } else {
        attributes.push_back(builder_->getNamedAttr(
            "backend_config", builder_->getStringAttr(raw_backend_config)));
      }

      if (custom_call->HasLiteral()) {
        const Literal& literal = custom_call->literal();
        auto attr = CreateDenseElementsAttrFromLiteral(literal, *builder_);
        if (!attr.ok()) return attr.status();
        attributes.push_back(
            builder_->getNamedAttr("mhlo.literal", attr.value()));
      }

      // StableHLO CustomCall doesn't have a schedule attribute
      if (custom_call->custom_call_schedule() !=
          CustomCallSchedule::SCHEDULE_NONE) {
        attributes.push_back(
            ConvertCustomCallSchedule(custom_call->custom_call_schedule()));
        TF_ASSIGN_OR_RETURN(
            auto mlir_api_version,
            ConvertCustomCallApiVersion(custom_call->api_version()));
        attributes.push_back(builder_->getNamedAttr(
            "api_version", mlir::mhlo::CustomCallApiVersionAttr::get(
                               builder_->getContext(), mlir_api_version)));
        attributes.push_back(builder_->getNamedAttr(
            "output_operand_aliases",
            ConvertOutputOperandAliasing(instruction->output_operand_aliasing(),
                                         builder_)));
        // XLA Feature - MHLO Only
        return func_builder
            ->create<mlir::mhlo::CustomCallOp>(loc, result_type, operands,
                                               attributes)
            .getOperation();
      }

      // Valid StableHLO CustomCall
      TF_ASSIGN_OR_RETURN(
          auto mlir_api_version,
          stablehlo::ConvertCustomCallApiVersion(custom_call->api_version()));
      attributes.push_back(builder_->getNamedAttr(
          "api_version", mlir::stablehlo::CustomCallApiVersionAttr::get(
                             builder_->getContext(), mlir_api_version)));
      attributes.push_back(builder_->getNamedAttr(
          "output_operand_aliases",
          stablehlo::ConvertOutputOperandAliasing(
              instruction->output_operand_aliasing(), builder_)));
      return func_builder
          ->create<mlir::stablehlo::CustomCallOp>(loc, result_type, operands,
                                                  attributes)
          .getOperation();
    }
    case HloOpcode::kCompare: {
      auto compare = Cast<HloCompareInstruction>(instruction);
      attributes.push_back(ConvertComparisonDirection(compare->direction()));
      auto default_type = Comparison::DefaultComparisonType(
          compare->operand(0)->shape().element_type());
      if (compare->type() != default_type)
        attributes.push_back(ConvertComparisonType(compare->type()));
      return func_builder
          ->create<mlir::stablehlo::CompareOp>(loc, result_type, operands,
                                               attributes)
          .getOperation();
    }
    case HloOpcode::kCholesky: {
      attributes.push_back(builder_->getNamedAttr(
          "lower",
          builder_->getBoolAttr(instruction->cholesky_options().lower())));
      return func_builder
          ->create<mlir::stablehlo::CholeskyOp>(loc, result_type, operands,
                                                attributes)
          .getOperation();
    }
    case HloOpcode::kGather: {
      auto gather_instruction = Cast<HloGatherInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "dimension_numbers",
          stablehlo::ConvertGatherDimensionNumbers(
              gather_instruction->gather_dimension_numbers(), builder_)));

      std::vector<int64_t> slice_sizes(
          gather_instruction->gather_slice_sizes().begin(),
          gather_instruction->gather_slice_sizes().end());
      attributes.push_back(
          builder_->getNamedAttr("slice_sizes", ConvertArray(slice_sizes)));
      attributes.push_back(builder_->getNamedAttr(
          "indices_are_sorted",
          builder_->getBoolAttr(gather_instruction->indices_are_sorted())));

      return func_builder
          ->create<mlir::stablehlo::GatherOp>(loc, result_type, operands,
                                              attributes)
          .getOperation();
    }
    case HloOpcode::kDynamicSlice: {
      std::vector<int64_t> slice_sizes(
          instruction->dynamic_slice_sizes().begin(),
          instruction->dynamic_slice_sizes().end());
      return func_builder
          ->create<mlir::stablehlo::DynamicSliceOp>(
              loc, result_type, operands[0],
              llvm::ArrayRef(operands).drop_front(), ConvertArray(slice_sizes))
          .getOperation();
    }
    case HloOpcode::kDynamicUpdateSlice: {
      return func_builder
          ->create<mlir::stablehlo::DynamicUpdateSliceOp>(
              loc, result_type, operands[0], operands[1],
              llvm::ArrayRef<Value>(operands.begin() + 2, operands.end()))
          .getOperation();
    }
    case HloOpcode::kInfeed: {
      if (IsNestedTupleInData(result_type)) {
        llvm_unreachable(
            "Importing xla::kInfeed with nested tuple shape not supported");
      }

      attributes.push_back(builder_->getNamedAttr(
          "infeed_config",
          mlir::StringAttr::get(builder_->getContext(),
                                instruction->infeed_config())));

      llvm::SmallVector<mlir::Attribute> flattened_attr;
      TF_RETURN_IF_ERROR(
          ConvertShapeToMlirLayout(instruction->shape(), flattened_attr));
      attributes.push_back(builder_->getNamedAttr(
          "layout", builder_->getArrayAttr(llvm::ArrayRef(flattened_attr))));

      // Flatten the return-type if they are tuple-typed.
      llvm::SmallVector<Type> flattened_ret_types;
      FlattenTupleType(result_type, flattened_ret_types);

      auto op = func_builder->create<mlir::stablehlo::InfeedOp>(
          loc, flattened_ret_types, operands, attributes);

      return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                      result_type);
    }
    case HloOpcode::kOutfeed: {
      attributes.push_back(builder_->getNamedAttr(
          "outfeed_config",
          mlir::StringAttr::get(builder_->getContext(),
                                instruction->outfeed_config())));

      assert(operands.size() == 2 && "Expected 2 operands for HLO Infeed");

      // In case operands[0] is a tuple, flatten it.
      llvm::SmallVector<Value> flattened_operands;
      FlattenTupleValue(func_builder, loc, operands[0], flattened_operands);
      flattened_operands.push_back(operands[1]);

      auto op = func_builder->create<mlir::stablehlo::OutfeedOp>(
          loc, result_type, flattened_operands, attributes);

      return op.getOperation();
    }
    case HloOpcode::kPad: {
      const auto& padding_config = instruction->padding_config();
      llvm::SmallVector<int64_t, 4> edge_padding_low;
      llvm::SmallVector<int64_t, 4> edge_padding_high;
      llvm::SmallVector<int64_t, 4> interior_padding;
      edge_padding_low.reserve(padding_config.dimensions_size());
      edge_padding_high.reserve(padding_config.dimensions_size());
      interior_padding.reserve(padding_config.dimensions_size());

      for (const auto& dimension : padding_config.dimensions()) {
        edge_padding_low.push_back(dimension.edge_padding_low());
        edge_padding_high.push_back(dimension.edge_padding_high());
        interior_padding.push_back(dimension.interior_padding());
      }

      return func_builder
          ->create<mlir::stablehlo::PadOp>(
              loc, result_type, operands[0], operands[1],
              ConvertArray(edge_padding_low), ConvertArray(edge_padding_high),
              ConvertArray(interior_padding))
          .getOperation();
    }
    case HloOpcode::kScatter: {
      auto scatter = Cast<HloScatterInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "scatter_dimension_numbers",
          stablehlo::ConvertScatterDimensionNumbers(
              scatter->scatter_dimension_numbers(), builder_)));
      attributes.push_back(builder_->getNamedAttr(
          "indices_are_sorted",
          builder_->getBoolAttr(scatter->indices_are_sorted())));
      attributes.push_back(builder_->getNamedAttr(
          "unique_indices", builder_->getBoolAttr(scatter->unique_indices())));

      llvm::SmallVector<Type> flattened_types;
      FlattenTupleType(result_type, flattened_types);

      auto scatter_op = func_builder->create<mlir::stablehlo::ScatterOp>(
          loc, flattened_types, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*scatter->to_apply(),
                                        &scatter_op.getUpdateComputation()));
      TF_ASSIGN_OR_RETURN(auto result_type,
                          ConvertShapeToType<RankedTensorType>(
                              instruction->shape(), *builder_));
      return CreateTupleFromOpResults(func_builder, loc,
                                      scatter_op.getOperation(), result_type);
    }
    case HloOpcode::kSelectAndScatter: {
      auto select_scatter = Cast<HloSelectAndScatterInstruction>(instruction);
      llvm::SmallVector<int64_t, 4> window_strides, window_dimensions;
      llvm::SmallVector<int64_t, 8> padding;
      for (const auto& dim : select_scatter->window().dimensions()) {
        window_strides.push_back(dim.stride());
        window_dimensions.push_back(dim.size());
        padding.push_back(dim.padding_low());
        padding.push_back(dim.padding_high());
      }
      attributes.push_back(builder_->getNamedAttr(
          "window_strides", ConvertArray(window_strides)));
      attributes.push_back(builder_->getNamedAttr(
          "window_dimensions", ConvertArray(window_dimensions)));
      attributes.push_back(ConvertPadding(padding));
      auto select_scatter_op =
          func_builder->create<mlir::stablehlo::SelectAndScatterOp>(
              loc, result_type, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*select_scatter->select(),
                                        &select_scatter_op.getSelect()));
      TF_RETURN_IF_ERROR(ImportAsRegion(*select_scatter->scatter(),
                                        &select_scatter_op.getScatter()));
      return select_scatter_op.getOperation();
    }
    case HloOpcode::kSetDimensionSize: {
      attributes.push_back(builder_->getNamedAttr(
          "dimension", builder_->getI64IntegerAttr(instruction->dimension())));
      return func_builder
          ->create<mlir::stablehlo::SetDimensionSizeOp>(loc, result_type,
                                                        operands, attributes)
          .getOperation();
    }
    case HloOpcode::kSlice: {
      return func_builder
          ->create<mlir::stablehlo::SliceOp>(
              loc, result_type, operands[0],
              ConvertArray(instruction->slice_starts()),
              ConvertArray(instruction->slice_limits()),
              ConvertArray(instruction->slice_strides()))
          .getOperation();
    }
    case HloOpcode::kSort: {
      auto sort_instruction = Cast<HloSortInstruction>(instruction);

      llvm::SmallVector<Type, 4> return_types = {result_type};
      if (mlir::TupleType tuple_ty =
              mlir::dyn_cast<mlir::TupleType>(result_type)) {
        return_types = llvm::to_vector<6>(tuple_ty.getTypes());
      }

      auto sort_op = func_builder->create<mlir::stablehlo::SortOp>(
          loc, return_types, operands,
          builder_->getI64IntegerAttr(sort_instruction->sort_dimension()),
          builder_->getBoolAttr(sort_instruction->is_stable()));
      TF_RETURN_IF_ERROR(ImportAsRegion(*sort_instruction->to_apply(),
                                        &sort_op.getComparator()));

      // Check if the output needs to be tupled.
      if (return_types.size() == 1 && return_types.front() == result_type) {
        return sort_op.getOperation();
      }

      return WrapInTuple(func_builder, sort_op);
    }
    case HloOpcode::kTopK: {
      auto topk_instruction = Cast<HloTopKInstruction>(instruction);
      // XLA Feature -- MHLO Only
      auto topk_op = func_builder->create<mlir::mhlo::TopKOp>(
          loc, mlir::dyn_cast<mlir::TupleType>(result_type).getTypes(),
          operands[0], builder_->getI64IntegerAttr(topk_instruction->k()),
          builder_->getBoolAttr(topk_instruction->largest()));
      return WrapInTuple(func_builder, topk_op);
    }
    case HloOpcode::kCopyStart: {
      return ImportCopyStart(instruction, loc, operands, attributes,
                             result_type, func_builder, symbol_table_);
    }
    case HloOpcode::kCopyDone: {
      return ImportAsyncOpDone(instruction, loc, operands, attributes,
                               result_type, func_builder);
    }
    case HloOpcode::kSend: {
      return ImportSend(instruction, loc, operands, attributes, result_type,
                        func_builder, symbol_table_);
    }
    case HloOpcode::kSendDone: {
      return ImportAsyncOpDone(instruction, loc, operands, attributes,
                               result_type, func_builder, HloOpcode::kSend);
    }
    case HloOpcode::kRecv: {
      return ImportRecv(instruction, loc, operands, attributes, result_type,
                        func_builder, symbol_table_);
    }
    case HloOpcode::kRecvDone: {
      return ImportAsyncOpDone(instruction, loc, operands, attributes,
                               result_type, func_builder, HloOpcode::kRecv);
    }
    case HloOpcode::kConditional: {
      llvm::SmallVector<Type, 4> rets;
      llvm::SmallVector<Value> flattened_operands =
          FlattenTupleValues(func_builder, loc, operands);

      // If/Case Op has a single operand; we collect the other operands to
      // replace the corresponding block arguments.
      llvm::ArrayRef<Value> implicit_operands(flattened_operands.begin() + 1,
                                              flattened_operands.end());

      mlir::Type pred_or_index_type =
          mlir::cast<mlir::TensorType>(operands[0].getType()).getElementType();
      // It is a predicated conditional if first argument is a boolean and
      // should be mapped to If op.
      if (pred_or_index_type.isInteger(1)) {
        TF_RETURN_IF_ERROR(GetMlirTypes(
            {instruction->true_computation()->root_instruction()}, &rets));

        // Flatten the return-type.
        llvm::SmallVector<Type> flattened_ret_types;
        assert(rets.size() == 1);
        FlattenTupleType(rets[0], flattened_ret_types);

        auto op = func_builder->create<mlir::stablehlo::IfOp>(
            loc, flattened_ret_types, flattened_operands[0], attributes);
        TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->true_computation(),
                                          &op.getTrueBranch()));
        TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->false_computation(),
                                          &op.getFalseBranch()));

        // Replace the uses of block-arguments of the IfOp with the
        // implicit_operands.
        ReplaceBlockArgumentsWithImplicitOperands(op.getOperation(),
                                                  implicit_operands);

        return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                        rets[0]);
      }

      // Otherwise, it is a indexed conditional and should be mapped to Case
      // op.
      TF_RETURN_IF_ERROR(GetMlirTypes(
          {instruction->branch_computation(0)->root_instruction()}, &rets));

      // Flatten the return-type.
      llvm::SmallVector<Type> flattened_ret_types;
      assert(rets.size() == 1);
      FlattenTupleType(rets[0], flattened_ret_types);

      int num_branches = instruction->branch_count();
      auto op = func_builder->create<mlir::stablehlo::CaseOp>(
          loc, flattened_ret_types, flattened_operands[0], attributes,
          num_branches);
      for (const auto& index_and_computation :
           llvm::enumerate(instruction->branch_computations())) {
        auto index = index_and_computation.index();
        HloComputation* computation = index_and_computation.value();
        TF_RETURN_IF_ERROR(
            ImportAsRegion(*computation, &op.getBranches()[index]));
      }

      // Replace the uses of block-arguments of the CaseOp with the
      // implicit_operands.
      ReplaceBlockArgumentsWithImplicitOperands(op.getOperation(),
                                                implicit_operands);

      return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                      rets[0]);
    }
    case HloOpcode::kConcatenate: {
      return func_builder
          ->create<mlir::stablehlo::ConcatenateOp>(
              loc, result_type, operands,
              builder_->getI64IntegerAttr(instruction->concatenate_dimension()))
          .getOperation();
    }
    case HloOpcode::kAllGather: {
      auto all_gather = Cast<HloAllGatherInstruction>(instruction);
      auto result_tuple_ty = mlir::dyn_cast<mlir::TupleType>(result_type);

      llvm::SmallVector<Type> result_types = {result_type};
      if (result_tuple_ty) {
        result_types = llvm::to_vector(result_tuple_ty.getTypes());
      }
      attributes.push_back(builder_->getNamedAttr(
          "all_gather_dim",
          builder_->getI64IntegerAttr(all_gather->all_gather_dimension())));
      attributes.push_back(
          ConvertReplicaGroups(all_gather->replica_groups(), builder_));
      if (all_gather->channel_id().has_value())
        attributes.push_back(stablehlo::ConvertChannelHandle(
            all_gather->channel_id().value(), builder_));
      if (all_gather->use_global_device_ids())
        attributes.push_back(ConvertUseGlobalDeviceIds(builder_));
      auto all_gather_op = func_builder->create<mlir::stablehlo::AllGatherOp>(
          loc, result_types, operands, attributes);
      if (result_tuple_ty) {
        return WrapInTuple(func_builder, all_gather_op);
      }
      return all_gather_op.getOperation();
    }
    case HloOpcode::kAllGatherStart: {
      return ImportAllGatherStart(instruction, loc, operands, attributes,
                                  result_type, func_builder, symbol_table_);
    }
    case HloOpcode::kAllGatherDone: {
      return ImportAsyncOpDone(instruction, loc, operands, attributes,
                               result_type, func_builder);
    }
    case HloOpcode::kAllReduce: {
      auto all_reduce = Cast<HloAllReduceInstruction>(instruction);
      auto result_tuple_ty = mlir::dyn_cast<mlir::TupleType>(result_type);

      llvm::SmallVector<Type> result_types = {result_type};
      if (result_tuple_ty) {
        result_types = llvm::to_vector(result_tuple_ty.getTypes());
      }

      attributes.push_back(
          ConvertReplicaGroups(all_reduce->replica_groups(), builder_));
      if (all_reduce->channel_id().has_value())
        attributes.push_back(stablehlo::ConvertChannelHandle(
            all_reduce->channel_id().value(), builder_));
      if (all_reduce->use_global_device_ids())
        attributes.push_back(ConvertUseGlobalDeviceIds(builder_));
      auto all_reduce_op = func_builder->create<mlir::stablehlo::AllReduceOp>(
          loc, result_types, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*all_reduce->to_apply(),
                                        &all_reduce_op.getComputation()));
      if (result_tuple_ty) {
        return WrapInTuple(func_builder, all_reduce_op);
      }
      return all_reduce_op.getOperation();
    }
    case HloOpcode::kAllReduceStart: {
      auto appendRegion = [&](mlir::stablehlo::AllReduceOp all_reduce_sync) {
        TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->to_apply(),
                                          &all_reduce_sync.getComputation()));
        return absl::OkStatus();
      };

      return ImportAllReduceStart(instruction, loc, operands, attributes,
                                  result_type, func_builder, appendRegion,
                                  symbol_table_);
    }
    case HloOpcode::kAllReduceDone: {
      return ImportAsyncOpDone(instruction, loc, operands, attributes,
                               result_type, func_builder);
    }
    case HloOpcode::kAllToAll: {
      auto all_to_all = Cast<HloAllToAllInstruction>(instruction);

      auto replica_groups_attr = llvm::cast<DenseIntElementsAttr>(
          ConvertReplicaGroups(all_to_all->replica_groups(), builder_)
              .getValue());

      // Check invariants of array all-to-all. This is a sanity check and is
      // verified by the HLO verifier.
      auto result_tuple_ty = llvm::dyn_cast<mlir::TupleType>(result_type);
      if (result_tuple_ty) {
        // Tuple all-to-all is MHLO only for now.
        if (all_to_all->split_dimension().has_value()) {
          return InvalidArgument(
              "Tuple all-to-all should not have a split dimension");
        }
        llvm::SmallVector<Type, 4> return_types =
            llvm::to_vector<4>(result_tuple_ty.getTypes());
        mlir::mhlo::ChannelHandleAttr channel_handle_attr{};
        if (all_to_all->channel_id().has_value()) {
          channel_handle_attr = llvm::cast<mlir::mhlo::ChannelHandleAttr>(
              ConvertChannelHandle(all_to_all->channel_id().value(), builder_)
                  .getValue());
        }

        // TODO(b/408024772): Fix StableHLO AllToAll to support tuple types the
        // way that XLA expects it. Currently it is mis-designed.
        // XLA Feature -- MHLO Only
        auto result = func_builder->create<mlir::mhlo::AllToAllOp>(
            loc, return_types, operands, nullptr, nullptr, nullptr,
            replica_groups_attr, channel_handle_attr);
        return WrapInTuple(func_builder, result);
      }
      // Array AllToAll
      if (!all_to_all->split_dimension().has_value() || operands.size() != 1 ||
          all_to_all->replica_groups().empty()) {
        return InvalidArgument(
            "Array all-to-all should have a split dimension, one operand and "
            "non-empty replica groups");
      }

      mlir::stablehlo::ChannelHandleAttr channel_handle_attr{};
      if (all_to_all->channel_id().has_value()) {
        channel_handle_attr = llvm::cast<mlir::stablehlo::ChannelHandleAttr>(
            stablehlo::ConvertChannelHandle(all_to_all->channel_id().value(),
                                            builder_)
                .getValue());
      }

      auto result = func_builder->create<mlir::stablehlo::AllToAllOp>(
          loc, result_type, operands[0], all_to_all->split_dimension().value(),
          all_to_all->split_dimension().value(),
          all_to_all->replica_groups()[0].replica_ids_size(),
          replica_groups_attr, channel_handle_attr);

      return result.getOperation();
    }
    case HloOpcode::kReduce: {
      // Operands in the first half are reduction inputs and the remaining
      // operands are corresponding initial values.
      size_t num_inputs = operands.size() / 2;
      llvm::SmallVector<Type, 4> return_types = {result_type};
      if (mlir::TupleType tuple_ty =
              mlir::dyn_cast<mlir::TupleType>(result_type)) {
        return_types = llvm::to_vector<6>(tuple_ty.getTypes());
      }

      auto reduce = func_builder->create<mlir::stablehlo::ReduceOp>(
          loc, return_types, mlir::ValueRange(operands).take_front(num_inputs),
          mlir::ValueRange(operands).drop_front(num_inputs),
          ConvertArray(ToArrayRef(instruction->dimensions())));
      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->to_apply(), &reduce.getBody()));

      // Check if the output needs to be tupled.
      if (return_types.size() == 1 && return_types.front() == result_type) {
        for (auto attr : attributes) {
          reduce->setAttr(attr.getName(), attr.getValue());
        }
        return reduce.getOperation();
      }

      mlir::Operation* operation = WrapInTuple(func_builder, reduce);
      for (auto attr : attributes) {
        operation->setAttr(attr.getName(), attr.getValue());
      }
      return operation;
    }
    case HloOpcode::kReverse: {
      return func_builder
          ->create<mlir::stablehlo::ReverseOp>(
              loc, result_type, operands[0],
              ConvertArray(ToArrayRef(instruction->dimensions())))
          .getOperation();
    }
    case HloOpcode::kRng: {
      auto shape = func_builder->create<mlir::stablehlo::ConstantOp>(
          loc, Convert(mlir::cast<RankedTensorType>(result_type).getShape()));
      switch (instruction->random_distribution()) {
        case RNG_UNIFORM:
          return func_builder
              ->create<mlir::stablehlo::RngOp>(
                  loc, result_type, operands[0], operands[1], shape,
                  ::mlir::stablehlo::RngDistribution::UNIFORM)
              .getOperation();

        case RNG_NORMAL:
          return func_builder
              ->create<mlir::stablehlo::RngOp>(
                  loc, result_type, operands[0], operands[1], shape,
                  ::mlir::stablehlo::RngDistribution::NORMAL)
              .getOperation();

        default:
          return InvalidArgument(
              "Unsupported distribution: %s",
              RandomDistributionToString(instruction->random_distribution()));
      }
    }
    case HloOpcode::kRngBitGenerator: {
      // HloRngBitGeneratorInstruction can have two kinds of shapes, (1)
      // tuple(output_state, output_data), and (2) output_data.
      // stablehlo::RngBitGeneratorOp has only one shape, (output_state,
      // output_data).
      auto rng_op = Cast<HloRngBitGeneratorInstruction>(instruction);

      auto algorithm_attr = mlir::stablehlo::RngAlgorithmAttr::get(
          builder_->getContext(),
          *mlir::stablehlo::symbolizeRngAlgorithm(rng_op->algorithm()));
      attributes.push_back(
          builder_->getNamedAttr("rng_algorithm", algorithm_attr));

      // Flatten the return type if they are tuple-typed.
      llvm::SmallVector<Type> flattened_ret_types;
      FlattenTupleType(result_type, flattened_ret_types);
      if (rng_op->shape().IsArray()) {
        TF_ASSIGN_OR_RETURN(auto state_type,
                            ConvertShapeToType<RankedTensorType>(
                                rng_op->operand(0)->shape(), *builder_));
        flattened_ret_types.insert(flattened_ret_types.begin(), state_type);

        if (instruction->has_sharding()) {
          Shape tuple_shape = ShapeUtil::MakeTupleShape(
              {rng_op->operand(0)->shape(), instruction->shape()});
          HloSharding tuple_sharding = HloSharding::Tuple(
              tuple_shape, {HloSharding::Replicate(), instruction->sharding()});
          CHECK_EQ(attributes.front().getName().str(), kShardingAttr);
          attributes.front() = builder_->getNamedAttr(
              kShardingAttr, ConvertSharding(tuple_sharding, builder_));
        }
      }
      CHECK_EQ(flattened_ret_types.size(), 2);

      auto op = func_builder->create<mlir::stablehlo::RngBitGeneratorOp>(
          loc, flattened_ret_types, operands[0], attributes);
      if (rng_op->shape().IsArray()) {
        return op.getOperation();
      }
      return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                      result_type);
    }
    case HloOpcode::kRngGetAndUpdateState: {
      // XLA Feature -- MHLO Only
      return func_builder
          ->create<mlir::mhlo::XlaRngGetAndUpdateStateOp>(
              loc, result_type,
              func_builder->getI64IntegerAttr(
                  Cast<HloRngGetAndUpdateStateInstruction>(instruction)
                      ->delta()))
          .getOperation();
    }
    case HloOpcode::kWhile: {
      llvm::SmallVector<Value> flattened_operands;
      llvm::SmallVector<Type> flattened_operand_types;
      FlattenTupleType(operands[0].getType(), flattened_operand_types);
      FlattenTupleValue(func_builder, loc, operands[0], flattened_operands);

      auto op = func_builder->create<mlir::stablehlo::WhileOp>(
          loc, flattened_operand_types, flattened_operands);

      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->while_condition(), &op.getCond()));
      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->while_body(), &op.getBody()));
      return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                      operands[0].getType());
    }
    case HloOpcode::kGetTupleElement: {
      return GetTupleElementOp(func_builder, operands[0],
                               instruction->tuple_index(),
                               std::move(attributes));
    };
    case HloOpcode::kGetDimensionSize: {
      attributes.push_back(builder_->getNamedAttr(
          "dimension", builder_->getI64IntegerAttr(instruction->dimension())));
      return func_builder
          ->create<mlir::stablehlo::GetDimensionSizeOp>(loc, result_type,
                                                        operands, attributes)
          .getOperation();
    };
    case HloOpcode::kTranspose: {
      attributes.push_back(builder_->getNamedAttr(
          "permutation", ConvertArray(ToArrayRef(instruction->dimensions()))));
      return func_builder
          ->create<mlir::stablehlo::TransposeOp>(loc, result_type, operands,
                                                 attributes)
          .getOperation();
    }
    case HloOpcode::kTriangularSolve: {
      attributes.push_back(builder_->getNamedAttr(
          "left_side",
          builder_->getBoolAttr(
              instruction->triangular_solve_options().left_side())));
      attributes.push_back(builder_->getNamedAttr(
          "lower", builder_->getBoolAttr(
                       instruction->triangular_solve_options().lower())));
      attributes.push_back(builder_->getNamedAttr(
          "unit_diagonal",
          builder_->getBoolAttr(
              instruction->triangular_solve_options().unit_diagonal())));
      auto transpose_a = mlir::stablehlo::TransposeAttr::get(
          builder_->getContext(),
          mlir::stablehlo::symbolizeTranspose(
              TriangularSolveOptions::Transpose_Name(
                  instruction->triangular_solve_options().transpose_a()))
              .value());

      attributes.push_back(builder_->getNamedAttr("transpose_a", transpose_a));
      return func_builder
          ->create<mlir::stablehlo::TriangularSolveOp>(loc, result_type,
                                                       operands, attributes)
          .getOperation();
    }
    case HloOpcode::kReduceScatter: {
      auto reduce_scatter = Cast<HloReduceScatterInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "scatter_dimension",
          builder_->getI64IntegerAttr(reduce_scatter->scatter_dimension())));
      attributes.push_back(
          ConvertReplicaGroups(reduce_scatter->replica_groups(), builder_));
      if (reduce_scatter->channel_id().has_value())
        attributes.push_back(stablehlo::ConvertChannelHandle(
            reduce_scatter->channel_id().value(), builder_));
      if (reduce_scatter->use_global_device_ids())
        attributes.push_back(ConvertUseGlobalDeviceIds(builder_));
      auto reduce_scatter_op =
          func_builder->create<mlir::stablehlo::ReduceScatterOp>(
              loc, result_type, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*reduce_scatter->to_apply(),
                                        &reduce_scatter_op.getComputation()));

      return reduce_scatter_op.getOperation();
    }
    case HloOpcode::kReduceWindow: {
      llvm::SmallVector<Type, 4> return_types = {result_type};
      if (mlir::TupleType tuple_ty =
              mlir::dyn_cast<mlir::TupleType>(result_type)) {
        return_types = llvm::to_vector<6>(tuple_ty.getTypes());
      }
      llvm::SmallVector<int64_t, 4> sizes, strides, base_dilations,
          win_dilations;
      llvm::SmallVector<int64_t, 8> padding;
      for (const auto& dim : instruction->window().dimensions()) {
        sizes.push_back(dim.size());
        strides.push_back(dim.stride());
        base_dilations.push_back(dim.base_dilation());
        win_dilations.push_back(dim.window_dilation());
        padding.push_back(dim.padding_low());
        padding.push_back(dim.padding_high());
      }
      attributes.push_back(
          builder_->getNamedAttr("window_dimensions", ConvertArray(sizes)));
      attributes.push_back(
          builder_->getNamedAttr("window_strides", ConvertArray(strides)));
      attributes.push_back(builder_->getNamedAttr(
          "base_dilations", ConvertArray(base_dilations)));
      attributes.push_back(builder_->getNamedAttr("window_dilations",
                                                  ConvertArray(win_dilations)));
      attributes.push_back(ConvertPadding(padding));
      auto reduce = func_builder->create<mlir::stablehlo::ReduceWindowOp>(
          loc, return_types, operands, attributes);
      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->to_apply(), &reduce.getBody()));

      // Check if the output needs to be tupled.
      if (return_types.size() == 1 && return_types.front() == result_type) {
        return reduce.getOperation();
      }

      return WrapInTuple(func_builder, reduce);
    }
    case HloOpcode::kMap: {
      auto op = func_builder->create<mlir::stablehlo::MapOp>(
          loc, result_type, operands,
          ConvertArray(ToArrayRef(instruction->dimensions())));
      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->to_apply(), &op.getComputation()));
      return op.getOperation();
    }
    case HloOpcode::kConvolution: {
      llvm::SmallVector<int64_t, 4> strides, lhs_dilations, rhs_dilations;
      llvm::SmallVector<bool, 4> reversals;
      llvm::SmallVector<int64_t, 8> paddings;
      for (const auto& dim : instruction->window().dimensions()) {
        strides.push_back(dim.stride());
        lhs_dilations.push_back(dim.base_dilation());
        rhs_dilations.push_back(dim.window_dilation());
        paddings.push_back(dim.padding_low());
        paddings.push_back(dim.padding_high());
        reversals.push_back(dim.window_reversal());
      }

      attributes.push_back(
          builder_->getNamedAttr("window_strides", ConvertArray(strides)));
      attributes.push_back(ConvertPadding(paddings));
      attributes.push_back(
          builder_->getNamedAttr("lhs_dilation", ConvertArray(lhs_dilations)));
      attributes.push_back(
          builder_->getNamedAttr("rhs_dilation", ConvertArray(rhs_dilations)));
      attributes.push_back(
          builder_->getNamedAttr("window_reversal", ConvertArray(reversals)));
      attributes.push_back(builder_->getNamedAttr(
          "dimension_numbers",
          stablehlo::ConvertConvDimensionNumbers(
              instruction->convolution_dimension_numbers(), builder_)));
      attributes.push_back(builder_->getNamedAttr(
          "feature_group_count",
          builder_->getI64IntegerAttr(instruction->feature_group_count())));
      attributes.push_back(builder_->getNamedAttr(
          "batch_group_count",
          builder_->getI64IntegerAttr(instruction->batch_group_count())));
      attributes.push_back(builder_->getNamedAttr(
          "precision_config", stablehlo::ConvertPrecisionConfig(
                                  &instruction->precision_config(), builder_)));

      // If the element types of the operands for convolution are different,
      // insert a convert op to convert the operands to the common element type
      // while preserving the values.
      auto lhs = operands[0];
      auto rhs = operands[1];
      auto lhs_element_type = instruction->operand(0)->shape().element_type();
      auto rhs_element_type = instruction->operand(1)->shape().element_type();
      if (lhs_element_type != rhs_element_type) {
        // Cast LHS or RHS to the common element type.
        if (primitive_util::CastPreservesValues(lhs_element_type,
                                                rhs_element_type)) {
          auto convert_op_return_type =
              mlir::cast<mlir::ShapedType>(lhs.getType())
                  .clone(mlir::getElementTypeOrSelf(rhs));
          lhs = func_builder->create<mlir::stablehlo::ConvertOp>(
              loc, convert_op_return_type, lhs);
        } else if (primitive_util::CastPreservesValues(rhs_element_type,
                                                       lhs_element_type)) {
          auto convert_op_return_type =
              mlir::cast<mlir::ShapedType>(rhs.getType())
                  .clone(mlir::getElementTypeOrSelf(lhs));
          rhs = func_builder->create<mlir::stablehlo::ConvertOp>(
              loc, convert_op_return_type, rhs);
        } else {
          return InvalidArgument(
              "Unsupported conversion between element types of operands (%s "
              "and %s) for convolution.",
              instruction->operand(0)->shape().ToString(),
              instruction->operand(1)->shape().ToString());
        }
      }

      return func_builder
          ->create<mlir::stablehlo::ConvolutionOp>(
              loc, result_type, std::vector<mlir::Value>{lhs, rhs}, attributes)
          .getOperation();
    }

    case HloOpcode::kFft: {
      auto fft_type = mlir::stablehlo::FftTypeAttr::get(
          builder_->getContext(), mlir::stablehlo::symbolizeFftType(
                                      FftType_Name(instruction->fft_type()))
                                      .value());

      std::vector<int64_t> fft_length(instruction->fft_length().begin(),
                                      instruction->fft_length().end());

      attributes.push_back(builder_->getNamedAttr("fft_type", fft_type));
      attributes.push_back(
          builder_->getNamedAttr("fft_length", ConvertArray(fft_length)));
      return func_builder
          ->create<mlir::stablehlo::FftOp>(loc, result_type, operands,
                                           attributes)
          .getOperation();
    }

    case HloOpcode::kAdd: {
      // HLO add ops on PRED elements are actually boolean or, but MHLO dialect
      // AddOps on i1 are just addition with overflow; so, we have to implement
      // the special behavior of HLO add ops on PRED here by creating an
      // arith::OrIOp instead.
      if (instruction->shape().element_type() == PRED) {
        return func_builder
            ->create<mlir::stablehlo::OrOp>(loc, result_type, operands,
                                            attributes)
            .getOperation();
      } else {
        return func_builder
            ->create<mlir::stablehlo::AddOp>(loc, result_type, operands,
                                             attributes)
            .getOperation();
      }
    }
    case HloOpcode::kAfterAll: {
      // HLO AfterAll ops without any token input are used to create a token.
      // TODO(b/408024772): Remove CreateTokenOp, it is redundant.
      if (instruction->operands().empty()) {
        return func_builder
            ->create<mlir::stablehlo::CreateTokenOp>(loc, result_type, operands,
                                                     attributes)
            .getOperation();
      } else {
        return func_builder
            ->create<mlir::stablehlo::AfterAllOp>(loc, result_type, operands,
                                                  attributes)
            .getOperation();
      }
    }

    case HloOpcode::kConvert: {
      // Convert to boolean is special, it requires a comparison to 0 instead of
      // a truncation to i1, otherwise it is a 1-1 translation.
      auto ranked_type = mlir::dyn_cast<mlir::RankedTensorType>(result_type);
      mlir::IntegerType integer_type =
          (ranked_type)
              ? mlir::dyn_cast<mlir::IntegerType>(ranked_type.getElementType())
              : nullptr;
      if (!integer_type || integer_type.getWidth() != 1) {
        // Simple case: 1-1 mapping.
        return {func_builder->create<mlir::stablehlo::ConvertOp>(
            loc, result_type, operands, attributes)};
      }

      // Return type is boolean, let's use `operand != 0` instead of Convert.
      Shape input_shape = instruction->operand(0)->shape();
      TF_ASSIGN_OR_RETURN(
          mlir::Value zero,
          createConstantZeroLike(operands[0], input_shape, func_builder, loc));
      std::vector<mlir::Value> compare_operands = {operands[0], zero};
      std::vector<mlir::NamedAttribute> attributes = {builder_->getNamedAttr(
          "comparison_direction",
          mlir::stablehlo::ComparisonDirectionAttr::get(
              func_builder->getContext(),
              mlir::stablehlo::ComparisonDirection::NE))};
      return {func_builder->create<mlir::stablehlo::CompareOp>(
          loc, result_type, compare_operands, attributes)};
    }
    case HloOpcode::kOptimizationBarrier: {
      llvm::SmallVector<Value> flattened_operands;
      llvm::SmallVector<Type> flattened_operand_types;
      FlattenTupleType(operands[0].getType(), flattened_operand_types);
      FlattenTupleValue(func_builder, loc, operands[0], flattened_operands);

      auto op = func_builder->create<mlir::stablehlo::OptimizationBarrierOp>(
          loc, flattened_operand_types, flattened_operands);

      return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                      operands[0].getType());
    }
    case HloOpcode::kDomain: {
      auto domain_kind = mlir::mhlo::symbolizeDomainKind(
          ToStringRef(instruction->user_side_metadata().Kind()));
      if (!domain_kind || *domain_kind != mlir::mhlo::DomainKind::sharding) {
        return InvalidArgument(
            "Invalid domain kind in hlo -> mhlo import. Only 'sharding' is "
            "supported");
      }
      attributes.push_back(builder_->getNamedAttr(
          "kind", mlir::mhlo::DomainKindAttr::get(func_builder->getContext(),
                                                  *domain_kind)));

      // In XLA, DomainMetadata is open-world, but in the proto, it is hardcoded
      // to be ShardingMetadata. Thankfully, the only other implementation of
      // DomainMetadata is OpName, which is generally used for debugging and
      // never for compiling production models.
      //
      // Since this is hardcoded as such in the proto, we must follow suit.
      auto exit_metadata = ShardingMetadata::ToShardingMetadata(
          &instruction->operand_side_metadata());
      auto entry_metadata = ShardingMetadata::ToShardingMetadata(
          &instruction->user_side_metadata());
      attributes.push_back(builder_->getNamedAttr(
          "exit_metadata",
          ConvertSharding(*(*exit_metadata)->sharding(), builder_)));
      attributes.push_back(builder_->getNamedAttr(
          "entry_metadata",
          ConvertSharding(*(*entry_metadata)->sharding(), builder_)));

      // XLA Feature -- MHLO Only
      return func_builder
          ->create<mlir::mhlo::DomainOp>(loc, result_type, operands, attributes)
          .getOperation();
    }

#define NO_ATTRIBUTE_CASE(hlo_op_code, mlir_op)                        \
  case HloOpcode::hlo_op_code: {                                       \
    return func_builder                                                \
        ->create<mlir::stablehlo::mlir_op>(loc, result_type, operands, \
                                           attributes)                 \
        .getOperation();                                               \
  }

      // broadcast dimensions are never added here because they don't exist as
      // part of the HLO instruction. They are only a convenience in the XLA
      // builder API.
      NO_ATTRIBUTE_CASE(kAbs, AbsOp);
      NO_ATTRIBUTE_CASE(kAnd, AndOp);
      NO_ATTRIBUTE_CASE(kAtan2, Atan2Op);
      NO_ATTRIBUTE_CASE(kBitcastConvert, BitcastConvertOp);
      NO_ATTRIBUTE_CASE(kClz, ClzOp);
      NO_ATTRIBUTE_CASE(kCeil, CeilOp);
      NO_ATTRIBUTE_CASE(kClamp, ClampOp);
      NO_ATTRIBUTE_CASE(kComplex, ComplexOp);
      NO_ATTRIBUTE_CASE(kDivide, DivOp);
      NO_ATTRIBUTE_CASE(kFloor, FloorOp);
      NO_ATTRIBUTE_CASE(kIsFinite, IsFiniteOp);
      NO_ATTRIBUTE_CASE(kImag, ImagOp);
      NO_ATTRIBUTE_CASE(kMaximum, MaxOp);
      NO_ATTRIBUTE_CASE(kMinimum, MinOp);
      NO_ATTRIBUTE_CASE(kMultiply, MulOp);
      NO_ATTRIBUTE_CASE(kNegate, NegOp);
      NO_ATTRIBUTE_CASE(kNot, NotOp);
      NO_ATTRIBUTE_CASE(kOr, OrOp);
      NO_ATTRIBUTE_CASE(kPartitionId, PartitionIdOp);
      NO_ATTRIBUTE_CASE(kPopulationCount, PopulationCountOp);
      NO_ATTRIBUTE_CASE(kPower, PowOp);
      NO_ATTRIBUTE_CASE(kReal, RealOp);
      NO_ATTRIBUTE_CASE(kRemainder, RemOp);
      NO_ATTRIBUTE_CASE(kReplicaId, ReplicaIdOp);
      // The dimensions attribute is not present on the HLO Reshape
      // instruction. If dimensions are non-default, the XLA builder
      // implements it as a separate transpose.
      NO_ATTRIBUTE_CASE(kReshape, ReshapeOp);
      NO_ATTRIBUTE_CASE(kRoundNearestAfz, RoundOp);
      NO_ATTRIBUTE_CASE(kRoundNearestEven, RoundNearestEvenOp);
      NO_ATTRIBUTE_CASE(kSelect, SelectOp);
      NO_ATTRIBUTE_CASE(kShiftLeft, ShiftLeftOp);
      NO_ATTRIBUTE_CASE(kShiftRightArithmetic, ShiftRightArithmeticOp);
      NO_ATTRIBUTE_CASE(kShiftRightLogical, ShiftRightLogicalOp);
      NO_ATTRIBUTE_CASE(kSign, SignOp);
      NO_ATTRIBUTE_CASE(kSubtract, SubtractOp);
      NO_ATTRIBUTE_CASE(kTuple, TupleOp);
      NO_ATTRIBUTE_CASE(kXor, XorOp);

#undef NO_ATTRIBUTE_CASE

#define NO_ATTRIBUTE_CASE_MHLO(hlo_op_code, mlir_op)                          \
  case HloOpcode::hlo_op_code: {                                              \
    return func_builder                                                       \
        ->create<mlir::mhlo::mlir_op>(loc, result_type, operands, attributes) \
        .getOperation();                                                      \
  }
      NO_ATTRIBUTE_CASE_MHLO(kAddDependency, AddDependencyOp);
      NO_ATTRIBUTE_CASE_MHLO(kCopy, CopyOp);
      NO_ATTRIBUTE_CASE_MHLO(kErf, ErfOp);
      NO_ATTRIBUTE_CASE_MHLO(kStochasticConvert, StochasticConvertOp);

#undef NO_ATTRIBUTE_CASE_MHLO

#define RESULT_ACCURACY_CASE(hlo_op_code, mlir_op)                            \
  case HloOpcode::hlo_op_code: {                                              \
    if (instruction->has_result_accuracy()) {                                 \
      attributes.push_back(builder_->getNamedAttr(                            \
          "result_accuracy", stablehlo::ConvertResultAccuracy(                \
                                 instruction->result_accuracy(), builder_))); \
    }                                                                         \
    return func_builder                                                       \
        ->create<mlir::stablehlo::mlir_op>(loc, result_type, operands,        \
                                           attributes)                        \
        .getOperation();                                                      \
  }

      RESULT_ACCURACY_CASE(kCbrt, CbrtOp);
      RESULT_ACCURACY_CASE(kCos, CosineOp);
      RESULT_ACCURACY_CASE(kExp, ExpOp);
      RESULT_ACCURACY_CASE(kExpm1, Expm1Op);
      RESULT_ACCURACY_CASE(kLog, LogOp);
      RESULT_ACCURACY_CASE(kLog1p, Log1pOp);
      RESULT_ACCURACY_CASE(kLogistic, LogisticOp);
      RESULT_ACCURACY_CASE(kRsqrt, RsqrtOp);
      RESULT_ACCURACY_CASE(kSin, SineOp);
      RESULT_ACCURACY_CASE(kSqrt, SqrtOp);
      RESULT_ACCURACY_CASE(kTan, TanOp);
      RESULT_ACCURACY_CASE(kTanh, TanhOp);

#undef RESULT_ACCURACY_CASE

    case HloOpcode::kFusion: {
      // Flatten the tuple-typed operands.
      llvm::SmallVector<Value> flattened_operands =
          FlattenTupleValues(func_builder, loc, operands);

      // Flatten the return type if they are tuple-typed.
      llvm::SmallVector<Type> flattened_ret_types;
      FlattenTupleType(result_type, flattened_ret_types);

      auto fusion_kind = mlir::mhlo::symbolizeFusionKind(
          ToStringRef(ToString(instruction->fusion_kind())));
      attributes.push_back(builder_->getNamedAttr(
          "fusion_kind", mlir::mhlo::FusionKindAttr::get(
                             func_builder->getContext(), fusion_kind.value())));
      attributes.push_back(builder_->getNamedAttr(
          "output_operand_aliasing",
          ConvertOutputOperandAliasing(instruction->output_operand_aliasing(),
                                       builder_)));
      // XLA Feature -- MHLO Only
      auto fusion = func_builder->create<mlir::mhlo::FusionOp>(
          loc, flattened_ret_types, flattened_operands, attributes);
      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->fused_instructions_computation(),
                         &fusion.getFusedComputation()));

      return CreateTupleFromOpResults(func_builder, loc, fusion.getOperation(),
                                      result_type);
    }
    case HloOpcode::kBitcast: {
      // XLA Feature -- MHLO Only
      auto bitcast = func_builder->create<mlir::mhlo::BitcastOp>(
          loc, result_type, operands, attributes);
      // Store the source and result layout as attributes. Although the MHLO
      // Bitcast operates on tensors, these layouts are relevant as they define
      // the mapping between the elements of the source and result.
      SetLayoutForMlir(bitcast, instruction->shape(), "result_layout");
      SetLayoutForMlir(bitcast, instruction->operand(0)->shape(),
                       "source_layout");
      return bitcast.getOperation();
    }
    case HloOpcode::kReducePrecision: {
      auto op = func_builder->create<mlir::stablehlo::ReducePrecisionOp>(
          loc, result_type, operands[0], attributes);
      op.setExponentBitsAttr(func_builder->getIntegerAttr(
          func_builder->getI32Type(), instruction->exponent_bits()));
      op.setMantissaBitsAttr(func_builder->getIntegerAttr(
          func_builder->getI32Type(), instruction->mantissa_bits()));
      return op.getOperation();
    }
    default: {
      mlir::OperationState result(loc, "mhlo.unknown");
      result.addOperands(operands);
      result.addTypes(result_type);
      for (auto attr : attributes) {
        result.attributes.push_back(attr);
      }

      return func_builder->create(result);
    }
  }
}

void SetXlaShape(mlir::Operation* op, const Shape& shape) {
  op->setAttr("xla_shape",
              mlir::Builder(op->getContext())
                  .getStringAttr(shape.ToString(/*print_layout=*/true)));
}

absl::StatusOr<mlir::Operation*>
HloFunctionImporter::ImportInstructionWithLayout(
    const HloInstruction* instruction,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    mlir::OpBuilder* func_builder, DynamicShapeHandlingMode mode) {
  LLVM_DEBUG(llvm::dbgs() << "Importing instruction: "
                          << HloOpcodeString(instruction->opcode()) << '\n');
  LLVM_DEBUG({
    llvm::dbgs() << "  operands: (";
    llvm::interleaveComma(operands, llvm::dbgs(),
                          [](Value v) { llvm::dbgs() << v.getType(); });
    llvm::dbgs() << ")\n";
  });
  TF_ASSIGN_OR_RETURN(
      mlir::Operation * op,
      ImportInstructionImpl(instruction, operands, func_builder, mode));
  if (op == nullptr) {
    LLVM_DEBUG(llvm::dbgs() << "  instruction skipped.\n");
    return op;
  }

  // Print generic in debug since module may be invalid while printing.
  LLVM_DEBUG({
    op->print(llvm::dbgs(), mlir::OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  // See MlirToHloConversionOptions for more about layouts.
  //
  // Minor-to-major is a permutation of [0, rank), presenting tensor dimensions
  // in physical minor-to-major order.
  const Shape& shape = instruction->shape();
  bool custom_layout = HasCustomLayout(shape);
  if (!shape.IsArray() || custom_layout) {
    SetXlaShape(op, shape);
  }
  if (custom_layout) {
    SetLayoutForMlir(op, shape, "result_layout");
  }
  return op;
}

absl::StatusOr<llvm::SmallVector<mlir::Value, 4>>
HloFunctionImporter::GetOperands(const HloInstruction* instruction) {
  llvm::SmallVector<mlir::Value, 4> operands;
  for (const auto& operand : instruction->operands()) {
    auto input_it = instruction_value_map_.find(operand);
    if (input_it == instruction_value_map_.end()) {
      return Internal("Could not find input value: %s for instruction %s",
                      operand->name(), instruction->name());
    }
    operands.push_back(input_it->second);
  }
  return operands;
}

absl::Status HloFunctionImporter::GetMlirTypes(
    absl::Span<const HloInstruction* const> instructions,
    llvm::SmallVectorImpl<mlir::Type>* types) {
  for (auto instruction : instructions) {
    TF_ASSIGN_OR_RETURN(auto ret_type, ConvertShapeToType<RankedTensorType>(
                                           instruction->shape(), *builder_));
    types->push_back(ret_type);
  }
  return absl::OkStatus();
}

absl::StatusOr<Value> HloFunctionImporter::GetMlirValue(
    const HloInstruction* instruction) {
  auto lookup = instruction_value_map_.find(instruction);
  if (lookup != instruction_value_map_.end()) {
    return lookup->second;
  }

  return Internal("Unable to find value for input: %s",
                  instruction->ToString());
}

mlir::NamedAttribute HloFunctionImporter::ConvertComparisonDirection(
    ComparisonDirection direction) {
  return builder_->getNamedAttr(
      "comparison_direction",
      mlir::stablehlo::ComparisonDirectionAttr::get(
          builder_->getContext(), mlir::stablehlo::symbolizeComparisonDirection(
                                      ComparisonDirectionToString(direction))
                                      .value()));
}

mlir::NamedAttribute HloFunctionImporter::ConvertComparisonType(
    Comparison::Type type) {
  return builder_->getNamedAttr(
      "compare_type",
      mlir::stablehlo::ComparisonTypeAttr::get(
          builder_->getContext(),
          mlir::stablehlo::symbolizeComparisonType(ComparisonTypeToString(type))
              .value()));
}

mlir::DenseIntElementsAttr HloFunctionImporter::ConvertDimensions(
    absl::Span<const int64_t> op_dimensions) {
  llvm::SmallVector<APInt, 8> dimensions;
  dimensions.reserve(op_dimensions.size());
  for (auto value : op_dimensions) dimensions.emplace_back(APInt(64, value));

  return DenseIntElementsAttr::get(
      RankedTensorType::get(dimensions.size(), builder_->getIntegerType(64)),
      dimensions);
}

mlir::DenseI64ArrayAttr HloFunctionImporter::ConvertArray(
    llvm::ArrayRef<int64_t> elements) {
  return builder_->getDenseI64ArrayAttr(elements);
}

mlir::DenseBoolArrayAttr HloFunctionImporter::ConvertArray(
    llvm::ArrayRef<bool> elements) {
  return builder_->getDenseBoolArrayAttr(elements);
}

mlir::DenseIntElementsAttr HloFunctionImporter::Convert(
    llvm::ArrayRef<int64_t> elements) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(elements.size(), builder_->getIntegerType(64)),
      elements);
}

mlir::DenseIntElementsAttr HloFunctionImporter::Convert(
    llvm::ArrayRef<bool> elements) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(elements.size(), builder_->getI1Type()), elements);
}

mlir::NamedAttribute HloFunctionImporter::ConvertCustomCallSchedule(
    CustomCallSchedule schedule) {
  auto converted_schedule = ::mlir::mhlo::CustomCallSchedule::NONE;
  switch (schedule) {
    case SCHEDULE_LATEST:
      converted_schedule = ::mlir::mhlo::CustomCallSchedule::LATEST;
      break;
    case SCHEDULE_EARLIEST:
      converted_schedule = ::mlir::mhlo::CustomCallSchedule::EARLIEST;
      break;
    case SCHEDULE_NONE:
      converted_schedule = ::mlir::mhlo::CustomCallSchedule::NONE;
      break;
    default:
      assert(false && "Unrecognized custom call schedule hint");
  }
  return builder_->getNamedAttr(
      "custom_call_schedule", ::mlir::mhlo::CustomCallScheduleAttr::get(
                                  builder_->getContext(), converted_schedule));
}

mlir::NamedAttribute HloFunctionImporter::ConvertPadding(
    llvm::ArrayRef<int64_t> padding) {
  auto ty =
      mlir::RankedTensorType::get({static_cast<int64_t>(padding.size()) / 2, 2},
                                  builder_->getIntegerType(64));
  auto attr = DenseIntElementsAttr::get(ty, padding);
  return builder_->getNamedAttr("padding", attr);
}

void HloFunctionImporter::SetLayoutForMlir(mlir::Operation* op,
                                           const Shape& shape,
                                           llvm::StringRef attr_name) {
  mlir::Builder b(op->getContext());
  op->setAttr(attr_name, GetLayoutAttribute(b, shape).first);
}

absl::Status HloFunctionImporter::ConvertShapeToMlirLayout(
    const Shape& shape,
    llvm::SmallVectorImpl<mlir::Attribute>& flattened_attr) {
  if (shape.IsToken()) {
    return absl::OkStatus();
  }
  if (shape.IsTuple()) {
    std::vector<mlir::Attribute> tuple_layouts;
    for (int i = 0; i < shape.tuple_shapes().size(); i++) {
      TF_RETURN_IF_ERROR(
          ConvertShapeToMlirLayout(shape.tuple_shapes(i), flattened_attr));
    }
    return absl::OkStatus();
  }
  if (shape.IsArray()) {
    const Layout l = shape.layout();
    std::vector<mlir::Attribute> minor_to_major;
    for (int64_t i : l.minor_to_major()) {
      minor_to_major.push_back(builder_->getI64IntegerAttr(i));
    }
    llvm::ArrayRef<mlir::Attribute> array_ref(minor_to_major);
    flattened_attr.push_back(builder_->getArrayAttr(array_ref));
    return absl::OkStatus();
  }
  return Internal("Couldn't convert layout.");
}

mlir::Attribute ConvertInputOutputAlias(const HloInputOutputAliasConfig& alias,
                                        mlir::Builder* builder) {
  llvm::SmallVector<mlir::Attribute> element_attrs;
  alias.ForEachAlias([&](const ShapeIndex& output_index,
                         const HloInputOutputAliasConfig::Alias& alias) {
    std::string kindToString;
    switch (alias.kind) {
      case HloInputOutputAliasConfig::AliasKind::kMayAlias:
        kindToString = "may_alias";
        break;
      case HloInputOutputAliasConfig::AliasKind::kMustAlias:
        kindToString = "must_alias";
        break;
      default:
        kindToString = "undefined_alias";
    }
    mlir::NamedAttribute alias_named_attributes[3] = {
        builder->getNamedAttr(
            "parameter_index",
            builder->getDenseI64ArrayAttr(ArrayRef<int64_t>(
                alias.parameter_index.begin(), alias.parameter_index.end()))),
        builder->getNamedAttr("parameter_number", builder->getI64IntegerAttr(
                                                      alias.parameter_number)),
        builder->getNamedAttr("kind", builder->getStringAttr(kindToString))};

    mlir::NamedAttribute named_attributes[2] = {
        builder->getNamedAttr("output_index",
                              builder->getDenseI64ArrayAttr(ArrayRef<int64_t>(
                                  output_index.begin(), output_index.end()))),
        builder->getNamedAttr(
            "alias", builder->getDictionaryAttr(alias_named_attributes))};
    element_attrs.push_back(builder->getDictionaryAttr(named_attributes));
  });
  return builder->getArrayAttr(element_attrs);
}

mlir::Attribute ConvertSharding(const HloSharding& sharding,
                                mlir::Builder* builder) {
  return builder->getStringAttr(sharding.ToString(/*include_metadata=*/true));
}

mlir::Attribute ConvertSharding(const OpSharding& sharding,
                                mlir::Builder* builder) {
  auto hlo_sharding = HloSharding::FromProto(sharding);
  if (!hlo_sharding.ok()) return {};
  return ConvertSharding(hlo_sharding.value(), builder);
}

}  // namespace xla
