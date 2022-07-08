/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/hlo_function_importer.h"

#include <unordered_map>

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/xla/attribute_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/statusor.h"

using llvm::APInt;
using llvm::makeArrayRef;
using mlir::DenseIntElementsAttr;
using mlir::NamedAttribute;
using mlir::Operation;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::Value;
using mlir::func::FuncOp;

namespace xla {

namespace {

constexpr char kShardingAttr[] = "mhlo.sharding";

// Note: This sanitization function causes an irreversible many-to-one mapping
// and any solution to mitigate this would cause issues with the reverse
// direction. Longterm solution is to add a function attribute to maintain the
// original HLO naming.
std::string SanitizeFunctionName(llvm::StringRef name) {
  std::string output(name);
  llvm::for_each(output, [](char& x) { x = x == '-' ? '_' : x; });
  return output;
}

// Returns whether the instruction is a default dot operation.
bool DotIsDefault(const HloInstruction* instruction) {
  const auto& operands = instruction->operands();
  // eg. vector[3] dot matrix[3, 2] => [2] not default dot
  if (operands[0]->shape().rank() < operands[1]->shape().rank()) {
    return false;
  }
  auto dnums = instruction->dot_dimension_numbers();
  DotDimensionNumbers default_dimension_numbers;
  default_dimension_numbers.add_lhs_contracting_dimensions(
      instruction->operand(0)->shape().dimensions_size() == 1 ? 0 : 1);
  default_dimension_numbers.add_rhs_contracting_dimensions(0);
  return xla::protobuf_util::ProtobufEquals(dnums, default_dimension_numbers);
}

// Returns an MLIR Location generated from HLO Instruction. Uses instruction
// metadata if present or instruction name.
mlir::Location GenerateInstructionLocation(const HloInstruction* instruction,
                                           mlir::OpBuilder* func_builder) {
  const std::string& op_name = instruction->metadata().op_name();
  if (op_name.empty()) {
    return mlir::NameLoc::get(func_builder->getStringAttr(instruction->name()));
  }

  mlir::Location op_name_loc =
      mlir::NameLoc::get(func_builder->getStringAttr(op_name));
  const std::string& source_file = instruction->metadata().source_file();
  if (source_file.empty()) {
    return op_name_loc;
  }

  return func_builder->getFusedLoc(
      {op_name_loc,
       mlir::FileLineColLoc::get(func_builder->getContext(), source_file,
                                 instruction->metadata().source_line(), 0)});
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
      if (llvm::isa<mlir::mhlo::GetTupleElementOp>(op)) {
        folded_results.clear();
        if (failed(builder->tryFold(&op, folded_results))) continue;
        op.replaceAllUsesWith(folded_results);
        op.erase();
        changed = true;
      } else if (llvm::isa<mlir::mhlo::TupleOp>(op) &&
                 mlir::isOpTriviallyDead(&op)) {
        op.erase();
        changed = true;
      }
    }
  }
}

}  // namespace

void HloFunctionImporter::ReplaceBlockArgumentsWithImplicitOperands(
    mlir::Operation* op, llvm::ArrayRef<mlir::Value> implicit_operands) {
  assert((mlir::dyn_cast<mlir::mhlo::IfOp>(*op) ||
          mlir::dyn_cast<mlir::mhlo::CaseOp>(*op)) &&
         "Unexpected mlir op in "
         "HloFunctionImporter::ReplaceBlockArgumentsWithImplicitOperands!");

  int implicit_operand_index = 0;
  for (auto& region : op->getRegions()) {
    for (auto arg : region.getArguments()) {
      assert(implicit_operand_index < implicit_operands.size());
      arg.replaceAllUsesWith(implicit_operands[implicit_operand_index++]);
    }
    region.front().eraseArguments(
        llvm::to_vector(llvm::seq<unsigned>(0, region.getNumArguments())));
  }
}

mlir::Operation* HloFunctionImporter::CreateTupleFromOpResults(
    mlir::OpBuilder* func_builder, mlir::Location loc, mlir::Operation* op,
    mlir::Type type) {
  if (!type.isa<mlir::TupleType>()) return op;

  llvm::SmallVector<Value> flattened_results = op->getResults();
  llvm::MutableArrayRef<mlir::Value> flattened_results_ref(flattened_results);
  auto result =
      CreateTupleValue(func_builder, loc, flattened_results_ref, type);
  auto defining_tuple_op = result.getDefiningOp<mlir::mhlo::TupleOp>();
  assert(defining_tuple_op && "builder didn't return the right type");
  auto tupleOp = defining_tuple_op.getOperation();
  return tupleOp;
}

static bool IsNestedTupleInData(Type type) {
  auto tuple_type = type.dyn_cast<mlir::TupleType>();
  if (!tuple_type) return false;

  assert(tuple_type.getType(1).isa<mlir::mhlo::TokenType>() &&
         "Infeed: Non token type");
  auto data_type = tuple_type.getType(0);

  auto data_tuple_type = data_type.dyn_cast<mlir::TupleType>();
  if (!data_tuple_type) return false;

  for (auto child_type : data_tuple_type.getTypes()) {
    if (child_type.isa<mlir::TupleType>()) return true;
  }

  return false;
}

void HloFunctionImporter::FlattenTupleType(
    Type type, llvm::SmallVectorImpl<Type>& flattened_types) {
  auto tuple_type = type.dyn_cast<mlir::TupleType>();
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
  auto tuple_type = value.getType().dyn_cast<mlir::TupleType>();
  if (!tuple_type) {
    flattened_values.push_back(value);
    return;
  }

  int flattenIdx = 0;
  for (auto child_type : tuple_type.getTypes()) {
    auto sub_value = func_builder->create<mlir::mhlo::GetTupleElementOp>(
        loc, child_type, value, func_builder->getI32IntegerAttr(flattenIdx++));
    FlattenTupleValue(func_builder, loc, sub_value, flattened_values);
  }
}

Value HloFunctionImporter::CreateTupleValue(
    mlir::OpBuilder* func_builder, mlir::Location loc,
    llvm::MutableArrayRef<Value>& flatten_values, Type type) {
  auto tuple_type = type.dyn_cast<mlir::TupleType>();
  if (!tuple_type) {
    assert(!flatten_values.empty());
    auto retval = flatten_values.front();
    flatten_values = flatten_values.drop_front();
    return retval;
  }

  llvm::SmallVector<mlir::Value> flatten_sub_values;
  for (auto child_type : tuple_type.getTypes())
    flatten_sub_values.push_back(
        CreateTupleValue(func_builder, loc, flatten_values, child_type));

  return func_builder->create<mlir::mhlo::TupleOp>(loc, flatten_sub_values)
      .getResult();
}

Status HloFunctionImporter::ImportAsFunc(
    const HloComputation& computation, mlir::ModuleOp module,
    std::unordered_map<const HloComputation*, FuncOp>* function_map,
    mlir::Builder* builder) {
  HloFunctionImporter importer(module, function_map, builder);
  return importer.ImportAsFunc(computation).status();
}

Status HloFunctionImporter::ImportAsRegion(
    const xla::HloComputation& computation, mlir::Region* region,
    mlir::Builder* builder, bool flatten_region_arg_tuple) {
  HloFunctionImporter importer(region->getParentOfType<mlir::ModuleOp>(), {},
                               builder);
  return importer.ImportAsRegion(computation, region, flatten_region_arg_tuple);
}

StatusOr<FuncOp> HloFunctionImporter::ImportAsFunc(
    const HloComputation& computation) {
  auto& imported = (*function_map_)[&computation];
  if (imported) return imported;
  llvm::SmallVector<Type, 4> args, rets;
  TF_RETURN_IF_ERROR(GetMlirTypes(computation.parameter_instructions(), &args));
  TF_RETURN_IF_ERROR(GetMlirTypes({computation.root_instruction()}, &rets));
  auto func_type = mlir::FunctionType::get(context_, args, rets);

  std::string computation_name =
      computation.parent()->entry_computation() == &computation
          ? "main"
          : SanitizeFunctionName(computation.name());

  // Construct the MLIR function and map arguments.
  llvm::ArrayRef<mlir::NamedAttribute> attrs;
  auto function = FuncOp::create(mlir::UnknownLoc::get(context_),
                                 computation_name, func_type, attrs);
  auto visibility = computation_name == "main" ? FuncOp::Visibility::Public
                                               : FuncOp::Visibility::Private;
  function.setVisibility(visibility);

  for (auto& entry : llvm::enumerate(computation.parameter_instructions())) {
    HloInstruction* parameter = entry.value();
    if (parameter->has_sharding()) {
      function.setArgAttr(
          entry.index(), kShardingAttr,
          builder_->getStringAttr(
              parameter->sharding().ToProto().SerializeAsString()));
    }
  }
  if (computation.root_instruction()->has_sharding()) {
    auto result = computation.root_instruction();
    if (function.getNumResults() != 1) {
      return tensorflow::errors::Internal(absl::StrCat(
          "Expected only a single result but got ", function.getNumResults()));
    }
    function.setResultAttr(
        0, kShardingAttr,
        builder_->getStringAttr(
            result->sharding().ToProto().SerializeAsString()));
  }

  module_.push_back(function);

  // Add to the map right away for function calls.
  imported = function;

  mlir::Block* block = function.addEntryBlock();
  TF_RETURN_IF_ERROR(ImportInstructions(computation, block,
                                        /*flatten_region_arg_tuple=*/false));

  return function;
}

tensorflow::Status HloFunctionImporter::ImportAsRegion(
    const HloComputation& computation, mlir::Region* region,
    bool flatten_region_arg_tuple) {
  auto loc = region->getLoc();
  // TODO(hinsu): Store computation name as an attribute for round-trip.
  auto* block = new mlir::Block;
  region->push_back(block);

  llvm::SmallVector<Type, 4> args;
  TF_RETURN_IF_ERROR(GetMlirTypes(computation.parameter_instructions(), &args));

  // Flatten the tuple-typed arguments.
  if (flatten_region_arg_tuple) {
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

  return ImportInstructions(computation, block, flatten_region_arg_tuple);
}

StatusOr<Value> HloFunctionImporter::ImportInstructionsImpl(
    const xla::HloComputation& computation,
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
      instruction_value_map_[instruction] = new_operation->getResult(0);
    }
  }

  // Setup the return type (HLO only supports a single return value).
  return GetMlirValue(computation.root_instruction());
}

Status HloFunctionImporter::ImportInstructions(
    const HloComputation& computation, mlir::Block* block,
    bool flatten_region_arg_tuple) {
  llvm::SmallVector<Value, 4> arguments(block->args_begin(), block->args_end());
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockEnd(block);

  // TODO(suderman): Add location tracking details.
  mlir::Location loc = builder.getUnknownLoc();

  Value result;
  if (!llvm::isa<FuncOp>(block->getParentOp()) && flatten_region_arg_tuple) {
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
          computation_arg_type.dyn_cast<mlir::TupleType>();

      // If the computation-parameter type is non-tuple, no action is needed.
      if (!orig_tuple_arg_type) {
        effective_arguments.push_back(arguments[flatten_idx]);
        flatten_idx++;
        continue;
      }

      // For each tuple-typed computation parameter, create a mhlo::TupleOp
      // value in the region body, using the already flattened values in
      // 'arguments'. For example: With computation parameters: [tuple<T1>,
      // tuple<T2, T4>] We have, 'arguments' = [T1 arg1, T2 arg2, T3 arg3] and
      // we need to create two tuples tuples, one using arg1, and the other
      // using arg2 and arg3.
      llvm::SmallVector<Type> flattened_arg_type;
      FlattenTupleType(orig_tuple_arg_type, flattened_arg_type);

      llvm::MutableArrayRef<Value> sub_args(
          arguments.begin() + flatten_idx,
          arguments.begin() + flatten_idx + flattened_arg_type.size());

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
  if (llvm::isa<FuncOp>(block->getParentOp())) {
    builder.create<mlir::func::ReturnOp>(loc, result);
  } else {
    if (flatten_region_arg_tuple) {
      // Flatten tuples in results of this region.
      llvm::SmallVector<Value> flattened_return_operands;
      FlattenTupleValue(&builder, loc, result, flattened_return_operands);
      builder.create<mlir::mhlo::ReturnOp>(loc, flattened_return_operands);
    } else {
      builder.create<mlir::mhlo::ReturnOp>(loc, result);
    }
  }

  CleanUpTupleOps(block, &builder);

  return ::tensorflow::OkStatus();
}

StatusOr<Value> HloFunctionImporter::ImportInstructions(
    const xla::HloComputation& computation,
    const llvm::SmallVectorImpl<Value>& arguments, mlir::OpBuilder* builder) {
  mlir::Block* block = builder->getBlock();
  if (block == nullptr)
    return InvalidArgument(
        "ImportInstructions requires a valid block in the builder");

  HloFunctionImporter importer(
      block->getParent()->getParentOfType<mlir::ModuleOp>(), {}, builder);
  return importer.ImportInstructionsImpl(computation, arguments, builder);
}

StatusOr<mlir::Operation*> HloFunctionImporter::ImportInstruction(
    const xla::HloInstruction* instr,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    mlir::OpBuilder* builder, DynamicShapeHandlingMode mode) {
  mlir::Block* block = builder->getBlock();
  if (block == nullptr)
    return InvalidArgument(
        "ImportInstructions requires a valid block in the builder");

  HloFunctionImporter importer(
      block->getParent()->getParentOfType<mlir::ModuleOp>(), {}, builder);

  return importer.ImportInstructionWithLayout(instr, operands, builder, mode);
}

StatusOr<mlir::Operation*> HloFunctionImporter::ImportInstructionImpl(
    const HloInstruction* instruction,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    mlir::OpBuilder* func_builder, DynamicShapeHandlingMode mode) {
  const Shape& instruction_shape = instruction->shape();
  const Shape& shape = mode == DynamicShapeHandlingMode::kConvertToStatic
                           ? xla::ShapeUtil::MakeStaticShape(instruction_shape)
                           : instruction_shape;
  TF_ASSIGN_OR_RETURN(auto result_type,
                      ConvertShapeToType<RankedTensorType>(shape, *builder_));
  mlir::Location loc = GenerateInstructionLocation(instruction, func_builder);

  llvm::SmallVector<NamedAttribute, 10> attributes;
  if (instruction->has_sharding()) {
    attributes.push_back(builder_->getNamedAttr(
        kShardingAttr,
        builder_->getStringAttr(
            instruction->sharding().ToProto().SerializeAsString())));
  }

  switch (instruction->opcode()) {
    case HloOpcode::kParameter: {
      return nullptr;
    }
    case HloOpcode::kConstant: {
      const Literal& literal = instruction->literal();
      auto attr = CreateDenseElementsAttrFromLiteral(literal, *builder_);
      if (!attr.ok()) return attr.status();
      mlir::Operation* new_operation =
          func_builder->create<mlir::mhlo::ConstantOp>(loc, attr.ValueOrDie());
      for (auto attr : attributes) {
        new_operation->setAttr(attr.getName(), attr.getValue());
      }
      return new_operation;
    }
    case HloOpcode::kIota: {
      return func_builder
          ->create<mlir::mhlo::IotaOp>(
              loc, result_type,
              func_builder->getI64IntegerAttr(
                  Cast<HloIotaInstruction>(instruction)->iota_dimension()))
          .getOperation();
    }
    case HloOpcode::kBroadcast: {
      // Note that the HLO broadcast is more powerful than the XLA broadcast
      // op. BroadcastInDim offers a superset of the HLO op's functionality.
      attributes.push_back(
          builder_->getNamedAttr("broadcast_dimensions",
                                 ConvertDimensions(instruction->dimensions())));
      return func_builder
          ->create<mlir::mhlo::BroadcastInDimOp>(loc, result_type, operands,
                                                 attributes)
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
                      ->create<mlir::mhlo::BatchNormGradOp>(
                          loc, flattened_ret_types, operands, attributes)
                      .getOperation();

        return CreateTupleFromOpResults(func_builder, loc, op, result_type);
      } else if (instruction->opcode() == HloOpcode::kBatchNormInference) {
        return func_builder
            ->create<mlir::mhlo::BatchNormInferenceOp>(loc, result_type,
                                                       operands, attributes)
            .getOperation();
      } else {
        assert(instruction->opcode() == HloOpcode::kBatchNormTraining);

        // Flatten the return type if they are tuple-typed.
        llvm::SmallVector<Type> flattened_ret_types;
        FlattenTupleType(result_type, flattened_ret_types);

        auto op = func_builder
                      ->create<mlir::mhlo::BatchNormTrainingOp>(
                          loc, flattened_ret_types, operands, attributes)
                      .getOperation();

        return CreateTupleFromOpResults(func_builder, loc, op, result_type);
      }

    case HloOpcode::kDot: {
      attributes.push_back(builder_->getNamedAttr(
          "precision_config",
          ConvertPrecisionConfig(&instruction->precision_config(), builder_)));

      // Consider consolidating DotOps together.
      if (DotIsDefault(instruction)) {
        return func_builder
            ->create<mlir::mhlo::DotOp>(loc, result_type, operands, attributes)
            .getOperation();
      }

      attributes.push_back(builder_->getNamedAttr(
          "dot_dimension_numbers",
          ConvertDotDimensionNumbers(instruction->dot_dimension_numbers(),
                                     builder_)));
      return func_builder
          ->create<mlir::mhlo::DotGeneralOp>(loc, result_type, operands,
                                             attributes)
          .getOperation();
    }
    case HloOpcode::kCall: {
      TF_ASSIGN_OR_RETURN(FuncOp function,
                          ImportAsFunc(*instruction->to_apply()));
      mlir::Operation* new_operation =
          func_builder->create<mlir::func::CallOp>(loc, function, operands);
      return new_operation;
    }
    case HloOpcode::kCollectivePermute: {
      attributes.push_back(ConvertSourceTargetPairs(
          instruction->source_target_pairs(), builder_));
      return func_builder
          ->create<mlir::mhlo::CollectivePermuteOp>(loc, result_type, operands,
                                                    attributes)
          .getOperation();
    }
    case HloOpcode::kCustomCall: {
      auto custom_call = Cast<HloCustomCallInstruction>(instruction);
      const auto& called_computations = custom_call->called_computations();
      if (!called_computations.empty()) {
        llvm::SmallVector<mlir::Attribute> callees;
        callees.reserve(called_computations.size());
        for (HloComputation* callee : called_computations) {
          TF_ASSIGN_OR_RETURN(FuncOp function, ImportAsFunc(*callee));
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

      TF_ASSIGN_OR_RETURN(
          auto mlir_api_version,
          ConvertCustomCallApiVersion(custom_call->api_version()));
      attributes.push_back(builder_->getNamedAttr(
          "call_target_name",
          builder_->getStringAttr(custom_call->custom_call_target())));
      attributes.push_back(builder_->getNamedAttr(
          "has_side_effect",
          builder_->getBoolAttr(custom_call->custom_call_has_side_effect())));
      attributes.push_back(builder_->getNamedAttr(
          "backend_config",
          builder_->getStringAttr(custom_call->raw_backend_config_string())));
      attributes.push_back(builder_->getNamedAttr(
          "api_version", mlir::mhlo::CustomCallApiVersionAttr::get(
                             builder_->getContext(), mlir_api_version)));
      return func_builder
          ->create<mlir::mhlo::CustomCallOp>(loc, result_type, operands,
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
          ->create<mlir::mhlo::CompareOp>(loc, result_type, operands,
                                          attributes)
          .getOperation();
    }
    case HloOpcode::kCholesky: {
      attributes.push_back(builder_->getNamedAttr(
          "lower",
          builder_->getBoolAttr(instruction->cholesky_options().lower())));
      return func_builder
          ->create<mlir::mhlo::CholeskyOp>(loc, result_type, operands,
                                           attributes)
          .getOperation();
    }
    case HloOpcode::kGather: {
      auto gather_instruction = Cast<HloGatherInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "dimension_numbers",
          ConvertGatherDimensionNumbers(
              gather_instruction->gather_dimension_numbers(), builder_)));

      std::vector<int64_t> slice_sizes(
          gather_instruction->gather_slice_sizes().begin(),
          gather_instruction->gather_slice_sizes().end());
      attributes.push_back(
          builder_->getNamedAttr("slice_sizes", Convert(slice_sizes)));
      attributes.push_back(builder_->getNamedAttr(
          "indices_are_sorted",
          builder_->getBoolAttr(gather_instruction->indices_are_sorted())));

      return func_builder
          ->create<mlir::mhlo::GatherOp>(loc, result_type, operands, attributes)
          .getOperation();
    }
    case HloOpcode::kDynamicSlice: {
      std::vector<int64_t> slice_sizes(
          instruction->dynamic_slice_sizes().begin(),
          instruction->dynamic_slice_sizes().end());
      return func_builder
          ->create<mlir::mhlo::DynamicSliceOp>(
              loc, result_type, operands[0],
              makeArrayRef(operands).drop_front(), Convert(slice_sizes))
          .getOperation();
    }
    case HloOpcode::kDynamicUpdateSlice: {
      return func_builder
          ->create<mlir::mhlo::DynamicUpdateSliceOp>(
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
          "layout", builder_->getArrayAttr(makeArrayRef(flattened_attr))));

      // Flatten the return-type if they are tuple-typed.
      llvm::SmallVector<Type> flattened_ret_types;
      FlattenTupleType(result_type, flattened_ret_types);

      auto op = func_builder->create<mlir::mhlo::InfeedOp>(
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

      auto op = func_builder->create<mlir::mhlo::OutfeedOp>(
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
          ->create<mlir::mhlo::PadOp>(loc, result_type, operands[0],
                                      operands[1], Convert(edge_padding_low),
                                      Convert(edge_padding_high),
                                      Convert(interior_padding))
          .getOperation();
    }
    case HloOpcode::kScatter: {
      auto scatter = Cast<HloScatterInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "scatter_dimension_numbers",
          ConvertScatterDimensionNumbers(scatter->scatter_dimension_numbers(),
                                         builder_)));
      attributes.push_back(builder_->getNamedAttr(
          "indices_are_sorted",
          builder_->getBoolAttr(scatter->indices_are_sorted())));
      attributes.push_back(builder_->getNamedAttr(
          "unique_indices", builder_->getBoolAttr(scatter->unique_indices())));

      llvm::SmallVector<Type> flattened_types;
      FlattenTupleType(result_type, flattened_types);

      auto scatter_op = func_builder->create<mlir::mhlo::ScatterOp>(
          loc, flattened_types, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*scatter->to_apply(),
                                        &scatter_op.update_computation(),
                                        /*flatten_region_arg_tuple=*/true));
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
      attributes.push_back(
          builder_->getNamedAttr("window_strides", Convert(window_strides)));
      attributes.push_back(builder_->getNamedAttr("window_dimensions",
                                                  Convert(window_dimensions)));
      attributes.push_back(ConvertPadding(padding));
      auto select_scatter_op =
          func_builder->create<mlir::mhlo::SelectAndScatterOp>(
              loc, result_type, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*select_scatter->select(),
                                        &select_scatter_op.select(),
                                        /*flatten_region_arg_tuple=*/true));
      TF_RETURN_IF_ERROR(ImportAsRegion(*select_scatter->scatter(),
                                        &select_scatter_op.scatter(),
                                        /*flatten_region_arg_tuple=*/true));
      return select_scatter_op.getOperation();
    }
    case HloOpcode::kSetDimensionSize: {
      attributes.push_back(builder_->getNamedAttr(
          "dimension", builder_->getI64IntegerAttr(instruction->dimension())));
      return func_builder
          ->create<mlir::mhlo::SetDimensionSizeOp>(loc, result_type, operands,
                                                   attributes)
          .getOperation();
    }
    case HloOpcode::kSlice: {
      return func_builder
          ->create<mlir::mhlo::SliceOp>(
              loc, result_type, operands[0],
              ConvertDimensions(instruction->slice_starts()),
              ConvertDimensions(instruction->slice_limits()),
              ConvertDimensions(instruction->slice_strides()))
          .getOperation();
    }
    case HloOpcode::kSort: {
      auto sort_instruction = Cast<HloSortInstruction>(instruction);

      llvm::SmallVector<Type, 4> return_types = {result_type};
      if (mlir::TupleType tuple_ty = result_type.dyn_cast<mlir::TupleType>()) {
        return_types = llvm::to_vector<6>(tuple_ty.getTypes());
      }

      auto sort_op = func_builder->create<mlir::mhlo::SortOp>(
          loc, return_types, operands,
          builder_->getI64IntegerAttr(sort_instruction->sort_dimension()),
          builder_->getBoolAttr(sort_instruction->is_stable()));
      TF_RETURN_IF_ERROR(ImportAsRegion(*sort_instruction->to_apply(),
                                        &sort_op.comparator(),
                                        /*flatten_region_arg_tuple=*/true));

      // Check if the output needs to be tupled.
      if (return_types.size() == 1 && return_types.front() == result_type) {
        return sort_op.getOperation();
      }

      return func_builder
          ->create<mlir::mhlo::TupleOp>(loc, result_type, sort_op.getResults())
          .getOperation();
    }
    case HloOpcode::kConditional: {
      llvm::SmallVector<Type, 4> rets;

      // Flatten the tuple-typed operands.
      llvm::SmallVector<Value> flattened_operands;
      for (auto& operand : operands)
        FlattenTupleValue(func_builder, loc, operand, flattened_operands);

      // If/Case Op has a single operand; we collect the other operands to
      // replace the corresponding block arguments.
      llvm::ArrayRef<Value> implicit_operands(flattened_operands.begin() + 1,
                                              flattened_operands.end());

      mlir::Type pred_or_index_type =
          operands[0].getType().cast<mlir::TensorType>().getElementType();
      // It is a predicated conditional if first argument is a boolean and
      // should be mapped to If op.
      if (pred_or_index_type.isInteger(1)) {
        TF_RETURN_IF_ERROR(GetMlirTypes(
            {instruction->true_computation()->root_instruction()}, &rets));

        // Flatten the return-type.
        llvm::SmallVector<Type> flattened_ret_types;
        assert(rets.size() == 1);
        FlattenTupleType(rets[0], flattened_ret_types);

        auto op = func_builder->create<mlir::mhlo::IfOp>(
            loc, flattened_ret_types, flattened_operands[0], attributes);
        TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->true_computation(),
                                          &op.true_branch(),
                                          /*flatten_region_arg_tuple=*/true));
        TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->false_computation(),
                                          &op.false_branch(),
                                          /*flatten_region_arg_tuple=*/true));

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
      auto op = func_builder->create<mlir::mhlo::CaseOp>(
          loc, flattened_ret_types, flattened_operands[0], attributes,
          num_branches);
      for (const auto& index_and_computation :
           llvm::enumerate(instruction->branch_computations())) {
        auto index = index_and_computation.index();
        HloComputation* computation = index_and_computation.value();
        TF_RETURN_IF_ERROR(ImportAsRegion(*computation, &op.branches()[index],
                                          /*flatten_region_arg_tuple=*/true));
      }

      // Replace the uses of block-arguments of the CaseOp with the
      // implicit_operands.
      ReplaceBlockArgumentsWithImplicitOperands(op.getOperation(),
                                                implicit_operands);

      return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                      rets[0]);
    }
    case HloOpcode::kConcatenate: {
      // TODO(b/132057942): Support taking an uint64_t instead of an
      // IntegerAttr for concatenate dimension.
      return func_builder
          ->create<mlir::mhlo::ConcatenateOp>(
              loc, result_type, operands,
              builder_->getI64IntegerAttr(instruction->concatenate_dimension()))
          .getOperation();
    }
    case HloOpcode::kAllGather: {
      auto all_gather = Cast<HloAllGatherInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "all_gather_dim",
          builder_->getI64IntegerAttr(all_gather->all_gather_dimension())));
      attributes.push_back(
          ConvertReplicaGroups(all_gather->replica_groups(), builder_));
      if (all_gather->channel_id().has_value())
        attributes.push_back(
            ConvertChannelHandle(all_gather->channel_id().value()));
      return func_builder
          ->create<mlir::mhlo::AllGatherOp>(loc, result_type, operands,
                                            attributes)
          .getOperation();
    }
    case HloOpcode::kAllReduce: {
      auto all_reduce = Cast<HloAllReduceInstruction>(instruction);
      attributes.push_back(
          ConvertReplicaGroups(all_reduce->replica_groups(), builder_));
      if (all_reduce->channel_id().has_value())
        attributes.push_back(
            ConvertChannelHandle(all_reduce->channel_id().value()));
      auto all_reduce_op = func_builder->create<mlir::mhlo::AllReduceOp>(
          loc, result_type, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*all_reduce->to_apply(),
                                        &all_reduce_op.computation()));
      return all_reduce_op.getOperation();
    }
    case HloOpcode::kAllToAll: {
      // TODO(b/207152612): all-to-all HLO can either have pre-split operands
      // (and returns a tuple) or a single operand that is split across
      // `split_dimension` into the number of replicas in a group. Only the
      // latter case (array all-to-all) is supported in importer right now and
      // the former (tuple all-to-all) is not supported yet.
      auto all_to_all = Cast<HloAllToAllInstruction>(instruction);
      if (all_to_all->shape().IsTuple())
        return tensorflow::errors::Unimplemented(
            "Importing tuple all-to-all HLO is not supported yet");

      // Check invariants of array all-to-all. This is a sanity check and is
      // verified by the HLO verifier.
      if (!all_to_all->split_dimension().has_value() || operands.size() != 1 ||
          all_to_all->replica_groups().empty())
        return tensorflow::errors::InvalidArgument(
            "Array all-to-all should have a split dimension, one operand and "
            "non-empty replica groups");

      auto replica_groups_attr =
          ConvertReplicaGroups(all_to_all->replica_groups(), builder_)
              .getValue()
              .cast<DenseIntElementsAttr>();
      uint64_t split_dim = all_to_all->split_dimension().value();
      uint64_t concat_dim = split_dim;
      uint64_t split_count = all_to_all->replica_groups()[0].replica_ids_size();

      return func_builder
          ->create<mlir::mhlo::AllToAllOp>(loc, result_type, operands[0],
                                           split_dim, concat_dim, split_count,
                                           replica_groups_attr)
          .getOperation();
    }
    case HloOpcode::kReduce: {
      // Operands in the first half are reduction inputs and the remaining
      // operands are corresponding initial values.
      size_t num_inputs = operands.size() / 2;
      llvm::SmallVector<Type, 4> return_types = {result_type};
      if (mlir::TupleType tuple_ty = result_type.dyn_cast<mlir::TupleType>()) {
        return_types = llvm::to_vector<6>(tuple_ty.getTypes());
      }

      auto reduce = func_builder->create<mlir::mhlo::ReduceOp>(
          loc, return_types,
          llvm::makeArrayRef(operands).take_front(num_inputs),
          llvm::makeArrayRef(operands).drop_front(num_inputs),
          ConvertDimensions(instruction->dimensions()));
      TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->to_apply(),
                                        &reduce.body(),
                                        /*flatten_region_arg_tuple=*/true));

      // Check if the output needs to be tupled.
      if (return_types.size() == 1 && return_types.front() == result_type) {
        return reduce.getOperation();
      }

      return func_builder
          ->create<mlir::mhlo::TupleOp>(loc, result_type, reduce.getResults())
          .getOperation();
    }
    case HloOpcode::kReverse: {
      return func_builder
          ->create<mlir::mhlo::ReverseOp>(
              loc, result_type, operands[0],
              ConvertDimensions(instruction->dimensions()))
          .getOperation();
    }
    case HloOpcode::kRng: {
      auto shape = func_builder->create<mlir::mhlo::ConstantOp>(
          loc, Convert(result_type.cast<RankedTensorType>().getShape()));
      switch (instruction->random_distribution()) {
        case xla::RNG_UNIFORM:
          return func_builder
              ->create<mlir::mhlo::RngOp>(
                  loc, result_type, operands[0], operands[1], shape,
                  ::mlir::mhlo::RngDistribution::UNIFORM)
              .getOperation();

        case xla::RNG_NORMAL:
          return func_builder
              ->create<mlir::mhlo::RngOp>(loc, result_type, operands[0],
                                          operands[1], shape,
                                          ::mlir::mhlo::RngDistribution::NORMAL)
              .getOperation();

        default:
          return tensorflow::errors::InvalidArgument(absl::StrCat(
              "Unsupported distribution: ",
              RandomDistributionToString(instruction->random_distribution())));
      }
    }
    case HloOpcode::kRngBitGenerator: {
      auto rng_op = Cast<HloRngBitGeneratorInstruction>(instruction);

      // Flatten the return type if they are tuple-typed.
      llvm::SmallVector<Type> flattened_ret_types;
      FlattenTupleType(result_type, flattened_ret_types);

      auto algorithm_attr = mlir::mhlo::RngAlgorithmAttr::get(
          builder_->getContext(),
          *mlir::mhlo::symbolizeRngAlgorithm(rng_op->algorithm()));
      auto op = func_builder->create<mlir::mhlo::RngBitGeneratorOp>(
          loc, flattened_ret_types, algorithm_attr, operands[0]);

      return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                      result_type);
    }
    case HloOpcode::kRngGetAndUpdateState: {
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

      auto op = func_builder->create<mlir::mhlo::WhileOp>(
          loc, flattened_operand_types, flattened_operands);

      TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->while_condition(),
                                        &op.cond(),
                                        /*flatten_region_arg_tuple=*/true));
      TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->while_body(), &op.body(),
                                        /*flatten_region_arg_tuple=*/true));
      return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                      operands[0].getType());
    }
    case HloOpcode::kGetTupleElement: {
      attributes.push_back(builder_->getNamedAttr(
          "index", builder_->getIntegerAttr(builder_->getIntegerType(32),
                                            instruction->tuple_index())));
      return func_builder
          ->create<mlir::mhlo::GetTupleElementOp>(loc, result_type, operands,
                                                  attributes)
          .getOperation();
    };
    case HloOpcode::kGetDimensionSize: {
      attributes.push_back(builder_->getNamedAttr(
          "dimension", builder_->getI64IntegerAttr(instruction->dimension())));
      return func_builder
          ->create<mlir::mhlo::GetDimensionSizeOp>(loc, result_type, operands,
                                                   attributes)
          .getOperation();
    };
    case HloOpcode::kTranspose: {
      attributes.push_back(builder_->getNamedAttr(
          "permutation", ConvertDimensions(instruction->dimensions())));
      return func_builder
          ->create<mlir::mhlo::TransposeOp>(loc, result_type, operands,
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
      auto transpose_a = mlir::mhlo::TransposeAttr::get(
          builder_->getContext(),
          mlir::mhlo::symbolizeTranspose(
              TriangularSolveOptions::Transpose_Name(
                  instruction->triangular_solve_options().transpose_a()))
              .getValue());

      attributes.push_back(builder_->getNamedAttr("transpose_a", transpose_a));
      return func_builder
          ->create<mlir::mhlo::TriangularSolveOp>(loc, result_type, operands,
                                                  attributes)
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
        attributes.push_back(
            ConvertChannelHandle(reduce_scatter->channel_id().value()));
      auto reduce_scatter_op =
          func_builder->create<mlir::mhlo::ReduceScatterOp>(
              loc, result_type, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*reduce_scatter->to_apply(),
                                        &reduce_scatter_op.computation(),
                                        /*flatten_region_arg_tuple=*/true));

      return reduce_scatter_op.getOperation();
    }
    case HloOpcode::kReduceWindow: {
      llvm::SmallVector<Type, 4> return_types = {result_type};
      if (mlir::TupleType tuple_ty = result_type.dyn_cast<mlir::TupleType>()) {
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
      attributes.push_back(builder_->getNamedAttr("window_dimensions",
                                                  ConvertDimensions(sizes)));
      attributes.push_back(
          builder_->getNamedAttr("window_strides", ConvertDimensions(strides)));
      attributes.push_back(builder_->getNamedAttr(
          "base_dilations", ConvertDimensions(base_dilations)));
      attributes.push_back(builder_->getNamedAttr(
          "window_dilations", ConvertDimensions(win_dilations)));
      attributes.push_back(ConvertPadding(padding));
      auto reduce = func_builder->create<mlir::mhlo::ReduceWindowOp>(
          loc, return_types, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->to_apply(),
                                        &reduce.body(),
                                        /*flatten_region_arg_tuple=*/true));

      // Check if the output needs to be tupled.
      if (return_types.size() == 1 && return_types.front() == result_type) {
        return reduce.getOperation();
      }

      return func_builder
          ->create<mlir::mhlo::TupleOp>(loc, result_type, reduce.getResults())
          .getOperation();
    }
    case HloOpcode::kMap: {
      auto op = func_builder->create<mlir::mhlo::MapOp>(
          loc, result_type, operands,
          ConvertDimensions(instruction->dimensions()));
      TF_RETURN_IF_ERROR(ImportAsRegion(*instruction->to_apply(),
                                        &op.computation(),
                                        /*flatten_region_arg_tuple=*/true));
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
          builder_->getNamedAttr("window_strides", Convert(strides)));
      attributes.push_back(ConvertPadding(paddings));
      attributes.push_back(
          builder_->getNamedAttr("lhs_dilation", Convert(lhs_dilations)));
      attributes.push_back(
          builder_->getNamedAttr("rhs_dilation", Convert(rhs_dilations)));
      attributes.push_back(
          builder_->getNamedAttr("window_reversal", Convert(reversals)));
      attributes.push_back(builder_->getNamedAttr(
          "dimension_numbers",
          ConvertConvDimensionNumbers(
              instruction->convolution_dimension_numbers(), builder_)));
      attributes.push_back(builder_->getNamedAttr(
          "feature_group_count",
          builder_->getI64IntegerAttr(instruction->feature_group_count())));
      attributes.push_back(builder_->getNamedAttr(
          "batch_group_count",
          builder_->getI64IntegerAttr(instruction->batch_group_count())));
      attributes.push_back(builder_->getNamedAttr(
          "precision_config",
          ConvertPrecisionConfig(&instruction->precision_config(), builder_)));

      return func_builder
          ->create<mlir::mhlo::ConvolutionOp>(loc, result_type, operands,
                                              attributes)
          .getOperation();
    }

    case HloOpcode::kFft: {
      auto fft_type = mlir::mhlo::FftTypeAttr::get(
          builder_->getContext(),
          mlir::mhlo::symbolizeFftType(FftType_Name(instruction->fft_type()))
              .getValue());

      std::vector<int64_t> fft_length(instruction->fft_length().begin(),
                                      instruction->fft_length().end());

      attributes.push_back(builder_->getNamedAttr("fft_type", fft_type));
      attributes.push_back(
          builder_->getNamedAttr("fft_length", Convert(fft_length)));
      return func_builder
          ->create<mlir::mhlo::FftOp>(loc, result_type, operands, attributes)
          .getOperation();
    }

    case HloOpcode::kAdd: {
      // HLO add ops on PRED elements are actually boolean or, but MHLO dialect
      // AddOps on i1 are just addition with overflow; so, we have to implement
      // the special behavior of HLO add ops on PRED here by creating an
      // arith::OrIOp instead.
      if (instruction->shape().element_type() == PRED) {
        return func_builder
            ->create<mlir::mhlo::OrOp>(loc, result_type, operands, attributes)
            .getOperation();
      } else {
        return func_builder
            ->create<mlir::mhlo::AddOp>(loc, result_type, operands, attributes)
            .getOperation();
      }
    }
    case HloOpcode::kAfterAll: {
      // HLO AfterAll ops without any token input are used to just create a
      // token. MHLO has a special op CreateToken for this case.
      if (instruction->operands().empty()) {
        return func_builder
            ->create<mlir::mhlo::CreateTokenOp>(loc, result_type, operands,
                                                attributes)
            .getOperation();
      } else {
        return func_builder
            ->create<mlir::mhlo::AfterAllOp>(loc, result_type, operands,
                                             attributes)
            .getOperation();
      }
    }

    case HloOpcode::kConvert: {
      // Convert to boolean is special, it requires a comparison to 0 instead of
      // a truncation to i1, otherwise it is a 1-1 translation.
      auto ranked_type = result_type.dyn_cast<mlir::RankedTensorType>();
      mlir::IntegerType integer_type =
          (ranked_type)
              ? ranked_type.getElementType().dyn_cast<mlir::IntegerType>()
              : nullptr;
      if (!integer_type || integer_type.getWidth() != 1) {
        // Simple case: 1-1 mapping.
        return {func_builder->create<mlir::mhlo::ConvertOp>(
            loc, result_type, operands, attributes)};
      }

      // Return type is boolean, let's use `operand != 0` instead of Convert.
      xla::Shape input_shape = instruction->operand(0)->shape();
      TF_ASSIGN_OR_RETURN(mlir::Type type,
                          ConvertTensorShapeToType<mlir::RankedTensorType>(
                              input_shape, *func_builder));
      auto zero = func_builder->create<mlir::mhlo::ConstantOp>(
          loc, func_builder->getZeroAttr(type));
      return {func_builder->create<mlir::mhlo::CompareOp>(
          loc, operands[0], zero, mlir::mhlo::ComparisonDirection::NE)};
    }
    case HloOpcode::kOptimizationBarrier: {
      llvm::SmallVector<Value> flattened_operands;
      llvm::SmallVector<Type> flattened_operand_types;
      FlattenTupleType(operands[0].getType(), flattened_operand_types);
      FlattenTupleValue(func_builder, loc, operands[0], flattened_operands);

      auto op = func_builder->create<mlir::mhlo::OptimizationBarrierOp>(
          loc, flattened_operand_types, flattened_operands);

      return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                      operands[0].getType());
    }
    case HloOpcode::kDomain: {
      auto domain_kind = mlir::mhlo::symbolizeDomainKind(
          instruction->user_side_metadata().Kind());
      if (!domain_kind || *domain_kind != mlir::mhlo::DomainKind::sharding) {
        return tensorflow::errors::InvalidArgument(
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
      // TODO(b/208783683): The one improvement we can make on this is to move
      // from the a serialized proto representation to a parsable string
      auto exit_metadata = ShardingMetadata::ToShardingMetadata(
          &instruction->operand_side_metadata());
      auto entry_metadata = ShardingMetadata::ToShardingMetadata(
          &instruction->user_side_metadata());
      attributes.push_back(builder_->getNamedAttr(
          "exit_metadata",
          builder_->getStringAttr(
              (*exit_metadata)->sharding()->ToProto().SerializeAsString())));
      attributes.push_back(builder_->getNamedAttr(
          "entry_metadata",
          builder_->getStringAttr(
              (*entry_metadata)->sharding()->ToProto().SerializeAsString())));

      return func_builder
          ->create<mlir::mhlo::DomainOp>(loc, result_type, operands, attributes)
          .getOperation();
    }

#define NO_ATTRIBUTE_CASE(hlo_op_code, mlir_op)                               \
  case HloOpcode::hlo_op_code: {                                              \
    return func_builder                                                       \
        ->create<mlir::mhlo::mlir_op>(loc, result_type, operands, attributes) \
        .getOperation();                                                      \
  }

      // broadcast dimensions are never added here because they don't exist as
      // part of the HLO instruction. They are only a convenience in the XLA
      // builder API.
      NO_ATTRIBUTE_CASE(kAbs, AbsOp);
      NO_ATTRIBUTE_CASE(kAddDependency, AddDependencyOp);
      NO_ATTRIBUTE_CASE(kAnd, AndOp);
      NO_ATTRIBUTE_CASE(kAtan2, Atan2Op);
      NO_ATTRIBUTE_CASE(kBitcastConvert, BitcastConvertOp);
      NO_ATTRIBUTE_CASE(kCbrt, CbrtOp);
      NO_ATTRIBUTE_CASE(kClz, ClzOp);
      NO_ATTRIBUTE_CASE(kCeil, CeilOp);
      NO_ATTRIBUTE_CASE(kClamp, ClampOp);
      NO_ATTRIBUTE_CASE(kComplex, ComplexOp);
      NO_ATTRIBUTE_CASE(kCos, CosOp);
      NO_ATTRIBUTE_CASE(kDivide, DivOp);
      NO_ATTRIBUTE_CASE(kExp, ExpOp);
      NO_ATTRIBUTE_CASE(kExpm1, Expm1Op);
      NO_ATTRIBUTE_CASE(kFloor, FloorOp);
      NO_ATTRIBUTE_CASE(kIsFinite, IsFiniteOp);
      NO_ATTRIBUTE_CASE(kImag, ImagOp);
      NO_ATTRIBUTE_CASE(kLog, LogOp);
      NO_ATTRIBUTE_CASE(kLog1p, Log1pOp);
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
      NO_ATTRIBUTE_CASE(kLogistic, LogisticOp);
      // The dimensions attribute is not present on the HLO Reshape
      // instruction. If dimensions are non-default, the XLA builder
      // implements it as a separate transpose.
      NO_ATTRIBUTE_CASE(kReshape, ReshapeOp);
      NO_ATTRIBUTE_CASE(kRoundNearestAfz, RoundOp);
      NO_ATTRIBUTE_CASE(kRoundNearestEven, RoundNearestEvenOp);
      NO_ATTRIBUTE_CASE(kRsqrt, RsqrtOp);
      NO_ATTRIBUTE_CASE(kSelect, SelectOp);
      NO_ATTRIBUTE_CASE(kShiftLeft, ShiftLeftOp);
      NO_ATTRIBUTE_CASE(kShiftRightArithmetic, ShiftRightArithmeticOp);
      NO_ATTRIBUTE_CASE(kShiftRightLogical, ShiftRightLogicalOp);
      NO_ATTRIBUTE_CASE(kSign, SignOp);
      NO_ATTRIBUTE_CASE(kSin, SinOp);
      NO_ATTRIBUTE_CASE(kSqrt, SqrtOp);
      NO_ATTRIBUTE_CASE(kSubtract, SubOp);
      NO_ATTRIBUTE_CASE(kTanh, TanhOp);
      NO_ATTRIBUTE_CASE(kTuple, TupleOp);
      NO_ATTRIBUTE_CASE(kXor, XorOp);
      // TODO(b/129422361) Copy needs special handling because it is not
      // defined in tensorflow/compiler/xla/client/xla_builder.h. See
      // operation semantics in
      // g3doc/platforms/xla/g3doc/internal/hlo_semantics#copy
      NO_ATTRIBUTE_CASE(kCopy, CopyOp);

#undef NO_ATTRIBUTE_CASE

    case HloOpcode::kFusion: {
      // Flatten the tuple-typed operands.
      llvm::SmallVector<Value> flattened_operands;
      for (auto& operand : operands)
        FlattenTupleValue(func_builder, loc, operand, flattened_operands);

      // Flatten the return type if they are tuple-typed.
      llvm::SmallVector<Type> flattened_ret_types;
      FlattenTupleType(result_type, flattened_ret_types);

      auto fusion_kind = mlir::mhlo::symbolizeFusionKind(
          xla::ToString(instruction->fusion_kind()));
      auto fusion = func_builder->create<mlir::mhlo::FusionOp>(
          loc, flattened_ret_types, flattened_operands,
          mlir::mhlo::FusionKindAttr::get(func_builder->getContext(),
                                          fusion_kind.getValue()));
      TF_RETURN_IF_ERROR(ImportAsRegion(
          *instruction->fused_instructions_computation(),
          &fusion.fused_computation(), /*flatten_region_arg_tuple=*/true));

      return CreateTupleFromOpResults(func_builder, loc, fusion.getOperation(),
                                      result_type);
    }
    case HloOpcode::kBitcast: {
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
      auto op = func_builder->create<mlir::mhlo::ReducePrecisionOp>(
          loc, result_type, operands[0], attributes);
      op.exponent_bitsAttr(func_builder->getIntegerAttr(
          func_builder->getI32Type(), instruction->exponent_bits()));
      op.mantissa_bitsAttr(func_builder->getIntegerAttr(
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

StatusOr<mlir::Operation*> HloFunctionImporter::ImportInstructionWithLayout(
    const HloInstruction* instruction,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    mlir::OpBuilder* func_builder, DynamicShapeHandlingMode mode) {
  TF_ASSIGN_OR_RETURN(
      mlir::Operation * op,
      ImportInstructionImpl(instruction, operands, func_builder, mode));
  if (op == nullptr) return op;

  // See MlirToHloConversionOptions for more about layouts.
  //
  // Minor-to-major is a permutation of [0, rank), presenting tensor dimensions
  // in physical minor-to-major order.
  if (instruction->shape().IsArray()) {
    if (!instruction->shape().layout().minor_to_major().empty() &&
        instruction->shape().layout() !=
            LayoutUtil::MakeDescendingLayout(
                instruction->shape().dimensions().size())) {
      SetXlaShape(op, instruction->shape());
    }
  } else {
    SetXlaShape(op, instruction->shape());
  }
  return op;
}

StatusOr<llvm::SmallVector<mlir::Value, 4>> HloFunctionImporter::GetOperands(
    const HloInstruction* instruction) {
  llvm::SmallVector<mlir::Value, 4> operands;
  for (const auto& operand : instruction->operands()) {
    auto input_it = instruction_value_map_.find(operand);
    if (input_it == instruction_value_map_.end()) {
      return tensorflow::errors::Internal(
          absl::StrCat("Could not find input value: ", operand->name(),
                       " for instruction ", instruction->name()));
    }
    operands.push_back(input_it->second);
  }
  return operands;
}

tensorflow::Status HloFunctionImporter::GetMlirTypes(
    const std::vector<HloInstruction*>& instructions,
    llvm::SmallVectorImpl<mlir::Type>* types) {
  for (auto instruction : instructions) {
    TF_ASSIGN_OR_RETURN(auto ret_type, ConvertShapeToType<RankedTensorType>(
                                           instruction->shape(), *builder_));
    types->push_back(ret_type);
  }
  return ::tensorflow::OkStatus();
}

StatusOr<Value> HloFunctionImporter::GetMlirValue(
    const HloInstruction* instruction) {
  auto lookup = instruction_value_map_.find(instruction);
  if (lookup != instruction_value_map_.end()) {
    return lookup->second;
  }

  return tensorflow::errors::Internal(absl::StrCat(
      "Unable to find value for input: ", instruction->ToString()));
}

mlir::NamedAttribute HloFunctionImporter::ConvertComparisonDirection(
    ComparisonDirection direction) {
  return builder_->getNamedAttr(
      "comparison_direction",
      mlir::mhlo::ComparisonDirectionAttr::get(
          builder_->getContext(), mlir::mhlo::symbolizeComparisonDirection(
                                      ComparisonDirectionToString(direction))
                                      .getValue()));
}

mlir::NamedAttribute HloFunctionImporter::ConvertComparisonType(
    Comparison::Type type) {
  return builder_->getNamedAttr(
      "compare_type",
      mlir::mhlo::ComparisonTypeAttr::get(
          builder_->getContext(),
          mlir::mhlo::symbolizeComparisonType(ComparisonTypeToString(type))
              .getValue()));
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

mlir::NamedAttribute HloFunctionImporter::ConvertPadding(
    llvm::ArrayRef<int64_t> padding) {
  auto ty =
      mlir::RankedTensorType::get({static_cast<int64_t>(padding.size()) / 2, 2},
                                  builder_->getIntegerType(64));
  auto attr = DenseIntElementsAttr::get(ty, padding);
  return builder_->getNamedAttr("padding", attr);
}

mlir::NamedAttribute HloFunctionImporter::ConvertSourceTargetPairs(
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    mlir::Builder* builder) {
  std::vector<int64_t> attr(source_target_pairs.size() * 2);
  for (const auto& p : llvm::enumerate(source_target_pairs)) {
    attr[2 * p.index()] = p.value().first;
    attr[2 * p.index() + 1] = p.value().second;
  }
  auto type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(attr.size() / 2), 2}, builder->getIntegerType(64));
  return builder->getNamedAttr("source_target_pairs",
                               DenseIntElementsAttr::get(type, attr));
}

mlir::NamedAttribute HloFunctionImporter::ConvertReplicaGroups(
    absl::Span<const ReplicaGroup> replica_groups, mlir::Builder* builder) {
  const int64_t num_groups = replica_groups.size();
  // Replica groups in HLO can be non-uniform in size, for example:
  // replica_groups={{0},{1,2},{3}}. Since we are representing them as a 2D
  // tensor, pad the smaller sized replica groups with -1.
  const int64_t group_size = absl::c_accumulate(
      replica_groups, int64_t(0), [](int64_t current, const ReplicaGroup& g) {
        return std::max<int64_t>(current, g.replica_ids_size());
      });
  // Initialize all elements to -1 to support non-uniform replica groups.
  std::vector<int64_t> attr(num_groups * group_size, -1);
  for (int i = 0; i < num_groups; ++i) {
    int index = i * group_size;
    for (const int64_t& id : replica_groups[i].replica_ids())
      attr[index++] = id;
  }
  auto type = mlir::RankedTensorType::get({num_groups, group_size},
                                          builder->getIntegerType(64));
  return builder->getNamedAttr("replica_groups",
                               DenseIntElementsAttr::get(type, attr));
}

mlir::NamedAttribute HloFunctionImporter::ConvertChannelHandle(
    std::optional<int64_t> channel_id) {
  xla::ChannelHandle channel_handle;
  if (channel_id) channel_handle.set_handle(*channel_id);
  return ConvertChannelHandle(channel_handle);
}

mlir::NamedAttribute HloFunctionImporter::ConvertChannelHandle(
    const xla::ChannelHandle& channel) {
  return builder_->getNamedAttr(
      "channel_handle", mlir::mhlo::ChannelHandleAttr::get(
                            context_, channel.handle(), channel.type()));
}

void HloFunctionImporter::SetLayoutForMlir(mlir::Operation* op,
                                           const Shape& shape,
                                           llvm::StringRef attr_name) {
  llvm::SmallVector<int64_t, 4> minor_to_major(
      shape.layout().minor_to_major().begin(),
      shape.layout().minor_to_major().end());
  op->setAttr(
      attr_name,
      mlir::Builder(op->getContext()).getIndexTensorAttr(minor_to_major));
}

Status HloFunctionImporter::ConvertShapeToMlirLayout(
    const xla::Shape& shape,
    llvm::SmallVectorImpl<mlir::Attribute>& flattened_attr) {
  if (shape.IsToken()) {
    return ::tensorflow::OkStatus();
  }
  if (shape.IsTuple()) {
    std::vector<mlir::Attribute> tuple_layouts;
    for (int i = 0; i < shape.tuple_shapes_size(); i++) {
      TF_RETURN_IF_ERROR(
          ConvertShapeToMlirLayout(shape.tuple_shapes(i), flattened_attr));
    }
    return ::tensorflow::OkStatus();
  }
  if (shape.IsArray()) {
    const xla::Layout l = shape.layout();
    std::vector<mlir::Attribute> minor_to_major;
    for (int64_t i : l.minor_to_major()) {
      minor_to_major.push_back(builder_->getI64IntegerAttr(i));
    }
    llvm::ArrayRef<mlir::Attribute> array_ref(minor_to_major);
    flattened_attr.push_back(builder_->getArrayAttr(array_ref));
    return ::tensorflow::OkStatus();
  }
  return tensorflow::errors::Internal("Couldn't convert layout.");
}

}  // namespace xla
