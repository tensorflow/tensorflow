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

#include "xla/translate/hlo_to_mhlo/hlo_function_importer.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/protobuf_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/translate/hlo_to_mhlo/attribute_importer.h"
#include "xla/translate/hlo_to_mhlo/custom_call_importer.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/translate/hlo_to_mhlo/location_importer.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

using llvm::APInt;
using llvm::ArrayRef;
using mlir::DenseIntElementsAttr;
using mlir::NamedAttribute;
using mlir::Operation;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::Value;
using mlir::func::FuncOp;

namespace xla {

namespace {

constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kParameterReplicationAttr[] = "mhlo.parameter_replication";

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
// Supports vector.vector, vector.matrix, matrix.vector, and matrix.matrix.
// Default operations have lhs_contracting dimension is 1 (or zero for vector)
// and the rhs_contracting dimension is zero, and there are no batch dimensions.
bool DotIsDefault(const HloInstruction* instruction) {
  // If LHS/RHS has rank greater than 2, not default dot
  const auto& operands = instruction->operands();
  if (operands[0]->shape().rank() > 2 || operands[1]->shape().rank() > 2) {
    return false;
  }

  auto dnums = instruction->dot_dimension_numbers();
  DotDimensionNumbers default_dimension_numbers;
  default_dimension_numbers.add_lhs_contracting_dimensions(
      instruction->operand(0)->shape().dimensions_size() == 1 ? 0 : 1);
  default_dimension_numbers.add_rhs_contracting_dimensions(0);
  return protobuf_util::ProtobufEquals(dnums, default_dimension_numbers);
}

ArrayRef<HloSharding> FlattenTupleSharding(const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    return sharding.tuple_elements();
  }
  return sharding;
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

Operation* createReturnOp(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::ValueRange operands, bool is_func) {
  if (is_func) {
    return builder.create<mlir::func::ReturnOp>(loc, operands);
  }
  return builder.create<mlir::mhlo::ReturnOp>(loc, operands);
}

}  // namespace

mlir::TypeRange Untuple(const mlir::Type& type) {
  if (type.isa<mlir::TupleType>()) {
    return llvm::dyn_cast<mlir::TupleType>(type).getTypes();
  }
  return type;
}

template <typename sync_op>
absl::StatusOr<mlir::Operation*> HloFunctionImporter::ImportOldStyleAsyncStart(
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    const llvm::SmallVectorImpl<mlir::Value>& operands, mlir::Location loc,
    mlir::Type result_type, mlir::OpBuilder* func_builder,
    std::string func_name, std::function<absl::Status(sync_op)> mutate_op) {
  auto result_types = result_type.cast<mlir::TupleType>().getTypes();
  if (result_types.size() < 2) {
    return tsl::errors::InvalidArgument(
        "async_bundle must contain at least two values");
  }
  auto func_type = mlir::FunctionType::get(context_, Untuple(result_types[0]),
                                           Untuple(result_types[1]));
  auto function = FuncOp::create(loc, func_name, func_type);

  // The new function doesn't need to be inserted in the beginning but is done
  // to make testing easier and preserve the original behavior.
  mlir::Block& block = symbol_table_.getOp()->getRegion(0).front();
  symbol_table_.insert(function, mlir::Block::iterator(block.begin()));

  function.setPrivate();
  auto async_builder = mlir::OpBuilder(function.getBody());

  llvm::SmallVector<mlir::NamedAttribute> async_attributes;
  async_attributes.push_back(builder_->getNamedAttr(
      "called_computation", mlir::FlatSymbolRefAttr::get(builder_->getContext(),
                                                         function.getName())));
  async_attributes.push_back(builder_->getNamedAttr(
      "execution_thread", builder_->getStringAttr("main")));

  // Attach the frontend_attributes and sharding attributes to the async op
  // instead of the sync op. First, semantically sharding attributes cannot be
  // attached to the sync op since the sync op may not produce the same number
  // of results as the sharding's tuple element count, e.g., `mhlo.send` vs. HLO
  // `send`. Second, `mlir_hlo_to_hlo.cc` imports these attributes from the
  // `mhlo.async_start` ops, so attaching them to the sync op will make them
  // disappear during MHLO to HLO lowering.
  for (auto it = attributes.begin(); it != attributes.end();) {
    if (it->getName() == kShardingAttr ||
        it->getName() == kFrontendAttributesAttr) {
      async_attributes.push_back(*it);
      it = attributes.erase(it);
    } else {
      ++it;
    }
  }

  llvm::SmallVector<mlir::Location, 1> locs(Untuple(result_types[0]).size(),
                                            loc);
  auto sync_operand =
      async_builder
          .createBlock(&function.getBody(), {}, Untuple(result_types[0]), locs)
          ->getArguments();
  auto sync_operation = async_builder.create<sync_op>(
      loc, Untuple(result_types[1]), sync_operand, attributes);
  async_builder.create<mlir::func::ReturnOp>(loc, sync_operation->getResults());
  TF_RETURN_IF_ERROR(mutate_op(sync_operation));

  function->setAttr("execution_thread", builder_->getStringAttr("main"));

  auto bundle_result_type =
      mlir::mhlo::AsyncBundleType::get(context_, result_types);
  return func_builder
      ->create<mlir::mhlo::AsyncStartOp>(loc, bundle_result_type, operands,
                                         async_attributes)
      .getOperation();
}

absl::StatusOr<mlir::Operation*> HloFunctionImporter::ImportOldStyleAsyncDone(
    llvm::SmallVectorImpl<NamedAttribute>& attributes,
    const llvm::SmallVectorImpl<mlir::Value>& operands, mlir::Location loc,
    mlir::Type result_type, mlir::OpBuilder* func_builder) {
  if (operands.size() != 1) {
    return InvalidArgument(
        "async-done must take only a single async_bundle operand");
  }
  auto async_start = operands[0].getDefiningOp<mlir::mhlo::AsyncStartOp>();
  if (!async_start) return InvalidArgument("*-start requires *-done as input");
  attributes.push_back(builder_->getNamedAttr(
      "called_computation",
      mlir::FlatSymbolRefAttr::get(builder_->getContext(),
                                   async_start.getCalledComputation())));
  attributes.push_back(builder_->getNamedAttr("execution_thread",
                                              builder_->getStringAttr("main")));

  auto start_tuple = async_start.getResult()
                         .getType()
                         .cast<mlir::mhlo::AsyncBundleType>()
                         .getTypes()[1]
                         .dyn_cast<mlir::TupleType>();
  if (start_tuple && start_tuple.getType(0).isa<mlir::TupleType>()) {
    auto op = func_builder->create<mlir::mhlo::AsyncDoneOp>(
        loc, result_type, operands, attributes);
    return {op};
  } else {
    auto op = func_builder->create<mlir::mhlo::AsyncDoneOp>(
        loc, Untuple(result_type), operands, attributes);
    return CreateTupleFromOpResults(func_builder, loc, op.getOperation(),
                                    result_type);
  }
}

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
    region.front().eraseArguments(0, region.getNumArguments());
  }
}

mlir::Operation* HloFunctionImporter::CreateTupleFromOpResults(
    mlir::OpBuilder* func_builder, mlir::Location loc, mlir::Operation* op,
    mlir::Type type) {
  if (!type.isa<mlir::TupleType>()) return op;

  mlir::ValueRange flattened_results_ref(op->getResults());
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

static bool HasCustomLayout(const Shape& shape) {
  if (shape.IsTuple()) {
    return llvm::any_of(shape.tuple_shapes(), HasCustomLayout);
  }
  return shape.has_layout() && !shape.layout().minor_to_major().empty() &&
         shape.layout() != LayoutUtil::GetDefaultLayoutForShape(shape);
}

static mlir::Attribute GetLayoutAttribute(mlir::Builder& b,
                                          const Shape& shape) {
  if (shape.IsTuple()) {
    llvm::SmallVector<mlir::Attribute> element_attrs;
    for (const auto& tuple_shape : shape.tuple_shapes()) {
      element_attrs.push_back(GetLayoutAttribute(b, tuple_shape));
    }
    return b.getArrayAttr(element_attrs);
  }

  llvm::SmallVector<int64_t> layout;
  if (shape.has_layout()) {
    layout = {shape.layout().minor_to_major().begin(),
              shape.layout().minor_to_major().end()};
  } else {
    Layout layout_for_shape = LayoutUtil::GetDefaultLayoutForShape(shape);
    layout = {layout_for_shape.minor_to_major().begin(),
              layout_for_shape.minor_to_major().end()};
  }
  return b.getIndexTensorAttr(layout);
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

Value HloFunctionImporter::CreateTupleValue(mlir::OpBuilder* func_builder,
                                            mlir::Location loc,
                                            mlir::ValueRange& flatten_values,
                                            Type type) {
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
      is_main ? "main" : SanitizeFunctionName(computation.name());

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
      ++arg_index;
    }
  }
  if (computation.root_instruction()->has_sharding()) {
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
    function->setAttr("execution_thread",
                      builder_->getStringAttr(computation.execution_thread()));
  }

  // The MLIR CPU pipeline assumes default layouts throughout the program. At
  // the boundaries, this may not be the case, so layout information needs to
  // be propagated to adapt the data layouts.
  if (computation.IsEntryComputation()) {
    const auto& computation_layout =
        computation.parent()->entry_computation_layout();
    if (computation_layout.LayoutIsSet()) {
      if (HasCustomLayout(computation_layout.result_layout().shape())) {
        function->setAttr(
            "xla_entry_computation_result_layout",
            GetLayoutAttribute(*builder_,
                               computation_layout.result_layout().shape()));
      }
      if (llvm::any_of(computation_layout.parameter_layouts(),
                       [](const ShapeLayout& shape) {
                         return HasCustomLayout(shape.shape());
                       })) {
        llvm::SmallVector<mlir::Attribute> parameter_layouts;
        for (auto& layout : computation_layout.parameter_layouts()) {
          parameter_layouts.push_back(
              GetLayoutAttribute(*builder_, layout.shape()));
        }
        function->setAttr("xla_entry_computation_parameter_layouts",
                          builder_->getArrayAttr(parameter_layouts));
      }
    }
  }

  symbol_table_.insert(function);

  // Add to the map right away for function calls if map is set.
  if (imported) {
    *imported = function;
  }

  mlir::Block* block = function.addEntryBlock();
  TF_RETURN_IF_ERROR(ImportInstructions(computation, block));

  return function;
}

absl::Status HloFunctionImporter::ImportAsRegion(
    const HloComputation& computation, mlir::Region* region) {
  auto loc = region->getLoc();
  // TODO(hinsu): Store computation name as an attribute for round-trip.
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

  // TODO(suderman): Add location tracking details.
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
    createReturnOp(builder, loc, flattened_return_operands, is_func);
  } else {
    createReturnOp(builder, loc, result, is_func);
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
  mlir::Location loc = mlir::mhlo::GenerateInstructionLocation(
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
  if (!frontend_attributes.empty()) {
    attributes.push_back(builder_->getNamedAttr(
        kFrontendAttributesAttr,
        builder_->getDictionaryAttr(frontend_attributes)));
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
          func_builder->create<mlir::mhlo::ConstantOp>(loc, attr.value());
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
      auto execution_thread = async_op->async_execution_thread();
      attributes.push_back(builder_->getNamedAttr(
          "execution_thread", builder_->getStringAttr(execution_thread)));
      function->setAttr("execution_thread",
                        builder_->getStringAttr(execution_thread));

      if (instruction->opcode() == HloOpcode::kAsyncStart) {
        auto bundle_result_type = mlir::mhlo::AsyncBundleType::get(
            context_, result_type.cast<mlir::TupleType>().getTypes());
        return func_builder
            ->create<mlir::mhlo::AsyncStartOp>(loc, bundle_result_type,
                                               operands, attributes)
            .getOperation();
      } else if (instruction->opcode() == HloOpcode::kAsyncUpdate) {
        auto bundle_result_type = mlir::mhlo::AsyncBundleType::get(
            context_, result_type.cast<mlir::TupleType>().getTypes());
        return func_builder
            ->create<mlir::mhlo::AsyncUpdateOp>(loc, bundle_result_type,
                                                operands, attributes)
            .getOperation();
      } else {
        assert(instruction->opcode() == HloOpcode::kAsyncDone);
        return func_builder
            ->create<mlir::mhlo::AsyncDoneOp>(loc, result_type, operands,
                                              attributes)
            .getOperation();
      }
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
      auto dot = Cast<HloDotInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "precision_config",
          ConvertPrecisionConfig(&instruction->precision_config(), builder_)));

      // Consider consolidating DotOps together.
      if (DotIsDefault(instruction) && !dot->sparse_operands()) {
        return func_builder
            ->create<mlir::mhlo::DotOp>(loc, result_type, operands, attributes)
            .getOperation();
      }

      attributes.push_back(builder_->getNamedAttr(
          "dot_dimension_numbers",
          ConvertDotDimensionNumbers(instruction->dot_dimension_numbers(),
                                     builder_)));
      if (!dot->sparse_operands()) {
        return func_builder
            ->create<mlir::mhlo::DotGeneralOp>(loc, result_type, operands,
                                               attributes)
            .getOperation();
      }

      for (const SparsityDescriptor& descriptor : dot->sparsity()) {
        TF_ASSIGN_OR_RETURN(auto sparsity,
                            ConvertSparsityDescriptor(descriptor, builder_));
        attributes.push_back(builder_->getNamedAttr(
            descriptor.index() == 0 ? "lhs_sparsity" : "rhs_sparsity",
            sparsity));
      }
      return func_builder
          ->create<mlir::mhlo::SparseDotOp>(loc, result_type, operands,
                                            attributes)
          .getOperation();
    }
    case HloOpcode::kCall: {
      TF_ASSIGN_OR_RETURN(
          FuncOp function,
          ImportAsFunc(*instruction->to_apply(), /*is_main=*/false));
      mlir::Operation* new_operation;
      if (flatten_computation_args_result_) {
        // Flatten the tuple-typed operands.
        llvm::SmallVector<Value> flattened_operands = FlattenTupleValues(
            func_builder, loc, operands, function.getNumArguments());
        new_operation = func_builder->create<mlir::func::CallOp>(
            loc, function, flattened_operands);
        for (auto attr : attributes) {
          new_operation->setAttr(attr.getName(), attr.getValue());
        }
        // Flatten the tuple-typed results.
        mlir::ValueRange flattened_results_ref(new_operation->getResults());
        TF_ASSIGN_OR_RETURN(auto result_type,
                            ConvertShapeToType<RankedTensorType>(
                                instruction->shape(), *builder_));
        new_operation = CreateTupleValue(func_builder, loc,
                                         flattened_results_ref, result_type)
                            .getDefiningOp();
      } else {
        new_operation =
            func_builder->create<mlir::func::CallOp>(loc, function, operands);
        for (auto attr : attributes) {
          new_operation->setAttr(attr.getName(), attr.getValue());
        }
      }
      return new_operation;
    }
    case HloOpcode::kCollectiveBroadcast: {
      auto collective_broadcast = Cast<HloChannelInstruction>(instruction);
      attributes.push_back(ConvertReplicaGroups(
          collective_broadcast->replica_groups(), builder_));
      if (collective_broadcast->channel_id().has_value())
        attributes.push_back(
            ConvertChannelHandle(collective_broadcast->channel_id().value()));
      return func_builder
          ->create<mlir::mhlo::CollectiveBroadcastOp>(loc, result_type,
                                                      operands, attributes)
          .getOperation();
    }

    case HloOpcode::kCollectivePermute: {
      auto collective_permute = Cast<HloChannelInstruction>(instruction);
      attributes.push_back(ConvertSourceTargetPairs(
          collective_permute->source_target_pairs(), builder_));
      if (collective_permute->channel_id().has_value())
        attributes.push_back(
            ConvertChannelHandle(collective_permute->channel_id().value()));
      return func_builder
          ->create<mlir::mhlo::CollectivePermuteOp>(loc, result_type, operands,
                                                    attributes)
          .getOperation();
    }
    case HloOpcode::kCollectivePermuteStart: {
      attributes.push_back(ConvertSourceTargetPairs(
          instruction->source_target_pairs(), builder_));
      return ImportOldStyleAsyncStart<mlir::mhlo::CollectivePermuteOp>(
          attributes, operands, loc, result_type, func_builder,
          "collective_permute_", [&](auto) { return absl::OkStatus(); });
    }
    case HloOpcode::kCollectivePermuteDone: {
      return ImportOldStyleAsyncDone(attributes, operands, loc, result_type,
                                     func_builder);
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

      attributes.push_back(
          ConvertCustomCallSchedule(custom_call->custom_call_schedule()));
      TF_ASSIGN_OR_RETURN(
          auto mlir_api_version,
          ConvertCustomCallApiVersion(custom_call->api_version()));
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
          if (!attr.isa<mlir::DictionaryAttr>())
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

      attributes.push_back(builder_->getNamedAttr(
          "api_version", mlir::mhlo::CustomCallApiVersionAttr::get(
                             builder_->getContext(), mlir_api_version)));
      attributes.push_back(builder_->getNamedAttr(
          "output_operand_aliases",
          ConvertOutputOperandAliasing(instruction->output_operand_aliasing(),
                                       builder_)));
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
              llvm::ArrayRef(operands).drop_front(), Convert(slice_sizes))
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
          "layout", builder_->getArrayAttr(llvm::ArrayRef(flattened_attr))));

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
      attributes.push_back(
          builder_->getNamedAttr("window_strides", Convert(window_strides)));
      attributes.push_back(builder_->getNamedAttr("window_dimensions",
                                                  Convert(window_dimensions)));
      attributes.push_back(ConvertPadding(padding));
      auto select_scatter_op =
          func_builder->create<mlir::mhlo::SelectAndScatterOp>(
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
                                        &sort_op.getComparator()));

      // Check if the output needs to be tupled.
      if (return_types.size() == 1 && return_types.front() == result_type) {
        return sort_op.getOperation();
      }

      return func_builder
          ->create<mlir::mhlo::TupleOp>(loc, result_type, sort_op.getResults())
          .getOperation();
    }
    case HloOpcode::kTopK: {
      auto topk_instruction = Cast<HloTopKInstruction>(instruction);
      auto topk_op = func_builder->create<mlir::mhlo::TopKOp>(
          loc, result_type.dyn_cast<mlir::TupleType>().getTypes(), operands[0],
          builder_->getI64IntegerAttr(topk_instruction->k()),
          builder_->getBoolAttr(topk_instruction->largest()));
      return func_builder
          ->create<mlir::mhlo::TupleOp>(loc, result_type, topk_op.getResults())
          .getOperation();
    }
    case HloOpcode::kCopyStart: {
      auto copy_start_instruction = Cast<HloCopyStartInstruction>(instruction);
      if (auto cross_program_prefetch_index =
              copy_start_instruction->cross_program_prefetch_index()) {
        attributes.push_back(builder_->getNamedAttr(
            "cross_program_prefetch_index",
            builder_->getIntegerAttr(builder_->getIntegerType(32),
                                     *cross_program_prefetch_index)));
        // Cross-program prefetch allows copy ops to accept tuples, in which
        // case, we need to double-wrap inputs and outputs in tuples.
        if (operands[0].getType().isa<mlir::TupleType>()) {
          auto result_types = result_type.cast<mlir::TupleType>().getTypes();
          result_type = mlir::TupleType::get(
              context_, {mlir::TupleType::get(context_, {result_types[0]}),
                         mlir::TupleType::get(context_, {result_types[1]}),
                         result_types[2]});
        }
      }
      return ImportOldStyleAsyncStart<mlir::mhlo::CopyOp>(
          attributes, operands, loc, result_type, func_builder, "copy_",
          [](auto) { return absl::OkStatus(); });
    }
    case HloOpcode::kCopyDone: {
      return ImportOldStyleAsyncDone(attributes, operands, loc, result_type,
                                     func_builder);
    }
    case HloOpcode::kSend: {
      // old-style send returns a bundle of (arg, sync flag, token) to be passed
      // along to send-done.
      // However, the new-style async ops have a shared bundle
      // format of (args, results, scratchpad), so to rewrite the `send` and
      // `send-done` ops to use the new-style async API, we need to reorder the
      // arguments to be in (args, token, sync flag) order.
      auto result_types = result_type.cast<mlir::TupleType>().getTypes();
      if (result_types.size() != 3)
        return InvalidArgument("send should return a 3-tuple");
      auto async_arg_type =
          mlir::TupleType::get(context_, {result_types[0], result_types[2]});
      auto async_bundled_tuple = mlir::TupleType::get(
          context_, {async_arg_type, result_types[2], result_types[1]});
      auto send_op = Cast<HloSendInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "is_host_transfer",
          builder_->getBoolAttr(send_op->is_host_transfer())));
      if (send_op->channel_id().has_value()) {
        ChannelHandle channel_handle;
        channel_handle.set_handle(send_op->channel_id().value());
        channel_handle.set_type(send_op->is_host_transfer()
                                    ? ChannelHandle::DEVICE_TO_HOST
                                    : ChannelHandle::DEVICE_TO_DEVICE);
        attributes.push_back(ConvertChannelHandle(channel_handle));
      }
      return ImportOldStyleAsyncStart<mlir::mhlo::SendOp>(
          attributes, operands, loc, async_bundled_tuple, func_builder, "send_",
          [](auto) { return absl::OkStatus(); });
    }
    case HloOpcode::kSendDone: {
      return ImportOldStyleAsyncDone(attributes, operands, loc, result_type,
                                     func_builder);
    }
    case HloOpcode::kRecv: {
      // Old-style `recv` returns a bundle of (result, sync flag, token) to be
      // passed along to recv-done.
      // However, the new-style async ops have a shared
      // bundle format of (args, results, scratchpad), so to rewrite the `recv`
      // and `recv-done` ops to use the new-style async API, we need to reorder
      // the arguments to be in (token, (result, token), sync flag) order.
      auto result_types = result_type.cast<mlir::TupleType>().getTypes();
      if (result_types.size() != 3)
        return InvalidArgument("recv should return a 3-tuple");
      auto async_result_type =
          mlir::TupleType::get(context_, {result_types[0], result_types[2]});
      auto async_bundled_tuple = mlir::TupleType::get(
          context_, {result_types[2], async_result_type, result_types[1]});
      auto recv_op = Cast<HloRecvInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "is_host_transfer",
          builder_->getBoolAttr(recv_op->is_host_transfer())));
      if (recv_op->channel_id().has_value()) {
        ChannelHandle channel_handle;
        channel_handle.set_handle(recv_op->channel_id().value());
        channel_handle.set_type(recv_op->is_host_transfer()
                                    ? ChannelHandle::HOST_TO_DEVICE
                                    : ChannelHandle::DEVICE_TO_DEVICE);
        attributes.push_back(ConvertChannelHandle(channel_handle));
      }
      return ImportOldStyleAsyncStart<mlir::mhlo::RecvOp>(
          attributes, operands, loc, async_bundled_tuple, func_builder, "recv_",
          [](auto) { return absl::OkStatus(); });
    }
    case HloOpcode::kRecvDone: {
      return ImportOldStyleAsyncDone(attributes, operands, loc, result_type,
                                     func_builder);
    }
    case HloOpcode::kConditional: {
      llvm::SmallVector<Type, 4> rets;

      // Flatten the tuple-typed operands.
      llvm::SmallVector<Value> flattened_operands =
          FlattenTupleValues(func_builder, loc, operands);

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
      auto op = func_builder->create<mlir::mhlo::CaseOp>(
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
      auto result_tuple_ty = result_type.dyn_cast<mlir::TupleType>();

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
        attributes.push_back(
            ConvertChannelHandle(all_gather->channel_id().value()));
      if (all_gather->use_global_device_ids())
        attributes.push_back(ConvertUseGlobalDeviceIds());
      auto all_gather_op = func_builder->create<mlir::mhlo::AllGatherOp>(
          loc, result_types, operands, attributes);
      if (result_tuple_ty) {
        return func_builder
            ->create<mlir::mhlo::TupleOp>(loc, result_type,
                                          all_gather_op.getResults())
            .getOperation();
      }
      return all_gather_op.getOperation();
    }
    case HloOpcode::kAllGatherStart: {
      auto all_gather_start = Cast<HloAllGatherInstruction>(instruction);
      attributes.push_back(builder_->getNamedAttr(
          "all_gather_dim", builder_->getI64IntegerAttr(
                                all_gather_start->all_gather_dimension())));
      attributes.push_back(
          ConvertReplicaGroups(all_gather_start->replica_groups(), builder_));
      if (all_gather_start->channel_id().has_value())
        attributes.push_back(
            ConvertChannelHandle(all_gather_start->channel_id().value()));
      if (all_gather_start->use_global_device_ids())
        attributes.push_back(ConvertUseGlobalDeviceIds());
      if (all_gather_start->operands().size() > 1)
        return InvalidArgument(
            "Async tuple all-gather is not supported in MHLO");

      return ImportOldStyleAsyncStart<mlir::mhlo::AllGatherOp>(
          attributes, operands, loc, result_type, func_builder, "all_gather_",
          [](auto) { return absl::OkStatus(); });
    }
    case HloOpcode::kAllGatherDone: {
      return ImportOldStyleAsyncDone(attributes, operands, loc, result_type,
                                     func_builder);
    }
    case HloOpcode::kAllReduce: {
      auto all_reduce = Cast<HloAllReduceInstruction>(instruction);
      auto result_tuple_ty = result_type.dyn_cast<mlir::TupleType>();

      llvm::SmallVector<Type> result_types = {result_type};
      if (result_tuple_ty) {
        result_types = llvm::to_vector(result_tuple_ty.getTypes());
      }

      attributes.push_back(
          ConvertReplicaGroups(all_reduce->replica_groups(), builder_));
      if (all_reduce->channel_id().has_value())
        attributes.push_back(
            ConvertChannelHandle(all_reduce->channel_id().value()));
      if (all_reduce->use_global_device_ids())
        attributes.push_back(ConvertUseGlobalDeviceIds());
      auto all_reduce_op = func_builder->create<mlir::mhlo::AllReduceOp>(
          loc, result_types, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*all_reduce->to_apply(),
                                        &all_reduce_op.getComputation()));
      if (result_tuple_ty) {
        return func_builder
            ->create<mlir::mhlo::TupleOp>(loc, result_type,
                                          all_reduce_op.getResults())
            .getOperation();
      }
      return all_reduce_op.getOperation();
    }
    case HloOpcode::kAllReduceStart: {
      auto all_reduce_start = Cast<HloAllReduceInstruction>(instruction);
      attributes.push_back(
          ConvertReplicaGroups(all_reduce_start->replica_groups(), builder_));
      if (all_reduce_start->channel_id().has_value())
        attributes.push_back(
            ConvertChannelHandle(all_reduce_start->channel_id().value()));
      if (all_reduce_start->use_global_device_ids())
        attributes.push_back(ConvertUseGlobalDeviceIds());
      if (all_reduce_start->operands().size() > 1)
        return InvalidArgument(
            "Async tuple all-reduce is not supported in MHLO");

      return ImportOldStyleAsyncStart<mlir::mhlo::AllReduceOp>(
          attributes, operands, loc, result_type, func_builder, "all_reduce_",
          [&](auto all_reduce_sync) {
            TF_RETURN_IF_ERROR(ImportAsRegion(
                *instruction->to_apply(), &all_reduce_sync.getComputation()));
            return absl::OkStatus();
          });
    }
    case HloOpcode::kAllReduceDone: {
      return ImportOldStyleAsyncDone(attributes, operands, loc, result_type,
                                     func_builder);
    }
    case HloOpcode::kAllToAll: {
      auto all_to_all = Cast<HloAllToAllInstruction>(instruction);
      auto result_tuple_ty = result_type.dyn_cast<mlir::TupleType>();

      // Check invariants of array all-to-all. This is a sanity check and is
      // verified by the HLO verifier.
      if (result_tuple_ty) {
        if (all_to_all->split_dimension().has_value()) {
          return InvalidArgument(
              "Tuple all-to-all should not have a split dimension");
        }
      } else {
        if (!all_to_all->split_dimension().has_value() ||
            operands.size() != 1 || all_to_all->replica_groups().empty()) {
          return InvalidArgument(
              "Array all-to-all should have a split dimension, one operand and "
              "non-empty replica groups");
        }
      }

      auto replica_groups_attr =
          ConvertReplicaGroups(all_to_all->replica_groups(), builder_)
              .getValue()
              .cast<DenseIntElementsAttr>();

      llvm::SmallVector<Type, 4> return_types = {result_type};
      if (result_tuple_ty) {
        return_types = llvm::to_vector<4>(result_tuple_ty.getTypes());
      }

      auto result = func_builder->create<mlir::mhlo::AllToAllOp>(
          loc, return_types, operands, nullptr, nullptr, nullptr,
          replica_groups_attr);

      if (all_to_all->channel_id().has_value()) {
        auto handle = ConvertChannelHandle(all_to_all->channel_id().value());
        result.setChannelHandleAttr(
            handle.getValue().cast<mlir::mhlo::ChannelHandleAttr>());
      }

      if (result_tuple_ty) {
        return func_builder
            ->create<mlir::mhlo::TupleOp>(loc, result_type, result.getResults())
            .getOperation();
      }

      result.setSplitDimension(all_to_all->split_dimension().value());
      result.setConcatDimension(all_to_all->split_dimension().value());
      result.setSplitCount(all_to_all->replica_groups()[0].replica_ids_size());
      return result.getOperation();
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
          loc, return_types, llvm::ArrayRef(operands).take_front(num_inputs),
          llvm::ArrayRef(operands).drop_front(num_inputs),
          ConvertDimensions(instruction->dimensions()));
      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->to_apply(), &reduce.getBody()));

      // Check if the output needs to be tupled.
      if (return_types.size() == 1 && return_types.front() == result_type) {
        for (auto attr : attributes) {
          reduce->setAttr(attr.getName(), attr.getValue());
        }
        return reduce.getOperation();
      }

      mlir::Operation* operation =
          func_builder
              ->create<mlir::mhlo::TupleOp>(loc, result_type,
                                            reduce.getResults())
              .getOperation();
      for (auto attr : attributes) {
        operation->setAttr(attr.getName(), attr.getValue());
      }
      return operation;
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
        case RNG_UNIFORM:
          return func_builder
              ->create<mlir::mhlo::RngOp>(
                  loc, result_type, operands[0], operands[1], shape,
                  ::mlir::mhlo::RngDistribution::UNIFORM)
              .getOperation();

        case RNG_NORMAL:
          return func_builder
              ->create<mlir::mhlo::RngOp>(loc, result_type, operands[0],
                                          operands[1], shape,
                                          ::mlir::mhlo::RngDistribution::NORMAL)
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
      // mhlo::RngBitGeneratorOp has only one shape, (output_state,
      // output_data).
      auto rng_op = Cast<HloRngBitGeneratorInstruction>(instruction);

      auto algorithm_attr = mlir::mhlo::RngAlgorithmAttr::get(
          builder_->getContext(),
          *mlir::mhlo::symbolizeRngAlgorithm(rng_op->algorithm()));
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

      auto op = func_builder->create<mlir::mhlo::RngBitGeneratorOp>(
          loc, flattened_ret_types, operands[0], attributes);
      if (rng_op->shape().IsArray()) {
        return op.getOperation();
      }
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

      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->while_condition(), &op.getCond()));
      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->while_body(), &op.getBody()));
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
              .value());

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
      if (reduce_scatter->use_global_device_ids())
        attributes.push_back(ConvertUseGlobalDeviceIds());
      auto reduce_scatter_op =
          func_builder->create<mlir::mhlo::ReduceScatterOp>(
              loc, result_type, operands, attributes);
      TF_RETURN_IF_ERROR(ImportAsRegion(*reduce_scatter->to_apply(),
                                        &reduce_scatter_op.getComputation()));

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
      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->to_apply(), &reduce.getBody()));

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
              .value());

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
      Shape input_shape = instruction->operand(0)->shape();
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
      // TODO(b/208783683): The one improvement we can make on this is to move
      // from the a serialized proto representation to a parsable string
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
      NO_ATTRIBUTE_CASE(kCos, CosineOp);
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
      NO_ATTRIBUTE_CASE(kStochasticConvert, StochasticConvertOp);
      NO_ATTRIBUTE_CASE(kLogistic, LogisticOp);
      NO_ATTRIBUTE_CASE(kErf, ErfOp);
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
      NO_ATTRIBUTE_CASE(kSin, SineOp);
      NO_ATTRIBUTE_CASE(kSqrt, SqrtOp);
      NO_ATTRIBUTE_CASE(kSubtract, SubtractOp);
      NO_ATTRIBUTE_CASE(kTan, TanOp);
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
      llvm::SmallVector<Value> flattened_operands =
          FlattenTupleValues(func_builder, loc, operands);

      // Flatten the return type if they are tuple-typed.
      llvm::SmallVector<Type> flattened_ret_types;
      FlattenTupleType(result_type, flattened_ret_types);

      auto fusion_kind =
          mlir::mhlo::symbolizeFusionKind(ToString(instruction->fusion_kind()));
      attributes.push_back(builder_->getNamedAttr(
          "fusion_kind", mlir::mhlo::FusionKindAttr::get(
                             func_builder->getContext(), fusion_kind.value())));
      attributes.push_back(builder_->getNamedAttr(
          "output_operand_aliasing",
          ConvertOutputOperandAliasing(instruction->output_operand_aliasing(),
                                       builder_)));
      auto fusion = func_builder->create<mlir::mhlo::FusionOp>(
          loc, flattened_ret_types, flattened_operands, attributes);
      TF_RETURN_IF_ERROR(
          ImportAsRegion(*instruction->fused_instructions_computation(),
                         &fusion.getFusedComputation()));

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
  TF_ASSIGN_OR_RETURN(
      mlir::Operation * op,
      ImportInstructionImpl(instruction, operands, func_builder, mode));
  if (op == nullptr) return op;

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
      mlir::mhlo::ComparisonDirectionAttr::get(
          builder_->getContext(), mlir::mhlo::symbolizeComparisonDirection(
                                      ComparisonDirectionToString(direction))
                                      .value()));
}

mlir::NamedAttribute HloFunctionImporter::ConvertComparisonType(
    Comparison::Type type) {
  return builder_->getNamedAttr(
      "compare_type",
      mlir::mhlo::ComparisonTypeAttr::get(
          builder_->getContext(),
          mlir::mhlo::symbolizeComparisonType(ComparisonTypeToString(type))
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
  ChannelHandle channel_handle;
  if (channel_id) channel_handle.set_handle(*channel_id);
  return ConvertChannelHandle(channel_handle);
}

mlir::NamedAttribute HloFunctionImporter::ConvertChannelHandle(
    const ChannelHandle& channel) {
  return builder_->getNamedAttr(
      "channel_handle", mlir::mhlo::ChannelHandleAttr::get(
                            context_, channel.handle(), channel.type()));
}

mlir::NamedAttribute HloFunctionImporter::ConvertUseGlobalDeviceIds() {
  return builder_->getNamedAttr("use_global_device_ids",
                                builder_->getUnitAttr());
}

void HloFunctionImporter::SetLayoutForMlir(mlir::Operation* op,
                                           const Shape& shape,
                                           llvm::StringRef attr_name) {
  mlir::Builder b(op->getContext());
  op->setAttr(attr_name, GetLayoutAttribute(b, shape));
}

absl::Status HloFunctionImporter::ConvertShapeToMlirLayout(
    const Shape& shape,
    llvm::SmallVectorImpl<mlir::Attribute>& flattened_attr) {
  if (shape.IsToken()) {
    return absl::OkStatus();
  }
  if (shape.IsTuple()) {
    std::vector<mlir::Attribute> tuple_layouts;
    for (int i = 0; i < shape.tuple_shapes_size(); i++) {
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
