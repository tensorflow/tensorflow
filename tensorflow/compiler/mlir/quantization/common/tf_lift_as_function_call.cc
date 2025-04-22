/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/common/tf_lift_as_function_call.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>
#include <stack>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/Version.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/stablehlo_type_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/quantization_unit_loc.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/platform/mutex.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace mlir::tf_quant {

using ::stablehlo::quantization::Method;
using ::tsl::protobuf::TextFormat;

// Default version number for native serialization.
constexpr int64_t kDefaultVersion = 9;
// Default platform for XlaCallModuleOp.
constexpr StringRef kPlatformCpu = "CPU";
// Name of `tf.XlaCallModule`'s dictionary attribute for keeping the
// deserialized stablehlo module's attributes.
constexpr StringRef kStablehloModuleAttrsAttrName = "_stablehlo_module_attrs";
// Attribute required for running shape refinement pass enabled in XlaCallModule
// version 8 and above.
constexpr StringRef kUsesShapePolymorphismAttr = "jax.uses_shape_polymorphism";

bool IsInLiftedFunc(Operation* op) {
  if (op == nullptr) return false;
  return op->getParentOfType<func::FuncOp>()->hasAttr(kFusedFunctionAttr);
}

bool IsInStableHloOpRegion(Operation* op) {
  if (op == nullptr) return false;
  auto parent_op = op->getParentOp();
  return parent_op != nullptr && quant::stablehlo::IsStablehloOp(parent_op);
}

// Inserts the function to the symbol table of the module thread-safely.
StringAttr InsertToSymbolTable(Operation& module, Operation& function,
                               const StringRef func_name) {
  static tensorflow::mutex* mtx = new tensorflow::mutex();
  tensorflow::mutex_lock lock(*mtx);

  SymbolTable symbol_table(&module);
  std::string unique_name = func_name.str();
  int32_t uniquing_counter = 0;
  while (symbol_table.lookup(unique_name) != nullptr) {
    ++uniquing_counter;
    unique_name = absl::StrCat(func_name.str(), "_", uniquing_counter);
  }
  function.setAttr("sym_name",
                   StringAttr::get(module.getContext(), unique_name));
  return symbol_table.insert(&function);
}

// Creates the TF::PartitionedCallOp with the given arguments and output types.
// This function call op is for invoking the TF subgraphs.
ValueRange CreateTFPartitionedCallOp(OpBuilder& builder,
                                     const Location location,
                                     const StringRef func_name,
                                     const TypeRange output_types,
                                     const ValueRange args) {
  TF::PartitionedCallOp call_op = builder.create<TF::PartitionedCallOp>(
      location, output_types, args,
      /*args_attrs=*/nullptr, /*res_attrs=*/nullptr,
      FlatSymbolRefAttr::get(builder.getStringAttr(func_name)),
      /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");

  // Set the attribute to annotate this function call op as a quantizable spot.
  call_op->setAttr(
      kQuantTraitAttrName,
      builder.getStringAttr(StringRef(
          std::string(QuantTraitValues[QuantizationTrait::FullyQuantizable]))));

  return call_op.getOutput();
}

// Creates the TF::XlaCallModuleOp with the given arguments and output types.
// This function call op is for invoking the StableHLO subgraphs.
ValueRange CreateTFXlaCallModuleOp(OpBuilder& builder, const Location location,
                                   const StringRef func_name,
                                   const TypeRange output_types,
                                   const ValueRange args) {
  MLIRContext* ctx = builder.getContext();
  // Collect the shapes of the output to fill up the Sout attribute.
  SmallVector<Attribute> shape_attrs;
  for (const Type result_type : output_types) {
    shape_attrs.push_back(
        tf_type::ShapeAttr::get(ctx, mlir::cast<ShapedType>(result_type)));
  }
  auto empty_array_attr = ArrayAttr::get(ctx, {});
  auto platforms = ArrayAttr::get(ctx, {StringAttr::get(ctx, kPlatformCpu)});

  auto call_op = builder.create<TF::XlaCallModuleOp>(
      location,
      /*output=*/output_types,
      /*args=*/args,
      /*version=*/kDefaultVersion, /*module=*/"",
      /*Sout=*/ArrayAttr::get(ctx, shape_attrs),
      /*dim_args_spec=*/empty_array_attr,
      /*platforms=*/platforms,
      /*function_list=*/empty_array_attr,
      /*has_token_input_output=*/false,
      /*disabled_checks=*/empty_array_attr);

  // Set the function name. This will be controlled by the
  // XlaCallModuleSerialization related passes directly, which means that the
  // function name can be changed by those passes.
  call_op->setAttr(TF::kStablehloEntryFunctionAttrName,
                   FlatSymbolRefAttr::get(builder.getStringAttr(func_name)));

  // Set target version to WEEK_4 since this is an offline quantizer.
  std::string target_version =
      mlir::vhlo::Version::fromCompatibilityRequirement(
          vhlo::Version::CompatibilityRequirement::WEEK_4)
          .toString();
  call_op->setAttr(TF::kStablehloVersionAttrName,
                   builder.getStringAttr(target_version));

  // Store the custom attribute to restore the function name when loading it
  // back in the post calibration stage. As mentioned above, the above entry
  // function attribute is not reliable.
  call_op->setAttr(kOriginalStablehloEntryFunctionAttrName,
                   builder.getStringAttr(func_name));

  // Set the attribute to annotate this function call op as a quantizable spot.
  call_op->setAttr(
      kQuantTraitAttrName,
      builder.getStringAttr(StringRef(
          std::string(QuantTraitValues[QuantizationTrait::FullyQuantizable]))));

  // Set jax.uses_shape_polymorphism=true to enable shape refinement at runtime.
  // This is needed for native serialization version >= 8.
  call_op->setAttr(kStablehloModuleAttrsAttrName,
                   builder.getDictionaryAttr(builder.getNamedAttr(
                       kUsesShapePolymorphismAttr, builder.getBoolAttr(true))));

  return call_op.getOutput();
}

// Creates the function call op based on the given call_op_type argument.
ValueRange CreateFunctionCallOp(OpBuilder& builder, const Location location,
                                const FunctionCallOpType call_op_type,
                                const StringRef func_name,
                                const TypeRange output_types,
                                const ValueRange args) {
  switch (call_op_type) {
    case FunctionCallOpType::TFXlaCallModuleOp:
      return CreateTFXlaCallModuleOp(builder, location, func_name, output_types,
                                     args);
    case FunctionCallOpType::TFPartitionedCallOp:
      return CreateTFPartitionedCallOp(builder, location, func_name,
                                       output_types, args);
  }
}

// Finds ops in the paths from arguments to results. The ops is listed in an
// order that the former ops shouldn't have any dependencies on the later ones.
SmallVector<Operation*> FindOpsFromArgumentsToResults(
    const ArrayRef<Value> arguments, const ArrayRef<Value> results) {
  std::queue<Value> value_queue;
  for (Value result : results) {
    value_queue.push(result);
  }
  absl::flat_hash_set<mlir::detail::ValueImpl*> argument_set;
  for (Value argument : arguments) {
    argument_set.insert(argument.getImpl());
  }

  // Searching for ops from results to arguments. Duplicate ops in the op stack
  // are intentional in order to make sure the op on the top of the stack
  // doesn't depends on any ops below it.
  std::stack<Operation*> op_stack;
  while (!value_queue.empty()) {
    Value current_value = value_queue.front();
    value_queue.pop();

    Operation* defining_node = current_value.getDefiningOp();
    if (defining_node == nullptr) continue;
    op_stack.push(defining_node);
    for (Value arg : defining_node->getOperands()) {
      if (!argument_set.contains(arg.getImpl())) {
        value_queue.push(arg);
      }
    }
  }

  // Remove duplicate ops from the op stack.
  SmallVector<Operation*> sorted_ops;
  absl::flat_hash_set<Operation*> unique_ops;
  while (!op_stack.empty()) {
    Operation* current_op = op_stack.top();
    op_stack.pop();
    if (unique_ops.contains(current_op)) continue;
    sorted_ops.push_back(current_op);
    unique_ops.insert(current_op);
  }
  return sorted_ops;
}

// Finds the name of each attribute in `attributes` and set the attr_map
// attribute which maps an attribute identifier to its attribute name. The
// identifier is the order of that attribute in `attributes`. This map
// is then used to set attributes in the quantized functions in the
// QuantizeCompositeFunctionsPass.
// For example, for tf.MatMul with `attributes` = {{"transpose_a", false},
// {"transpose_b", false}}, the generated attr_map is
// "0:transpose_a,1:transpose_b", where 0 and 1 are the respective attribute
// identifiers.
// This function returns success if all attributes could be found.
LogicalResult SetAttributeMap(MLIRContext& context,
                              const ArrayRef<NamedAttribute> attributes,
                              const ArrayRef<Operation*> ops) {
  // A map to find which operation an attribute belongs to.
  // The key for this map uses the entire NamedAttribute object, i.e. the
  // {attribute_name, attribute_value} pair.
  llvm::SmallDenseMap<NamedAttribute, Operation*> attr_to_op_map;
  for (Operation* op : ops) {
    for (const NamedAttribute named_attr : op->getAttrs()) {
      attr_to_op_map.insert({named_attr, op});
    }
  }

  for (int idx : llvm::seq<int>(0, attributes.size())) {
    const NamedAttribute& attribute = attributes[idx];
    // Skip the following steps if the attribute value is `NullAttribute`.
    if (const auto string_attr =
            mlir::dyn_cast_or_null<StringAttr>(attribute.getValue());
        string_attr != nullptr &&
        string_attr.getValue() == kNullAttributeValue) {
      continue;
    }

    if (std::find_if(
            attr_to_op_map.begin(), attr_to_op_map.end(), [&](auto attr_op) {
              return std::get<0>(attr_op).getName() == attribute.getName();
            }) == attr_to_op_map.end()) {
      emitError(UnknownLoc::get(&context),
                "Could not find attribute: " + attribute.getName().str());
      return failure();
    }

    Operation* owner_op;
    for (const auto& [attr, val] : attr_to_op_map) {
      if (attr.getName() == attribute.getName()) owner_op = val;
    }
    if (quant::stablehlo::IsStablehloOp(owner_op)) {
      owner_op->setAttr(StringRef(attribute.getName()), attribute.getValue());
    } else {
      owner_op = attr_to_op_map[attribute];

      std::string new_attr_map_str{};
      if (owner_op->hasAttr(kAttrMapAttribute)) {
        new_attr_map_str =
            owner_op->getAttrOfType<StringAttr>(kAttrMapAttribute).str();
        absl::StrAppend(&new_attr_map_str, ",");
      }

      // Append "<identifier>:<attribute_name>". Ex) "0:transpose_a".
      const std::string identifier = std::to_string(idx);
      const StringAttr attribute_name = attribute.getName();
      absl::StrAppend(&new_attr_map_str, identifier, ":", attribute_name.str());
      owner_op->setAttr(kAttrMapAttribute,
                        StringAttr::get(&context, new_attr_map_str));
    }
  }
  return success();
}

// Creates a function to wrap the section between arguments and results.
SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder& builder, const Location location,
    const FunctionCallOpType call_op_type, const StringRef func_name,
    const ArrayRef<Value> arguments, const ArrayRef<Value> results,
    const ArrayRef<NamedAttribute> attributes) {
  MLIRContext* context = builder.getContext();
  if (results.empty()) {
    emitError(UnknownLoc::get(context), "No result values specified");
    return {};
  }
  Operation* result_op = results[0].getDefiningOp();
  auto module = result_op->getParentOfType<ModuleOp>();

  // Create a private function and copy all ops between arguments and results.
  auto current_func = result_op->getParentOfType<func::FuncOp>();
  auto guard = OpBuilder::InsertionGuard(builder);
  builder.setInsertionPointAfter(current_func);
  TypeRange arg_types{ValueRange{arguments}};
  TypeRange result_types{ValueRange{results}};
  auto func_type = FunctionType::get(context, arg_types, result_types);

  SmallVector<Location> arg_locs;
  for (Value arg : arguments) {
    arg_locs.push_back(arg.getLoc());
  }

  auto wrap_func = builder.create<func::FuncOp>(location, func_name, func_type);
  wrap_func.setVisibility(SymbolTable::Visibility::Private);
  // The callee function for TF::XlaCallModuleOp must have this attribute.
  if (call_op_type == FunctionCallOpType::TFXlaCallModuleOp) {
    wrap_func->setAttr(TF::kFromXlaCallModuleAttrName, builder.getUnitAttr());
  }
  wrap_func->setAttr(kFusedFunctionAttr, builder.getUnitAttr());
  builder.createBlock(&wrap_func.getBody(), wrap_func.begin(), arg_types,
                      arg_locs);

  IRMapping mapping;
  for (int32_t i : llvm::seq<int32_t>(0, arguments.size())) {
    mapping.map(arguments[i], wrap_func.getArgument(i));
  }

  auto cloning_ops = FindOpsFromArgumentsToResults(arguments, results);
  // Set the location of call op to QuantizationUnitLoc if found.
  Location call_op_loc = location;
  for (Operation* op : cloning_ops) {
    std::optional<quant::QuantizationUnitLoc::QuantizationUnit> unit =
        quant::FindQuantizationUnitFromLoc(op->getLoc());
    if (unit.has_value()) {
      call_op_loc =
          quant::QuantizationUnitLoc(builder.getContext(), unit.value());
    }
  }

  if (failed(SetAttributeMap(*context, attributes, cloning_ops))) {
    current_func.emitError() << "Some attributes couldn't be found.";
  }
  for (Operation* op : cloning_ops) {
    builder.clone(*op, mapping);
  }

  SmallVector<Value> return_values;
  for (Value result : results) {
    return_values.push_back(mapping.lookupOrNull(result));
  }
  builder.create<func::ReturnOp>(location, return_values);

  // Create a function call to the newly created function.
  StringAttr new_func_name =
      InsertToSymbolTable(*module, *wrap_func, func_name);
  builder.setInsertionPointAfter(result_op);
  ValueRange new_results =
      CreateFunctionCallOp(builder, call_op_loc, call_op_type,
                           new_func_name.getValue(), result_types, arguments);
  return SmallVector<Value, 4>(new_results.begin(), new_results.end());
}

SmallVector<Value, 4> LiftAsFunctionCall(OpBuilder& builder,
                                         const Location location,
                                         const FunctionCallOpType call_op_type,
                                         const StringRef func_name,
                                         const ArrayRef<Value> arguments,
                                         const ArrayRef<Value> results) {
  SmallVector<NamedAttribute> attributes;
  return LiftAsFunctionCall(builder, location, call_op_type, func_name,
                            arguments, results, attributes);
}

SmallVector<Value> AppendToVector(const ArrayRef<Value> arguments,
                                  Value append) {
  SmallVector<Value> ret(arguments);
  ret.push_back(append);
  return ret;
}

// Check if the given einsum equation is supported by XlaDotV2.
// Conditions:
// 1. Two inputs & one output.
// 2. No ... in the equation.
// 3. Batch dimensions should be the same, or only the left equation should have
//    the batch dimension. This condition is from the XlaDotV2 specification. It
//    could process the following equation by setting the attributes properly:
//    abc,cd->abd.
// 4. The output should be in the form: [batch dims][lhs dims][rhs dims]
bool IsEinsumSupportedByXlaDotV2(StringAttr equation_attr) {
  StringRef equation = equation_attr.getValue();

  if (!absl::StrContains(equation, "->") || !absl::StrContains(equation, ",") ||
      absl::StrContains(equation, ".")) {
    return false;
  }

  // Parse equation.
  int idx_arrow = equation.find("->");
  StringRef calc_eq = equation.substr(0, idx_arrow);
  StringRef out_eq = equation.substr(idx_arrow + 2);

  int idx_comma = calc_eq.find(',');
  StringRef lhs_eq = calc_eq.substr(0, idx_comma);
  StringRef rhs_eq = calc_eq.substr(idx_comma + 1);

  if (absl::StrContains(rhs_eq, ",")) return false;

  int lhs_out_idx_start = out_eq.size();
  int lhs_out_idx_end = -1;
  int rhs_out_idx_start = out_eq.size();
  int rhs_out_idx_end = -1;
  int lhs_batch_dim_size = 0;
  int rhs_batch_dim_size = 0;
  for (const char c : lhs_eq) {
    if (absl::StrContains(out_eq, c) && absl::StrContains(rhs_eq, c)) {
      lhs_batch_dim_size++;
    } else if (absl::StrContains(out_eq, c)) {
      const int out_idx = out_eq.find(c);
      if (out_idx < lhs_out_idx_end) {
        // Left-hand equation is reversed in the output.
        return false;
      }
      lhs_out_idx_start = std::min(lhs_out_idx_start, out_idx);
      lhs_out_idx_end = std::max(lhs_out_idx_end, out_idx);
    }
  }

  for (const char c : rhs_eq) {
    if (absl::StrContains(out_eq, c) && absl::StrContains(lhs_eq, c)) {
      rhs_batch_dim_size++;
    } else if (absl::StrContains(out_eq, c)) {
      int out_idx = out_eq.find(c);
      if (out_idx < rhs_out_idx_end) {
        return false;
      }
      if (out_idx < rhs_out_idx_start) rhs_out_idx_start = out_idx;
      if (out_idx > rhs_out_idx_end) rhs_out_idx_end = out_idx;
    }
  }

  if (lhs_batch_dim_size != rhs_batch_dim_size && lhs_batch_dim_size != 0 &&
      rhs_batch_dim_size != 0) {
    // Batch dimension does not match.
    return false;
  }

  // All the lhs equations should come first.
  if (lhs_out_idx_end > rhs_out_idx_start) return false;

  // All the lhs out dim and rhs out dim should be larger than the batch dims,
  // and they should not be mixed.
  int batch_dim_size = std::max(rhs_batch_dim_size, lhs_batch_dim_size);
  return lhs_out_idx_start >= batch_dim_size &&
         rhs_out_idx_start >= batch_dim_size;
}

absl::StatusOr<Method> GetQuantizationMethod(Operation* /*absl_nonnull*/ op) {
  const auto quantization_method_attr =
      op->getAttrOfType<StringAttr>(kQuantizationMethodAttr);
  if (!quantization_method_attr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Attribute ", kQuantizationMethodAttr.str(), " is not found."));
  }

  Method quantization_method;
  const std::string method_txtpb = quantization_method_attr.getValue().str();
  if (!TextFormat::ParseFromString(method_txtpb, &quantization_method)) {
    return absl::InternalError(
        absl::StrCat("Failed to parse Method from textproto: ", method_txtpb));
  }

  return quantization_method;
}

Method GetQuantizationMethodOrDefault(Operation* /*absl_nonnull*/ op) {
  absl::StatusOr<Method> method = GetQuantizationMethod(op);
  if (method.status().code() == absl::StatusCode::kInternal) {
    // This indicates that the `Method` protobuf string is corrupt, but this
    // function ignores it and returns the default instance.
    op->emitError(absl::StrCat("Failed to get quantization method: ",
                               method.status().ToString()));
  }
  return method.ok() ? *method : Method::default_instance();
}

bool HasWeightOnlyPtqMethod(TF::XlaCallModuleOp xla_call_module_op) {
  Method method = GetQuantizationMethodOrDefault(xla_call_module_op);
  return method.has_weight_only_ptq();
}

bool IsWeightOnlyQuantizableOp(const Operation& op) {
  if (auto call_op = dyn_cast<TF::XlaCallModuleOp>(op)) {
    StringRef entry_function_name = GetEntryFunctionName(call_op);
    absl::StatusOr<Method> quantization_method = GetQuantizationMethod(call_op);
    return ContainsConvOrDot(entry_function_name) && quantization_method.ok() &&
           quantization_method->has_weight_only_ptq();
  }
  return false;
}

SmallVector<func::FuncOp> GetSortedFunctions(ModuleOp module_op) {
  auto iterator_range = module_op.getOps<func::FuncOp>();
  SmallVector<func::FuncOp> func_ops(iterator_range.begin(),
                                     iterator_range.end());
  absl::c_sort(func_ops, [](func::FuncOp op1, func::FuncOp op2) {
    return op1.getName() < op2.getName();
  });
  return func_ops;
}

}  // namespace mlir::tf_quant
