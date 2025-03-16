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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "json/reader.h"
#include "json/value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {

#define GEN_PASS_DEF_BUILDSTABLEHLOCOMPOSITEPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

namespace {

// Checks if this operation is a MarkTensor operation used to mark the
// boundaries of a composite.
static bool IsMarkTensorOp(mlir::Operation* op) {
  if (op == nullptr) {
    return false;
  }
  if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
    return false;
  }
  if (!llvm::isa<mlir::stablehlo::CustomCallOp>(op)) {
    return false;
  }
  auto target_name =
      mlir::dyn_cast<mlir::StringAttr>(op->getAttr("call_target_name"));
  if (target_name == nullptr || target_name.str() != "mark_tensor") {
    return false;
  }
  return true;
}

struct BoundaryMetadata {
  std::string name;
  std::string id;
  int64_t pos;
  bool is_input;
  std::unordered_map<std::string, Json::Value> attrs;

  auto boundary_key() const { return absl::StrCat(name, "__@@__", id); }

  auto uid() const { return std::forward_as_tuple(name, id, pos, is_input); }

  bool operator==(const BoundaryMetadata& other) const {
    return uid() == other.uid();
  }
  bool operator<(const BoundaryMetadata& other) const {
    return uid() < other.uid();
  }

  static std::unique_ptr<BoundaryMetadata> Parse(llvm::StringRef str_ref) {
    Json::Value root;
    Json::Reader reader;
    if (!reader.parse(str_ref.str(), root)) {
      return nullptr;
    }
    return Build(root);
  }

 private:
  template <typename T>
  static bool CopyJsonValue(const Json::Value& json, llvm::StringRef key,
                            Json::ValueType expected_type, T* to) {
    if (!json.isMember(key.str()) || json[key.str()].type() != expected_type) {
      return false;
    }

    *to = json[key.str()].as<T>();
    return true;
  }

  static std::unique_ptr<BoundaryMetadata> Build(const Json::Value& json) {
    BoundaryMetadata metadata;

    bool is_valid_metadata_json =
        CopyJsonValue(json, "name", Json::stringValue, &metadata.name) &&
        CopyJsonValue(json, "id", Json::stringValue, &metadata.id) &&
        CopyJsonValue(json, "pos", Json::intValue, &metadata.pos) &&
        CopyJsonValue(json, "is_input", Json::booleanValue, &metadata.is_input);

    if (!is_valid_metadata_json) {
      return nullptr;
    }

    Json::Value attrs_value = json["attr"];
    if (attrs_value.type() == Json::objectValue) {
      for (const auto& key_value : attrs_value.getMemberNames()) {
        metadata.attrs.insert({key_value, attrs_value[key_value]});
      }
    }
    return std::make_unique<BoundaryMetadata>(std::move(metadata));
  }
};

class BuildStableHLOCompositePass
    : public impl::BuildStableHLOCompositePassBase<
          BuildStableHLOCompositePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuildStableHLOCompositePass);

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    llvm::SmallVector<mlir::func::FuncOp> func_ops(
        module_op.getOps<mlir::func::FuncOp>());
    for (mlir::func::FuncOp& func_op : func_ops) {
      llvm::DenseMap<const mlir::Operation*, size_t> op_order_map =
          BuildOpOrderMap(func_op);
      llvm::SmallVector<llvm::SmallVector<mlir::Operation*>>
          boundary_output_ops_map = BuildBoundaryOutputOpsMap(func_op);

      for (const auto& ops : boundary_output_ops_map) {
        if (mlir::failed(BuildStableHLOComposite(ops, op_order_map))) {
          func_op.emitError() << "failed to build composite.";
          return signalPassFailure();
        }
      }
    }

    // Remove mark_tensor custom_call ops.
    getOperation()->walk([](mlir::stablehlo::CustomCallOp op) {
      if (!IsMarkTensorOp(op.getOperation())) {
        return;
      }
      mlir::Value original_value = op.getOperand(0);

      for (mlir::Value result : op.getResults()) {
        result.replaceAllUsesWith(original_value);
      }
      op.erase();
    });

    getOperation()->walk([&](mlir::stablehlo::CompositeOp composite_op) {
      if (IsCompositeOpOutlined(composite_op)) {
        // According to the design of torch API, each boundary has one and only
        // one usage in the entire model, which corresponds to one unique
        // composite op in module. If the composite op is marked as
        // outlined, there should be a same composite op nested in another
        // composite op, and this composite op should be unused and safe to
        // erase.
        EraseCompositeOp(composite_op);
      }
    });
  }

 private:
  void MarkCompositeOpOutlined(mlir::stablehlo::CompositeOp composite_op) {
    composite_op->setAttr("_outlined_to_be_erased",
                          mlir::UnitAttr::get(composite_op.getContext()));
  }

  bool IsCompositeOpOutlined(mlir::stablehlo::CompositeOp composite_op) {
    return composite_op->hasAttr("_outlined_to_be_erased");
  }

  void EraseCompositeOp(mlir::stablehlo::CompositeOp composite_op) {
    // CompositeOps have side effects and are not eligible for CSE. This
    // function replaces a CompositeOp with a semantically equivalent
    // 'dummy_to_be_erased' CustomCallOp that *lacks* side effects. This allows
    // the CSE pass to eliminate the dummy op and, transitively, any users of
    // the original CompositeOp.
    mlir::OpBuilder builder(&getContext());

    builder.setInsertionPointAfter(composite_op);
    auto dummy_op = builder.create<mlir::stablehlo::CustomCallOp>(
        composite_op.getLoc(), composite_op.getResultTypes(),
        composite_op.getOperands(),
        llvm::SmallVector<NamedAttribute>{
            builder.getNamedAttr("call_target_name",
                                 builder.getStringAttr("dummy_to_be_erased")),
            builder.getNamedAttr(
                "has_side_effect",
                builder.getIntegerAttr(builder.getI1Type(), 0)),
        });

    for (auto [old_result, new_result] :
         llvm::zip(composite_op.getResults(), dummy_op.getResults())) {
      old_result.replaceAllUsesWith(new_result);
    }
    composite_op.erase();
  }

  llvm::DenseMap<const mlir::Operation*, size_t> BuildOpOrderMap(
      mlir::func::FuncOp func_op) const {
    llvm::DenseMap<const mlir::Operation*, size_t> op_order_map;
    for (const auto& op : llvm::enumerate(func_op.getOps())) {
      op_order_map[&op.value()] = op.index();
    }
    return op_order_map;
  }

  llvm::SmallVector<llvm::SmallVector<mlir::Operation*>>
  BuildBoundaryOutputOpsMap(mlir::func::FuncOp func_op) {
    std::unordered_map<std::string, llvm::SmallVector<mlir::Operation*>>
        boundary_output_ops;

    llvm::SetVector<std::string, std::vector<std::string>,
                    std::unordered_set<std::string>>
        ordered_boundary_keys;

    for (auto op : func_op.getOps<mlir::stablehlo::CustomCallOp>()) {
      auto metadata_or = GetBoundaryMetadata(op);
      if (mlir::failed(metadata_or)) {
        continue;
      }

      std::unique_ptr<BoundaryMetadata> metadata = std::move(*metadata_or);
      if (metadata == nullptr || metadata->is_input) {
        continue;
      }

      std::string boundary_key = metadata->boundary_key();
      auto& output_ops = boundary_output_ops[boundary_key];
      if (metadata->pos >= output_ops.size()) {
        output_ops.resize(metadata->pos + 1, nullptr);
      }
      output_ops[metadata->pos] = op.getOperation();

      // Update the boundary order, which is determined by the order of the
      // last output op of each boundary.
      ordered_boundary_keys.remove(boundary_key);
      ordered_boundary_keys.insert(boundary_key);
    }

    llvm::SmallVector<llvm::SmallVector<mlir::Operation*>>
        ordered_boundary_output_ops;
    for (const auto& boundary_key : ordered_boundary_keys) {
      ordered_boundary_output_ops.push_back(
          std::move(boundary_output_ops[boundary_key]));
    }
    return ordered_boundary_output_ops;
  }

  mlir::FailureOr<std::unique_ptr<BoundaryMetadata>> GetBoundaryMetadata(
      mlir::Operation* op) {
    if (!IsMarkTensorOp(op)) {
      return mlir::FailureOr<std::unique_ptr<BoundaryMetadata>>(nullptr);
    }
    auto backend_config =
        mlir::dyn_cast<mlir::StringAttr>(op->getAttr("backend_config"));
    if (backend_config == nullptr) {
      return mlir::FailureOr<std::unique_ptr<BoundaryMetadata>>(nullptr);
    }
    std::unique_ptr<BoundaryMetadata> metadata =
        BoundaryMetadata::Parse(backend_config);
    if (metadata == nullptr) {
      return op->emitError() << "invalid boundary metadata JSON.";
    }
    return metadata;
  }

  mlir::FailureOr<mlir::Attribute> BuildAttrFromJson(
      mlir::OpBuilder& builder, mlir::Operation* op,
      const Json::Value& json_value) {
    switch (json_value.type()) {
      case Json::intValue:
      case Json::uintValue:
        return builder.getI64IntegerAttr(json_value.as<int64_t>());
      case Json::ValueType::realValue:
        return builder.getF32FloatAttr(json_value.as<float>());
      case Json::ValueType::booleanValue:
        return builder.getBoolAttr(json_value.as<bool>());
      case Json::ValueType::stringValue:
        return builder.getStringAttr(json_value.as<std::string>());
      case Json::ValueType::arrayValue: {
        if (json_value.empty()) {
          return builder.getArrayAttr({});
        }
        auto get_json_type = [](const Json::Value& json_value) {
          auto ty = json_value.type();
          if (ty == Json::uintValue) {
            return Json::intValue;
          }
          return ty;
        };

        auto head_type = get_json_type(json_value[0]);
        bool is_homogeneous = llvm::all_of(json_value, [&](auto& el) {
          return get_json_type(el) == head_type;
        });
        if (!is_homogeneous) {
          return op->emitError()
                 << "invalid JSON to MLIR, arrays must be homogeneous";
        }

        switch (head_type) {
          case Json::intValue: {
            llvm::SmallVector<int64_t> int_values;
            for (const auto& json_value : json_value) {
              int_values.push_back(json_value.as<int64_t>());
            }
            return builder.getI64TensorAttr(int_values);
          }
          case Json::realValue: {
            llvm::SmallVector<float> float_values;
            for (const auto& json_value : json_value) {
              float_values.push_back(json_value.as<float>());
            }
            return mlir::DenseFPElementsAttr::get(
                mlir::RankedTensorType::get(json_value.size(),
                                            builder.getF32Type()),
                float_values);
          }
          case Json::booleanValue: {
            llvm::SmallVector<bool> bool_values;
            for (const auto& json_value : json_value) {
              bool_values.push_back(json_value.as<bool>());
            }
            return mlir::DenseIntElementsAttr::get(
                mlir::RankedTensorType::get(json_value.size(),
                                            builder.getI1Type()),
                bool_values);
          }
          default:
            return op->emitError()
                   << "invalid JSON to MLIR: invalid array type. arrays must "
                      "be "
                      "1-D homogeneous arrays of supported primitive types";
        }
      }
      default:
        return op->emitError()
               << "invalid JSON to MLIR: unsupported json value type";
    }
  }

  mlir::FailureOr<mlir::DictionaryAttr> BuildDictionaryAttrFromJsonMap(
      mlir::OpBuilder& builder, mlir::Operation* op,
      const std::unordered_map<std::string, Json::Value>& json_map) {
    llvm::SmallVector<mlir::NamedAttribute> named_attrs;
    for (auto& [key, json] : json_map) {
      mlir::FailureOr<mlir::Attribute> attribute_or =
          BuildAttrFromJson(builder, op, json);
      if (mlir::failed(attribute_or)) {
        return mlir::failure();
      }
      named_attrs.push_back({builder.getStringAttr(key), *attribute_or});
    }
    return builder.getDictionaryAttr(named_attrs);
  }

  mlir::LogicalResult BuildStableHLOComposite(
      const llvm::SmallVector<mlir::Operation*>& output_ops,
      llvm::DenseMap<const mlir::Operation*, size_t>& op_order_map) {
    if (output_ops.empty()) {
      return mlir::success();
    }

    // Get the output op with minimum order num as the representative.
    mlir::Operation* first_output_op = output_ops[0];
    for (mlir::Operation* op : output_ops) {
      if (op_order_map.at(op) < op_order_map.at(first_output_op)) {
        first_output_op = op;
      }
    }

    auto metadata_or = GetBoundaryMetadata(first_output_op);
    if (mlir::failed(metadata_or)) {
      return mlir::failure();
    }

    std::unique_ptr<BoundaryMetadata> metadata = std::move(*metadata_or);
    if (metadata == nullptr || metadata->is_input) {
      // There should always be a valid boundary output metadata associated with
      // each op in output_ops.
      return mlir::failure();
    }

    auto args_ops_or =
        GetBoundaryArgsAndOps(output_ops, *metadata, op_order_map);
    if (mlir::failed(args_ops_or)) {
      return mlir::failure();
    }

    auto [args, impl_ops] = *args_ops_or;

    mlir::func::FuncOp impl_func = BuildStableHLOCompositeImplFunc(
        output_ops, absl::StrCat(metadata->name, ".impl"), args, impl_ops);
    mlir::FailureOr<mlir::Operation*> composite_op_or =
        BuildStableHLOCompositeOp(first_output_op, impl_func, args, *metadata);
    if (mlir::failed(composite_op_or)) {
      return mlir::failure();
    }
    mlir::Operation* composite_op = *composite_op_or;

    // Updates all users of this op's result(s) to use the results(s) of impl
    // func call.
    size_t composite_result_i = 0;
    for (mlir::Operation* op : output_ops) {
      for (size_t i = 0; i < op->getNumResults(); ++i) {
        mlir::OpResult result = op->getResult(i);
        result.replaceAllUsesWith(
            composite_op->getResult(composite_result_i++));
      }
    }

    // Composite op inherits the order of the first output op. The output ops
    // are not going to be used after this anyway so the duplication is fine.
    op_order_map[composite_op] = op_order_map[first_output_op];

    if (!mlir::sortTopologically(composite_op->getBlock())) {
      composite_op->emitError()
          << "The graph is not acyclic after BuildStableHLOCompositePass pass.";
      return mlir::failure();
    }
    // The unused impl_ops will be eliminated with canonicalizer.
    return mlir::success();
  }

  mlir::FailureOr<std::pair<llvm::SmallVector<mlir::Value>,
                            llvm::SmallVector<mlir::Operation*>>>
  GetBoundaryArgsAndOps(
      const llvm::SmallVector<mlir::Operation*> boundary_output_ops,
      const BoundaryMetadata& metadata,
      const llvm::DenseMap<const mlir::Operation*, size_t>& op_order_map) {
    llvm::SetVector<mlir::Operation*> impl_ops_setvec;
    llvm::SetVector<std::pair<mlir::Value, int64_t>> arg_pos_setvec;
    llvm::SmallVector<mlir::Operation*> processing(boundary_output_ops.begin(),
                                                   boundary_output_ops.end());

    // Reverse graph traversal: from boundary output op to boundary input op,
    // global function arg, or stablehlo constant.
    while (!processing.empty()) {
      mlir::Operation* curr_op = processing.back();
      processing.pop_back();
      if (impl_ops_setvec.contains(curr_op)) {
        continue;
      }

      auto curr_metadata_or = GetBoundaryMetadata(curr_op);
      if (mlir::failed(curr_metadata_or)) {
        return mlir::failure();
      }
      std::unique_ptr<BoundaryMetadata> curr_metadata =
          std::move(*curr_metadata_or);
      if (curr_metadata != nullptr) {
        if (curr_metadata->is_input &&
            curr_metadata->boundary_key() == metadata.boundary_key()) {
          // Terminal condition: boundary input op.

          arg_pos_setvec.insert(
              {mlir::dyn_cast<mlir::Value>(curr_op->getResult(0)),
               curr_metadata->pos});
          continue;
        }
      }

      impl_ops_setvec.insert(curr_op);
      for (mlir::Value value : curr_op->getOperands()) {
        mlir::Operation* def_op = value.getDefiningOp();
        if (def_op == nullptr) {
          // Terminal condition: global function arg
          arg_pos_setvec.insert({value, std::numeric_limits<int64_t>::max()});
        } else if (def_op->hasTrait<OpTrait::ConstantLike>()) {
          // Terminal condition: constant
          impl_ops_setvec.insert(def_op);
        } else {
          processing.push_back(def_op);
        }
      }
    }
    // Sorts all ops within the boundary by their line numbers in the input
    // MLIR. The ops will be duplicated to the impl function following this
    // order.
    llvm::SmallVector<mlir::Operation*> impl_ops = impl_ops_setvec.takeVector();
    for (auto& op : impl_ops) {
      if (!op_order_map.contains(op)) {
        return op->emitError()
               << "does not have a ordering number in its outer func.";
      }
    }
    std::sort(impl_ops.begin(), impl_ops.end(),
              [&op_order_map](const auto& a, const auto& b) {
                return op_order_map.at(a) < op_order_map.at(b);
              });

    // Sorts boundary args by their positions. Note that the args of the
    // composite and impl function may be more than the boundary inputs, because
    // the MLIR is lowered from the functionalized graph and additional args may
    // be Pytorch constants. In such case the position of those args would be
    // undetermined, while they would always come after boundary inputs.
    auto arg_pos_pairs = arg_pos_setvec.takeVector();
    std::stable_sort(
        arg_pos_pairs.begin(), arg_pos_pairs.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    llvm::SmallVector<mlir::Value> args;
    args.reserve(arg_pos_pairs.size());
    for (auto& [arg, unused] : arg_pos_pairs) {
      args.push_back(arg);
    }

    return std::make_pair(std::move(args), std::move(impl_ops));
  }

  mlir::func::FuncOp BuildStableHLOCompositeImplFunc(
      const llvm::SmallVector<mlir::Operation*> boundary_output_ops,
      llvm::StringRef func_name, const llvm::SmallVector<mlir::Value>& args,
      const llvm::SmallVector<mlir::Operation*>& impl_ops) {
    mlir::ModuleOp module_op = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::OpBuilder builder(context);

    // Creates composite impl function and duplicates all ops within the
    // boundary in the function.
    llvm::SmallVector<mlir::Location> arg_locs;
    llvm::SmallVector<mlir::Type> arg_types;
    for (auto& arg : args) {
      arg_types.push_back(arg.getType());
      arg_locs.push_back(arg.getLoc());
    }
    llvm::SmallVector<mlir::Type> result_types;
    for (mlir::Operation* op : boundary_output_ops) {
      result_types.append(op->getResultTypes().begin(),
                          op->getResultTypes().end());
    }

    mlir::func::FuncOp impl_func = builder.create<mlir::func::FuncOp>(
        module_op.getLoc(), func_name,
        mlir::FunctionType::get(context, arg_types, result_types));
    mlir::IRMapping mapping;
    builder.createBlock(&impl_func.getBody(), impl_func.begin(), arg_types,
                        arg_locs);
    for (const auto& arg : llvm::enumerate(args)) {
      mapping.map(arg.value(), impl_func.getArgument(arg.index()));
    }
    for (mlir::Operation* original_op : impl_ops) {
      mlir::Operation* cloned_op = builder.clone(*original_op, mapping);
      mapping.map(original_op, cloned_op);
      if (auto original_composite_op =
              mlir::dyn_cast<mlir::stablehlo::CompositeOp>(original_op);
          original_composite_op != nullptr) {
        MarkCompositeOpOutlined(original_composite_op);
      }
    }

    llvm::SmallVector<mlir::Value> results;
    for (mlir::Operation* op : boundary_output_ops) {
      results.append(mapping.lookup(op)->getResults().begin(),
                     mapping.lookup(op)->getResults().end());
    }
    builder.create<mlir::func::ReturnOp>(impl_func.getBody().getLoc(), results);

    // Adds the new function to symbol table.
    mlir::SymbolTable symbol_table(module_op);
    impl_func.setPrivate();
    symbol_table.insert(impl_func);

    return impl_func;
  }

  mlir::FailureOr<mlir::Operation*> BuildStableHLOCompositeOp(
      mlir::Operation* boundary_output_op, mlir::func::FuncOp impl_func,
      const llvm::SmallVector<mlir::Value>& args,
      const BoundaryMetadata& metadata) {
    mlir::MLIRContext* context = &getContext();
    mlir::OpBuilder builder(context);

    mlir::FailureOr<mlir::DictionaryAttr> attributes_or =
        BuildDictionaryAttrFromJsonMap(builder, boundary_output_op,
                                       metadata.attrs);
    if (mlir::failed(attributes_or)) {
      return boundary_output_op->emitError()
             << "failed to transform boundary attr "
                "JSON into composite attributes.";
    }

    // Creates and inserts composite call op.
    builder.setInsertionPointAfter(boundary_output_op);
    mlir::Operation* composite_op =
        builder.create<mlir::stablehlo::CompositeOp>(
            boundary_output_op->getLoc(),
            impl_func.getFunctionType().getResults(), args, metadata.name,
            *attributes_or, impl_func.getSymName());
    return composite_op;
  }
};

static PassRegistration<BuildStableHLOCompositePass> pass;

}  // namespace
}  // namespace odml
}  // namespace mlir
