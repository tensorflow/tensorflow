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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

#include <algorithm>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_saved_model {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool IsStrArrayAttr(Attribute attr) {
  auto array = mlir::dyn_cast<ArrayAttr>(attr);
  if (!array) return false;

  return llvm::all_of(
      array, [](Attribute attr) { return mlir::isa<StringAttr>(attr); });
}

//===----------------------------------------------------------------------===//
// TensorFlowSavedModelDialect Op's
//===----------------------------------------------------------------------===//

LogicalResult VerifyTensorTypesCompatible(Type t1, Type t2) {
  if (!mlir::isa<TensorType>(t1) || !mlir::isa<TensorType>(t2)) {
    return failure();
  }
  return verifyCompatibleShape(mlir::cast<TensorType>(t1),
                               mlir::cast<TensorType>(t2));
}

LogicalResult GlobalTensorOp::verify() {
  GlobalTensorOp global_tensor = *this;
  if (global_tensor.getValue()) {
    if (failed(VerifyTensorTypesCompatible(
            global_tensor.getType(), global_tensor.getValue()->getType()))) {
      return global_tensor.emitError()
             << "'type' and 'value' attributes should "
                "have compatible tensor types";
    }
  }
  if (!global_tensor.getIsMutable()) {
    if (!mlir::cast<TensorType>(global_tensor.getType()).hasStaticShape()) {
      return global_tensor.emitError()
             << "'type' attribute for immutable 'tf_saved_model.global_tensor' "
                "should have a static shape";
    }
  }
  return success();
}

LogicalResult SessionInitializerOp::verify() {
  SessionInitializerOp session_initializer = *this;
  mlir::SymbolTable symbol_table(
      session_initializer->getParentOfType<ModuleOp>());

  for (auto sym_ref : session_initializer.getInitializers()) {
    auto init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
        mlir::cast<FlatSymbolRefAttr>(sym_ref).getValue());

    if (!init_func_op)
      return session_initializer.emitOpError()
             << "the initializer function does not exist";

    if (!init_func_op.getFunctionType().getResults().empty())
      return session_initializer.emitOpError()
             << "the initializer function should have no output";

    auto exported_names = GetExportedNames(init_func_op);

    if (exported_names.empty())
      return session_initializer.emitOpError()
             << "the initializer function should be exported";

    if (exported_names.size() != 1)
      return session_initializer.emitOpError()
             << "the initializer function should have only one exported names";
  }

  return success();
}

}  // namespace tf_saved_model
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc.inc"

namespace mlir {
namespace tf_saved_model {

//===----------------------------------------------------------------------===//
// TensorFlowSavedModelDialect Dialect
//===----------------------------------------------------------------------===//

TensorFlowSavedModelDialect::TensorFlowSavedModelDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf_saved_model", context,
              TypeID::get<TensorFlowSavedModelDialect>()) {
  // The TensorFlow Dialect is needed in the verifier and other routines
  // associated to this dialect. It makes little sense anyway to use the
  // SavedModel dialect without the TensorFlow Dialect.
  context->loadDialect<TF::TensorFlowDialect>();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc.inc"
      >();
}

static LogicalResult VerifyIndexPath(Operation *op, NamedAttribute named_attr) {
  auto attr = mlir::dyn_cast<ArrayAttr>(named_attr.getValue());
  if (!attr) {
    return op->emitError() << "'" << kTfSavedModelIndexPathAttr
                           << "' attribute should be an ArrayAttr";
  }
  for (auto element : attr) {
    if (mlir::isa<StringAttr>(element)) {
      continue;
    }
    if (auto integer = mlir::dyn_cast<IntegerAttr>(element)) {
      if (integer.getValue().getBitWidth() == 64) {
        continue;
      }
    }
    return op->emitError() << "'" << kTfSavedModelIndexPathAttr
                           << "' elements should be strings or 64-bit integers";
  }
  return mlir::success();
}

Type GetBoundInputArgTypeFor(mlir::Operation *op) {
  if (auto global_tensor = llvm::dyn_cast<GlobalTensorOp>(op)) {
    auto type = mlir::cast<TensorType>(global_tensor.getType());
    return RankedTensorType::get(
        {}, TF::ResourceType::get({type}, type.getContext()));
  }

  if (auto asset = llvm::dyn_cast<AssetOp>(op)) {
    return RankedTensorType::get({}, TF::StringType::get(asset.getContext()));
  }

  op->emitError() << "unknown symbol operation";
  return {};
}

static LogicalResult VerifyBoundInputArgType(Operation *op_for_diagnostics,
                                             Type arg_type,
                                             mlir::Operation *symbol_op) {
  auto expected_type = GetBoundInputArgTypeFor(symbol_op);
  if (!expected_type) return failure();

  if (arg_type != expected_type) {
    return op_for_diagnostics->emitError()
           << "bound input with type " << arg_type << " expected to have type "
           << expected_type;
  }
  return success();
}

LogicalResult TensorFlowSavedModelDialect::verifyRegionArgAttribute(
    Operation *op, unsigned region_index, unsigned arg_index,
    NamedAttribute named_attr) {
  if (named_attr.getName() == "tf_saved_model.bound_input") {
    if (!mlir::isa<FlatSymbolRefAttr>(named_attr.getValue())) {
      return op->emitError() << "'tf_saved_model.bound_input' attribute should "
                                "be a FlatSymbolRefAttr";
    }
    auto symbol_name =
        mlir::cast<FlatSymbolRefAttr>(named_attr.getValue()).getValue();
    auto module = op->getParentOfType<ModuleOp>();
    mlir::Operation *symbol_op = module.lookupSymbol(symbol_name);
    if (!symbol_op) {
      return op->emitError() << "'tf_saved_model.bound_input' attribute must "
                                "reference a valid symbol, got invalid symbol '"
                             << symbol_name << "'";
    }
    auto arg_type = cast<func::FuncOp>(op).getArgument(arg_index).getType();
    return VerifyBoundInputArgType(op, arg_type, symbol_op);
  }
  if (named_attr.getName() == kTfSavedModelIndexPathAttr) {
    return VerifyIndexPath(op, named_attr);
  }

  return op->emitError() << "unknown tf_saved_model dialect arg attribute '"
                         << named_attr.getName().getValue() << "'";
}

LogicalResult TensorFlowSavedModelDialect::verifyRegionResultAttribute(
    Operation *op, unsigned region_index, unsigned result_index,
    NamedAttribute named_attr) {
  if (named_attr.getName() == kTfSavedModelIndexPathAttr) {
    return VerifyIndexPath(op, named_attr);
  }

  return op->emitError() << "unknown tf_saved_model dialect result attribute '"
                         << named_attr.getName().getValue() << "'";
}

// Verifies the session_init op. There should be no duplicate
// tf_saved_model.initializer_type values.
LogicalResult VerifySessionInitOp(SessionInitializerOp session_init_op,
                                  SymbolTable &symbol_table) {
  // Stores previously seen init types.
  llvm::SmallDenseSet<StringAttr> init_types{};

  for (auto init_sym :
       session_init_op.getInitializers().getAsValueRange<FlatSymbolRefAttr>()) {
    auto init_func = symbol_table.lookup<func::FuncOp>(init_sym);
    if (!init_func) continue;

    auto init_type =
        init_func->getAttrOfType<StringAttr>(kTfSavedModelInitializerTypeAttr);
    if (!init_type) continue;

    if (init_types.contains(init_type)) {
      return init_func->emitError()
             << "Attribute tf_saved_model.initializer_type should not have "
                "duplicate values. Found duplicate: "
             << init_type;
    }

    init_types.insert(init_type);
  }

  return success();
}

static bool HasAnyTfSavedModelArgAttr(func::FuncOp func) {
  for (int i = 0, e = func.getNumArguments(); i < e; i++) {
    if (func.getArgAttr(i, kTfSavedModelIndexPathAttr) ||
        func.getArgAttr(i, "tf_saved_model.bound_input")) {
      return true;
    }
  }
  for (int i = 0, e = func.getNumResults(); i < e; i++) {
    if (func.getResultAttr(i, kTfSavedModelIndexPathAttr) ||
        func.getResultAttr(i, "tf_saved_model.bound_input")) {
      return true;
    }
  }
  return false;
}

static LogicalResult VerifySavedModelModule(
    ModuleOp module, TensorFlowSavedModelDialect *dialect) {
  auto exported_names_ident =
      StringAttr::get(dialect->getContext(), kTfSavedModelExportedNamesAttr);
  // Check that there are no duplicated exported_names.
  DenseMap<StringRef, Operation *> exported_name_to_op;
  for (auto &op : module) {
    auto attr = op.getAttr(exported_names_ident);
    if (!attr) continue;
    // If this verifier is called before we verify the
    // 'tf_saved_model.exported_names' attribute, then it might be invalid.
    // Forward to the dialect's verification to establish that precondition.
    if (failed(dialect->verifyOperationAttribute(
            &op, {exported_names_ident, attr}))) {
      return failure();
    }
    for (auto str : mlir::cast<ArrayAttr>(attr)) {
      auto exported_name = mlir::cast<StringAttr>(str).getValue();
      auto p = exported_name_to_op.insert({exported_name, &op});
      if (!p.second) {
        return op.emitError()
            .append("duplicate exported name '", exported_name, "'")
            .attachNote(p.first->getSecond()->getLoc())
            .append("previously seen here");
      }
    }
  }
  for (auto func : module.getOps<func::FuncOp>()) {
    const bool is_exported = IsExported(func);

    if (is_exported && !func.isPublic()) {
      return func.emitError()
             << "exported function @" << func.getName() << " should be public";
    }

    if (!is_exported && func.isPublic()) {
      return func.emitError() << "non-exported function @" << func.getName()
                              << " should be private";
    }
    if (!is_exported && HasAnyTfSavedModelArgAttr(func)) {
      return func.emitError() << "can only apply 'tf_saved_model' argument "
                                 "attributes to exported functions";
    }
  }

  SymbolTable symbol_table(module);

  auto session_initializers = module.getOps<SessionInitializerOp>();
  if (!session_initializers.empty()) {
    if (!llvm::hasSingleElement(session_initializers)) {
      return (*++session_initializers.begin()).emitError()
             << "there must be no more than one session_initializer op";
    }

    if (failed(
            VerifySessionInitOp(*session_initializers.begin(), symbol_table))) {
      return failure();
    }
  }

  auto is_init = [&session_initializers](mlir::func::FuncOp func) {
    if (session_initializers.empty()) return false;
    auto init_syms = (*session_initializers.begin()).getInitializers();
    return std::any_of(
        init_syms.begin(), init_syms.end(), [&](Attribute sym_ref) {
          return mlir::cast<FlatSymbolRefAttr>(sym_ref).getValue() ==
                 func.getName();
        });
  };

  auto symbol_uses = SymbolTable::getSymbolUses(&module.getBodyRegion());
  if (!symbol_uses.has_value()) {
    return module.emitError() << "modules with 'tf_saved_model.semantics' must "
                                 "have analyzable symbol uses";
  }
  for (auto symbol_use : *symbol_uses) {
    auto func = symbol_table.lookupNearestSymbolFrom<func::FuncOp>(
        symbol_use.getUser(), symbol_use.getSymbolRef());
    if (func && IsExported(func)) {
      // If it is an init function, then it can be used by the unique
      // session_initializer op.
      if (is_init(func) &&
          llvm::isa<SessionInitializerOp>(symbol_use.getUser())) {
        if (!func->getAttr(kTfSavedModelInitializerTypeAttr)) {
          LOG(WARNING)
              << "func op in session_initializer op's initializers attribute "
              << "should have tf_saved_model.initializer_type attribute.";
        }
        continue;
      }

      return symbol_use.getUser()
          ->emitError("exported function cannot be internally referenced")
          .attachNote(func.getLoc())
          .append("references this exported function");
    }
  }
  return success();
}

LogicalResult VerifyExportedFunc(func::FuncOp func) {
  bool reached_bound_inputs = false;
  auto module = func->getParentOfType<ModuleOp>();
  for (int i = 0, e = func.getNumArguments(); i < e; i++) {
    if (func.getArgAttr(i, "tf_saved_model.bound_input")) {
      reached_bound_inputs = true;
      continue;
    }
    if (func.getArgAttr(i, kTfSavedModelIndexPathAttr)) {
      if (reached_bound_inputs) {
        return func.emitError()
               << "all '" << kTfSavedModelIndexPathAttr
               << "' arg attributes should precede all "
                  "'tf_saved_model.bound_input' arg attributes";
      }
      continue;
    }
    if (func.getArgAttr(i, "tf.resource_name")) {
      if (module->getAttr("tf_saved_model.under_construction")) continue;
      return func.emitError() << "'tf.resource_name' attribute is not allowed "
                                 "unless it is being under construction";
    }
    return func.emitError()
           << "all arguments should have '" << kTfSavedModelIndexPathAttr
           << "', 'tf_saved_model.bound_input' or 'tf.resource_name' "
              "attributes";
  }
  llvm::SmallDenseSet<StringRef, 8> unique_bound_inputs;
  for (int i = 0, e = func.getNumArguments(); i < e; i++) {
    if (auto attr = func.getArgAttrOfType<FlatSymbolRefAttr>(
            i, "tf_saved_model.bound_input")) {
      if (!unique_bound_inputs.insert(attr.getValue()).second) {
        if (module->getAttr("tf_saved_model.under_construction")) continue;
        return func.emitError()
               << "duplicate 'tf_saved_model.bound_input' binding";
      }
    }
  }

  for (int i = 0, e = func.getNumResults(); i < e; i++) {
    if (!func.getResultAttr(i, kTfSavedModelIndexPathAttr)) {
      return func.emitError() << "all results should have '"
                              << kTfSavedModelIndexPathAttr << "' attributes";
    }
  }

  return success();
}

bool IsValidInitializerType(StringRef initializer_type) {
  return initializer_type == kTfSavedModelInitializerRestoreType ||
         initializer_type == kTfSavedModelInitializerInitType;
}

// Verifies the "tf_saved_model.initializer_type" attribute.
LogicalResult VerifyInitializerTypeAttr(Operation *op,
                                        const NamedAttribute named_attr) {
  if (!isa<func::FuncOp>(op)) {
    return op->emitError() << "Attribute tf_saved_model.initializer_type "
                           << "should be on a func::FuncOp.";
  }

  // Validate the attribute value.
  auto initializer_type_attr_value =
      mlir::dyn_cast_or_null<StringAttr>(named_attr.getValue());
  if (!initializer_type_attr_value) {
    return op->emitError() << "Attribute tf_saved_model.initializer_type "
                           << "should be a StringAttr.";
  }

  if (!IsValidInitializerType(initializer_type_attr_value)) {
    return op->emitError() << "tf_saved_model.initializer_type should be one "
                              "of 'restore_op' or 'init_op'. Got: "
                           << initializer_type_attr_value.str();
  }

  return success();
}

LogicalResult TensorFlowSavedModelDialect::verifyOperationAttribute(
    Operation *op, NamedAttribute named_attr) {
  if (named_attr.getName() == kTfSavedModelExportedNamesAttr) {
    if (!isa<func::FuncOp, GlobalTensorOp>(op)) {
      return op->emitError()
             << "'" << kTfSavedModelExportedNamesAttr
             << "' must be on a 'func' or 'tf_saved_model.global_tensor' op";
    }
    if (!IsStrArrayAttr(named_attr.getValue())) {
      return op->emitError() << "'" << kTfSavedModelExportedNamesAttr
                             << "' must be an array of strings";
    }
    if (!op->getParentOp()->getAttr("tf_saved_model.semantics")) {
      return op->emitError() << "'" << kTfSavedModelExportedNamesAttr
                             << "' must be on an op whose immediate parent has "
                                "attribute 'tf_saved_model.semantics'";
    }
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      if (failed(VerifyExportedFunc(func))) {
        return failure();
      }
    }
    return success();
  }
  if (named_attr.getName() == "tf_saved_model.semantics") {
    auto module = dyn_cast<ModuleOp>(op);
    if (!module) {
      return op->emitError() << "'tf_saved_model.semantics' must "
                                "be on a module op";
    }
    return VerifySavedModelModule(module, this);
  }
  if (named_attr.getName() == "tf_saved_model.under_construction") {
    return success();
  }

  if (named_attr.getName() == kTfSavedModelInitializerTypeAttr) {
    return VerifyInitializerTypeAttr(op, named_attr);
  }

  return op->emitError() << "unknown tf_saved_model dialect attribute '"
                         << named_attr.getName().getValue() << "'";
}

SmallVector<StringRef, 2> GetExportedNames(Operation *op) {
  SmallVector<StringRef, 2> ret;
  auto exported_names =
      op->getAttrOfType<ArrayAttr>(kTfSavedModelExportedNamesAttr);
  if (exported_names) {
    for (auto name : exported_names) {
      ret.push_back(mlir::cast<StringAttr>(name).getValue());
    }
  }
  return ret;
}

bool IsExported(Operation *op) {
  auto exported_names =
      op->getAttrOfType<ArrayAttr>(kTfSavedModelExportedNamesAttr);
  return exported_names && !exported_names.empty();
}

bool HasTfSavedModelSemantics(ModuleOp module) {
  return module->getAttr("tf_saved_model.semantics") != nullptr;
}

Operation *LookupBoundInput(func::FuncOp func, int arg_index,
                            const SymbolTable &symbol_table) {
  auto attr = func.getArgAttrOfType<FlatSymbolRefAttr>(
      arg_index, "tf_saved_model.bound_input");
  if (!attr) return nullptr;
  return symbol_table.lookup(attr.getValue());
}

SessionInitializerOp GetSessionInitializerOp(mlir::ModuleOp op) {
  auto initializers = op.getOps<SessionInitializerOp>();
  if (initializers.empty()) return {};
  return *initializers.begin();
}

class OptimizeSessionInitializerPattern
    : public OpRewritePattern<SessionInitializerOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SessionInitializerOp op,
                                PatternRewriter &rewriter) const override {
    SymbolTable symbol_table(op->getParentOfType<ModuleOp>());

    SmallVector<func::FuncOp, 2> to_remove;
    SmallVector<mlir::Attribute, 2> to_keep;
    for (auto sym_ref : op.getInitializers()) {
      auto init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
          mlir::cast<FlatSymbolRefAttr>(sym_ref).getValue());

      // The init function can only be referenced from the SessionInitializerOp.
      // And there is at most one SessionInitializerOp in the module. So if both
      // ops have no other uses or have one NoOp only, they can be simply
      // erased.
      auto &operations = init_func_op.front().getOperations();
      if ((operations.size() == 1 &&
           operations.front().hasTrait<OpTrait::IsTerminator>()) ||
          (operations.size() == 2 &&
           dyn_cast<mlir::TF::NoOp>(operations.front()) &&
           operations.back().hasTrait<OpTrait::IsTerminator>())) {
        to_remove.push_back(init_func_op);
      } else {
        to_keep.push_back(sym_ref);
      }
    }

    for (auto func_op : to_remove) rewriter.eraseOp(func_op);

    if (to_keep.empty())
      rewriter.eraseOp(op);
    else
      op->setAttr("initializers", rewriter.getArrayAttr(to_keep));

    return success();
  }
};

void SessionInitializerOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<OptimizeSessionInitializerPattern>(context);
}

SmallVector<StringRef, 2> GetSessionInitializerExportedName(ModuleOp op) {
  auto session_initializer_op = GetSessionInitializerOp(op);
  if (!session_initializer_op) return {};

  SymbolTable symbol_table(op);

  SmallVector<StringRef, 2> results;
  for (auto sym_ref : session_initializer_op.getInitializers()) {
    auto init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
        mlir::cast<FlatSymbolRefAttr>(sym_ref).getValue());
    auto exported_names = GetExportedNames(init_func_op);
    assert(exported_names.size() == 1);
    results.push_back(exported_names[0]);
  }

  return results;
}

SmallVector<func::FuncOp, 2> GetInitializerFunctions(ModuleOp module_op) {
  SessionInitializerOp session_initializer_op =
      GetSessionInitializerOp(module_op);
  if (!session_initializer_op) return {};

  SymbolTable symbol_table(module_op);

  SmallVector<func::FuncOp, 2> init_func_ops;
  for (auto init_func_sym : session_initializer_op.getInitializers()
                                .getAsValueRange<FlatSymbolRefAttr>()) {
    auto init_func_op = symbol_table.lookup<func::FuncOp>(init_func_sym);
    // `init_func_op` is guaranteed to be not null in a valid module.
    init_func_ops.push_back(init_func_op);
  }

  return init_func_ops;
}

func::FuncOp GetInitializerFunction(ModuleOp module_op,
                                    const StringRef initializer_type) {
  SmallVector<func::FuncOp, 2> init_func_ops =
      GetInitializerFunctions(module_op);

  auto init_func_itr = absl::c_find_if(
      init_func_ops, [initializer_type](const func::FuncOp init_func_op) {
        const auto init_type_attr = init_func_op->getAttrOfType<StringAttr>(
            kTfSavedModelInitializerTypeAttr);
        return init_type_attr && init_type_attr == initializer_type;
      });

  return init_func_itr == init_func_ops.end() ? nullptr : *init_func_itr;
}

bool IsRestoreGraph(ModuleOp module) {
  return module
      .walk([](mlir::Operation *op) {
        if (llvm::isa<mlir::TF::RestoreV2Op>(op)) {
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      })
      .wasInterrupted();
}
}  // namespace tf_saved_model
}  // namespace mlir
