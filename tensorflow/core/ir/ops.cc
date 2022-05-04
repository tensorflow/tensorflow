/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/ops.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/FunctionImplementation.h"  // from @llvm-project
#include "mlir/IR/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/ir/utility.h"

// Generated definitions.
#include "tensorflow/core/ir/dialect.cc.inc"

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// TFGraph dialect.
//===----------------------------------------------------------------------===//

// Name operation results with the operation name, except control outputs which
// are named "ctl". MLIR will automatically use a numerical suffix to unique.
static void GenericGetAsmResultNames(Operation *op,
                                     OpAsmSetValueNameFn set_name_fn) {
  // We only name the results when there are results to name, an op like `print`
  // which does not have results will just use the `ctl` name for the control
  // output.
  if (op->getNumResults() > 1 && !op->getResult(0).getType().isa<ControlType>())
    set_name_fn(op->getResult(0), op->getName().stripDialect());
  for (Value result : op->getResults()) {
    if (result.getType().isa<ControlType>()) {
      set_name_fn(op->getResult(op->getNumResults() - 1), "ctl");
      break;
    }
  }
}

// TFGraph support for interacting with the AsmPrinter.
// Gives prettier names to SSA values.
struct TFGraphOpAsmInterface
    : public OpAsmOpInterface::FallbackModel<TFGraphOpAsmInterface> {
  static bool classof(Operation *op) { return true; }

  void getAsmResultNames(Operation *op, OpAsmSetValueNameFn set_name_fn) const {
    GenericGetAsmResultNames(op, set_name_fn);
  }
  void getAsmBlockArgumentNames(Operation *op, Region &region,
                                OpAsmSetValueNameFn setNameFn) const {}
  void getAsmBlockNames(Operation *op,
                        mlir::OpAsmSetBlockNameFn setNameFn) const {}
};

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void TFGraphDialect::initialize() {
  getContext()->getOrLoadDialect<tf_type::TFTypeDialect>();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/core/ir/ops.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tensorflow/core/ir/attributes.cc.inc"
      >();

  // Support unknown operations because not all TensorFlow operations are
  // registered.
  allowUnknownOperations();

  // Create the fallback OpAsmOpInterface instance.
  fallbackOpAsmInterface_ = new TFGraphOpAsmInterface;

  // Caching some often used context-owned informations for fast-access.
  name_key_ = StringAttr::get(getContext(), getNameAttrKey());
  device_key_ = StringAttr::get(getContext(), getDeviceAttrKey());
  assigned_device_key_ =
      StringAttr::get(getContext(), getAssignedDeviceAttrKey());
  fulltype_key_ = StringAttr::get(getContext(), getFullTypeAttrKey());
  tfg_name_key_ = StringAttr::get(getContext(), getTfgNameAttrKey());
  tfg_description_key_ =
      StringAttr::get(getContext(), getTfgDescriptionAttrKey());
  tfg_is_ref_key_ = StringAttr::get(getContext(), getTfgIsRefAttrKey());
  tfg_handle_data_key_ =
      StringAttr::get(getContext(), getTfgHandleDataAttrKey());
  tfg_full_type_key_ = StringAttr::get(getContext(), getTfgFullTypeAttrKey());

  control_ty_ = ControlType::get(getContext());
}

// Provides a hook for op interface.
void *TFGraphDialect::getRegisteredInterfaceForOp(TypeID interface,
                                                  OperationName opName) {
  if (interface == TypeID::get<OpAsmOpInterface>()) {
    return fallbackOpAsmInterface_;
  } else if (interface == TypeID::get<TensorFlowRegistryInterface>()) {
    if (auto *instance =
            getRegisteredInterface<TensorFlowRegistryInterfaceBase>()) {
      // Important: cast to (Concept *) to shift the pointer off the vtable.
      return static_cast<TensorFlowRegistryInterfaceBase::Concept *>(
          const_cast<TensorFlowRegistryInterfaceBase *>(instance));
    }
  }

  return nullptr;
}

TFGraphDialect::~TFGraphDialect() { delete fallbackOpAsmInterface_; }

// The name of certain optional attributes.
static std::array<StringRef, 3> keyword_attrs{
    "_mlir_device", "_mlir_assigned_device", "_mlir_name"};

static void PrintKeywordAttributes(Operation *op, OpAsmPrinter &printer,
                                   ArrayRef<StringRef> elided_attrs = {}) {
  // Handles the optional "device" and "name" attribute.
  for (StringRef keyword : keyword_attrs) {
    if (StringAttr value_attr = op->getAttrOfType<StringAttr>(keyword)) {
      assert(!value_attr.getValue().empty());
      printer << " " << keyword.drop_front(/*len(_mlir_)*/ 6) << "(\""
              << value_attr.getValue() << "\")";
    }
  }

  // Print attributes (other than name and device).
  SmallVector<StringRef> attrs_to_elide = llvm::to_vector(elided_attrs);
  llvm::append_range(attrs_to_elide, keyword_attrs);
  printer.printOptionalAttrDict(op->getAttrs(), attrs_to_elide);
}

// Print an operation that belongs to this dialect, if unregistered.
// The general syntax is:
//   tfg.OpName(%input1, %input2, %input3) [%control_dep1, %control_dep2]
//           name("<node_name>") device("<device>") { attribute-dict } :
//           (input types) -> (result_types)
void TFGraphDialect::printCustomTfOp(Operation *op,
                                     OpAsmPrinter &printer) const {
  ControlType control_ty = getControlType();

  // Check that all control dependencies are after the regular values,
  // otherwise print the generic form. We don't expect this to happen but
  // we're defensive in the printer since this may happen in "hard-to-debug"
  // issues.
  {
    bool has_control_dep = false;
    for (Value operand : op->getOperands()) {
      if (operand.getType() == control_ty) {
        has_control_dep = true;
        continue;
      }
      if (has_control_dep) {
        printer.printGenericOp(op);
        return;
      }
    }
    has_control_dep = false;
    for (Value result : op->getResults()) {
      if (result.getType() == control_ty) {
        has_control_dep = true;
        continue;
      }
      if (has_control_dep) {
        printer.printGenericOp(op);
        return;
      }
    }
  }

  // Print the inputs (other than the control dependencies), if any.
  TFOp tfg_op(op);
  OperandRange data = tfg_op.getNonControlOperands();
  if (!data.empty()) printer << '(' << data << ')';
  // Print the control dependencies (if any).
  OperandRange ctls = tfg_op.getControlOperands();
  if (!ctls.empty()) printer << " [" << ctls << ']';

  // Print the keyword attributes and optional attribute dictionary.
  PrintKeywordAttributes(op, printer);

  // Print the type, but omit control dependencies.
  // If there is a single control return, just print the list of input types,
  // otherwise print the complete type in a "function-style" way: (operands)
  // -> (results).
  ResultRange results = tfg_op.getNonControlResults();
  if (results.empty()) {
    if (!data.empty()) printer << " : " << data.getTypes();
  } else {
    printer << " : (" << data.getTypes() << ") -> (" << results.getTypes()
            << ")";
  }
}

// Print a custom TFG op.
static void PrintCustomTfOp(Operation *op, OpAsmPrinter &printer) {
  cast<TFGraphDialect>(op->getDialect())->printCustomTfOp(op, printer);
}

llvm::unique_function<void(Operation *, OpAsmPrinter &)>
TFGraphDialect::getOperationPrinter(Operation *op) const {
  return [this](Operation *op, OpAsmPrinter &printer) {
    this->printCustomTfOp(op, printer);
  };
}

// Try to parse optional keyword attributes and prefix them with `_mlir_`, of
// `device`, `assigned_device`, and `name`.
static ParseResult ParseKeywordAttributes(OpAsmParser &parser,
                                          OperationState &result) {
  for (const char *keyword : {"device", "assigned_device", "name"}) {
    if (succeeded(parser.parseOptionalKeyword(keyword))) {
      StringAttr value;
      if (parser.parseLParen() ||
          parser.parseAttribute<StringAttr>(
              value, NoneType::get(parser.getContext())) ||
          parser.parseRParen())
        return failure();
      result.addAttribute((Twine("_mlir_") + keyword).str(), value);
    }
  }
  return parser.parseOptionalAttrDict(result.attributes);
}

// Parse an operation that belongs to this dialect, if unregistered.
// The general syntax is:
//   tfg.OpName(%input1, %input2, %input3) [%control_dep1, %control_dep2]
//           name("<node_name>") device("<device>") { attribute-dict } :
//           (input types) -> (result_types)
static ParseResult ParseCustomTfOp(OpAsmParser &parser,
                                   OperationState &result) {
  MLIRContext *context = parser.getBuilder().getContext();
  // Parse optional argument list
  SmallVector<OpAsmParser::UnresolvedOperand, 4> op_infos;
  if (parser.parseOperandList(op_infos, AsmParser::Delimiter::OptionalParen))
    return failure();
  unsigned numNonControlOperands = op_infos.size();
  // Optional control list, in between brackets.
  if (parser.parseOperandList(op_infos, AsmParser::Delimiter::OptionalSquare))
    return failure();

  // Parse the optional keyword attributes and optional attribute dictionary.
  if (ParseKeywordAttributes(parser, result)) return failure();

  // Parse the functional type.
  SmallVector<Type> arg_types;
  arg_types.reserve(op_infos.size());
  llvm::SMLoc loc = parser.getCurrentLocation();
  Type control_type = ControlType::get(context);
  if (failed(parser.parseOptionalColonTypeList(arg_types))) return failure();
  if (arg_types.size() == 1 && arg_types.front().isa<FunctionType>()) {
    auto funcType = arg_types.front().cast<FunctionType>();
    if (funcType.getNumInputs() != numNonControlOperands)
      return parser.emitError(loc)
             << "got " << numNonControlOperands
             << " non-control operands, but the type defines "
             << funcType.getNumInputs() << " input types";
    arg_types.clear();
    arg_types.append(funcType.getInputs().begin(), funcType.getInputs().end());
    result.types.append(funcType.getResults().begin(),
                        funcType.getResults().end());
  }

  // The control input are elided from the type list, add them here.
  arg_types.resize(op_infos.size(), control_type);
  if (!arg_types.empty())
    parser.resolveOperands(op_infos, arg_types, loc, result.operands);
  if (result.name.getStringRef() != "tfg.return")
    result.types.push_back(control_type);
  return success();
}

Optional<Dialect::ParseOpHook> TFGraphDialect::getParseOperationHook(
    StringRef opName) const {
  return ParseOpHook(ParseCustomTfOp);
}

static bool VerifyGenericTFGOperation(Operation &op) {
  TFGraphDialect *dialect = dyn_cast<TFGraphDialect>(op.getDialect());
  if (!dialect) return true;
  ControlType control_ty = dialect->getControlType();

  // verifies that control operands (or results) are always after regular
  // inputs (or results).
  auto check_ctl_at_end = [&](TypeRange types, StringRef input_or_output) {
    int has_control_dep = -1;
    for (auto &indexed_operand : llvm::enumerate(types)) {
      if (indexed_operand.value() == control_ty) {
        has_control_dep = indexed_operand.index();
        continue;
      }
      if (has_control_dep != -1) {
        op.emitOpError() << "found non-control " << input_or_output
                         << " in position #" << indexed_operand.index()
                         << " after control " << input_or_output
                         << " in position #" << has_control_dep;
        return false;
      }
    }
    return true;
  };
  if (!check_ctl_at_end(op.getOperandTypes(), "input")) return false;
  if (!check_ctl_at_end(op.getResultTypes(), "result")) return false;

  // Certain attributes are supposed to be inserted with non-empty value.
  for (StringRef keyword : keyword_attrs) {
    if (StringAttr value_attr = op.getAttrOfType<StringAttr>(keyword)) {
      if (value_attr.getValue().empty()) {
        op.emitOpError() << keyword
                         << " has empty value. Only insert this attribute when "
                            "it has a value";
      }
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Graph Operation
//===----------------------------------------------------------------------===//

LogicalResult GraphOp::verify() {
  GraphOp op = *this;
  // Check all ops in the body.
  if (!all_of(*op.getBody(), VerifyGenericTFGOperation)) return failure();

  return success();
}
//===----------------------------------------------------------------------===//
// Func Operation
//===----------------------------------------------------------------------===//

bool GraphFuncOp::isMarkedForCompilation() {
  auto is_enabled = [this](StringRef attr_name) -> bool {
    Attribute attr = (*this)->getAttr(attr_name);
    if (!attr) return false;
    if (auto bool_attr = attr.dyn_cast<BoolAttr>()) return bool_attr.getValue();
    if (auto str_attr = attr.dyn_cast<StringAttr>())
      return !str_attr.getValue().empty();
    return false;
  };
  return is_enabled("_xla_compile_id") || is_enabled("_tpu_replicate") ||
         is_enabled("_XlaMustCompile");
}

// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
// attribute is present and checks if it holds a function type. Ensures
// getType, getNumFuncArguments, and getNumFuncResults can be called safely
LogicalResult GraphFuncOp::verifyType() {
  auto type = getFunctionTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

// Hook for OpTrait::FunctionLike, called after verifying the function
// type and the presence of the (potentially empty) function body.
LogicalResult GraphFuncOp::verifyBody() {
  FunctionType type = getFunctionType();
  // Check that the body is terminated with a tfg.return.
  if (getRegion().empty() || getBody()->empty())
    return emitOpError() << "expects a non empty body";

  if (getBody()->getNumArguments() != type.getNumInputs())
    return emitOpError() << "function type indicated " << type.getNumInputs()
                         << " args but block has "
                         << getBody()->getNumArguments();

  for (auto &arg_types : llvm::enumerate(
           llvm::zip(type.getInputs(), getBody()->getArgumentTypes()))) {
    Type signature_arg = std::get<0>(arg_types.value());
    Type block_arg = std::get<1>(arg_types.value());
    if (signature_arg != block_arg)
      return emitOpError() << "type mismatch for arg #" << arg_types.index()
                           << ", signature defines " << signature_arg
                           << " block arg is " << block_arg;
  }

  if (!isa<ReturnOp>(getBody()->back()))
    return emitOpError()
           << "expects body to be terminated with a tfg.return, but got: "
           << getBody()->back().getName().getStringRef();

  ReturnOp return_op = cast<ReturnOp>(getBody()->getTerminator());

  if (type.getNumResults() > return_op->getNumOperands())
    return emitOpError() << "expects " << type.getNumResults()
                         << " returned values but tfg.return has "
                         << return_op->getNumOperands() << " operands";
  for (auto &indexed_type : llvm::enumerate(type.getResults())) {
    Type expected_type = indexed_type.value();
    int res_num = indexed_type.index();
    Type actual_type = return_op->getOperand(res_num).getType();
    if (expected_type == actual_type) continue;
    return emitOpError() << "type mismatch for returned value #" << res_num
                         << ", expected " << expected_type << " but got "
                         << actual_type;
  }
  Type control_type = getDialect()->getControlType();
  for (auto &indexed_type : llvm::enumerate(llvm::drop_begin(
           return_op->getOperandTypes(), type.getNumResults()))) {
    Type actual_type = indexed_type.value();
    if (actual_type != control_type) {
      return emitOpError() << "returned value #" << indexed_type.index()
                           << " overflow the expected " << type.getNumResults()
                           << " returned value for function " << getName()
                           << ", expected a ControlType but got "
                           << actual_type;
    }
  }

  // Check all ops in the body.
  if (!all_of(*getBody(), VerifyGenericTFGOperation)) return failure();

  return success();
}

LogicalResult GraphFuncOp::canonicalize(GraphFuncOp func_op,
                                        PatternRewriter &rewriter) {
  // Prune function body: the body is a graph where feeds/fetches a materialized
  // with function arguments and returned values. As such any operation not
  // reachable from the "fetches" can be pruned. The return statement also has
  // control input so that side-effecting operations without results (print for
  // example) aren't pruned.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op :
         llvm::make_early_inc_range(llvm::reverse(*func_op.getBody()))) {
      if (isa<ReturnOp>(op)) continue;
      if (op.getUses().empty()) {
        rewriter.eraseOp(&op);
        changed = true;
      }
    }
  }
  return failure();
}

LogicalResult GraphFuncOp::verify() {
  GraphFuncOp func_op = *this;
  if (func_op.getNumArguments() % 2)
    return func_op.emitOpError() << "expects an even number of arguments";
  ArrayAttr args_attrs = func_op.getAllArgAttrs();
  if (!args_attrs)
    return func_op.emitOpError() << "missing argument attributes array";
  if (args_attrs.size() != func_op.getNumArguments())
    return func_op.emitOpError()
           << "expects argument attributes for each argument ("
           << args_attrs.size() << " vs " << func_op.getNumArguments() << ")";
  ArrayAttr res_attrs = func_op.getAllResultAttrs();
  if (!res_attrs)
    return func_op.emitOpError() << "missing results attributes array";
  if (res_attrs.size() != func_op.getNumResults())
    return func_op.emitOpError()
           << "expects results attributes for each result (" << res_attrs.size()
           << " vs " << func_op.getNumResults() << ")";
  return success();
}

ParseResult GraphFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> entry_args;
  SmallVector<Attribute> arg_attrs;
  SmallVector<Attribute> result_attrs;
  SmallVector<Type> arg_types;
  SmallVector<Type> result_types;
  auto &builder = parser.getBuilder();
  MLIRContext *context = builder.getContext();

  // Parse visibility.
  StringRef visibility;
  if (!parser.parseOptionalKeyword(&visibility,
                                   {"public", "private", "nested"})) {
    StringAttr visibility_attr = parser.getBuilder().getStringAttr(visibility);
    result.attributes.push_back(parser.getBuilder().getNamedAttr(
        SymbolTable::getVisibilityAttrName(), visibility_attr));
  }

  if (succeeded(parser.parseOptionalKeyword("generic")))
    result.addAttribute("generic", builder.getUnitAttr());

  // Parse the name as a symbol.
  StringAttr name_attr;
  if (parser.parseSymbolName(name_attr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  // The difference with usual functions, is that for every single argument
  // parsed, we create two block arguments: one for the expected value and one
  // for the control dependency.
  if (parser.parseLParen()) return failure();
  Type control_ty = ControlType::get(builder.getContext());
  std::list<std::string> control_operand_names;

  // Helper to parse a single argument and its attributes.
  auto parse_argument = [&]() -> ParseResult {
    // Parse argument name if present.
    entry_args.emplace_back();
    arg_types.emplace_back();
    if (parser.parseOperand(entry_args.back(), /*allowResultNumber=*/false) ||
        parser.parseColonType(arg_types.back()))
      return failure();

    // Parse any argument attributes.
    NamedAttrList attrs;
    if (parser.parseOptionalAttrDict(attrs)) return failure();
    arg_attrs.push_back(attrs.getDictionary(context));

    // Define the control input: it's not printed but is added as block
    // argument. Note the name computed here (suffixed ".ctl") is coupled to the
    // implementation of:
    //   TFGraphOpAsmInterface::getAsmBlockArgumentNames()
    // at the top of this file.
    OpAsmParser::UnresolvedOperand control_operand = entry_args.back();
    control_operand_names.push_back((control_operand.name + ".ctl").str());
    control_operand.name = control_operand_names.back();
    entry_args.push_back(control_operand);
    arg_types.push_back(control_ty);
    arg_attrs.push_back(DictionaryAttr::get(context));
    return success();
  };

  // Parse the function arguments and their attributes.
  if (failed(parser.parseOptionalRParen())) {
    do {
      if (parse_argument()) return failure();
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen()) return failure();
  }

  // Parse the result types and their attributes.
  if (succeeded(parser.parseOptionalArrow())) {
    if (failed(parser.parseLParen())) return failure();
    // Parse individual function results.
    do {
      result_types.emplace_back();
      NamedAttrList result_attr;
      if (parser.parseType(result_types.back()) ||
          parser.parseOptionalAttrDict(result_attr)) {
        return failure();
      }
      result_attrs.push_back(builder.getDictionaryAttr(result_attr));
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen()) return failure();
  }

  auto type = builder.getFunctionType(arg_types, result_types);
  result.addAttribute(GraphFuncOp::getTypeAttrName(), TypeAttr::get(type));

  // If function attributes are present, parse them.
  NamedAttrList parsed_attributes;
  if (parser.parseOptionalAttrDictWithKeyword(parsed_attributes))
    return failure();
  result.attributes.append(parsed_attributes);

  // Add the attributes to the function arguments.
  assert(arg_attrs.size() == arg_types.size());
  assert(result_attrs.size() == result_types.size());
  result.attributes.append(
      builder.getNamedAttr(FunctionOpInterface::getArgDictAttrName(),
                           builder.getArrayAttr(arg_attrs)));
  result.attributes.append(
      builder.getNamedAttr(FunctionOpInterface::getResultDictAttrName(),
                           builder.getArrayAttr(result_attrs)));

  // Parse the function body.
  auto *body = result.addRegion();
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::Argument> args;
  if (entry_args.size()) {
    for (auto argAndType : llvm::zip(entry_args, arg_types)) {
      auto &arg = args.emplace_back();
      arg.ssaName = std::get<0>(argAndType);
      arg.type = std::get<1>(argAndType);
    }
  }

  if (failed(parser.parseRegion(*body, args, /*enableNameShadowing=*/false)))
    return failure();

  // Function body was parsed, make sure it's not empty.
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  return success();
}

void GraphFuncOp::print(OpAsmPrinter &p) {
  // Print the operation and the function name.
  Operation *op = *this;
  p << " ";
  int argIndentSize = op->getName().getStringRef().size() + 3;
  StringRef visibility_attr_name = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibility_attr_name)) {
    p << visibility.getValue() << ' ';
    argIndentSize += visibility.getValue().size() + 1;
  }
  if (generic()) p << "generic ";
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p.printSymbolName(funcName);
  argIndentSize += funcName.size();
  std::string indent(argIndentSize, ' ');
  FunctionType fnType = getFunctionType();
  ArrayRef<Type> arg_types = fnType.getInputs();
  ArrayRef<Type> result_types = fnType.getResults();
  assert((arg_types.size() % 2) == 0);
  // Print operand list with attributes.
  p << '(';
  ArrayAttr args_attr = getAllArgAttrs();
  for (unsigned i = 0, e = arg_types.size(); i < e; i += 2) {
    // Args come by pair: input+control.
    p.printOperand(getArgument(i));
    p << ": ";
    p.printType(arg_types[i]);
    if (auto arg_attrs = args_attr[i].dyn_cast<DictionaryAttr>())
      p.printOptionalAttrDict(arg_attrs.getValue());
    if (i != e - 2) {
      p << ", ";
      p.printNewline();
      p << indent;
    }
  }
  p << ')';

  // Print result types, if any.
  if (!result_types.empty()) {
    p.printNewline();
    p.getStream() << "     -> (";
    indent = std::string(9, ' ');
    ArrayAttr results_attr = getAllResultAttrs();
    for (int i = 0, e = result_types.size(); i < e; ++i) {
      p.printType(result_types[i]);
      if (auto result_attrs = results_attr[i].dyn_cast<DictionaryAttr>())
        p.printOptionalAttrDict(result_attrs.getValue());
      if (i != e - 1) {
        p << ", ";
        p.printNewline();
        p << indent;
      }
    }
    p << ")";
  }
  // Print attributes.
  if (!op->getAttrs().empty()) {
    p.printNewline();
    function_interface_impl::printFunctionAttributes(
        p, *this, fnType.getNumInputs(), fnType.getNumResults(),
        {"generic", SymbolTable::getVisibilityAttrName()});
  }
  // Print body.
  p << ' ';
  p.printRegion(body(), /*printEntryBlockArgs=*/false);
}

GraphFuncOp GraphFuncOp::getCalledFunction(Operation *op,
                                           SymbolTable &symbol_table) {
  // Check if a node does indirect function call via PartitionedCallOp.
  // TODO(aminim): consider replacing with isa<...> when possible.
  if (op->getName().getStringRef() == "tfg.PartitionCall" ||
      op->getName().getStringRef() == "tfg.StatefulPartitionedCall") {
    auto func_attr = op->getAttrOfType<FuncAttr>("f");
    if (!func_attr) return {};
    GraphFuncOp callee = symbol_table.lookup<GraphFuncOp>(
        func_attr.getName().getLeafReference());
    if (callee) return callee;
  }
  return symbol_table.lookup<GraphFuncOp>(op->getName().stripDialect());
}

BlockArgument GraphFuncOp::getDataValueOf(BlockArgument ctl) {
  return ctl.getOwner()->getArgument(ctl.getArgNumber() - 1);
}

BlockArgument GraphFuncOp::getControlTokenOf(BlockArgument data) {
  return data.getOwner()->getArgument(data.getArgNumber() + 1);
}

BlockArgument GraphFuncOp::getDataValue(Region &region, unsigned idx) {
  return region.getArgument(idx * 2);
}

// This is naming block arguments for GraphFuncOp, we rely on the arg attributes
// for computing the names.
void GraphFuncOp::getAsmBlockArgumentNames(Region &region,
                                           OpAsmSetValueNameFn set_name_fn) {
  ArrayRef<BlockArgument> args = getBody()->getArguments();
  ControlType control_ty = ControlType::get(getContext());
  // Sanity checking: this is verified by the op but this may be called before
  // the verifier or in some diagnostic/debug context, let's not crash.
  // We expect the function block operands to come as pair: tensor+control.
  if (args.size() % 2) return;
  for (unsigned i = 0, e = args.size(); i < e; i += 2)
    if (args[i].getType() == control_ty || args[i + 1].getType() != control_ty)
      return;

  // Name the values based on the `tfg.name` arg attribute retrieved from the
  // func_op.
  ArrayAttr args_attr = getAllArgAttrs();
  if (!args_attr || args_attr.size() != args.size()) return;
  for (int arg_num = 0, e = args.size(); arg_num < e; arg_num += 2) {
    DictionaryAttr arg_attrs = args_attr[arg_num].dyn_cast<DictionaryAttr>();
    if (!arg_attrs) continue;
    if (auto strAttr = arg_attrs.getAs<StringAttr>("tfg.name")) {
      set_name_fn(args[arg_num], strAttr.getValue());
      set_name_fn(args[arg_num + 1], (strAttr.getValue() + ".ctl").str());
    }
  }
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  ReturnOp op = *this;
  // If the control result attributes are present, there must be the same number
  // of entries as control results.
  if (op.control_ret_attrs().size() != TFOp(op).getControlOperands().size()) {
    return op.emitOpError(
        "expected as many control result attributes as there are control "
        "operands");
  }
  return success();
}

ParseResult ReturnOp::parse(OpAsmParser &parser, OperationState &result) {
  // ReturnOp has the same assembly format as generic TFG ops except that the
  // control result attributes are embedded with the control operands:
  // [%ctl {tfg.name = "foo"}, %ctl_0 {tfg.name = "bar"}]
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands, AsmParser::Delimiter::OptionalParen))
    return failure();

  SmallVector<Attribute> control_ret_attrs;
  if (succeeded(parser.parseOptionalLSquare())) {
    OpAsmParser::UnresolvedOperand operand;
    do {
      NamedAttrList attrs;
      OptionalParseResult parse_result = parser.parseOptionalOperand(operand);
      if (!parse_result.hasValue()) break;
      if (failed(parse_result.getValue())) return failure();
      if (parser.parseOptionalAttrDict(attrs)) return failure();
      control_ret_attrs.push_back(attrs.getDictionary(result.getContext()));
      operands.push_back(std::move(operand));
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare()) return failure();
  }

  if (ParseKeywordAttributes(parser, result)) return failure();
  result.addAttribute(ReturnOp::control_ret_attrsAttrName(result.name),
                      ArrayAttr::get(result.getContext(), control_ret_attrs));

  SmallVector<Type> types;
  if (parser.parseOptionalColonTypeList(types)) return failure();
  types.resize(operands.size(), ControlType::get(result.getContext()));
  if (parser.resolveOperands(operands, types, parser.getCurrentLocation(),
                             result.operands))
    return failure();
  return success();
}

void ReturnOp::print(OpAsmPrinter &printer) {
  TFOp tfg_op(*this);
  OperandRange data = tfg_op.getNonControlOperands();
  if (!data.empty()) printer << '(' << data << ')';

  OperandRange ctls = tfg_op.getControlOperands();
  if (!ctls.empty()) {
    printer << " [";
    llvm::interleave(
        llvm::zip(ctls, control_ret_attrs().getAsRange<DictionaryAttr>()),
        printer,
        [&](auto it) {
          printer << std::get<0>(it);
          if (!std::get<1>(it).empty()) printer << ' ' << std::get<1>(it);
        },
        ", ");
    printer << ']';
  }

  PrintKeywordAttributes(*this, printer, {"control_ret_attrs"});

  if (!data.empty()) printer << " : " << data.getTypes();
}

void ReturnOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     ValueRange operands, ValueRange control_operands) {
  odsState.addOperands(operands);
  odsState.addOperands(control_operands);
  // Populate `control_ret_attrs` with empty dictionaries.
  odsState.addAttribute(
      ReturnOp::control_ret_attrsAttrName(odsState.name),
      odsBuilder.getArrayAttr(SmallVector<Attribute>(
          control_operands.size(), odsBuilder.getDictionaryAttr({}))));
}

//===----------------------------------------------------------------------===//
// Concrete Ops
//===----------------------------------------------------------------------===//

// The ODS definitions of TFG ops can be autogenerated TODO(jeffniu) as well as
// parts of their verifiers. These hand-written verifiers focus on verifying the
// ops' operand and result types with respect to their functions' types, the
// logic for which is slightly different between operations.

// Verify that all control operands follow non-control operands, and return the
// subrange of non-control operands.
static FailureOr<TypeRange> VerifyOperands(Operation *op) {
  ControlType control_ty =
      cast<TFGraphDialect>(op->getDialect())->getControlType();
  Operation::operand_type_iterator it =
      llvm::find(op->getOperandTypes(), control_ty);
  if (!std::all_of(it, op->operand_type_end(),
                   [&](Type type) { return type == control_ty; })) {
    return op->emitOpError(
        "not all control tokens come after non-control operands");
  }
  return {Operation::operand_type_range(op->operand_type_begin(), it)};
}

// Verify that the last result of an operation is the only control result, and
// return a subrange of the non-control results.
static FailureOr<TypeRange> VerifyResults(Operation *op) {
  ControlType control_ty =
      cast<TFGraphDialect>(op->getDialect())->getControlType();
  Operation::result_type_iterator it =
      llvm::find(op->getResultTypes(), control_ty);
  if (it == op->result_type_end())
    return op->emitOpError("does not define a control result");
  if (it != std::prev(op->result_type_end())) {
    return op->emitOpError(
        "must have a control token result as and only as its last result");
  }
  return {Operation::result_type_range(op->result_type_begin(), it)};
}

// Verify that the signature of the function matches the operation's operands
// and results.
static LogicalResult VerifySignature(GraphFuncOp func, Operation *op,
                                     TypeRange operands, TypeRange results,
                                     const Twine &func_name) {
  auto attach_func = [&](InFlightDiagnostic diag) -> LogicalResult {
    return diag.attachNote(func.getLoc()).appendOp(*func, OpPrintingFlags())
           << "\nsee referenced function";
  };

  ArrayRef<Type> arguments = func.getFunctionType().getInputs();
  ArrayRef<Type> returns = func.getFunctionType().getResults();
  if (operands.size() * 2 != arguments.size()) {
    return attach_func(op->emitOpError(func_name)
                       << " function has " << arguments.size() / 2
                       << " arguments but was provided " << operands.size());
  }
  if (results.size() != returns.size()) {
    return attach_func(op->emitOpError(func_name)
                       << " function has " << returns.size()
                       << " results but expected " << results.size());
  }

  if (func.generic()) return success();

  for (auto &it : llvm::enumerate(operands)) {
    Type arg_type = arguments[it.index() * 2];
    Type op_type = it.value();
    if (!tf_type::HasCompatibleElementTypes(arg_type, op_type)) {
      return attach_func(
          op->emitOpError(func_name)
          << " function argument #" << it.index() << " type " << arg_type
          << " is not compatible with corresponding operand type: " << op_type);
    }
  }
  for (auto &it : llvm::enumerate(results)) {
    Type ret_type = returns[it.index()];
    Type res_type = it.value();
    if (!tf_type::HasCompatibleElementTypes(ret_type, res_type)) {
      return attach_func(
          op->emitOpError(func_name)
          << " function result #" << it.index() << " type " << ret_type
          << " is not compatible with corresponding result type: " << res_type);
    }
  }
  return success();
}

// This function verifies that the types of `values`, which are either operands
// or results of `op`, match the types specified in `types`, which is expected
// to be an array of type attributes.
static LogicalResult VerifyTypeArray(Operation *op, ValueRange values,
                                     ArrayAttr types, StringRef kind) {
  // Don't verify if the types are not present.
  if (!types) return success();
  if (values.size() != types.size()) {
    return op->emitOpError("has ") << values.size() << " " << kind << "s but "
                                   << types.size() << " " << kind << " types";
  }
  for (auto it :
       llvm::zip(llvm::enumerate(values), types.getAsRange<TypeAttr>())) {
    Type type = std::get<0>(it).value().getType();
    Type dtype = std::get<1>(it).getValue();
    if (!tf_type::HasCompatibleElementTypes(type,
                                            UnrankedTensorType::get(dtype))) {
      return op->emitOpError(kind)
             << " #" << std::get<0>(it).index()
             << " is incompatible with dtype " << dtype << ", got: " << type;
    }
  }
  return success();
}

namespace detail {
// Check if the op type has `T`.
template <typename OpT>
using has_T = decltype(std::declval<OpT>().T());
template <typename OpT>
using detect_has_T = llvm::is_detected<has_T, OpT>;

// Get the input and output type arrays. If the op has a single type array,
// use it for both input and output. Otherwise, return separate type arrays.
template <typename OpT, bool = detect_has_T<OpT>::value>
struct GetTypeArray {
  static ArrayAttr getInputTypes(OpT op) { return op.TinAttr(); }
  static ArrayAttr getOutputTypes(OpT op) { return op.ToutAttr(); }
};
template <typename OpT>
struct GetTypeArray<OpT, true> {
  static ArrayAttr getInputTypes(OpT op) { return op.TAttr(); }
  static ArrayAttr getOutputTypes(OpT op) { return op.TAttr(); }
};
}  // namespace detail

// Verify a functional op's inputs and outputs against its data type arrays. For
// loop ops, this also checks that the number of inputs and outputs match. This
// is guaranteed to be valid on import but may be violated by a transformation.
template <typename OpT>
static LogicalResult VerifyTypeArrayAttributes(OpT op) {
  using GetTypeArray = typename detail::GetTypeArray<OpT>;
  ValueRange args =
      SplitDataAndControlValues(op.args(), ControlType::get(op.getContext()))
          .first;
  return success(
      succeeded(VerifyTypeArray(op, args, GetTypeArray::getInputTypes(op),
                                "argument")) &&
      succeeded(VerifyTypeArray(op, op.outs(), GetTypeArray::getOutputTypes(op),
                                "result")));
}

//===----------------------------------------------------------------------===//
// If-Like Ops

template <typename IfLikeOp>
static LogicalResult VerifyIfLikeOp(IfLikeOp op,
                                    SymbolTableCollection &symbol_table) {
  if (failed(op.verifyInvariants())) return failure();
  FailureOr<TypeRange> ins = VerifyOperands(op);
  if (failed(ins)) return failure();
  FailureOr<TypeRange> outs = VerifyResults(op);
  if (failed(outs)) return failure();

  // The first operand is the condition and is not passed to the functions.
  TypeRange func_args = ins->drop_front();

  auto then_func = symbol_table.lookupNearestSymbolFrom<GraphFuncOp>(
      op, op.then_branch().getName());
  if (then_func &&
      failed(VerifySignature(then_func, op, func_args, *outs, "then")))
    return failure();

  auto else_func = symbol_table.lookupNearestSymbolFrom<GraphFuncOp>(
      op, op.else_branch().getName());
  if (else_func &&
      failed(VerifySignature(else_func, op, func_args, *outs, "else")))
    return failure();

  return VerifyTypeArrayAttributes(op);
}

//===----------------------------------------------------------------------===//
// Case-Like Ops

template <typename CaseLikeOp>
static LogicalResult VerifyCaseLikeOp(CaseLikeOp op,
                                      SymbolTableCollection &symbol_table) {
  if (failed(op.verifyInvariants())) return failure();
  FailureOr<TypeRange> ins = VerifyOperands(op);
  if (failed(ins)) return failure();
  FailureOr<TypeRange> outs = VerifyResults(op);
  if (failed(outs)) return failure();

  // The first operand is the branch index and is not passed to the functions.
  TypeRange func_args = ins->drop_front();

  for (auto &it : llvm::enumerate(op.branches())) {
    SymbolRefAttr func_name = it.value().template cast<FuncAttr>().getName();
    auto func =
        symbol_table.lookupNearestSymbolFrom<GraphFuncOp>(op, func_name);
    if (func && failed(VerifySignature(func, op, func_args, *outs,
                                       "branch #" + Twine(it.index()))))
      return failure();
  }

  return VerifyTypeArrayAttributes(op);
}

//===----------------------------------------------------------------------===//
// While-Like Ops

template <typename WhileLikeOp>
static LogicalResult VerifyWhileLikeOp(WhileLikeOp op,
                                       SymbolTableCollection &symbol_table) {
  if (failed(op.verifyInvariants())) return failure();
  FailureOr<TypeRange> ins = VerifyOperands(op);
  if (failed(ins)) return failure();
  FailureOr<TypeRange> outs = VerifyResults(op);
  if (failed(outs)) return failure();

  SymbolRefAttr body_name = op.body().getName();

  auto cond_func = symbol_table.lookupNearestSymbolFrom<GraphFuncOp>(
      op, op.cond().getName());
  auto i1_type = UnrankedTensorType::get(Builder(op.getContext()).getI1Type());
  if (cond_func &&
      failed(VerifySignature(cond_func, op, *ins, i1_type, "cond")))
    return failure();

  auto body_func = symbol_table.lookupNearestSymbolFrom<GraphFuncOp>(
      op, op.body().getName());
  if (body_func && failed(VerifySignature(body_func, op, *ins, *outs, "body")))
    return failure();

  return VerifyTypeArrayAttributes(op);
}

//===----------------------------------------------------------------------===//
// ForOp

LogicalResult ForOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(verifyInvariants())) return failure();
  FailureOr<TypeRange> ins = VerifyOperands(*this);
  if (failed(ins)) return failure();
  FailureOr<TypeRange> outs = VerifyResults(*this);
  if (failed(outs)) return failure();

  auto body_func =
      symbolTable.lookupNearestSymbolFrom<GraphFuncOp>(*this, body().getName());
  // The first three arguments are the for-loop indices, but the current loop
  // index is passed in.
  TypeRange func_args = llvm::drop_begin(*ins, /*N=*/2);
  if (body_func &&
      failed(VerifySignature(body_func, *this, func_args, *outs, "body")))
    return failure();

  return VerifyTypeArrayAttributes(*this);
}

//===----------------------------------------------------------------------===//
// Region Ops and Terminators
//===----------------------------------------------------------------------===//

// If a region op has preserved attributes, verify that they match the number of
// results and block arguments.
static LogicalResult VerifyPreservedAttrs(Operation *op,
                                          ArrayRef<Attribute> preserved_attrs) {
  assert(op->getNumRegions() == preserved_attrs.size());
  for (auto it : llvm::zip(preserved_attrs, op->getRegions())) {
    // Preserved attributes for a particular region may not exist.
    auto attrs = std::get<0>(it).dyn_cast_or_null<RegionAttr>();
    if (!attrs) continue;
    Region &region = std::get<1>(it);

    const auto emit_region_error = [&](StringRef msg) {
      return op->emitOpError("region #")
             << region.getRegionNumber() << " " << msg;
    };

    unsigned num_args = GetLoopRegionDataArgs(region).size();
    if (num_args != attrs.getArg_attrs().size()) {
      return emit_region_error("has ")
             << num_args << " argument(s) but preserved attributes has "
             << attrs.getArg_attrs().size();
    }

    // All regions are terminated by either a YieldOp or a ConditionOp. In the
    // latter case, the function will only have one result.
    unsigned num_rets;
    Operation *terminator = region.front().getTerminator();
    if (isa<ConditionOp>(terminator)) {
      num_rets = 1;
    } else {
      num_rets = cast<RegionBranchTerminatorOpInterface>(terminator)
                     .getMutableSuccessorOperands(region.getRegionNumber())
                     .size();
    }
    if (num_rets != attrs.getRes_attrs().size()) {
      return emit_region_error("has ")
             << num_rets << " result(s) but preserved attributes has "
             << attrs.getRes_attrs().size();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp

MutableOperandRange YieldOp::getMutableSuccessorOperands(
    Optional<unsigned> index) {
  // Get the subrange of non-control operands.
  return argsMutable();
}

static bool TerminatedByYield(Block &block) {
  return isa<YieldOp>(block.getTerminator());
}

//===----------------------------------------------------------------------===//
// IfLikeRegionOp

// Verify an if-like region op.
template <typename IfLikeRegionOp>
static LogicalResult VerifyIfLikeRegionOp(IfLikeRegionOp op) {
  // Verify terminators.
  if (!TerminatedByYield(op.then_block()))
    return op.emitOpError("then region must be terminated by a 'tfg.yield'");
  if (!TerminatedByYield(op.else_block()))
    return op.emitOpError("else region must be terminated by a 'tfg.yield'");
  return VerifyPreservedAttrs(
      op, {op.then_region_attrsAttr(), op.else_region_attrsAttr()});
}

// Given an potentially null attribute that would represent a constant value,
// try to narrow it to a statically known condition.
// TODO(jeffniu): Incorporate the other cases of `tf.ToBool`.
static Optional<bool> GetStaticallyKnownBranch(Attribute cond_attr) {
  // Only handle the case of a scalar tensor of i1.
  auto cond = cond_attr.dyn_cast_or_null<ElementsAttr>();
  if (cond && cond.getNumElements() == 1 &&
      cond.getElementType().isSignlessInteger(1))
    return cond.getSplatValue<bool>();
  return {};
}

// Get the successor of the regions of an if-like op.
template <typename IfLikeRegionOp>
void GetIfLikeRegionOpSuccessorRegions(
    IfLikeRegionOp op, Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  assert(index.hasValue() ||
         !operands.empty() && "if-like op expected at least 1 operand");
  // Both regions branch back to the parent op.
  if (index.hasValue()) {
    // Ignore the control token.
    regions.emplace_back(
        ResultRange(op->result_begin(), std::prev(op->result_end())));
  } else if (auto cond = GetStaticallyKnownBranch(operands[0])) {
    // Add only 1 possible successor if the condition is known.
    Region &region = *cond ? op.then_region() : op.else_region();
    regions.emplace_back(&region, GetLoopRegionDataArgs(region));
  } else {
    // Unknown successor.
    regions.emplace_back(&op.then_region(),
                         GetLoopRegionDataArgs(op.then_region()));
    regions.emplace_back(&op.else_region(),
                         GetLoopRegionDataArgs(op.else_region()));
  }
}

//===----------------------------------------------------------------------===//
// CaseLikeRegionOp

// Verify a case-like region op.
template <typename CaseLikeRegionOp>
static LogicalResult VerifyCaseLikeRegionOp(CaseLikeRegionOp op) {
  for (auto &it : llvm::enumerate(op.branches())) {
    if (!TerminatedByYield(it.value().front())) {
      return op.emitOpError("branch region #")
             << it.index() << " is not terminated by a 'tfg.yield' op";
    }
  }

  if (op.branch_attrs() && op.branches().size() != op.branch_attrs()->size()) {
    return op.emitOpError("has ")
           << op.branches().size() << " regions but "
           << op.branch_attrs()->size() << " branch function attributes";
  }
  if (auto region_attrs = op.region_attrsAttr()) {
    if (region_attrs.size() != op.getNumRegions()) {
      return op.emitOpError("expected ")
             << op.getNumRegions() << " region attribute(s) but got "
             << region_attrs.size();
    }
    if (failed(VerifyPreservedAttrs(op, region_attrs.getValue())))
      return failure();
  }
  return success();
}

// Given a potentially null attribute that would represent a constant value,
// try to narrow it to a statically known branch index.
static Optional<unsigned> GetStaticallyKnownCaseBranch(Attribute branch_attr) {
  auto branch = branch_attr.dyn_cast_or_null<ElementsAttr>();
  if (branch && branch.getNumElements() == 1 &&
      branch.getElementType().isSignlessInteger(32))
    return branch.getSplatValue<unsigned>();
  return {};
}

// Get the successor of the regions of a case-like op.
template <typename CaseLikeRegionOp>
void GetCaseLikeRegionOpSuccessorRegions(
    CaseLikeRegionOp op, Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  assert(index.hasValue() ||
         !operands.empty() && "case-like op expected at least 1 operand");
  // All branch regions branch back to the parent op.
  if (index.hasValue()) {
    // Ignore the control token.
    regions.emplace_back(
        ResultRange(op->result_begin(), std::prev(op->result_end())));
  } else if (auto branch_index = GetStaticallyKnownCaseBranch(operands[0])) {
    // Add only 1 possible successor if the condition is known.
    Region &region = op.branches()[*branch_index];
    regions.emplace_back(&region, GetLoopRegionDataArgs(region));
  } else {
    // Unknown successor. Add all of them.
    for (Region &branch : op.branches())
      regions.emplace_back(&branch, GetLoopRegionDataArgs(branch));
  }
}

//===----------------------------------------------------------------------===//
// ConditionOp

MutableOperandRange ConditionOp::getMutableSuccessorOperands(
    Optional<unsigned> index) {
  // Get the subrange of non-control operands that are forwarded to the
  // successor region.
  return argsMutable();
}

//===----------------------------------------------------------------------===//
// WhileLikeRegionOp

// Verify that the loop regions of a region-based loop op have N control tokens
// immediately following N data values in their entry block arguments.
// `RegionBranchOpInterface` will verify the number of arguments and their
// types.
static LogicalResult VerifyLoopRegionArgs(Operation *op, Region &region) {
  const auto arg_error = [&](BlockArgument arg) {
    return op->emitOpError("region #")
           << region.getRegionNumber() << " argument #" << arg.getArgNumber()
           << " ";
  };

  // The interface trait verifies the number of data and control arguments. If
  // the first half of the arguments are not control tokens, then we know for
  // sure that the second half is only control tokens.
  for (BlockArgument data : GetLoopRegionDataArgs(region))
    if (data.getType().isa<ControlType>())
      return arg_error(data) << "should not be a control token";
  return success();
}

// Verify a while-like region op.
template <typename WhileLikeRegionOp>
static LogicalResult VerifyWhileLikeRegionOp(WhileLikeRegionOp op) {
  // Verify terminators.
  if (!isa<ConditionOp>(op.cond_block().getTerminator())) {
    return op.emitOpError(
        "condition region must be terminated by a 'tfg.condition' op");
  }
  if (!TerminatedByYield(op.body_block()))
    op.emitOpError("body region must be terminated by a 'tfg.yield' op");

  if (failed(VerifyLoopRegionArgs(op, op.cond_region())) ||
      failed(VerifyLoopRegionArgs(op, op.body_region())))
    return failure();
  if (failed(VerifyPreservedAttrs(
          op, {op.cond_region_attrsAttr(), op.body_region_attrsAttr()})))
    return failure();

  return success();
}

template <typename WhileLikeRegionOp>
static void GetWhileLikeRegionOpSuccessorRegions(
    WhileLikeRegionOp op, Optional<unsigned> index,
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  // The parent op and the body region always branch to the condion region.
  if (!index || *index == 1) {
    regions.emplace_back(&op.cond_region(),
                         GetLoopRegionDataArgs(op.cond_region()));
    return;
  }
  assert(*index == 0 && "invalid region index");
  // The condition regions branches to the loop body or back to the parent.
  // Try to narrow the condition value to a constant.
  auto condition = cast<ConditionOp>(op.cond_region().front().getTerminator());
  Attribute cond_attr;
  matchPattern(condition.cond(), m_Constant(&cond_attr));
  Optional<bool> cond = GetStaticallyKnownBranch(cond_attr);
  if (!cond || *cond) {
    regions.emplace_back(&op.body_region(),
                         GetLoopRegionDataArgs(op.body_region()));
  }
  if (!cond || !*cond) {
    // Drop the control token.
    regions.emplace_back(op.getResults().drop_back());
  }
}

//===----------------------------------------------------------------------===//
// ForRegionOp

LogicalResult ForRegionOp::verify() {
  if (!TerminatedByYield(body_block())) {
    return emitOpError("body region must be terminated by a 'tfg.yield' op");
  }

  Block::BlockArgListType args = body_block().getArguments();
  if (args.empty()) {
    return emitOpError(
        "expected the body block to have at least have the loop index as an "
        "argument");
  }
  auto index = args.front().getType().dyn_cast<TensorType>();
  if (!index || !index.getElementType().isSignlessInteger(32)) {
    return emitOpError(
        "expected first body block argument to be an i32 tensor");
  }

  if (failed(VerifyLoopRegionArgs(*this, body_region()))) return failure();
  return VerifyPreservedAttrs(*this, {region_attrsAttr()});
}

OperandRange ForRegionOp::getSuccessorEntryOperands(unsigned index) {
  return init();
}

void ForRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Both the parent op and the body region branch to the body. Ignore the loop
  // index block argument, as it is not modified by the loop body itself.
  regions.emplace_back(&body_region(),
                       GetLoopRegionDataArgs(body_region()).drop_front());
  if (!index) return;
  // The body might branch back to the parent. Drop the control token.
  regions.emplace_back((*this)->getResults().drop_back());
}

BlockArgument ForRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument ForRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument ForRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument ForRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}

//===----------------------------------------------------------------------===//
// Function Table
//===----------------------------------------------------------------------===//

FunctionTable::FunctionTable(ModuleOp module) {
  // Collect function names (to be used for disambiguating legacy call
  // behavior).
  for (auto &op : module.getOps()) {
    if (auto func = dyn_cast<GraphFuncOp>(op)) functions.insert(func.getName());
  }
}

bool FunctionTable::MaybeCall(Operation *op) {
  if (functions.count(op->getName().stripDialect())) return true;
  for (NamedAttribute named_attr : op->getAttrs()) {
    // Treat any operation that references a FuncAttr as a call.
    if (named_attr.getValue().isa<FuncAttr>()) return true;
  }
  return false;
}

}  // namespace tfg
}  // namespace mlir

//===----------------------------------------------------------------------===//
// ODS Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/core/ir/ops.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "tensorflow/core/ir/attributes.cc.inc"
