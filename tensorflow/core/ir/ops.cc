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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/ir/utility.h"

// Generated definitions.
#include "tensorflow/core/ir/dialect.cc.inc"

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// TFGraph dialect.
//===----------------------------------------------------------------------===//

// TFGraph support for interacting with the AsmPrinter.
// Gives prettier names to SSA values.
struct TFGraphOpAsmInterface
    : public OpAsmOpInterface::FallbackModel<TFGraphOpAsmInterface> {
  static bool classof(Operation *op) { return true; }

  // Name operation results with the operation name, except control outputs
  // which are named "ctl". MLIR will automatically use a numerical suffix to
  // unique.
  void getAsmResultNames(Operation *op, OpAsmSetValueNameFn set_name_fn) const {
    // We only name the results when there are results to name, an op like
    // `print` which does not have results will just use the `ctl` name for the
    // control output.
    if (op->getNumResults() > 1 &&
        !op->getResult(0).getType().isa<ControlType>())
      set_name_fn(op->getResult(0), op->getName().stripDialect());
    for (Value result : op->getResults()) {
      if (result.getType().isa<ControlType>()) {
        set_name_fn(op->getResult(op->getNumResults() - 1), "ctl");
        break;
      }
    }
  }
  void getAsmBlockArgumentNames(Operation *op, Region &region,
                                OpAsmSetValueNameFn setNameFn) const {
    return;
  }
};

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void TFGraphDialect::initialize() {
  getContext()->getOrLoadDialect<tf_type::TFTypeDialect>();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/core/ir/ops.cc.inc"
      >();

  // Support unknown operations because not all TensorFlow operations are
  // registered.
  allowUnknownOperations();

  fallbackOpAsmInterface_ = new TFGraphOpAsmInterface;
  // Caching some often used context-owned informations for fast-access.
  name_key_ = StringAttr::get(getContext(), getNameAttrKey());
  device_key_ = StringAttr::get(getContext(), getDeviceAttrKey());
  assigned_device_key_ =
      StringAttr::get(getContext(), getAssignedDeviceAttrKey());
  control_ty_ = ControlType::get(getContext());
}

// Provides a hook for op interface.
void *TFGraphDialect::getRegisteredInterfaceForOp(mlir::TypeID interface,
                                                  mlir::OperationName opName) {
  if (interface == TypeID::get<mlir::OpAsmOpInterface>()) {
    return fallbackOpAsmInterface_;
  }

  return nullptr;
}

TFGraphDialect::~TFGraphDialect() { delete fallbackOpAsmInterface_; }

// Print an operation that belongs to this dialect, if unregistered.
// The general syntax is:
//   tfg.OpName(%input1, %input2, %input3) [%control_dep1, %control_dep2]
//           name("<node_name>") device("<device>") { attribute-dict } :
//           (input types) -> (result_types)
void TFGraphDialect::printCustomTfOp(Operation *op,
                                     OpAsmPrinter &printer) const {
  raw_ostream &os = printer.getStream();
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
  if (llvm::any_of(op->getOperandTypes(),
                   [&](Type ty) { return ty != control_ty; })) {
    os << "(";
    llvm::interleaveComma(
        llvm::make_filter_range(
            op->getOperands(),
            [&](Value operand) { return operand.getType() != control_ty; }),
        os, [&](Value operand) { printer.printOperand(operand); });
    os << ")";
  }
  // Print the control dependencies (if any).
  if (llvm::any_of(op->getOperands(), [&](Value operand) {
        return operand.getType() == control_ty;
      })) {
    os << " [";
    llvm::interleaveComma(
        llvm::make_filter_range(
            op->getOperands(),
            [&](Value operand) { return operand.getType() == control_ty; }),
        os, [&](Value operand) { printer.printOperand(operand); });
    os << "]";
  }

  // Handles the optional "device" and "name" attribute.
  std::array<llvm::StringRef, 3> keywords{
      "_mlir_device", "_mlir_assigned_device", "_mlir_name"};
  for (StringRef keyword : keywords) {
    if (StringAttr value_attr = op->getAttrOfType<StringAttr>(keyword))
      if (!value_attr.getValue().empty())
        os << " " << keyword.drop_front(/*len(_mlir_)*/ 6) << "(\""
           << value_attr.getValue() << "\")";
  }

  // Print attributes (other than name and device).
  printer.printOptionalAttrDict(op->getAttrs(), keywords);

  // Print the type, but omit control dependencies.
  // If there is a single control return, just print the list of input types,
  // otherwise print the complete type in a "function-style" way: (operands)
  // -> (results).
  auto is_not_control_dep = [&](Type t) { return t != control_ty; };
  if (isa<ReturnOp>(op) ||
      (op->getNumResults() == 1 && op->getResult(0).getType() == control_ty)) {
    if (llvm::any_of(op->getOperandTypes(), is_not_control_dep)) {
      os << " : ";
      llvm::interleaveComma(
          make_filter_range(op->getOperandTypes(), is_not_control_dep), os);
    }
  } else {
    if (op->getNumResults() > 1 ||
        llvm::any_of(op->getOperandTypes(), is_not_control_dep)) {
      os << " : (";
      llvm::interleaveComma(
          make_filter_range(op->getOperandTypes(), is_not_control_dep), os);
      os << ") -> (";
      llvm::interleaveComma(
          make_filter_range(op->getResultTypes(), is_not_control_dep), os);
      os << ")";
    }
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

// Parse an operation that belongs to this dialect, if unregistered.
// The general syntax is:
//   tfg.OpName(%input1, %input2, %input3) [%control_dep1, %control_dep2]
//           name("<node_name>") device("<device>") { attribute-dict } :
//           (input types) -> (result_types)
static ParseResult ParseCustomTfOp(OpAsmParser &parser,
                                   OperationState &result) {
  MLIRContext *context = parser.getBuilder().getContext();
  // Parse optional argument list
  SmallVector<OpAsmParser::OperandType, 4> op_infos;
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      OpAsmParser::OperandType operand;
      if (failed(parser.parseOperand(operand))) return failure();
      op_infos.push_back(operand);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen()) return failure();
  }
  int numNonControlOperands = op_infos.size();
  // Optional control list, in between brackets.
  if (succeeded(parser.parseOptionalLSquare())) {
    do {
      OpAsmParser::OperandType operand;
      if (failed(parser.parseOperand(operand))) return failure();
      op_infos.push_back(operand);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare()) return failure();
  }

  // Parse the optional device and name attributes.
  for (const char *keyword : {"device", "assigned_device", "name"}) {
    if (succeeded(parser.parseOptionalKeyword(keyword))) {
      StringAttr value;
      if (parser.parseLParen() ||
          parser.parseAttribute<StringAttr>(value, NoneType::get(context)) ||
          parser.parseRParen())
        return failure();
      result.addAttribute((Twine("_mlir_") + keyword).str(), value);
    }
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  // Parse the functional type.
  SmallVector<Type> arg_types;
  arg_types.reserve(op_infos.size());
  llvm::SMLoc loc = parser.getCurrentLocation();
  Type control_type = ControlType::get(context);
  if (failed(parser.parseOptionalColonTypeList(arg_types))) return failure();
  if (arg_types.size() == 1 && arg_types.front().isa<FunctionType>()) {
    auto funcType = arg_types.front().dyn_cast<FunctionType>();
    if (!funcType)
      return parser.emitError(loc)
             << "expected functional type, got " << arg_types.front();
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
  return true;
}

//===----------------------------------------------------------------------===//
// Graph Operation
//===----------------------------------------------------------------------===//

static void PrintGraphOp(OpAsmPrinter &p, GraphOp op) {
  p << ' ' << op.version();
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), {"version"});
  p << ' ';
  p.printRegion(op.getBodyRegion());
}

static ParseResult ParseGraphOp(OpAsmParser &parser, OperationState &result) {
  VersionAttr version;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseAttribute(version)) {
    parser.emitError(loc) << "expected a version attribute";
    return failure();
  }
  result.addAttribute("version", version);

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*result.addRegion()))
    return failure();
  if (result.regions.front()->empty())
    result.regions.front()->push_back(new Block);
  return success();
}

static LogicalResult VerifyGraph(GraphOp graph_op) {
  // Check all ops in the body.
  if (!all_of(*graph_op.getBody(), VerifyGenericTFGOperation)) return failure();

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
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

// Hook for OpTrait::FunctionLike, called after verifying the function
// type and the presence of the (potentially empty) function body.
LogicalResult GraphFuncOp::verifyBody() {
  // Check that the body is terminated with a tfg.return.
  if (getRegion().empty() || getBody()->empty())
    return emitOpError() << "expects a non empty body";

  if (getBody()->getNumArguments() != getType().getNumInputs())
    return emitOpError() << "function type indicated "
                         << getType().getNumInputs() << " args but block has "
                         << getBody()->getNumArguments();

  for (auto &arg_types : llvm::enumerate(
           llvm::zip(getType().getInputs(), getBody()->getArgumentTypes()))) {
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
  FunctionType type = getType();

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

static LogicalResult VerifyGraphFunc(GraphFuncOp func_op) {
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

static ParseResult ParseGraphFunc(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType> entry_args;
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
    if (parser.parseRegionArgument(entry_args.back()) ||
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
    OpAsmParser::OperandType control_operand = entry_args.back();
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
  if (failed(parser.parseRegion(
          *body, entry_args, entry_args.empty() ? ArrayRef<Type>() : arg_types,
          /*argLocations=*/{},
          /*enableNameShadowing=*/false)))
    return failure();

  // Function body was parsed, make sure it's not empty.
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  return success();
}

static void PrintGraphFunc(GraphFuncOp op, OpAsmPrinter &p) {
  // Print the operation and the function name.
  p << " ";
  int argIndentSize = op->getName().getStringRef().size() + 3;
  StringRef visibility_attr_name = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibility_attr_name)) {
    p << visibility.getValue() << ' ';
    argIndentSize += visibility.getValue().size() + 1;
  }
  if (op.generic()) p << "generic ";
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p.printSymbolName(funcName);
  argIndentSize += funcName.size();
  std::string indent(argIndentSize, ' ');
  Region &body = op->getRegion(0);
  FunctionType fnType = op.getType();
  ArrayRef<Type> arg_types = fnType.getInputs();
  ArrayRef<Type> result_types = fnType.getResults();
  assert((arg_types.size() % 2) == 0);
  // Print operand list with attributes.
  p << '(';
  ArrayAttr args_attr = op.getAllArgAttrs();
  for (unsigned i = 0, e = arg_types.size(); i < e; i += 2) {
    // Args come by pair: input+control.
    p.printOperand(body.getArgument(i));
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
    ArrayAttr results_attr = op.getAllResultAttrs();
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
        p, op, fnType.getNumInputs(), fnType.getNumResults(),
        {"generic", SymbolTable::getVisibilityAttrName()});
  }
  // Print body.
  p << ' ';
  p.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false);
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
    return {op->emitOpError(
        "not all control tokens come after non-control operands")};
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
    return {op->emitOpError("does not define a control result")};
  if (it != std::prev(op->result_type_end())) {
    return {op->emitOpError(
        "must have a control token result as and only as its last result")};
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

  ArrayRef<Type> arguments = func.getType().getInputs();
  ArrayRef<Type> returns = func.getType().getResults();
  if (operands.size() * 2 != arguments.size()) {
    return attach_func(op->emitOpError(func_name)
                       << " function expected to have " << operands.size() * 2
                       << " arguments but got " << arguments.size());
  }
  if (results.size() != returns.size()) {
    return attach_func(op->emitOpError(func_name)
                       << " function expected to have " << results.size()
                       << " return values but got " << returns.size());
  }

  if (func.generic()) return success();

  for (auto &it : llvm::enumerate(operands)) {
    auto arg_type = getElementTypeOrSelf(arguments[it.index() * 2]);
    auto op_type = getElementTypeOrSelf(it.value());
    if (arg_type != op_type) {
      return attach_func(
          op->emitOpError(func_name)
          << " function argument #" << it.index() << " dtype " << arg_type
          << " does not match corresponding operand dtype: " << op_type);
    }
  }
  for (auto &it : llvm::enumerate(results)) {
    auto ret_type = getElementTypeOrSelf(returns[it.index()]);
    auto res_type = getElementTypeOrSelf(it.value());
    if (ret_type != res_type) {
      return attach_func(
          op->emitOpError(func_name)
          << " function result #" << it.index() << " dtype " << ret_type
          << " does not match corresponding op result dtype: " << res_type);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp

LogicalResult CastOp::fold(ArrayRef<Attribute> operands,
                           SmallVectorImpl<OpFoldResult> &results) {
  // If the op has control operands, we can't fold.
  if (!ctls().empty()) return failure();
  // GetResultOp gets the value from uninstantiated op and it can't be folded.
  if (x().getDefiningOp<GetResultOp>()) return failure();

  ShapedType x_shape = x().getType().dyn_cast<ShapedType>();
  ShapedType y_shape = y().getType().dyn_cast<ShapedType>();
  if (!x_shape || x_shape != y_shape || !x_shape.hasStaticShape())
    return failure();

  results.push_back(x());
  results.push_back(LookupControlDependency(x()));
  return success();
}

//===----------------------------------------------------------------------===//
// If-Like Ops

template <typename IfLikeOp>
static LogicalResult VerifyIfLikeOp(IfLikeOp op,
                                    SymbolTableCollection &symbol_table) {
  if (failed(op.verify())) return failure();
  FailureOr<TypeRange> ins = VerifyOperands(op);
  if (failed(ins)) return failure();
  FailureOr<TypeRange> outs = VerifyResults(op);
  if (failed(outs)) return failure();

  SymbolRefAttr then_name = op.then_branch().getName();
  SymbolRefAttr else_name = op.else_branch().getName();
  // The first operand is the condition and is not passed to the functions.
  TypeRange func_args = llvm::drop_begin(*ins);

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

  return success();
}

//===----------------------------------------------------------------------===//
// Case-Like Ops

template <typename CaseLikeOp>
static LogicalResult VerifyCaseLikeOp(CaseLikeOp op,
                                      SymbolTableCollection &symbol_table) {
  if (failed(op.verify())) return failure();
  FailureOr<TypeRange> ins = VerifyOperands(op);
  if (failed(ins)) return failure();
  FailureOr<TypeRange> outs = VerifyResults(op);
  if (failed(outs)) return failure();

  // The first operand is the branch index and is not passed to the functions.
  TypeRange func_args = llvm::drop_begin(*ins);

  for (auto &it : llvm::enumerate(op.branches())) {
    SymbolRefAttr func_name = it.value().template cast<FuncAttr>().getName();
    auto func =
        symbol_table.lookupNearestSymbolFrom<GraphFuncOp>(op, func_name);
    if (func && failed(VerifySignature(func, op, func_args, *outs,
                                       "branch #" + Twine(it.index()))))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// While-Like Ops

template <typename WhileLikeOp>
static LogicalResult VerifyWhileLikeOp(WhileLikeOp op,
                                       SymbolTableCollection &symbol_table) {
  if (failed(op.verify())) return failure();
  FailureOr<TypeRange> ins = VerifyOperands(op);
  if (failed(ins)) return failure();
  FailureOr<TypeRange> outs = VerifyResults(op);
  if (failed(outs)) return failure();

  SymbolRefAttr body_name = op.body().getName();

  auto cond_func = symbol_table.lookupNearestSymbolFrom<GraphFuncOp>(
      op, op.cond().getName());
  auto i1_type = Builder(op.getContext()).getI1Type();
  if (cond_func &&
      failed(VerifySignature(cond_func, op, *ins, i1_type, "cond")))
    return failure();

  auto body_func = symbol_table.lookupNearestSymbolFrom<GraphFuncOp>(
      op, op.body().getName());
  if (body_func && failed(VerifySignature(body_func, op, *ins, *outs, "body")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ForOp

LogicalResult ForOp::verifySymbolUses(SymbolTableCollection &symbol_table) {
  if (failed(verify())) return failure();
  FailureOr<TypeRange> ins = VerifyOperands(*this);
  if (failed(ins)) return failure();
  FailureOr<TypeRange> outs = VerifyResults(*this);
  if (failed(outs)) return failure();

  auto body_func = symbol_table.lookupNearestSymbolFrom<GraphFuncOp>(
      *this, body().getName());
  // The first three arguments are the for-loop indices, but the current loop
  // index is passed in.
  TypeRange func_args = llvm::drop_begin(*ins, /*N=*/2);
  if (body_func &&
      failed(VerifySignature(body_func, *this, func_args, *outs, "body")))
    return failure();
  return success();
}

}  // namespace tfg
}  // namespace mlir

//===----------------------------------------------------------------------===//
// ODS Op Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/core/ir/ops.cc.inc"
