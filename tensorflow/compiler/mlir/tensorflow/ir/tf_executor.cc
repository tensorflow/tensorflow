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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

#include <algorithm>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Traits.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/StandardOps/Ops.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_executor {
namespace {

// If the given tensor has elements of type variant, then returns a new type
// after dropping subtypes info. Otherwise, returns the original type as is.
Type DropVariantSubTypes(Type ty) {
  ShapedType shaped_ty = ty.cast<ShapedType>();
  Type element_ty = shaped_ty.getElementType();
  if (!element_ty.isa<TF::VariantType>()) return ty;

  Type variant_ty = TF::VariantType::get(ty.getContext());
  if (shaped_ty.hasRank()) {
    return RankedTensorType::get(shaped_ty.getShape(), variant_ty);
  }

  return UnrankedTensorType::get(variant_ty);
}

}  // namespace

//===----------------------------------------------------------------------===//
// TF Executor Dialect
//===----------------------------------------------------------------------===//

TensorFlowExecutorDialect::TensorFlowExecutorDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf_executor", context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc.inc"
      >();
  addTypes<ControlType, TokenType>();
}

Type TensorFlowExecutorDialect::parseType(StringRef data_type,
                                          Location loc) const {
  if (data_type == "control") return ControlType::get(getContext());
  if (data_type == "token") return TokenType::get(getContext());
  emitError(loc) << "unknown tf_executor type: " << data_type;
  return nullptr;
}

void TensorFlowExecutorDialect::printType(Type type, raw_ostream &os) const {
  if (type.isa<ControlType>()) {
    os << "control";
    return;
  }
  if (type.isa<TokenType>()) {
    os << "token";
    return;
  }
  os << "<unknown tf_executor type>";
}

//===----------------------------------------------------------------------===//
// Implementation for all the operations defined in ODS (op definition spec).
//===----------------------------------------------------------------------===//

namespace {

// Verifies that every control operands are at the end of the list.
// Used by the constraint `ControlOperandsAfterAllData` in ODS.
LogicalResult VerifyControlOperandsAfterAllData(Operation *op) {
  bool found_control = false;
  for (int operand_idx : llvm::seq<int>(0, op->getNumOperands())) {
    if (op->getOperand(operand_idx)->getType().isa<ControlType>()) {
      found_control = true;
      continue;
    }
    if (found_control)
      return op->emitOpError() << "found non-control operand #" << operand_idx
                               << " after control operand";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// tf_executor.graph
//===----------------------------------------------------------------------===//

LogicalResult Verify(GraphOp graph) {
  auto *executorDialect = graph.getDialect();

  if (graph.GetBody().empty())
    return graph.emitOpError() << "expects a non-empty body";

  // Only tf_executor dialect operations are allowed to be immediately nested
  // in a tf_executor.graph region.
  for (Operation &op : graph.GetBody()) {
    if (op.getDialect() != executorDialect)
      return op.emitOpError() << "unallowed inside a tf_executor.graph region";
    if (isa<GraphOp>(op))
      return op.emitOpError()
             << "unallowed directly inside another tf_executor.graph";
  }

  Operation &fetch = graph.GetBody().back();
  if (!isa<FetchOp>(fetch))
    return fetch.emitOpError()
           << "invalid tf_executor.graph terminator, fetch expected";

  // Ensure that the fetch terminator operands matches the graph result type.
  // All the non-control operands of the fetch operation must match the graph
  // returned value.
  if (fetch.getNumOperands() < graph.getNumResults())
    return fetch.emitOpError() << "does not have enough operands to cover the "
                                  "graph returned values";
  for (int i : llvm::seq<int>(0, fetch.getNumOperands())) {
    Value *operand = fetch.getOperand(i);
    // Break out of the loop at the first control operand encountered.
    if (operand->getType().isa<ControlType>()) {
      if (i != graph.getNumResults())
        return fetch.emitOpError()
               << "operand #" << i
               << " is a control type, can't be bound to a graph result";
      break;
    }
    if (i >= graph.getNumResults())
      return fetch.emitOpError()
             << "operand #" << i << " does not have a graph results to bind";
    if (graph.getResult(i)->getType() != operand->getType())
      return fetch.emitOpError()
             << "operand #" << i << " type mismatch graph results";
  }
  return success();
}

void Print(GraphOp graph, OpAsmPrinter *p) {
  *p << graph.getOperationName();
  p->printRegion(graph.getOperation()->getRegion(0));
  p->printOptionalAttrDict(graph.getAttrs());
}

ParseResult ParseGraphOp(OpAsmParser *parser, OperationState *result) {
  llvm::SMLoc loc = parser->getCurrentLocation();

  // Parse the body region.
  Region &body = *result->addRegion();
  if (parser->parseRegion(body, llvm::None, llvm::None)) return failure();

  if (body.getBlocks().size() > 1)
    return parser->emitError(loc) << "expects a single block region";

  // Ensure that the region is well formed: it contains at least a block with
  // a FetchOp terminator.
  GraphOp::ensureTerminator(body, parser->getBuilder(), result->location);

  // Get the results type from the terminator type inside the graph.
  Operation &fetch = body.back().back();
  if (!isa<FetchOp>(fetch))
    return parser->emitError(loc) << "expects a tf_executor.fetch terminator";

  // The return value of the graph operation are the non-control operands of
  // the fetch operation.
  result->types.reserve(fetch.getNumOperands());
  for (Type type : fetch.getOperandTypes()) {
    if (type.isa<ControlType>()) break;
    result->types.push_back(type);
  }

  // Parse the optional attribute list.
  if (parser->parseOptionalAttributeDict(result->attributes)) return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// tf_executor.fetch
//===----------------------------------------------------------------------===//

void Print(FetchOp fetch, OpAsmPrinter *p) {
  *p << fetch.getOperationName();
  if (fetch.getNumOperands() > 0) {
    *p << ' ';
    p->printOperands(fetch.operand_begin(), fetch.operand_end());
    *p << " : ";
    interleaveComma(fetch.getOperandTypes(), *p);
  }
  p->printOptionalAttrDict(fetch.getAttrs());
}

ParseResult ParseFetchOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser->getCurrentLocation();
  return failure(
      parser->parseOperandList(opInfo) ||
      (!opInfo.empty() && parser->parseColonTypeList(types)) ||
      parser->resolveOperands(opInfo, types, loc, result->operands) ||
      parser->parseOptionalAttributeDict(result->attributes)

  );
}

//===----------------------------------------------------------------------===//
// tf_executor.island
//===----------------------------------------------------------------------===//

LogicalResult Verify(IslandOp island) {
  if (island.GetBody().empty())
    return island.emitOpError() << "expects a non-empty body";

  Operation &yield = island.GetBody().back();
  if (!isa<YieldOp>(yield))
    return yield.emitOpError()
           << "invalid tf_executor.island terminator, yield expected";

  // Ensure that the yield terminator operands matches the island results type.
  int result_count = island.getNumResults() - 1;  // -1 for the control token
  if (yield.getNumOperands() != result_count)
    return yield.emitOpError()
           << "has " << yield.getNumOperands()
           << " operand, but island returns " << result_count;
  for (int operand_idx : llvm::seq<int>(0, yield.getNumOperands())) {
    if (island.getResult(operand_idx)->getType() !=
        yield.getOperand(operand_idx)->getType())
      return yield.emitOpError()
             << "operand #" << operand_idx << " type mismatch island results";
  }

  // Check that there aren't any control results other than the last one.
  Type control_type = ControlType::get(island.getContext());
  for (int operand_idx : llvm::seq<int>(0, island.getNumResults() - 1)) {
    if (island.getResult(operand_idx)->getType() == control_type)
      return yield.emitOpError()
             << "unexpected control type for operand #" << operand_idx;
  }
  return success();
}

void Print(IslandOp op, OpAsmPrinter *p) {
  *p << op.getOperationName();
  if (op.getNumOperands()) {
    // These are always control operand, no explicit type needed.
    *p << '(';
    p->printOperands(op.getOperands());
    *p << ')';
  }
  p->printRegion(op.getOperation()->getRegion(0));
  p->printOptionalAttrDict(op.getAttrs());
}

ParseResult ParseIslandOp(OpAsmParser *parser, OperationState *result) {
  llvm::SMLoc loc = parser->getCurrentLocation();
  Type control_type = ControlType::get(parser->getBuilder().getContext());

  // Parse optional argument list (control dependencies only).
  SmallVector<OpAsmParser::OperandType, 4> op_infos;
  if (parser->parseOperandList(op_infos, OpAsmParser::Delimiter::OptionalParen))
    return failure();
  if (!op_infos.empty()) {
    SmallVector<Type, 2> types(op_infos.size(), control_type);
    parser->resolveOperands(op_infos, types, loc, result->operands);
  }

  // Parse the body region.
  Region &body = *result->addRegion();

  // TODO(b/134773778): the custom parser is missing support to implement to
  // short syntax right now.
  // if (!parser->parseOptionalKeyword("wraps")) {
  //   body.push_back(new Block);
  //   Block &block = body.back();
  //   parser->getBuilder().setInsertionPointToEnd(&block);
  //   if (parser->parseOperation())
  //     return failure();
  // }

  if (parser->parseRegion(body, llvm::None, llvm::None)) return failure();

  IslandOp::ensureTerminator(body, parser->getBuilder(), result->location);

  // Get the results type for the island from the terminator operands.
  Operation &yield = body.back().back();
  result->types.reserve(yield.getNumOperands() + 1);
  result->types.append(yield.operand_type_begin(), yield.operand_type_end());
  result->types.push_back(control_type);

  // Parse the optional attribute list.
  if (parser->parseOptionalAttributeDict(result->attributes)) return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// tf_executor.yield
//===----------------------------------------------------------------------===//

void Print(YieldOp yield, OpAsmPrinter *p) {
  *p << yield.getOperationName();
  if (yield.getNumOperands() > 0) {
    *p << ' ';
    p->printOperands(yield.operand_begin(), yield.operand_end());
    *p << " : ";
    interleaveComma(yield.getOperandTypes(), *p);
  }
  p->printOptionalAttrDict(yield.getAttrs());
}

ParseResult ParseYieldOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> op_info;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser->getCurrentLocation();
  return failure(
      parser->parseOperandList(op_info) ||
      (!op_info.empty() && parser->parseColonTypeList(types)) ||
      parser->resolveOperands(op_info, types, loc, result->operands) ||
      parser->parseOptionalAttributeDict(result->attributes));
}

//===----------------------------------------------------------------------===//
// tf_executor.Switch
//===----------------------------------------------------------------------===//

ParseResult ParseSwitchOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> op_infos;
  SmallVector<Type, 1> types;
  if (parser->parseOperandList(op_infos, 2) ||
      parser->parseColonTypeList(types))
    return failure();
  if (types.size() != 1)
    return parser->emitError(parser->getNameLoc())
           << " expects only a single data type";

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // input and the outputs are all using this type).
  if (types.front().isa<FunctionType>()) {
    FunctionType type = types.front().cast<FunctionType>();
    if (type.getNumInputs() != 2)
      return parser->emitError(parser->getNameLoc())
             << " expects a single data type and a predicate";
    result->types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    Type control_type = ControlType::get(parser->getBuilder().getContext());
    result->types.append(2, types[0]);
    result->types.push_back(control_type);
    Type i1_type = parser->getBuilder().getI1Type();
    RankedTensorType predicate_type = RankedTensorType::get({}, i1_type);
    types.push_back(predicate_type);
    types.append(op_infos.size() - 2, control_type);
  }

  llvm::SMLoc loc = parser->getCurrentLocation();
  if (parser->resolveOperands(op_infos, types, loc, result->operands))
    return failure();

  return parser->parseOptionalAttributeDict(result->attributes);
}

void Print(SwitchOp switch_op, OpAsmPrinter *p) {
  *p << switch_op.getOperationName() << ' ';
  p->printOperands(switch_op.getOperands());
  Type data_operand_ty = switch_op.data()->getType();
  // If the types aren't perfectly matching, print the functional type syntax
  // else print the shorter single type.
  *p << " : ";
  if (switch_op.trueOutput()->getType() != data_operand_ty ||
      switch_op.falseOutput()->getType() != data_operand_ty) {
    p->printFunctionalType(switch_op.getOperation());
  } else {
    *p << switch_op.getType(0);
  }
  p->printOptionalAttrDict(switch_op.getAttrs());
}

//===----------------------------------------------------------------------===//
// tf_executor.SwitchN
//===----------------------------------------------------------------------===//

LogicalResult Verify(SwitchNOp switchn) {
  IntegerAttr num_outs = switchn.getAttrOfType<IntegerAttr>("num_outs");
  if (!num_outs)
    return switchn.emitOpError() << "expects a `num_outs` integer attribute";

  // Expect num_outs results + 1 control output.
  if (switchn.getNumResults() != num_outs.getInt() + 1)
    return switchn.emitOpError()
           << "expect `num_outs` (" << num_outs.getInt() << ") results but got "
           << (switchn.getNumResults() - 1);

  auto operand0_type = switchn.getOperand(0)->getType();
  for (Value *result : switchn.outputs())
    if (operand0_type != result->getType())
      return switchn.emitOpError()
             << "type mismatch between data operand and result: "
             << operand0_type << " vs " << result->getType();

  return success();
}

void Print(SwitchNOp switchn, OpAsmPrinter *p) {
  *p << switchn.getOperationName() << ' ';
  auto operands = switchn.getOperands();
  // Print the 2 data operands.
  p->printOperands(operands.begin(), std::next(operands.begin(), 2));
  *p << " of " << (switchn.getNumResults() - 1);
  // print control dependencies if any
  if (!llvm::empty(switchn.controlInputs())) {
    *p << " (";
    p->printOperands(switchn.controlInputs());
    *p << ")";
  }
  *p << " : " << switchn.getType(0);
  p->printOptionalAttrDict(switchn.getAttrs(), {"num_outs"});
}

ParseResult ParseSwitchNOp(OpAsmParser *parser, OperationState *result) {
  // Parsing:
  //       %2:6 = tf_executor.SwitchN %0, %1 by 5 : tensor<??xf32>
  // Where the first operand is the data to replicate, the second is an i32
  // indicating which output to populate, followed by the keyword `by` and the
  // number of outputs (+1 for the control token).
  SmallVector<OpAsmParser::OperandType, 2> op_infos;
  SmallVector<Type, 1> types;
  llvm::SMLoc loc = parser->getCurrentLocation();
  IntegerAttr num_outs;
  Type i64_type = parser->getBuilder().getIntegerType(64);
  if (parser->parseOperandList(op_infos, 2) || parser->parseKeyword("of") ||
      parser->parseAttribute(num_outs, i64_type, "num_outs",
                             result->attributes) ||
      parser->parseOperandList(op_infos,
                               OpAsmParser::Delimiter::OptionalParen) ||
      parser->parseColonTypeList(types))
    return failure();
  if (types.size() != 1)
    return parser->emitError(parser->getNameLoc())
           << " expects only a single data type";

  if (num_outs.getInt() <= 0)
    return parser->emitError(parser->getNameLoc())
           << " expects a positive number of outputs";

  // `types` already contains the type for the data, add an i32 for the
  // output_index, and then the optional control inputs.
  types.push_back(parser->getBuilder().getIntegerType(32));
  Type control_type = ControlType::get(parser->getBuilder().getContext());
  types.append(op_infos.size() - 2, control_type);

  if (parser->resolveOperands(op_infos, types, loc, result->operands))
    return failure();

  // Output result types is a replication `num_outs` times the data input type.
  result->types.append(num_outs.getInt(), types[0]);
  result->types.push_back(control_type);

  return parser->parseOptionalAttributeDict(result->attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.Merge
//===----------------------------------------------------------------------===//

LogicalResult Verify(MergeOp merge) {
  if (!merge.getNumOperands())
    return merge.emitOpError() << "expects at least one operand";

  Type data_type = merge.getOperand(0)->getType();
  if (data_type.isa<ControlType>())
    return merge.emitOpError() << "expects a non-control input";

  // Check that all operands can be broadcasted to a common type compatible with
  // the result type.
  Type broadcasted_type = merge.output()->getType();
  for (Type operand_type : merge.getOperandTypes()) {
    if (operand_type.isa<ControlType>()) break;

    // TODO(hinsu): Update ControlOperandsAfterAllData trait to verify this
    // constraint.
    if (!operand_type.isa<TensorType>())
      return merge.emitOpError("expects data operands to have tensor type");

    // Variant types may have opaque subtypes information that need not match
    // between the two types so drop them before computing the broadcasted type.
    Type new_broadcasted_type =
        OpTrait::util::getBroadcastedType(DropVariantSubTypes(broadcasted_type),
                                          DropVariantSubTypes(operand_type));
    if (!new_broadcasted_type)
      return merge.emitOpError()
             << "expects all operands to be broadcastable"
             << " but got " << broadcasted_type << " vs " << operand_type;
    // Use the broadcasted type unless we're losing the rank information here.
    // This is because for example starting with a result of tensor<4xf32>, if
    // the first operand is unranked, the broadcasted type will be unranked.
    // Then any tensor operand will be broadcastable to this unranked type.
    if (!broadcasted_type.cast<TensorType>().hasRank() ||
        new_broadcasted_type.cast<TensorType>().hasRank())
      broadcasted_type = new_broadcasted_type;
  }

  return success();
}

void Print(MergeOp merge, OpAsmPrinter *p) {
  // Use short form only when there are exactly two data operands and their
  // type matches the output type. Otherwise, use the generic printer.
  bool use_short_form = true;
  int num_data_operands = 0;

  Type output_type = merge.output()->getType();
  for (Type operand_type : merge.getOperandTypes()) {
    if (operand_type.isa<ControlType>()) break;
    num_data_operands++;

    if (operand_type != output_type) {
      use_short_form = false;
      break;
    }
  }

  *p << merge.getOperationName() << ' ';
  p->printOperands(merge.getOperands());

  // Print the type signature of the operation.
  *p << " : ";
  if (!use_short_form || num_data_operands != 2) {
    p->printFunctionalType(merge.getOperation());
  } else {
    *p << output_type;
  }

  p->printOptionalAttrDict(merge.getAttrs());
}

ParseResult ParseMergeOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> op_infos;
  SmallVector<Type, 1> types;
  llvm::SMLoc loc = parser->getCurrentLocation();
  if (parser->parseOperandList(op_infos) || parser->parseColonTypeList(types))
    return failure();
  if (types.size() != 1)
    return parser->emitError(parser->getNameLoc())
           << " expects only a single data type";

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // inputs and the output are all using this type).
  if (FunctionType type = types.front().dyn_cast<FunctionType>()) {
    result->types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    // In case of the short form, use the parsed type for both the operands and
    // the remaining operands are expected to be control inputs.
    types.push_back(types.front());
    Type control_type = ControlType::get(parser->getBuilder().getContext());
    types.append(op_infos.size() - 2, control_type);

    RankedTensorType i32_tensor =
        RankedTensorType::get({}, parser->getBuilder().getIntegerType(32));
    result->types = {types.front(), i32_tensor, control_type};
  }

  if (parser->resolveOperands(op_infos, types, loc, result->operands))
    return failure();

  return parser->parseOptionalAttributeDict(result->attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.Enter
//===----------------------------------------------------------------------===//

// Default number for the parallel_iterations attributes on Enter nodes.
constexpr int kDefaultParallelIterations = 10;

void Print(EnterOp enter, OpAsmPrinter *p) {
  *p << enter.getOperationName() << ' ';
  p->printOperands(enter.getOperands());

  *p << " frame \"";
  printEscapedString(enter.frame_name(), p->getStream());
  *p << "\"";
  if (enter.parallel_iterations() != kDefaultParallelIterations)
    *p << " parallel_iterations " << enter.parallel_iterations();
  if (enter.is_constant()) *p << " constant ";

  // If the types aren't perfectly matching, print the functional type syntax
  // else print the shorter single type.
  *p << " : ";
  if (enter.data()->getType() != enter.output()->getType()) {
    p->printFunctionalType(enter.getOperation());
  } else {
    *p << enter.getType(0);
  }

  p->printOptionalAttrDict(
      enter.getAttrs(), {"frame_name", "parallel_iterations", "is_constant"});
}

ParseResult ParseEnterOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> op_infos;
  llvm::SMLoc loc = parser->getCurrentLocation();
  MLIRContext *context = parser->getBuilder().getContext();
  if (parser->parseOperandList(op_infos)) return failure();
  if (op_infos.empty())
    return parser->emitError(loc) << " expects at least one data operand";

  Attribute frame;
  if (parser->parseKeyword("frame") ||
      parser->parseAttribute(frame, NoneType::get(context), "frame_name",
                             result->attributes))
    return failure();

  Type i64 = parser->getBuilder().getIntegerType(64);
  if (parser->parseOptionalKeyword("parallel_iterations")) {
    result->addAttribute("parallel_iterations",
                         IntegerAttr::get(i64, kDefaultParallelIterations));
  } else {
    IntegerAttr parallel_iterations;
    if (parser->parseAttribute(parallel_iterations, i64, "parallel_iterations",
                               result->attributes))
      return failure();
  }
  bool has_constant = succeeded(parser->parseOptionalKeyword("constant"));
  result->addAttribute("is_constant", BoolAttr::get(has_constant, context));

  SmallVector<Type, 1> types;
  if (parser->parseColonTypeList(types)) return failure();
  if (types.size() != 1)
    return parser->emitError(loc) << " expects only a single data type";

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // input and the outputs are all using this type).
  if (FunctionType type = types.front().dyn_cast<FunctionType>()) {
    if (type.getNumInputs() != 1)
      return parser->emitError(parser->getNameLoc())
             << " expects a single data type";
    result->types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    Type control_type = ControlType::get(context);
    types.append(op_infos.size() - 1, control_type);
    result->addTypes({types.front(), control_type});
  }

  // Extra operands are expected to be control inputs.

  if (parser->resolveOperands(op_infos, types, loc, result->operands))
    return failure();

  return parser->parseOptionalAttributeDict(result->attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.NextIteration.Source
//===----------------------------------------------------------------------===//

LogicalResult Verify(NextIterationSourceOp source) {
  Value *token = source.token();
  if (!token->hasOneUse())
    return source.emitOpError() << "expects a single user for produced token";
  if (!isa<NextIterationSinkOp>(*token->user_begin()))
    return source.emitOpError() << "token should be consumed by a sink op";
  return success();
}

void Print(NextIterationSourceOp next_iteration, OpAsmPrinter *p) {
  *p << next_iteration.getOperationName() << " : " << next_iteration.getType(0);
  p->printOptionalAttrDict(next_iteration.getAttrs());
}

ParseResult ParseNextIterationSourceOp(OpAsmParser *parser,
                                       OperationState *result) {
  SmallVector<Type, 1> types;
  if (parser->parseColonTypeList(types)) return failure();

  MLIRContext *context = parser->getBuilder().getContext();
  Type token_type = TokenType::get(context);
  Type control_type = ControlType::get(context);
  result->addTypes({types.front(), token_type, control_type});
  return parser->parseOptionalAttributeDict(result->attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.NextIteration.Sink
//===----------------------------------------------------------------------===//

LogicalResult Verify(NextIterationSinkOp sink) {
  Value *token = sink.token();
  Operation *definingOp = token->getDefiningOp();
  if (!definingOp)
    return sink.emitOpError() << "expects a token directly produced by a "
                                 "tf_executor.NextIteration.Source op: ";
  auto source = dyn_cast<NextIterationSourceOp>(definingOp);
  if (!source)
    return sink.emitOpError() << "expects a token produced by a "
                                 "tf_executor.NextIteration.Source op: ";
  if (source.output()->getType() != sink.input()->getType())
    return sink.emitOpError()
           << "input type " << sink.input()->getType()
           << " mismatch the tf_executor.NextIteration.Source output type: "
           << source.output()->getType();
  return success();
}

void Print(NextIterationSinkOp next_iteration, OpAsmPrinter *p) {
  *p << next_iteration.getOperationName() << " [";
  p->printOperand(next_iteration.getOperand(0));
  *p << "] ";
  p->printOperands(llvm::drop_begin(next_iteration.getOperands(), 1));
  *p << " : " << next_iteration.getOperand(1)->getType();
  p->printOptionalAttrDict(next_iteration.getAttrs());
}

ParseResult ParseNextIterationSinkOp(OpAsmParser *parser,
                                     OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> op_infos;
  llvm::SMLoc loc = parser->getCurrentLocation();

  // First type is always the token consumed from the NextIteration.source
  Type token_type = TokenType::get(parser->getBuilder().getContext());
  SmallVector<Type, 1> types = {token_type};

  if (parser->parseOperandList(op_infos, 1, OpAsmParser::Delimiter::Square) ||
      parser->parseOperandList(op_infos) || parser->parseColonTypeList(types))
    return failure();

  Type control_type = ControlType::get(parser->getBuilder().getContext());
  types.append(op_infos.size() - 2, control_type);
  if (parser->resolveOperands(op_infos, types, loc, result->operands))
    return failure();

  return parser->parseOptionalAttributeDict(result->attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.Exit
//===----------------------------------------------------------------------===//

void Print(ExitOp exit, OpAsmPrinter *p) {
  *p << exit.getOperationName() << ' ';
  p->printOperands(exit.getOperands());
  *p << " : " << exit.getType(0);
  p->printOptionalAttrDict(exit.getAttrs());
}

ParseResult ParseExitOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> op_infos;
  SmallVector<Type, 1> types;

  if (parser->parseOperandList(op_infos) || parser->parseColonTypeList(types))
    return failure();

  llvm::SMLoc loc = parser->getCurrentLocation();
  Type control_type = ControlType::get(parser->getBuilder().getContext());
  types.append(op_infos.size() - 1, control_type);
  if (parser->resolveOperands(op_infos, types, loc, result->operands))
    return failure();

  result->addTypes({types.front(), control_type});
  return parser->parseOptionalAttributeDict(result->attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.ControlTrigger
//===----------------------------------------------------------------------===//

void Print(ControlTriggerOp trigger, OpAsmPrinter *p) {
  *p << trigger.getOperationName() << ' ';
  p->printOperands(trigger.getOperands());
  p->printOptionalAttrDict(trigger.getAttrs());
}

ParseResult ParseControlTriggerOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> op_infos;
  SmallVector<Type, 1> types;
  llvm::SMLoc loc = parser->getCurrentLocation();

  if (parser->parseOperandList(op_infos)) return failure();
  Type control_type = ControlType::get(parser->getBuilder().getContext());
  types.append(op_infos.size(), control_type);
  if (parser->resolveOperands(op_infos, types, loc, result->operands))
    return failure();

  // Single control as the only output
  result->types.push_back(control_type);
  return parser->parseOptionalAttributeDict(result->attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.LoopCond
//===----------------------------------------------------------------------===//

void Print(LoopCondOp loop_cond, OpAsmPrinter *p) {
  *p << loop_cond.getOperationName() << ' ';
  p->printOperands(loop_cond.getOperands());

  // If the types aren't matching (broadcast), print the functional type syntax.
  if (loop_cond.input()->getType() != loop_cond.output()->getType()) {
    *p << " : ";
    p->printFunctionalType(loop_cond.getOperation());
  } else {
    *p << " : " << loop_cond.input()->getType();
  }

  p->printOptionalAttrDict(loop_cond.getAttrs());
}

ParseResult ParseLoopCondOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> op_infos;

  if (parser->parseOperandList(op_infos)) return failure();
  if (op_infos.empty())
    return parser->emitError(parser->getNameLoc())
           << "expects at least one operand";

  SmallVector<Type, 1> types;
  if (parser->parseColonTypeList(types)) return failure();

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // input and the outputs are all using this type).
  Type control_type = ControlType::get(parser->getBuilder().getContext());
  if (FunctionType type = types.front().dyn_cast<FunctionType>()) {
    if (llvm::count_if(type.getInputs(),
                       [=](Type type) { return type != control_type; }) != 1)
      return parser->emitError(parser->getNameLoc())
             << " expects a single data type";
    result->types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    if (types.size() != 1)
      return parser->emitError(parser->getNameLoc())
             << " expects a single data type";
    types.append(op_infos.size() - 1, control_type);
    result->addTypes({types.front(), control_type});
  }

  llvm::SMLoc loc = parser->getCurrentLocation();
  if (parser->resolveOperands(op_infos, types, loc, result->operands))
    return failure();

  return parser->parseOptionalAttributeDict(result->attributes);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

// TODO(lyandy): Add canonicalization for dedupping control inputs.

//===----------------------------------------------------------------------===//
// tf_executor.graph
//===----------------------------------------------------------------------===//

// TODO(lyandy): Add canonicalization for empty graphs.

namespace {
// This pattern matches GraphOps with only one island, pulls out all inner ops
// of the island to the block containing the GraphOp, and then removes the
// GraphOp.
struct HoistInnerOpsSingleIslandGraph : public OpRewritePattern<GraphOp> {
  using OpRewritePattern<GraphOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(GraphOp op,
                                     PatternRewriter &rewriter) const override {
    if (!HasSingleIslandInGraph(&op)) return matchFailure();
    auto fetch_op = llvm::cast<FetchOp>(op.GetBody().back());
    auto island_op = llvm::cast<IslandOp>(op.GetBody().front());
    auto yield_op = llvm::cast<YieldOp>(island_op.GetBody().back());

    // Mapping from island outputs to inner ops outputs.
    llvm::SmallDenseMap<Value *, Value *, 8> island_rets_to_ops;
    for (auto ops_and_ret_vals :
         llvm::zip(island_op.outputs(), yield_op.fetches())) {
      island_rets_to_ops.insert(
          {std::get<0>(ops_and_ret_vals), std::get<1>(ops_and_ret_vals)});
    }

    // Map graph outputs to inner ops outputs.
    llvm::SmallVector<Value *, 8> new_rets;
    for (auto *fetch : fetch_op.fetches()) {
      if (auto *op = island_rets_to_ops.lookup(fetch))
        new_rets.push_back(op);
      else
        new_rets.push_back(fetch);
    }

    // Move inner ops from island to block containing graph.
    auto &island_body = island_op.GetBody().getOperations();
    Operation *operation = op.getOperation();
    operation->getBlock()->getOperations().splice(
        operation->getIterator(), island_body, island_body.begin(),
        std::prev(island_body.end()));
    rewriter.replaceOp(op, new_rets);

    return matchSuccess();
  }

 private:
  // Finds in a block if the op of type `InnerOpT` is the first operation
  // and optionally followed by a terminator.
  template <typename InnerOpT>
  bool HasSingleOpInBlock(Block *block) const {
    if (block->empty()) return false;
    if (!llvm::isa<InnerOpT>(block->front())) return false;
    // Either InnerOpT is the only instruction in the block, or there is a
    // possible terminator.
    return std::next(block->begin()) == block->end() ||
           std::next(block->begin(), 2) == block->end();
  }

  // Checks if graph only has one island.
  bool HasSingleIslandInGraph(GraphOp *graph_op) const {
    return HasSingleOpInBlock<IslandOp>(&graph_op->GetBody());
  }
};
}  // anonymous namespace

void GraphOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<HoistInnerOpsSingleIslandGraph>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc.inc"

}  // namespace tf_executor
}  // namespace mlir
