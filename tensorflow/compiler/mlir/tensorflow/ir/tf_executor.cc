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
#include <iterator>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// TF Executor Dialect
//===----------------------------------------------------------------------===//

namespace {

struct TensorFlowExecutorInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // Override the inlining hook to determine if 'src' can be inlined into
  // 'dest'.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &value_mapping) const final {
    // Allow inlining into tf.island regions if the incoming region has a single
    // block.
    return llvm::isa<tf_executor::IslandOp>(dest->getParentOp()) &&
           llvm::hasSingleElement(*src);
  }
};

struct TensorFlowExecutorDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  // Registered hook to check if the given region, which is attached to an
  // operation that is *not* isolated from above (i.e. no internal regions
  // reference values defined in an enclosing region), should be used when
  // materializing constants.
  // In the executor dialect we materialize inside an island.
  bool shouldMaterializeInto(Region *region) const final {
    return isa<tf_executor::IslandOp>(region->getParentOp());
  }
};

}  // namespace

TensorFlowExecutorDialect::TensorFlowExecutorDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf_executor", context,
              TypeID::get<TensorFlowExecutorDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc.inc"
      >();

  addInterfaces<TensorFlowExecutorInlinerInterface,
                TensorFlowExecutorDialectFoldInterface>();

  addTypes<ControlType, TokenType>();
}

Type TensorFlowExecutorDialect::parseType(DialectAsmParser &parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();

  if (data_type == "control") return ControlType::get(getContext());
  if (data_type == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc())
      << "unknown tf_executor type: " << data_type;
  return nullptr;
}

void TensorFlowExecutorDialect::printType(Type type,
                                          DialectAsmPrinter &os) const {
  if (mlir::isa<ControlType>(type)) {
    os << "control";
    return;
  }
  if (mlir::isa<TokenType>(type)) {
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
    if (mlir::isa<ControlType>(op->getOperand(operand_idx).getType())) {
      found_control = true;
      continue;
    }
    if (found_control)
      return op->emitOpError() << "found non-control operand #" << operand_idx
                               << " after control operand";
  }
  return success();
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// tf_executor.graph
//===----------------------------------------------------------------------===//

FetchOp GraphOp::GetFetch() { return llvm::cast<FetchOp>(GetBody().back()); }

LogicalResult GraphOp::verify() {
  GraphOp graph = *this;
  auto *executorDialect = graph->getDialect();

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
    Value operand = fetch.getOperand(i);
    // Break out of the loop at the first control operand encountered.
    const int64_t num_results = graph.getNumResults();
    if (mlir::isa<ControlType>(operand.getType())) {
      if (i != num_results)
        return fetch.emitOpError()
               << "operand #" << i
               << " is a control type, can't be bound to a graph result";
      break;
    }
    if (i >= num_results)
      return fetch.emitOpError()
             << "operand #" << i << " does not have a graph results to bind";
    if (graph.getResult(i).getType() != operand.getType()) {
      return fetch.emitOpError()
             << "operand #" << i << " type mismatch graph results ("
             << graph.getResult(i).getType() << " != " << operand.getType()
             << ")";
    }
  }
  return success();
}

void GraphOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printRegion(getOperation()->getRegion(0));
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc loc = parser.getCurrentLocation();

  // Parse the body region.
  Region &body = *result.addRegion();
  if (parser.parseRegion(body)) return failure();

  // Ensure that the region is well formed: it contains at least a block with
  // a FetchOp terminator.
  GraphOp::ensureTerminator(body, parser.getBuilder(), result.location);

  if (!llvm::hasSingleElement(body))
    return parser.emitError(loc) << "expects a single block region";

  // Get the results type from the terminator type inside the graph.
  Operation &fetch = body.back().back();
  if (!isa<FetchOp>(fetch))
    return parser.emitError(loc) << "expects a tf_executor.fetch terminator";

  // The return value of the graph operation are the non-control operands of
  // the fetch operation.
  result.types.reserve(fetch.getNumOperands());
  for (Type type : fetch.getOperandTypes()) {
    if (mlir::isa<ControlType>(type)) break;
    result.types.push_back(type);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// tf_executor.fetch
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_executor.island
//===----------------------------------------------------------------------===//

YieldOp IslandOp::GetYield() { return llvm::cast<YieldOp>(GetBody().back()); }

// Checks if a tf_executor.island wraps a single operation and the single
// operation results are perfectly forwarded to the islands yield.
bool IslandOp::WrapsSingleOp() {
  auto body = GetBody().without_terminator();
  if (!hasSingleElement(body)) return false;

  Operation &wrapped_op = *body.begin();
  YieldOp yield = GetYield();
  return wrapped_op.getNumResults() == yield.getNumOperands() &&
         std::equal(wrapped_op.getResults().begin(),
                    wrapped_op.getResults().end(), yield.getOperands().begin());
}

mlir::LogicalResult IslandOp::verify() {
  IslandOp island = *this;
  if (!island.GetBody().args_empty())
    return island.emitOpError() << "expects body without any arguments";

  Operation &yield = island.GetBody().back();
  if (!isa<YieldOp>(yield))
    return yield.emitOpError()
           << "invalid tf_executor.island terminator, yield expected";

  // Ensure that the yield terminator operands matches the island results type.
  int result_count = island.getNumResults() - 1;  // -1 for the control token
  const int num_operands = yield.getNumOperands();
  if (num_operands != result_count)
    return yield.emitOpError()
           << "has " << yield.getNumOperands()
           << " operand, but island returns " << result_count;
  for (int operand_idx : llvm::seq<int>(0, yield.getNumOperands())) {
    if (island.getResult(operand_idx).getType() !=
        yield.getOperand(operand_idx).getType())
      return yield.emitOpError()
             << "operand #" << operand_idx << " type mismatch island results";
  }

  // Check that there aren't any control results other than the last one.
  Type control_type = ControlType::get(island.getContext());
  for (int operand_idx : llvm::seq<int>(0, island.getNumResults() - 1)) {
    if (island.getResult(operand_idx).getType() == control_type)
      return yield.emitOpError()
             << "unexpected control type for operand #" << operand_idx;
  }
  return success();
}

void IslandOp::print(OpAsmPrinter &p) {
  if (getNumOperands()) {
    // These are always control operand, no explicit type needed.
    p << '(';
    p.printOperands(getOperands());
    p << ')';
  }

  // Check if we can print the short "wraps" form: that is if the island
  // contains a single operation and the result of this operation are perfectly
  // forwarded to the yield.
  if (getOperation()->getAttrs().empty() && WrapsSingleOp()) {
    Operation &wrapped_op = GetBody().front();
    YieldOp yield_op = GetYield();
    // The "wraps" syntax only encodes a single location.
    // In order to correctly round-trip, we can only use this syntax when all
    // the locations are identical.
    if (wrapped_op.getLoc() == getLoc() && yield_op.getLoc() == getLoc()) {
      p << " wraps ";
      p.printGenericOp(&wrapped_op);
      return;
    }
  }
  p << ' ';
  p.printRegion(getOperation()->getRegion(0));
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult IslandOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Type control_type = ControlType::get(parser.getBuilder().getContext());

  // Parse optional argument list (control dependencies only).
  SmallVector<OpAsmParser::UnresolvedOperand, 4> op_infos;
  if (parser.parseOperandList(op_infos, OpAsmParser::Delimiter::OptionalParen))
    return failure();
  if (!op_infos.empty()) {
    SmallVector<Type, 2> types(op_infos.size(), control_type);
    if (parser.resolveOperands(op_infos, types, loc, result.operands))
      return failure();
  }

  // Parse the body region.
  Region &body = *result.addRegion();

  if (succeeded(parser.parseOptionalKeyword("wraps"))) {
    // If we parse the short version of the island, we have an operation in the
    // generic form that follows the "wraps" keyword. Parse it inside the region
    // and forward all of its results as-is to the yield operation.
    body.push_back(new Block);
    Block &block = body.back();
    Operation *wrapped_op = parser.parseGenericOperation(&block, block.begin());
    if (!wrapped_op) return failure();
    OpBuilder builder(parser.getBuilder().getContext());
    builder.setInsertionPointToEnd(&block);
    YieldOp::create(builder, wrapped_op->getLoc(), wrapped_op->getResults());
    result.location = wrapped_op->getLoc();
  } else if (parser.parseRegion(body)) {
    return failure();
  }

  IslandOp::ensureTerminator(body, parser.getBuilder(), result.location);

  // Get the results type for the island from the terminator operands.
  Operation &yield = body.back().back();
  result.types.reserve(yield.getNumOperands() + 1);
  result.types.append(yield.operand_type_begin(), yield.operand_type_end());
  result.types.push_back(control_type);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// tf_executor.yield
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_executor.Switch
//===----------------------------------------------------------------------===//

ParseResult SwitchOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  SmallVector<Type, 1> types;
  if (parser.parseOperandList(op_infos) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 1)
    return parser.emitError(parser.getNameLoc())
           << " expects only a single data type";

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // input and the outputs are all using this type and predicate is tensor<i1>
  // type).
  if (mlir::isa<FunctionType>(types.front())) {
    FunctionType type = mlir::cast<FunctionType>(types.front());
    if (type.getNumInputs() < 2)
      return parser.emitError(parser.getNameLoc())
             << " expects a single data type and a predicate";
    result.types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    if (op_infos.size() < 2)
      return parser.emitError(parser.getNameLoc())
             << " expects a single data type and a predicate";
    Type control_type = ControlType::get(parser.getBuilder().getContext());
    result.types.append(2, types[0]);
    result.types.push_back(control_type);
    Type i1_type = parser.getBuilder().getI1Type();
    RankedTensorType predicate_type = RankedTensorType::get({}, i1_type);
    types.push_back(predicate_type);
    types.append(op_infos.size() - 2, control_type);
  }

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

void SwitchOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getOperands());
  Type data_operand_ty = getData().getType();
  // If the types aren't perfectly matching, print the functional type syntax
  // else print the shorter single type.
  p << " : ";
  if (getTrueOutput().getType() != data_operand_ty ||
      getFalseOutput().getType() != data_operand_ty ||
      mlir::isa<UnrankedTensorType>(getPredicate().getType())) {
    p.printFunctionalType(getOperation());
  } else {
    p << getType(0);
  }
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

//===----------------------------------------------------------------------===//
// tf_executor.SwitchN
//===----------------------------------------------------------------------===//

LogicalResult SwitchNOp::verify() {
  SwitchNOp switchn = *this;
  IntegerAttr num_outs = switchn->getAttrOfType<IntegerAttr>("num_outs");
  if (!num_outs)
    return switchn.emitOpError() << "expects a `num_outs` integer attribute";

  // Expect num_outs results + 1 control output.
  if (switchn.getNumResults() != num_outs.getInt() + 1)
    return switchn.emitOpError()
           << "expect `num_outs` (" << num_outs.getInt() << ") results but got "
           << (switchn.getNumResults() - 1);

  // Check that operand can be broadcasted to each output type.
  auto operand0_type = switchn.getOperand(0).getType();
  TensorType operand0_tensor_type = mlir::dyn_cast<TensorType>(operand0_type);
  if (!operand0_tensor_type) {
    return switchn.emitOpError()
           << "expects data operand to have tensor type but got "
           << operand0_type;
  }
  for (Type output_type : switchn.getResultTypes()) {
    if (mlir::isa<ControlType>(output_type)) break;

    TensorType output_tensor_type = mlir::dyn_cast<TensorType>(output_type);
    if (!output_tensor_type) {
      return switchn.emitOpError()
             << "expects outputs to have tensor type but got " << output_type;
    }

    // If the output type is a ref type, then the operand type should also be of
    // the same ref type. However, if the output type is a non-ref type T, then
    // the operand can be tensor of type T or T_REF.
    bool is_output_ref = mlir::isa<tf_type::TensorFlowRefType>(
        output_tensor_type.getElementType());
    if (is_output_ref && !mlir::isa<tf_type::TensorFlowRefType>(
                             operand0_tensor_type.getElementType())) {
      return switchn.emitOpError()
             << "expects same operand and output element type but got "
             << operand0_tensor_type << " vs " << output_tensor_type;
    }
    Type broadcasted_type = OpTrait::util::getBroadcastedType(
        tf_type::DropRefAndSubTypes(operand0_tensor_type),
        tf_type::DropRefAndSubTypes(output_tensor_type));
    if (!broadcasted_type) {
      return switchn.emitOpError()
             << "expects data operand to be broadcastable with all output types"
             << " but got " << operand0_tensor_type << " vs "
             << output_tensor_type;
    }
  }
  return success();
}

void SwitchNOp::print(OpAsmPrinter &p) {
  p << ' ';
  auto operands = getOperands();
  // Print the 2 data operands.
  p.printOperands(operands.begin(), std::next(operands.begin(), 2));
  p << " of " << (getNumResults() - 1);
  // print control dependencies if any
  if (!getControlInputs().empty()) {
    p << " (";
    p.printOperands(getControlInputs());
    p << ")";
  }
  p << " : " << getType(0);
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"num_outs"});
}

ParseResult SwitchNOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parsing:
  //       %2:6 = tf_executor.SwitchN %0, %1 of 5 : tensor<??xf32>
  // Where the first operand is the data to replicate, the second is an i32
  // indicating which output to populate, followed by the keyword `of` and the
  // number of outputs (+1 for the control token).
  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  SmallVector<Type, 1> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  IntegerAttr num_outs;
  Type i64_type = parser.getBuilder().getIntegerType(64);
  if (parser.parseOperandList(op_infos, 2) || parser.parseKeyword("of") ||
      parser.parseAttribute(num_outs, i64_type, "num_outs",
                            result.attributes) ||
      parser.parseOperandList(op_infos,
                              OpAsmParser::Delimiter::OptionalParen) ||
      parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 1)
    return parser.emitError(parser.getNameLoc())
           << " expects only a single data type";

  if (num_outs.getInt() <= 0)
    return parser.emitError(parser.getNameLoc())
           << " expects a positive number of outputs";

  // `types` already contains the type for the data, add an i32 for the
  // output_index, and then the optional control inputs.
  auto builder = parser.getBuilder();
  types.push_back(RankedTensorType::get({}, builder.getIntegerType(32)));
  Type control_type = ControlType::get(builder.getContext());
  types.append(op_infos.size() - 2, control_type);

  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  // Output result types is a replication `num_outs` times the data input type.
  result.types.append(num_outs.getInt(), types[0]);
  result.types.push_back(control_type);

  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.Merge
//===----------------------------------------------------------------------===//

LogicalResult MergeOp::verify() {
  MergeOp merge = *this;
  if (!merge.getNumOperands())
    return merge.emitOpError() << "expects at least one operand";

  Type data_type = merge.getOperand(0).getType();
  if (mlir::isa<ControlType>(data_type))
    return merge.emitOpError() << "expects a non-control input";

  // Check that each operand can be individually broadcasted to the output type.
  Type output_type = merge.getOutput().getType();
  TensorType output_tensor_ty = mlir::dyn_cast<TensorType>(output_type);
  if (!output_tensor_ty) {
    return merge.emitOpError()
           << "expects output to have tensor type but got " << output_type;
  }
  bool is_output_ref =
      mlir::isa<tf_type::TensorFlowRefType>(output_tensor_ty.getElementType());
  for (Type operand_type : merge.getOperandTypes()) {
    if (mlir::isa<ControlType>(operand_type)) break;

    // TODO(hinsu): Update ControlOperandsAfterAllData trait to verify this
    // constraint.
    TensorType operand_tensor_ty = mlir::dyn_cast<TensorType>(operand_type);
    if (!operand_tensor_ty)
      return merge.emitOpError()
             << "expects data operands to have tensor type but got "
             << operand_type;

    // If output type is a ref type then all operand types should also be of the
    // same ref type. However, if the output type is a non-ref type T, operands
    // can be tensor of type T or T_REF.
    if (is_output_ref && !mlir::isa<tf_type::TensorFlowRefType>(
                             operand_tensor_ty.getElementType())) {
      return merge.emitOpError()
             << "expects same operand and output element type but got "
             << operand_tensor_ty << " vs " << output_tensor_ty;
    }
    Type broadcasted_type = OpTrait::util::getBroadcastedType(
        tf_type::DropRefAndSubTypes(output_tensor_ty),
        tf_type::DropRefAndSubTypes(operand_tensor_ty));
    if (!broadcasted_type)
      return merge.emitOpError()
             << "expects all operands to be broadcastable with output type"
             << " but got " << operand_tensor_ty << " vs " << output_tensor_ty;
  }
  return success();
}

void MergeOp::print(OpAsmPrinter &p) {
  // Use short form only when there are exactly two data operands and their
  // type matches the output type. Otherwise, use the generic printer.
  bool use_short_form = true;
  int num_data_operands = 0;

  Type output_type = getOutput().getType();
  for (Type operand_type : getOperandTypes()) {
    if (mlir::isa<ControlType>(operand_type)) break;
    num_data_operands++;

    if (operand_type != output_type) {
      use_short_form = false;
      break;
    }
  }

  p << ' ';
  p.printOperands(getOperands());

  // Print the type signature of the operation.
  p << " : ";
  if (!use_short_form || num_data_operands != 2) {
    p.printFunctionalType(getOperation());
  } else {
    p << output_type;
  }

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult MergeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  SmallVector<Type, 1> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOperandList(op_infos) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 1)
    return parser.emitError(parser.getNameLoc())
           << " expects only a single data type";

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // inputs and the output are all using this type).
  if (FunctionType type = mlir::dyn_cast<FunctionType>(types.front())) {
    result.types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    // In case of the short form, use the parsed type for both the operands and
    // the remaining operands are expected to be control inputs.
    types.push_back(Type(types.front()));
    Type control_type = ControlType::get(parser.getBuilder().getContext());
    types.append(op_infos.size() - 2, control_type);

    RankedTensorType i32_tensor =
        RankedTensorType::get({}, parser.getBuilder().getIntegerType(32));
    result.types = {types.front(), i32_tensor, control_type};
  }

  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.Enter
//===----------------------------------------------------------------------===//

// Default number for the parallel_iterations attributes on Enter nodes.
static constexpr int kDefaultParallelIterations = 10;

void EnterOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getOperands());

  p << " frame \"";
  printEscapedString(getFrameName(), p.getStream());
  p << "\"";
  if (getParallelIterations() != kDefaultParallelIterations)
    p << " parallel_iterations " << getParallelIterations();
  if (getIsConstant()) p << " constant ";

  // If the types aren't perfectly matching, print the functional type syntax
  // else print the shorter single type.
  p << " : ";
  if (getData().getType() != getOutput().getType()) {
    p.printFunctionalType(getOperation());
  } else {
    p << getType(0);
  }

  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {"frame_name", "parallel_iterations", "is_constant"});
}

ParseResult EnterOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  llvm::SMLoc loc = parser.getCurrentLocation();
  MLIRContext *context = parser.getBuilder().getContext();
  if (parser.parseOperandList(op_infos)) return failure();
  if (op_infos.empty())
    return parser.emitError(loc) << " expects at least one data operand";

  Attribute frame;
  if (parser.parseKeyword("frame") ||
      parser.parseAttribute(frame, NoneType::get(context), "frame_name",
                            result.attributes))
    return failure();

  Type i64 = parser.getBuilder().getIntegerType(64);
  if (parser.parseOptionalKeyword("parallel_iterations")) {
    result.addAttribute("parallel_iterations",
                        IntegerAttr::get(i64, kDefaultParallelIterations));
  } else {
    IntegerAttr parallel_iterations;
    if (parser.parseAttribute(parallel_iterations, i64, "parallel_iterations",
                              result.attributes))
      return failure();
  }
  bool has_constant = succeeded(parser.parseOptionalKeyword("constant"));
  result.addAttribute("is_constant", BoolAttr::get(context, has_constant));

  SmallVector<Type, 1> types;
  if (parser.parseColonTypeList(types)) return failure();
  if (types.size() != 1)
    return parser.emitError(loc) << " expects only a single data type";

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // input and the outputs are all using this type).
  if (FunctionType type = mlir::dyn_cast<FunctionType>(types.front())) {
    // One data input, and any number of control inputs.
    if (type.getNumInputs() >= 1) {
      result.types.assign(type.getResults().begin(), type.getResults().end());
      types.assign(type.getInputs().begin(), type.getInputs().end());
    } else {
      return parser.emitError(parser.getNameLoc()) << " expects a data input";
    }
  } else {
    Type control_type = ControlType::get(context);
    types.append(op_infos.size() - 1, control_type);
    result.addTypes({types.front(), control_type});
  }

  // Extra operands are expected to be control inputs.

  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.NextIteration.Source
//===----------------------------------------------------------------------===//

LogicalResult NextIterationSourceOp::verify() {
  NextIterationSourceOp source = *this;
  Value token = source.getToken();
  if (!token.hasOneUse())
    return source.emitOpError() << "expects a single user for produced token";
  if (!isa<NextIterationSinkOp>(*token.user_begin()))
    return source.emitOpError() << "token should be consumed by a sink op";
  return success();
}

//===----------------------------------------------------------------------===//
// tf_executor.NextIteration.Sink
//===----------------------------------------------------------------------===//

LogicalResult NextIterationSinkOp::verify() {
  NextIterationSinkOp sink = *this;
  Value token = sink.getToken();
  Operation *definingOp = token.getDefiningOp();
  if (!definingOp)
    return sink.emitOpError() << "expects a token directly produced by a "
                                 "tf_executor.NextIteration.Source op: ";
  auto source = dyn_cast<NextIterationSourceOp>(definingOp);
  if (!source)
    return sink.emitOpError() << "expects a token produced by a "
                                 "tf_executor.NextIteration.Source op: ";
  if (source.getOutput().getType() != sink.getInput().getType())
    return sink.emitOpError()
           << "input type " << sink.getInput().getType()
           << " mismatch the tf_executor.NextIteration.Source output type: "
           << source.getOutput().getType();
  return success();
}

NextIterationSourceOp NextIterationSinkOp::GetSource() {
  return cast<NextIterationSourceOp>(getToken().getDefiningOp());
}

//===----------------------------------------------------------------------===//
// tf_executor.Exit
//===----------------------------------------------------------------------===//

void ExitOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getOperands());
  p << " : " << getType(0);
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult ExitOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  SmallVector<Type, 1> types;

  if (parser.parseOperandList(op_infos) || parser.parseColonTypeList(types))
    return failure();

  llvm::SMLoc loc = parser.getCurrentLocation();
  Type control_type = ControlType::get(parser.getBuilder().getContext());
  types.append(op_infos.size() - 1, control_type);
  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  result.addTypes({types.front(), control_type});
  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.ControlTrigger
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_executor.LoopCond
//===----------------------------------------------------------------------===//

void LoopCondOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getOperands());

  // If the types aren't matching (broadcast), print the functional type syntax.
  if (getInput().getType() != getOutput().getType()) {
    p << " : ";
    p.printFunctionalType(getOperation());
  } else {
    p << " : " << getInput().getType();
  }

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult LoopCondOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;

  if (parser.parseOperandList(op_infos)) return failure();
  if (op_infos.empty())
    return parser.emitError(parser.getNameLoc())
           << "expects at least one operand";

  SmallVector<Type, 1> types;
  if (parser.parseColonTypeList(types)) return failure();

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // input and the outputs are all using this type).
  Type control_type = ControlType::get(parser.getBuilder().getContext());
  if (FunctionType type = mlir::dyn_cast<FunctionType>(types.front())) {
    if (llvm::count_if(type.getInputs(),
                       [=](Type type) { return type != control_type; }) != 1)
      return parser.emitError(parser.getNameLoc())
             << " expects a single data type";
    result.types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    if (types.size() != 1)
      return parser.emitError(parser.getNameLoc())
             << " expects a single data type";
    types.append(op_infos.size() - 1, control_type);
    result.addTypes({types.front(), control_type});
  }

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

// TODO(lyandy): Add canonicalization for dedupping control inputs.

//===----------------------------------------------------------------------===//
// tf_executor.graph
//===----------------------------------------------------------------------===//

namespace {
// Finds in a block if the op of type `InnerOpT` is the first operation and
// optionally followed by a terminator.
template <typename InnerOpT>
bool HasSingleOpInBlock(Block *block) {
  if (block->empty()) return false;
  if (!llvm::isa<InnerOpT>(block->front())) return false;
  // Either InnerOpT is the only instruction in the block, or there is a
  // possible terminator.
  return std::next(block->begin()) == block->end() ||
         std::next(block->begin(), 2) == block->end();
}

// This pattern matches GraphOps with only one FetchOp (empty) and remaps the
// results of the GraphOp to the operands of the FetchOp.
struct DropEmptyGraph : public OpRewritePattern<GraphOp> {
  using OpRewritePattern<GraphOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GraphOp op,
                                PatternRewriter &rewriter) const override {
    Block &block = op.GetBody();
    // Check if graph only has one fetch.
    if (&block.front() != &block.back()) return failure();

    // Map graph results to fetch operands.
    rewriter.replaceOp(op, op.GetFetch().getFetches());

    return success();
  }
};

// This pattern matches GraphOps with only one island, pulls out all inner ops
// of the island to the block containing the GraphOp, and then removes the
// GraphOp.
struct HoistInnerOpsSingleIslandGraph : public OpRewritePattern<GraphOp> {
  using OpRewritePattern<GraphOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GraphOp op,
                                PatternRewriter &rewriter) const override {
    Block &block = op.GetBody();
    // Check if graph only has one island.
    if (!HasSingleOpInBlock<IslandOp>(&block)) return failure();

    FetchOp fetch_op = op.GetFetch();
    auto island_op = llvm::cast<IslandOp>(block.front());
    YieldOp yield_op = island_op.GetYield();

    // Map graph results to inner ops results of single island.
    llvm::SmallVector<Value, 8> new_rets;
    for (Value operand : fetch_op.getFetches()) {
      // Control results should not be propagated out.
      if (mlir::isa<ControlType>(operand.getType())) break;

      if (operand.getDefiningOp() != island_op) {
        // Operand is not from island, simply propagate it out.
        new_rets.push_back(operand);
      } else {
        // Lookup yield operand in island for inner op result.
        auto result = mlir::cast<OpResult>(operand);
        new_rets.push_back(yield_op.getOperand(result.getResultNumber()));
      }
    }

    // Move inner ops from island to block containing graph.
    auto &island_body = island_op.GetBody().getOperations();
    Operation *operation = op.getOperation();
    operation->getBlock()->getOperations().splice(
        operation->getIterator(), island_body, island_body.begin(),
        std::prev(island_body.end()));
    rewriter.replaceOp(op, new_rets);

    return success();
  }
};
}  // anonymous namespace

void GraphOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<DropEmptyGraph, HoistInnerOpsSingleIslandGraph>(context);
}

//===----------------------------------------------------------------------===//
// tf_executor.island
//===----------------------------------------------------------------------===//

namespace {
// This pattern matches and removes IslandOps with no inner ops, no control
// operands and no data results. Control result users will have their relevant
// operands removed.
struct DropEmptyIslandNoOperandNoDataResult
    : public OpRewritePattern<IslandOp> {
  using OpRewritePattern<IslandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IslandOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 0 || op.getNumResults() != 1 ||
        !HasSingleOpInBlock<YieldOp>(&op.GetBody()))
      return failure();

    for (auto &use : llvm::make_early_inc_range(op.getControl().getUses()))
      use.getOwner()->eraseOperand(use.getOperandNumber());

    rewriter.eraseOp(op);

    return success();
  }
};

// This pattern matches and removes IslandOps with no inner ops, no control
// operands, one data result and no control result user. The single data result
// (from YieldOps first operand) is forwarded to the IslandOp single data result
// users.
struct DropEmptyIslandNoOperandOneDataResult
    : public OpRewritePattern<IslandOp> {
  using OpRewritePattern<IslandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IslandOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 0 || op.getNumResults() != 2 ||
        !op.getControl().use_empty() ||
        !HasSingleOpInBlock<YieldOp>(&op.GetBody()))
      return failure();

    rewriter.replaceOp(op, {op.GetYield().getOperand(0), nullptr});

    return success();
  }
};

// TODO(lyandy): Add canonicalization for empty IslandOps with more than one
// control operand and no data results.

}  // anonymous namespace

void IslandOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<DropEmptyIslandNoOperandNoDataResult,
              DropEmptyIslandNoOperandOneDataResult>(context);
}

//===----------------------------------------------------------------------===//
// tf_executor.ControlTrigger
//===----------------------------------------------------------------------===//

namespace {
// This pattern matches and removes ControlTriggerOps with no control operands.
// Control result users will have their relevant operands removed.
struct DropEmptyControlTrigger : public OpRewritePattern<ControlTriggerOp> {
  using OpRewritePattern<ControlTriggerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ControlTriggerOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 0) return failure();

    for (auto &use : llvm::make_early_inc_range(op.getControl().getUses()))
      use.getOwner()->eraseOperand(use.getOperandNumber());

    rewriter.eraseOp(op);

    return success();
  }
};
}  // anonymous namespace

void ControlTriggerOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<DropEmptyControlTrigger>(context);
}

//===----------------------------------------------------------------------===//
// Folders
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_executor.island
//===----------------------------------------------------------------------===//

LogicalResult IslandOp::fold(FoldAdaptor,
                             llvm::SmallVectorImpl<OpFoldResult> &results) {
  // This folds IslandOps with no inner ops, one control operand and no data
  // results. The single control operand is forwarded to the IslandOp control
  // result users.
  if (getNumOperands() != 1 || getNumResults() != 1 ||
      !HasSingleOpInBlock<YieldOp>(&GetBody()))
    return failure();

  results.emplace_back(getOperand(0));

  return success();
}

}  // namespace tf_executor
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc.inc"
