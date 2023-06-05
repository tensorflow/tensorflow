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
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/attributes.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/sync/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime

namespace tfrt {
namespace fallback_async {

namespace {

struct FallbackInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *dest, bool would_be_cloned,
                       IRMapping &) const final {
    return true;
  }
};

}  // namespace

FallbackAsyncDialect::FallbackAsyncDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_fallback_async", context,
              TypeID::get<FallbackAsyncDialect>()) {
  context->getOrLoadDialect<tfrt::fallback::FallbackDialect>();
  context->getOrLoadDialect<compiler::TFRTDialect>();
  context->getOrLoadDialect<corert::CoreRTDialect>();

  allowUnknownTypes();

  allowUnknownOperations();

  addInterfaces<FallbackInlinerInterface>();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cpp.inc"
      >();
}

static Type GetChainType(Builder *builder) {
  return builder->getType<compiler::ChainType>();
}

LogicalResult CreateOp::verify() {
  return fallback_common::VerifyFallbackExecuteOp(*this);
}
LogicalResult ExecuteOp::verify() {
  return fallback_common::VerifyFallbackExecuteOp(*this);
}
LogicalResult ExecuteOpSeq::verify() {
  return fallback_common::VerifyFallbackExecuteOp(*this);
}
LogicalResult ExecuteOpWithAllocator::verify() {
  return fallback_common::VerifyExecuteOpCommon(*this);
}
LogicalResult ExecuteOpSeqWithAllocator::verify() {
  return fallback_common::VerifyExecuteOpCommon(*this);
}
LogicalResult BatchFunctionOp::verify() {
  return fallback_common::VerifyExecuteOpCommon(*this);
}

ParseResult CreateOp::parse(OpAsmParser &parser, OperationState &result) {
  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = true;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = false;

  auto &builder = parser.getBuilder();
  if (mlir::failed(fallback_common::ParseExecuteOpCommon(
          parser, builder, result, builder.getType<fallback::TFTensorType>(),
          parse_options)))
    return mlir::failure();

  mlir::IntegerAttr num_args;
  if (parser.parseKeyword("num_args") || parser.parseLParen() ||
      parser.parseAttribute(num_args, "num_args", result.attributes) ||
      parser.parseRParen())
    return mlir::failure();

  return mlir::success();
}
ParseResult ExecuteOp::parse(OpAsmParser &parser, OperationState &result) {
  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = false;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = true;

  auto &builder = parser.getBuilder();
  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, builder.getType<fallback::TFTensorType>(),
      parse_options);
}
ParseResult ExecuteOpSeq::parse(OpAsmParser &parser, OperationState &result) {
  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = true;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = true;

  auto &builder = parser.getBuilder();
  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, builder.getType<fallback::TFTensorType>(),
      parse_options);
}
ParseResult ExecuteOpWithAllocator::parse(OpAsmParser &parser,
                                          OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> allocator;
  if (parser.parseOperandList(allocator,
                              /*requiredOperandCount=*/1,
                              mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  if (parser.resolveOperand(allocator.front(),
                            builder.getType<fallback::TFAllocatorType>(),
                            result.operands))
    return mlir::failure();

  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = false;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = true;

  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, builder.getType<fallback::TFTensorType>(),
      parse_options);
}
ParseResult ExecuteOpSeqWithAllocator::parse(OpAsmParser &parser,
                                             OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2>
      chain_and_allocator;
  if (parser.parseOperandList(chain_and_allocator,
                              /*requiredOperandCount=*/2,
                              mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  auto &chain = chain_and_allocator[0];
  auto &allocator = chain_and_allocator[1];

  if (parser.resolveOperand(chain, builder.getType<compiler::ChainType>(),
                            result.operands))
    return mlir::failure();

  if (parser.resolveOperand(allocator,
                            builder.getType<fallback::TFAllocatorType>(),
                            result.operands))
    return mlir::failure();

  // The first result is a chain.
  result.types.push_back(builder.getType<compiler::ChainType>());

  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = false;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = true;

  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, builder.getType<fallback::TFTensorType>(),
      parse_options);
}

ParseResult BatchFunctionOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = false;
  parse_options.has_key = false;
  parse_options.has_device = true;
  parse_options.has_func_attr = false;
  parse_options.has_cost = false;
  parse_options.has_op_name = false;
  parse_options.has_symbol_ref = true;

  auto &builder = parser.getBuilder();
  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, builder.getType<fallback::TFTensorType>(),
      parse_options);
}

void CreateOp::print(OpAsmPrinter &p) {
  CreateOp op = *this;
  p << "(" << op.getInCh() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") device("
    << op->getAttr("device") << ") " << op->getAttr("op_name") << "()";

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);

  p << " num_args(" << op->getAttrOfType<mlir::IntegerAttr>("num_args").getInt()
    << ')';
}

void ExecuteOp::print(OpAsmPrinter &p) {
  ExecuteOp op = *this;
  p << " key(" << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt()
    << ") cost(" << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.getArgs() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.getResults().empty()) p << " : " << op.getResults().size();
}

void ExecuteOpSeq::print(OpAsmPrinter &p) {
  ExecuteOpSeq op = *this;
  p << "(" << op.getInOpChain() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") cost("
    << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.getArgs() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.getResults().empty()) p << " : " << op.getResults().size();
}

void ExecuteOpWithAllocator::print(OpAsmPrinter &p) {
  ExecuteOpWithAllocator op = *this;
  p << "(" << op.getAllocator() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") cost("
    << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.getArgs() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.getResults().empty()) p << " : " << op.getResults().size();
}

void ExecuteOpSeqWithAllocator::print(OpAsmPrinter &p) {
  ExecuteOpSeqWithAllocator op = *this;
  p << "(" << op.getInOpChain() << ", " << op.getAllocator() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") cost("
    << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.getArgs() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.getResults().empty()) p << " : " << op.getResults().size();
}

void BatchFunctionOp::print(OpAsmPrinter &p) {
  BatchFunctionOp op = *this;
  p << " device(" << op->getAttr("device") << ") " << op->getAttr("f") << " ("
    << op.getArgs() << ")";

  fallback_common::PrintExecuteOpCommon(p, op);
}

void ExecuteOp::getOpAttrs(
    SmallVectorImpl<std::pair<StringRef, Attribute>> *op_attrs) {
  fallback_common::GetExecuteOpAttrsCommon(
      this->getContext(), this->getOpAttrs().getValue(), op_attrs);
}

//===----------------------------------------------------------------------===//
// ConstDenseTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstDenseTensorOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// CoreRTTensorHandleToFallbackTensorOp
//===----------------------------------------------------------------------===//

namespace {

// Simplifies pattern containing a corert const tensor op followed by a
// `tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor` op to a single
// tfrt_fallback_async const tensor.
struct ConstCoreRTTensorHandleToFallbackTensorCanonicalization
    : public OpRewritePattern<CoreRTTensorHandleToFallbackTensorOp> {
  using OpRewritePattern<
      CoreRTTensorHandleToFallbackTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CoreRTTensorHandleToFallbackTensorOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 1> new_values;
    bool should_rewrite = false;
    for (auto operand : op.getArgs()) {
      if (auto corert_const_dense_tensor_op =
              operand.getDefiningOp<corert::ConstDenseTensorOp>()) {
        new_values.push_back(
            rewriter.create<fallback_async::ConstDenseTensorOp>(
                op.getLoc(), rewriter.getType<fallback::TFTensorType>(),
                corert_const_dense_tensor_op.getValue()));
        should_rewrite = true;
        continue;
      }
      if (auto corert_const_string_tensor_op =
              operand.getDefiningOp<corert::ConstStringTensorOp>()) {
        new_values.push_back(
            rewriter.create<fallback_async::ConstStringTensorOp>(
                op.getLoc(), rewriter.getType<fallback::TFTensorType>(),
                corert_const_string_tensor_op.getShape(),
                corert_const_string_tensor_op.getValue()));
        should_rewrite = true;
        continue;
      }
      // To guarantee that the new values are in the same order as the old
      // ones, we create individual ops for the non-canonicalizable operands.
      // For simplicity, we don't consolidate these ops when all the
      // non-canonicalizable operands are adjacent.
      new_values.push_back(
          rewriter
              .create<fallback_async::CoreRTTensorHandleToFallbackTensorOp>(
                  op.getLoc(), rewriter.getType<fallback::TFTensorType>(),
                  operand, op->getAttrOfType<mlir::StringAttr>("device"))
              .getResult(0));
    }

    if (!should_rewrite) return failure();
    rewriter.replaceOp(op, new_values);
    return success();
  }
};

// Removes the following double tensor conversion:
//  %1 = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle %0
//  %2 = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %1
struct RemoveDoubleTensorConversion
    : mlir::OpRewritePattern<CoreRTTensorHandleToFallbackTensorOp> {
  using OpRewritePattern<
      CoreRTTensorHandleToFallbackTensorOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      CoreRTTensorHandleToFallbackTensorOp op,
      mlir::PatternRewriter &rewriter) const override {
    // Currently only handles the case where there is only one value in the
    // conversion op. This should be enough for most of the cases.
    if (op.getNumOperands() > 1) return mlir::failure();

    auto def =
        op.getOperand(0).getDefiningOp<FallbackTensorToCoreRTTensorHandleOp>();
    if (!def) return mlir::failure();

    if (def.getNumResults() > 1) return mlir::failure();

    rewriter.replaceOp(op, def.getOperand(0));

    return mlir::success();
  }
};

}  // namespace

void CoreRTTensorHandleToFallbackTensorOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<ConstCoreRTTensorHandleToFallbackTensorCanonicalization,
              RemoveDoubleTensorConversion>(context);
}

}  // namespace fallback_async
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cpp.inc"
