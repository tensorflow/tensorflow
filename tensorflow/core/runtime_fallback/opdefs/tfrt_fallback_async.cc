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
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_async.h"

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
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback.h"
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_common.h"
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
                       BlockAndValueMapping &) const final {
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
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_async.cpp.inc"
      >();
}

static Type GetChainType(Builder *builder) {
  return builder->getType<compiler::ChainType>();
}

static LogicalResult verify(CreateOp op) {
  return fallback_common::VerifyFallbackExecuteOp(op);
}
static LogicalResult verify(ExecuteOp op) {
  return fallback_common::VerifyFallbackExecuteOp(op);
}
static LogicalResult verify(ExecuteOpSeq op) {
  return fallback_common::VerifyFallbackExecuteOp(op);
}
static LogicalResult verify(ExecuteOpWithAllocator op) {
  return fallback_common::VerifyExecuteOpCommon(op);
}
static LogicalResult verify(ExecuteOpSeqWithAllocator op) {
  return fallback_common::VerifyExecuteOpCommon(op);
}
static LogicalResult verify(BatchFunctionOp op) {
  return fallback_common::VerifyExecuteOpCommon(op);
}

static ParseResult parseCreateOp(OpAsmParser &parser, OperationState &result) {
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
static ParseResult parseExecuteOp(OpAsmParser &parser, OperationState &result) {
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
static ParseResult parseExecuteOpSeq(OpAsmParser &parser,
                                     OperationState &result) {
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
static ParseResult parseExecuteOpWithAllocator(OpAsmParser &parser,
                                               OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 1> allocator;
  if (parser.parseOperandList(allocator,
                              /*requiredOperandCount=*/1,
                              mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  if (parser.resolveOperands(allocator.front(),
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
static ParseResult parseExecuteOpSeqWithAllocator(OpAsmParser &parser,
                                                  OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> chain_and_allocator;
  if (parser.parseOperandList(chain_and_allocator,
                              /*requiredOperandCount=*/2,
                              mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  auto &chain = chain_and_allocator[0];
  auto &allocator = chain_and_allocator[1];

  if (parser.resolveOperands(chain, builder.getType<compiler::ChainType>(),
                             result.operands))
    return mlir::failure();

  if (parser.resolveOperands(allocator,
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

static ParseResult parseBatchFunctionOp(OpAsmParser &parser,
                                        OperationState &result) {
  auto &builder = parser.getBuilder();
  auto chain_type = GetChainType(&builder);
  auto tensorhandle_type = builder.getType<corert::TensorHandleType>();

  FlatSymbolRefAttr f;
  SmallVector<OpAsmParser::OperandType, 4> in_chains;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  NamedAttrList op_attrs;
  auto loc = parser.getNameLoc();

  if (parser.parseOperandList(in_chains,
                              /*requiredOperandCount=*/1,
                              OpAsmParser::Delimiter::Paren))
    return failure();

  if (parser.parseAttribute(f, "f", result.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(op_attrs))
    return failure();

  int64_t num_results = 0;
  if (succeeded(parser.parseOptionalColon())) {
    IntegerAttr attr;
    mlir::NamedAttrList attrs;
    if (failed(parser.parseAttribute(attr, "num_results", attrs)))
      return failure();
    num_results = attr.getValue().getSExtValue();
  }

  SmallVector<Type, 4> operand_types;
  operand_types.push_back(chain_type);
  if (parser.resolveOperands(in_chains, operand_types, loc, result.operands) ||
      parser.resolveOperands(operands, tensorhandle_type, result.operands))
    return failure();

  result.types.push_back(chain_type);
  result.types.append(num_results, tensorhandle_type);

  SmallVector<Attribute, 4> op_attr_array;
  for (const auto &key_value : op_attrs) {
    auto key = key_value.getName();
    auto value = key_value.getValue();
    op_attr_array.push_back(builder.getArrayAttr({key, value}));
  }

  result.attributes.push_back(
      builder.getNamedAttr("op_attrs", builder.getArrayAttr(op_attr_array)));

  return success();
}

static void print(OpAsmPrinter &p, CreateOp op) {
  p << "(" << op.in_ch() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") device("
    << op->getAttr("device") << ") " << op->getAttr("op_name") << "()";

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);

  p << " num_args(" << op->getAttrOfType<mlir::IntegerAttr>("num_args").getInt()
    << ')';
}

static void print(OpAsmPrinter &p, ExecuteOp op) {
  p << " key(" << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt()
    << ") cost(" << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

static void print(OpAsmPrinter &p, ExecuteOpSeq op) {
  p << "(" << op.in_op_chain() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") cost("
    << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

static void print(OpAsmPrinter &p, ExecuteOpWithAllocator op) {
  p << "(" << op.allocator() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") cost("
    << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

static void print(OpAsmPrinter &p, ExecuteOpSeqWithAllocator op) {
  p << "(" << op.in_op_chain() << ", " << op.allocator() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") cost("
    << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

static void print(OpAsmPrinter &p, BatchFunctionOp op) {
  p << "(" << op.in_op_chain() << ") " << op->getAttr("f") << " ("
    << op.operands() << ") ";

  fallback_common::PrintExecuteOpCommon(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

void ExecuteOp::getOpAttrs(
    SmallVectorImpl<std::pair<StringRef, Attribute>> *op_attrs) {
  fallback_common::GetExecuteOpAttrsCommon(
      this->getContext(), this->op_attrs().getValue(), op_attrs);
}

//===----------------------------------------------------------------------===//
// ConstDenseTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstDenseTensorOp::fold(ArrayRef<Attribute> operands) {
  return value();
}

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
    for (auto operand : op.operands()) {
      if (auto corert_const_dense_tensor_op =
              operand.getDefiningOp<corert::ConstDenseTensorOp>()) {
        new_values.push_back(
            rewriter.create<fallback_async::ConstDenseTensorOp>(
                op.getLoc(), rewriter.getType<fallback::TFTensorType>(),
                corert_const_dense_tensor_op.value()));
        should_rewrite = true;
        continue;
      }
      if (auto corert_const_string_tensor_op =
              operand.getDefiningOp<corert::ConstStringTensorOp>()) {
        new_values.push_back(
            rewriter.create<fallback_async::ConstStringTensorOp>(
                op.getLoc(), rewriter.getType<fallback::TFTensorType>(),
                corert_const_string_tensor_op.shape(),
                corert_const_string_tensor_op.value()));
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
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ConstCoreRTTensorHandleToFallbackTensorCanonicalization,
                 RemoveDoubleTensorConversion>(context);
}

}  // namespace fallback_async
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_async.cpp.inc"
