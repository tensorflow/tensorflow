/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/mlir/xla_cpu/ir/xla_cpu.h"

#include <optional>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "xla/mlir/xla_cpu/ir/xla_cpu_dialect.cc.inc"
#include "xla/mlir/xla_cpu/ir/xla_cpu_enums.cc.inc"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#define GET_ATTRDEF_CLASSES
#include "xla/mlir/xla_cpu/ir/xla_cpu_attrdefs.cc.inc"

namespace mlir {
namespace xla_cpu {

using ::mlir::mhlo::TokenType;

void XlaCpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/mlir/xla_cpu/ir/xla_cpu.cc.inc"
#undef GET_OP_LIST
      >();
}

template <typename Op>
LogicalResult BufferizeOp(Op op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          int64_t num_inputs) {
  if (op.getOperands().front().getType().template isa<MemRefType>()) {
    return success();
  }
  SmallVector<Value> new_operands;
  std::optional<Value> token = std::nullopt;
  for (auto operand : op.getOperands()) {
    if (operand.getType().template isa<TokenType>()) {
      assert(operand == op.getOperands().back() &&
             "Expect token type only for last operand");
      assert(!token && "Expect at most only one token-typed operand");
      token = operand;
      continue;
    }
    FailureOr<Value> maybe_buffer = getBuffer(rewriter, operand, options);
    if (failed(maybe_buffer)) {
      return failure();
    }
    new_operands.push_back(*maybe_buffer);
  }
  rewriter.create<Op>(op.getLoc(), TypeRange{}, new_operands,
                      op.getOperation()->getAttrs());

  if (token) {
    new_operands.push_back(*token);
  }
  bufferization::replaceOpWithBufferizedValues(
      rewriter, op.getOperation(),
      llvm::ArrayRef(new_operands).drop_front(num_inputs));
  return success();
}

bool AllReduceOp::bufferizesToMemoryRead(OpOperand &opOperand,
                                         const bufferization::AnalysisState &) {
  return opOperand.getOperandNumber() < getNumOperands() / 2;
}

bool AllReduceOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return !bufferizesToMemoryRead(opOperand, state);
}

bufferization::AliasingValueList AllReduceOp::getAliasingValues(
    OpOperand &opOperand, const bufferization::AnalysisState &) {
  if (opOperand.getOperandNumber() < getNumOperands() / 2) {
    return {};
  }
  return {{getOperation()->getOpResult(opOperand.getOperandNumber() -
                                       getNumOperands() / 2),
           bufferization::BufferRelation::Equivalent}};
}

LogicalResult AllReduceOp::bufferize(
    RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {
  return BufferizeOp(*this, rewriter, options, this->getNumOperands() / 2);
}

LogicalResult CollectivePermuteOp::bufferize(
    RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {
  return BufferizeOp(*this, rewriter, options, this->getNumOperands() / 2);
}

LogicalResult AllToAllOp::bufferize(
    RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {
  return BufferizeOp(*this, rewriter, options, this->getNumOperands() / 2);
}

LogicalResult FftOp::bufferize(
    RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {
  return BufferizeOp(*this, rewriter, options, this->getNumOperands() / 2);
}

LogicalResult InfeedOp::bufferize(
    RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {
  return BufferizeOp(*this, rewriter, options, 0);
}

LogicalResult OutfeedOp::bufferize(
    RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {
  return BufferizeOp(*this, rewriter, options, this->getNumOperands());
}

LogicalResult RngBitGeneratorOp::bufferize(
    RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {
  return BufferizeOp(*this, rewriter, options, 1);
}

LogicalResult AddDependencyOp::bufferize(
    RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {
  FailureOr<Value> maybe_buffer =
      getBuffer(rewriter, this->getOperand(), options);
  if (failed(maybe_buffer)) {
    return rewriter.notifyMatchFailure(*this,
                                       "failed during bufferizing operand");
  }
  bufferization::replaceOpWithBufferizedValues(rewriter, this->getOperation(),
                                               *maybe_buffer);
  return success();
}

LogicalResult MemRefElementCastOp::verify() {
  auto src_memref_ty = getSrc().getType().cast<MemRefType>();
  auto dst_memref_ty = getDst().getType().cast<MemRefType>();
  if (src_memref_ty.getShape() != dst_memref_ty.getShape()) {
    return emitOpError() << "expects matching shapes";
  }

  unsigned src_width = src_memref_ty.getElementType().getIntOrFloatBitWidth();
  unsigned dst_width = dst_memref_ty.getElementType().getIntOrFloatBitWidth();
  if ((src_width + CHAR_BIT - 1) / CHAR_BIT !=
      (dst_width + CHAR_BIT - 1) / CHAR_BIT) {
    return emitOpError() << "cannot cast from "
                         << src_memref_ty.getElementType() << " to "
                         << dst_memref_ty.getElementType();
  }
  return success();
}

LogicalResult ConvolutionOp::bufferize(
    RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {
  return BufferizeOp(*this, rewriter, options, this->getNumOperands() - 1);
}

}  // namespace xla_cpu
}  // namespace mlir

#define GET_OP_CLASSES
#include "xla/mlir/xla_cpu/ir/xla_cpu.cc.inc"
