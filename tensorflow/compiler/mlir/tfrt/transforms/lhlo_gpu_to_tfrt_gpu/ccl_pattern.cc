// Copyright 2021 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- ccl_pattern.cc
//-------------------------------------------------------------------------===//
//
// Pattern to lower lmhlo collective ops to tfrt_gpu/xlir dialect.
//
//===----------------------------------------------------------------------===//
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/ccl_pattern.h"

#include <functional>
#include <string>

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/pass/pass.h"  // from @tf_runtime

namespace tensorflow {
namespace {

ncclRedOp_t ToNcclReduction(xla::ReductionKind kind) {
  switch (kind) {
    case xla::ReductionKind::SUM:
      return ncclSum;
    case xla::ReductionKind::PRODUCT:
      return ncclProd;
    case xla::ReductionKind::MIN:
      return ncclMin;
    case xla::ReductionKind::MAX:
      return ncclMax;
  }
}

FailureOr<ncclDataType_t> ToNcclDataType(xla::PrimitiveType element_type) {
  switch (element_type) {
    case xla::S8:
      return ncclInt8;
    case xla::PRED:
    case xla::U8:
      return ncclUint8;
    case xla::S32:
      return ncclInt32;
    case xla::U32:
      return ncclUint32;
    case xla::S64:
      return ncclInt64;
    case xla::U64:
      return ncclUint64;
    case xla::F16:
      return ncclFloat16;
    case xla::F32:
    case xla::C64:
      return ncclFloat32;
    case xla::F64:
    case xla::C128:
      return ncclFloat64;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case xla::BF16:
      return ncclBfloat16;
#endif
    default:
      return mlir::failure();
  }
}

FailureOr<Value> CclOpConversionRewrite(lmhlo::AllReduceOp srcOp, Value chain,
                                        Value stream,
                                        mlir::BlockAndValueMapping& mapping,
                                        ConversionPatternRewriter& rewriter) {
  const auto& operands = srcOp.operands();
  const auto& results = srcOp.results();
  if (operands.size() != results.size()) {
    return rewriter.notifyMatchFailure(
        srcOp, "Number of operands and results do not match.");
  }

  auto reduction_kind =
      xla::gpu::NcclAllReduceThunkBase::MatchAllReduceComputation(
          srcOp.computation());
  if (!reduction_kind.has_value()) {
    return rewriter.notifyMatchFailure(
        srcOp,
        "Failed to match the reduction computation to a reduction kind.");
  }
  ncclRedOp_t reduction_op = ToNcclReduction(*reduction_kind);

  auto context =
      rewriter.create<tfrt::gpu::StreamGetContextOp>(srcOp.getLoc(), stream)
          .getResult();

  auto handle = rewriter.create<xla::gpu::CclCreateOp>(srcOp.getLoc(), context)
                    .getResult();

  for (int i = 0; i < operands.size(); i++) {
    xla::Shape shape = xla::TypeToShape(operands[i].getType());
    auto nccl_data_type_or = ToNcclDataType(shape.element_type());
    if (mlir::failed(nccl_data_type_or)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to convert operand data type to ncclDataType_t.");
    }
    ncclDataType_t nccl_data_type = nccl_data_type_or.getValue();

    Value input = mapping.lookup(operands[i]);
    Value output = mapping.lookup(results[i]);

    chain = rewriter
                .create<tfrt::gpu::CclAllReduceOp>(
                    srcOp.getLoc(), handle, input, output, nccl_data_type,
                    reduction_op, chain)
                .getResult();
  }

  return rewriter
      .create<tfrt::gpu::CclExecuteOp>(srcOp.getLoc(), stream, handle, chain)
      .getResult();
}

// TODO(hanbinyoon): Support additional lmhlo collective ops (in addition to
// lmhlo::AllReduceOp).
struct CclRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<lmhlo::AllReduceOp> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      lmhlo::AllReduceOp>::GpuAsyncOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::AllReduceOp op, Value chain, Value stream,
      ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (!all_of(operands, [](Value operand) {
          return operand.getType().isa<tfrt::gpu::BufferType>();
        }))
      return rewriter.notifyMatchFailure(op, "expected buffer operands");

    BlockAndValueMapping mapping;
    for (auto pair : llvm::zip_first(op->getOperands(), operands))
      mapping.map(std::get<0>(pair), std::get<1>(pair));

    auto out_chain_or =
        CclOpConversionRewrite(op, chain, stream, mapping, rewriter);
    if (mlir::succeeded(out_chain_or)) {
      rewriter.eraseOp(op);
    }
    return out_chain_or;
  }
};

}  // namespace

void populateCclConversionPattern(RewritePatternSet& patterns) {
  patterns.add<CclRewritePattern>(patterns.getContext());
}

}  // namespace tensorflow
