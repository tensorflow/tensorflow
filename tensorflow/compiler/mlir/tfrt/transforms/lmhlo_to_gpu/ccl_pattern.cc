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
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime

#if XLA_ENABLE_XCCL
#include "tfrt/gpu/wrapper/ccl_wrapper.h"  // from @tf_runtime
#endif  // XLA_ENABLE_XCCL

namespace tensorflow {

#if XLA_ENABLE_XCCL
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

FailureOr<Value> CclOpConversionRewrite(
    lmhlo::AllGatherOp srcOp, Value chain, Value handle,
    xla::gpu::NcclCollectiveConfig& /*config*/,
    mlir::BlockAndValueMapping& mapping, ConversionPatternRewriter& rewriter) {
  const auto& operands = srcOp.operands();
  const auto& results = srcOp.results();

  for (int i = 0; i < operands.size(); i++) {
    xla::Shape shape = xla::TypeToShape(operands[i].getType());
    auto nccl_data_type_or = ToNcclDataType(shape.element_type());
    if (mlir::failed(nccl_data_type_or)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to convert operand data type to ncclDataType_t.");
    }
    tfrt::gpu::wrapper::CclDataType nccl_data_type = *nccl_data_type_or;

    Value input = mapping.lookup(operands[i]);
    Value output = mapping.lookup(results[i]);

    chain = rewriter.create<tfrt::gpu::CclAllGatherOp>(
        srcOp.getLoc(), handle, input, output, nccl_data_type, chain);
  }
  return chain;
}

FailureOr<Value> CclOpConversionRewrite(
    lmhlo::AllReduceOp srcOp, Value chain, Value handle,
    xla::gpu::NcclCollectiveConfig& /*config*/,
    mlir::BlockAndValueMapping& mapping, ConversionPatternRewriter& rewriter) {
  const auto& operands = srcOp.operands();
  const auto& results = srcOp.results();

  auto reduction_kind =
      xla::gpu::NcclAllReduceThunkBase::MatchAllReduceComputation(
          srcOp.computation());
  if (!reduction_kind.has_value()) {
    return rewriter.notifyMatchFailure(
        srcOp,
        "Failed to match the reduction computation to a reduction kind.");
  }
  ncclRedOp_t reduction_op = ToNcclReduction(*reduction_kind);

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

    chain = rewriter.create<tfrt::gpu::CclAllReduceOp>(
        srcOp.getLoc(), handle, input, output, nccl_data_type, reduction_op,
        chain);
  }
  return chain;
}

FailureOr<Value> CclOpConversionRewrite(
    lmhlo::ReduceScatterOp srcOp, Value chain, Value handle,
    xla::gpu::NcclCollectiveConfig& /*config*/,
    mlir::BlockAndValueMapping& mapping, ConversionPatternRewriter& rewriter) {
  const auto& operands = srcOp.operands();
  const auto& results = srcOp.results();

  auto reduction_kind =
      xla::gpu::NcclAllReduceThunkBase::MatchAllReduceComputation(
          srcOp.computation());
  if (!reduction_kind.has_value()) {
    return rewriter.notifyMatchFailure(
        srcOp,
        "Failed to match the reduction computation to a reduction kind.");
  }
  ncclRedOp_t reduction_op = ToNcclReduction(*reduction_kind);

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

    chain = rewriter.create<tfrt::gpu::CclReduceScatterOp>(
        srcOp.getLoc(), handle, input, output, nccl_data_type, reduction_op,
        chain);
  }
  return chain;
}

FailureOr<Value> CclOpConversionRewrite(
    lmhlo::AllToAllOp srcOp, Value chain, Value handle,
    xla::gpu::NcclCollectiveConfig& /*config*/,
    mlir::BlockAndValueMapping& mapping, ConversionPatternRewriter& rewriter) {
  const auto& operands = srcOp.operands();
  const auto& results = srcOp.results();

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

    if (srcOp.split_dimension().hasValue()) {
      chain = rewriter.create<tfrt::gpu::CclAllToAllOp>(
          srcOp.getLoc(), handle, input, output, nccl_data_type, chain);
    } else {
      Value peer = rewriter.create<tfrt::compiler::ConstantI32Op>(
          srcOp.getLoc(), rewriter.getI32Type(), i);

      chain = rewriter.create<tfrt::gpu::CclSendOp>(
          srcOp.getLoc(), handle, input, peer, nccl_data_type, chain);
      chain = rewriter.create<tfrt::gpu::CclRecvOp>(
          srcOp.getLoc(), handle, output, peer, nccl_data_type, chain);
    }
  }
  return chain;
}

FailureOr<Value> CclOpConversionRewrite(lmhlo::CollectivePermuteOp srcOp,
                                        Value chain, Value handle,
                                        xla::gpu::NcclCollectiveConfig& config,
                                        mlir::BlockAndValueMapping& mapping,
                                        ConversionPatternRewriter& rewriter) {
  const auto& operand = srcOp.operand();
  const auto& result = srcOp.output();

  xla::Shape shape = xla::TypeToShape(operand.getType());
  auto nccl_data_type_or = ToNcclDataType(shape.element_type());
  if (mlir::failed(nccl_data_type_or)) {
    return rewriter.notifyMatchFailure(
        srcOp, "Failed to convert operand data type to ncclDataType_t.");
  }
  ncclDataType_t nccl_data_type = nccl_data_type_or.getValue();

  Value input = mapping.lookup(operand);
  Value output = mapping.lookup(result);

  auto source_target_pairs_or =
      xla::ConvertNx2Attribute(srcOp.source_target_pairs());
  if (!source_target_pairs_or.ok()) {
    return rewriter.notifyMatchFailure(
        srcOp, source_target_pairs_or.status().error_message());
  }
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs =
      std::move(source_target_pairs_or.ValueOrDie());

  std::vector<int64_t> source_peers, target_peers;
  source_peers.reserve(source_target_pairs.size());
  target_peers.reserve(source_target_pairs.size());
  for (const auto& source_target_pair : source_target_pairs) {
    source_peers.push_back(source_target_pair.first);
    target_peers.push_back(source_target_pair.second);
  }
  return rewriter
      .create<xla::gpu::CclCollectivePermuteOp>(
          srcOp.getLoc(), handle, input, output, nccl_data_type,
          static_cast<int64_t>(config.group_mode),
          rewriter.getI64ArrayAttr(source_peers),
          rewriter.getI64ArrayAttr(target_peers), chain)
      .getResult();
}

template <class CclOpType>
LogicalResult BufferOperandsEqualsOpArguments(CclOpType op,
                                              ValueRange operands) {
  if (operands.size() != op.operands().size() + op.results().size()) {
    return mlir::failure();
  }
  return mlir::success();
}

LogicalResult BufferOperandsEqualsOpArguments(lmhlo::CollectivePermuteOp op,
                                              ValueRange operands) {
  // lmhlo::CollectivePermuteOp's input and output count are not variable.
  return mlir::success();
}

template <class CclOpType>
xla::gpu::NcclCollectiveConfig GetNcclCollectiveConfig(CclOpType op,
                                                       int /*replica_count*/,
                                                       int /*num_partitions*/) {
  return xla::gpu::GetNcclCollectiveConfigForMlir(op,
                                                  op.use_global_device_ids());
}

xla::gpu::NcclCollectiveConfig GetNcclCollectiveConfig(lmhlo::AllToAllOp op,
                                                       int /*replica_count*/,
                                                       int /*num_partitions*/) {
  // TODO(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  return xla::gpu::GetNcclCollectiveConfigForMlir(op, absl::nullopt);
}

xla::gpu::NcclCollectiveConfig GetNcclCollectiveConfig(
    lmhlo::CollectivePermuteOp op, int replica_count, int num_partitions) {
  return xla::gpu::NcclCollectivePermuteThunk::GetNcclCollectivePermuteConfig(
      op, replica_count, num_partitions);
}

template <class CclOpType>
FailureOr<Value> TryDegenerateToMemCopy(CclOpType op, Value chain, Value stream,
                                        xla::gpu::NcclCollectiveConfig& config,
                                        int replica_count, int num_partitions,
                                        mlir::BlockAndValueMapping& mapping,
                                        ConversionPatternRewriter& rewriter) {
  if (!config.IsDegenerate(replica_count, num_partitions)) return failure();

  for (int i = 0; i < op.operands().size(); i++) {
    Value dst = mapping.lookup(op.results()[i]);
    Value src = mapping.lookup(op.operands()[i]);
    chain =
        rewriter
            .create<tfrt::gpu::MemCopyOp>(op.getLoc(), dst, src, stream, chain)
            .getResult();
  }
  return chain;
}

FailureOr<Value> TryDegenerateToMemCopy(lmhlo::CollectivePermuteOp op,
                                        Value chain, Value stream,
                                        xla::gpu::NcclCollectiveConfig& config,
                                        int replica_count, int num_partitions,
                                        mlir::BlockAndValueMapping& mapping,
                                        ConversionPatternRewriter& rewriter) {
  if (!xla::gpu::NcclCollectivePermuteThunk::IsDegenerate(op, replica_count,
                                                          num_partitions))
    return failure();

  Value dst = mapping.lookup(op.output());
  Value src = mapping.lookup(op.operand());
  return rewriter
      .create<tfrt::gpu::MemCopyOp>(op.getLoc(), dst, src, stream, chain)
      .getResult();
}

bool CanImplement(lmhlo::AllGatherOp op) {
  return xla::gpu::NcclAllGatherThunk::CanImplement(op);
}

bool CanImplement(lmhlo::AllReduceOp op) {
  return xla::gpu::NcclAllReduceThunk::CanImplement(op);
}

bool CanImplement(lmhlo::ReduceScatterOp op) {
  return xla::gpu::NcclReduceScatterThunk::CanImplement(op);
}

bool CanImplement(lmhlo::AllToAllOp op) {
  return xla::gpu::NcclAllToAllThunk::CanImplement(op);
}

bool CanImplement(lmhlo::CollectivePermuteOp op) {
  return xla::gpu::NcclCollectivePermuteThunk::CanImplement(op);
}

template <class CclOpType>
struct CclRewritePattern : tfrt::gpu::StreamifyOpConversionPattern<CclOpType> {
  using typename tfrt::gpu::StreamifyOpConversionPattern<CclOpType>::OpAdaptor;
  using tfrt::gpu::StreamifyOpConversionPattern<
      CclOpType>::StreamifyOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      CclOpType op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    if (mlir::failed(
            BufferOperandsEqualsOpArguments(op, adaptor.getOperands()))) {
      return rewriter.notifyMatchFailure(
          op,
          "Number of buffer operands does not match the number of op inputs "
          "and outputs.");
    }

    BlockAndValueMapping mapping;
    for (auto pair : llvm::zip_first(op->getOperands(), adaptor.getOperands()))
      mapping.map(std::get<0>(pair), std::get<1>(pair));

    mlir::func::FuncOp func =
        op->template getParentOfType<mlir::func::FuncOp>();
    mlir::IntegerAttr replica_count_attr =
        func->getAttrOfType<mlir::IntegerAttr>("replica_count");
    mlir::IntegerAttr num_partitions_attr =
        func->getAttrOfType<mlir::IntegerAttr>("num_partitions");
    const int replica_count = replica_count_attr.getInt();
    const int num_partitions = num_partitions_attr.getInt();
    xla::gpu::NcclCollectiveConfig config =
        GetNcclCollectiveConfig(op, replica_count, num_partitions);

    // A given collective op can be degenerate if across all groups formed
    // by it are singleton. In such a case, we don't need to do any
    // communication and we can just copy the input to the output.
    auto out_chain_or =
        TryDegenerateToMemCopy(op, chain, stream, config, replica_count,
                               num_partitions, mapping, rewriter);
    if (mlir::succeeded(out_chain_or)) {
      rewriter.eraseOp(op);
      return out_chain_or;
    }
    if (!CanImplement(op)) {
      return rewriter.notifyMatchFailure(op, "Collective op not implemented.");
    }

    auto group_mode_attr = rewriter.getIntegerAttr(
        rewriter.getIntegerType(64), static_cast<int64_t>(config.group_mode));
    auto op_id_attr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), config.op_id);
    SmallVector<NamedAttribute, 8> attributes = {
        rewriter.getNamedAttr("group_mode", group_mode_attr),
        rewriter.getNamedAttr("op_id", op_id_attr),
    };
    for (int i = 0; i < config.replica_groups.size(); ++i) {
      SmallVector<int64_t, 8> replica_group;
      llvm::copy(config.replica_groups[i].replica_ids(),
                 std::back_inserter(replica_group));
      auto replica_group_attr = rewriter.getI64ArrayAttr(replica_group);
      attributes.push_back(rewriter.getNamedAttr(
          tfrt::StrCat("replica_group", i), replica_group_attr));
    }

    auto context =
        rewriter.create<tfrt::gpu::StreamGetContextOp>(op.getLoc(), stream);
    auto handle = rewriter.create<xla::gpu::CclCreateOp>(
        op.getLoc(), ValueRange{context, chain}, attributes);

    out_chain_or =
        CclOpConversionRewrite(op, chain, handle, config, mapping, rewriter);
    if (mlir::succeeded(out_chain_or)) {
      out_chain_or = rewriter
                         .create<tfrt::gpu::CclExecuteOp>(op.getLoc(), stream,
                                                          handle, *out_chain_or)
                         .getResult();
      rewriter.eraseOp(op);
    }
    return out_chain_or;
  }
};

}  // namespace

void populateCclConversionPattern(RewritePatternSet& patterns,
                                  TypeConverter& converter) {
  patterns.add<CclRewritePattern<lmhlo::AllGatherOp>,
               CclRewritePattern<lmhlo::AllReduceOp>,
               CclRewritePattern<lmhlo::ReduceScatterOp>,
               CclRewritePattern<lmhlo::AllToAllOp>,
               CclRewritePattern<lmhlo::CollectivePermuteOp>>(
      converter, patterns.getContext());
}

#else  // XLA_ENABLE_XCCL

void populateCclConversionPattern(RewritePatternSet& patterns,
                                  TypeConverter& converter) {}

#endif  // XLA_ENABLE_XCCL

}  // namespace tensorflow
