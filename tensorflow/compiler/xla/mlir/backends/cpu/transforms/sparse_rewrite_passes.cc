/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_SPARSECUSTOMCALLREWRITINGPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

DenseIntElementsAttr getDenseIntAttrFromConstant(Value v) {
  if (auto const_op = v.getDefiningOp<mhlo::ConstantOp>()) {
    return const_op.getValue().cast<DenseIntElementsAttr>();
  } else if (auto itoa_op = v.getDefiningOp<mhlo::IotaOp>()) {
    // MHLO canonicalizer canonicalizes constants like [0, 1, 2, .., n-1] to
    // mhlo.itoa {itoa_dimension=0}: tensor<n x i64>
    RankedTensorType rtt = itoa_op.getOutput().getType();
    // We only use 1-D tensors to encode constant parameters in custom calls.
    assert(itoa_op.getIotaDimension() == 0 && rtt.getRank() == 1);
    SmallVector<int64_t> const_values;
    const_values.reserve(rtt.getShape()[0]);
    for (int i = 0; i < rtt.getShape()[0]; ++i) {
      const_values.push_back(i);
    }
    return DenseIntElementsAttr::get(rtt, const_values);
  }
  llvm_unreachable("unrecognizable type of constant");
}

void getIntegersFromDenseElements(Value v, SmallVectorImpl<int64_t>& values) {
  auto attr = getDenseIntAttrFromConstant(v);
  values.reserve(values.size() + attr.size());
  auto range = llvm::map_range(attr, [](APInt i) { return i.getZExtValue(); });
  values.append(range.begin(), range.end());
}

struct SparsePackCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 && "Need two arrays (data/indices)");
    assert(op.getResults().size() == 1 && "Must be packing into one tensor");
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<sparse_tensor::PackOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0], op.getInputs()[1]);
    return success();
  }
};

struct SparseUnpackCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getResults().size() == 3 &&
           "Must be unpacking into data/indices/nnz");
    assert(op.getInputs().size() == 1 &&
           "Must be unpacking from one sparse tensor");

    SmallVector<Type, 3> unpack_ret_tp(op.getResults().getTypes());
    // A scalar is treated as a zero-ranked tensor type from frontend.
    auto nnz_type = unpack_ret_tp.back().cast<RankedTensorType>();
    assert(nnz_type.getRank() == 0 && "nnz tensor must be zero ranked");
    unpack_ret_tp.back() = nnz_type.getElementType();

    // Constructs the UnpackOp.
    auto unpack_op = rewriter.create<sparse_tensor::UnpackOp>(
        op.getLoc(), unpack_ret_tp, op.getInputs());

    // Converts the scalar nnz returned from UnpackOp back to tensor type.
    SmallVector<Value, 3> unpack_ret_v(unpack_op.getResults());
    auto scalar_nnz = unpack_op.getNse();
    Value tensor_nnz = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), ArrayRef<int64_t>{}, scalar_nnz.getType());
    tensor_nnz = rewriter.create<tensor::InsertOp>(op.getLoc(), scalar_nnz,
                                                   tensor_nnz, ValueRange{});
    unpack_ret_v.back() = tensor_nnz;
    rewriter.replaceOp(op, unpack_ret_v);
    return success();
  }
};

struct SparseTransposeCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 && "Need argument and permutation");
    assert(op.getResults().size() == 1 && "Need one output tensor");

    // The permutation is passed in as a constant of dense int elements.
    auto permutation_constant =
        op.getInputs()[1].getDefiningOp<mhlo::ConstantOp>();
    auto permutation =
        permutation_constant.getValue().cast<DenseIntElementsAttr>();

    // Reconstruct the transpose operation.
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0], permutation);
    return success();
  }
};

struct SparseDotCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 6 && "Need arguments and metadata");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    SmallVector<int64_t> lhs_contr, rhs_contr, lhs_batch, rhs_batch;
    getIntegersFromDenseElements(op.getInputs()[2], lhs_contr);
    getIntegersFromDenseElements(op.getInputs()[3], rhs_contr);
    getIntegersFromDenseElements(op.getInputs()[4], lhs_batch);
    getIntegersFromDenseElements(op.getInputs()[5], rhs_batch);
    auto dot_dims = mlir::mhlo::DotDimensionNumbersAttr::get(
        op.getContext(), lhs_batch, rhs_batch, lhs_contr, rhs_contr);
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0], op.getInputs()[1],
        dot_dims, /*defaultPrecision*/ ArrayAttr());
    return success();
  }
};

struct SparseConcatenateCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getResults().size() == 1 && "Need one output tensor");

    // The concatenation dimension.
    auto concat_dim = op.getInputs().back().getDefiningOp<mhlo::ConstantOp>();
    auto concat_dim_attr = concat_dim.getValue().cast<DenseIntElementsAttr>();
    // Reconstruct the concatenate operation.
    Value ret_sp_tensor = op.getResults()[0];
    // Depending on test setup, we can get either a 32-bit integer or a 64-bit
    // integer.
    if (concat_dim_attr.getElementType().isInteger(32)) {
      rewriter.replaceOpWithNewOp<sparse_tensor::ConcatenateOp>(
          op, ret_sp_tensor.getType(), op.getInputs().drop_back(),
          rewriter.getIndexAttr(concat_dim_attr.getValues<uint32_t>()[0]));
    } else {
      assert(concat_dim_attr.getElementType().isInteger(64));
      rewriter.replaceOpWithNewOp<sparse_tensor::ConcatenateOp>(
          op, ret_sp_tensor.getType(), op.getInputs().drop_back(),
          rewriter.getIndexAttr(concat_dim_attr.getValues<uint64_t>()[0]));
    }

    return success();
  }
};

struct SparseBroadcastInDimCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 &&
           "Need argument and broadcast dimensions");
    assert(op.getResults().size() == 1 && "Need one output tensor");

    // Broadcast dimensions are passed in as a constant of dense int elements.
    auto dims_constant = op.getInputs()[1].getDefiningOp<mhlo::ConstantOp>();
    auto broadcast_dimensions =
        dims_constant.getValue().cast<DenseIntElementsAttr>();

    // Reconstruct the broadcast_in_dim operation.
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0], broadcast_dimensions);
    return success();
  }
};

template <typename unaryChlo>
struct SparseUnaryChloCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 1 && "Need one argument");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    // Reconstruct the unary chlo operation.
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<unaryChlo>(op, ret_sp_tensor.getType(),
                                           op.getInputs()[0]);
    return success();
  }
};

struct SparseSliceCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 4 &&
           "Need one operand and three slicing parameters");
    assert(op.getResults().size() == 1 && "Need one output tensor");

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto retTp = op.getResults().getTypes()[0].cast<RankedTensorType>();

    auto offsets = getDenseIntAttrFromConstant(op.getInputs()[1]);
    auto strides = getDenseIntAttrFromConstant(op.getInputs()[3]);

    assert(offsets.getNumElements() == strides.getNumElements() &&
           offsets.getNumElements() == retTp.getRank());

    SmallVector<sparse_tensor::SparseTensorDimSliceAttr> slice_attrs;
    SmallVector<int64_t> static_offsets, static_sizes, static_strides;
    for (auto [offset, size, stride] :
         llvm::zip(offsets, retTp.getShape(), strides)) {
      int64_t o = offset.getZExtValue(), s = stride.getZExtValue();
      // Converts limits to sizes.
      slice_attrs.push_back(
          sparse_tensor::SparseTensorDimSliceAttr::get(ctx, o, size, s));
      static_offsets.push_back(o);
      static_sizes.push_back(size);
      static_strides.push_back(s);
    }

    auto srcEnc =
        retTp.getEncoding().cast<sparse_tensor::SparseTensorEncodingAttr>();
    // TODO(peiming): add a getSliceEncodingFrom into MLIR upstream.
    auto sliceEnc = sparse_tensor::SparseTensorEncodingAttr::get(
        ctx, srcEnc.getDimLevelType(), srcEnc.getDimOrdering(),
        srcEnc.getHigherOrdering(), srcEnc.getPosWidth(), srcEnc.getCrdWidth(),
        slice_attrs);
    auto sliceTp = RankedTensorType::get(retTp.getShape(),
                                         retTp.getElementType(), sliceEnc);

    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTp, op.getInputs()[0], ValueRange(), ValueRange(),
        ValueRange(), static_offsets, static_sizes, static_strides);

    // TODO(peiming): This weakens the performance benefit we get from the
    // sparse compiler by forcing every slice to be materizalized while the
    // sparse compiler supports view-based slice.
    rewriter.replaceOpWithNewOp<sparse_tensor::ConvertOp>(op, retTp, slice);
    return success();
  }
};

struct SparseDynSliceCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getResults().size() == 1 && "Need one output tensor");

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto retTp = op.getResults().getTypes()[0].cast<RankedTensorType>();
    // Strips the tensor operand at the front and the static_size array at
    // the end. Inputs in between specify the dynamic offsets.
    auto dyn_off_tensors = op.getInputs().drop_front().drop_back();
    auto sizes = getDenseIntAttrFromConstant(op.getInputs().back());

    assert(sizes.getNumElements() == retTp.getRank() &&
           dyn_off_tensors.size() == retTp.getRank());

    SmallVector<sparse_tensor::SparseTensorDimSliceAttr> slice_attrs;
    SmallVector<int64_t> static_offsets, static_sizes, static_strides;
    SmallVector<Value> dyn_offsets;
    constexpr auto dyn_v = sparse_tensor::SparseTensorDimSliceAttr::kDynamic;
    for (auto em : llvm::enumerate(sizes)) {
      // Populates sparse tensor slice attribute
      uint64_t sz = em.value().getZExtValue();
      slice_attrs.push_back(
          sparse_tensor::SparseTensorDimSliceAttr::get(ctx, dyn_v, sz, 1));
      // Populates arrays used for ExtractSliceOp.
      static_offsets.push_back(ShapedType::kDynamic);
      static_strides.push_back(1);  // dynamic_slice always uses stride == 1
      static_sizes.push_back(sz);
      // Populates dynamic offset value arrays for ExtractSliceOp.
      Value dyn_off = rewriter.create<tensor::ExtractOp>(
          loc, dyn_off_tensors[em.index()], ValueRange{});
      Value dyn_off_idx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), dyn_off);
      dyn_offsets.push_back(dyn_off_idx);
    }

    auto srcEnc =
        retTp.getEncoding().cast<sparse_tensor::SparseTensorEncodingAttr>();
    auto sliceEnc = sparse_tensor::SparseTensorEncodingAttr::get(
        ctx, srcEnc.getDimLevelType(), srcEnc.getDimOrdering(),
        srcEnc.getHigherOrdering(), srcEnc.getPosWidth(), srcEnc.getCrdWidth(),
        slice_attrs);
    auto sliceTp = RankedTensorType::get(retTp.getShape(),
                                         retTp.getElementType(), sliceEnc);

    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTp, op.getInputs()[0], dyn_offsets, /*sizes=*/ValueRange{},
        /*strides=*/ValueRange{}, static_offsets, static_sizes, static_strides);

    // TODO(peiming): This weakens the performance benefit we get from the
    // sparse compiler by forcing every slice to be materizalized while the
    // sparse compiler supports view-based slice.
    rewriter.replaceOpWithNewOp<sparse_tensor::ConvertOp>(op, retTp, slice);
    return success();
  }
};

struct SparseReshapeCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 1 && "Need one input tensor");
    assert(op.getResults().size() == 1 && "Need one output tensor");

    // Reconstruct the reshape operation.
    Value ret_sp_tensor = op.getResults()[0];
    // TODO(anlunx): Fix the issue that the reshape is rewritten to a collapse +
    // expand pair where the sparsity encoding is dropped in between.
    rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, ret_sp_tensor.getType(),
                                                 op.getInputs()[0]);
    return success();
  }
};

struct SparseConvertCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 1 && "Need one input tensor");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<sparse_tensor::ConvertOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0]);
    return success();
  }
};

class SparseCustomCallRewriter : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;
  using SparseCustomTargetRewriter = std::function<LogicalResult(
      mhlo::CustomCallOp op, PatternRewriter& rewriter)>;

  const llvm::StringMap<SparseCustomTargetRewriter> rewriter_map_{
      std::make_pair("sparse_tensor_sparse_pack", SparsePackCallRewriter()),
      std::make_pair("sparse_tensor_sparse_unpack", SparseUnpackCallRewriter()),
      std::make_pair("sparse_tensor_transpose", SparseTransposeCallRewriter()),
      std::make_pair("sparse_tensor_dot_general", SparseDotCallRewriter()),
      std::make_pair("sparse_tensor_concatenate",
                     SparseConcatenateCallRewriter()),
      std::make_pair("sparse_tensor_broadcast_in_dim",
                     SparseBroadcastInDimCallRewriter()),
      std::make_pair("sparse_tensor_asin",
                     SparseUnaryChloCallRewriter<chlo::AsinOp>()),
      std::make_pair("sparse_tensor_asinh",
                     SparseUnaryChloCallRewriter<chlo::AsinhOp>()),
      std::make_pair("sparse_tensor_atan",
                     SparseUnaryChloCallRewriter<chlo::AtanOp>()),
      std::make_pair("sparse_tensor_atanh",
                     SparseUnaryChloCallRewriter<chlo::AtanhOp>()),
      std::make_pair("sparse_tensor_bessel_i1e",
                     SparseUnaryChloCallRewriter<chlo::BesselI1eOp>()),
      std::make_pair("sparse_tensor_sinh",
                     SparseUnaryChloCallRewriter<chlo::SinhOp>()),
      std::make_pair("sparse_tensor_tan",
                     SparseUnaryChloCallRewriter<chlo::TanOp>()),
      std::make_pair("sparse_tensor_slice", SparseSliceCallRewriter()),
      std::make_pair("sparse_tensor_dynamic_slice",
                     SparseDynSliceCallRewriter()),
      std::make_pair("sparse_tensor_reshape", SparseReshapeCallRewriter()),
      std::make_pair("sparse_tensor_convert", SparseConvertCallRewriter()),
  };

  // Rewrites a CustomCallOp to corresponding sparse_tensor operation.
  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (auto it = rewriter_map_.find(op.getCallTargetName());
        it != rewriter_map_.end()) {
      return it->second(op, rewriter);
    }
    // Returns failure on unmatched call target.
    return failure();
  }
};

class SparseCustomCallRewritingPass
    : public impl::SparseCustomCallRewritingPassBase<
          SparseCustomCallRewritingPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<SparseCustomCallRewriter>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSparseCustomCallRewritingPass() {
  return std::make_unique<SparseCustomCallRewritingPass>();
}

}  // namespace cpu
}  // namespace xla
