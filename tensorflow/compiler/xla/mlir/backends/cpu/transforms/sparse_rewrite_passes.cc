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
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
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

Value getEmptyTensor(OpBuilder& b, Location loc, RankedTensorType type) {
  auto t = b.create<tensor::EmptyOp>(loc, type.getShape(),
                                     type.getElementType(), ValueRange{});
  auto zero = b.getZeroAttr(type.getElementType());
  auto c0 = b.create<arith::ConstantOp>(loc, zero);
  return b.create<linalg::FillOp>(loc, ValueRange{c0}, ValueRange{t})
      .getResult(0);
}

struct SparseBatchedPackCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getResults().size() == 1 && "Must be packing into one tensor");
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<sparse_tensor::PackOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0],  // sparse tensor values
        op.getInputs().drop_front());                    // sparse tensor levels
    return success();
  }
};

template <typename BinaryMhlo>
struct SparseBinaryCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 && "Need two argument");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    // Reconstruct the binary mhlo operation.
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<BinaryMhlo>(
        op, ret_sp_tensor.getType(), op.getInputs()[0], op.getInputs()[1]);
    return success();
  }
};

struct SparseBroadcastInDimCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 &&
           "Need argument and broadcast dimensions");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    // Broadcast dimensions are passed in as a constant of dense int elements.
    auto dims_constant = op.getInputs()[1];
    auto broadcast_dimensions = getDenseIntAttrFromConstant(dims_constant);
    // Reconstruct the broadcast_in_dim operation.
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0], broadcast_dimensions);
    return success();
  }
};

template <mhlo::ComparisonDirection CmpDir, mhlo::ComparisonType CmpType>
struct SparseCmpNoEqualCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 && "Need two arguments");
    assert(op.getResults().size() == 1 && "Need one output tensor");

    Value lhs = op.getInputs().front();
    Value rhs = op.getInputs().back();
    // Uses the explicit type in case this is a sparse tensor.
    Type ret_tp = op.getResultTypes().front();
    auto cmp_attr = mhlo::ComparisonTypeAttr::get(op.getContext(), CmpType);
    // Replaces the call with the compare operation.
    rewriter.replaceOpWithNewOp<mhlo::CompareOp>(op, ret_tp, lhs, rhs, CmpDir,
                                                 cmp_attr);
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

struct SparseConvCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 && "Need two input tensors");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    auto rtp = op.getResults()[0].getType().cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<linalg::Conv2DNchwFchwOp>(
        op, op.getInputs(), getEmptyTensor(rewriter, op.getLoc(), rtp));
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
    rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(op, ret_sp_tensor.getType(),
                                                    op.getInputs()[0],
                                                    op.getInputs()[1], dot_dims,
                                                    /*defaultPrecision*/
                                                    ArrayAttr());
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
        ctx, srcEnc.getLvlTypes(), srcEnc.getDimToLvl(), srcEnc.getPosWidth(),
        srcEnc.getCrdWidth(), slice_attrs);
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

template <typename ReduceOp>
struct SparseReduceCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 3 &&
           "Need one input tensor, identity, and axes");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    SmallVector<int64_t> axes;
    getIntegersFromDenseElements(op.getInputs()[2], axes);
    Value result = op.getResults()[0];
    auto resultType = result.getType().dyn_cast<RankedTensorType>();
    auto elementType = resultType.getElementType();

    Location loc = op.getLoc();
    RankedTensorType blockArgumentType = RankedTensorType::get({}, elementType);
    mhlo::ReduceOp reduce = rewriter.create<mhlo::ReduceOp>(
        loc, result.getType(), op.getInputs()[0], op.getInputs()[1],
        rewriter.getI64TensorAttr(axes));

    // Setup the body for mhlo.reduce. Note that sparse reductions like
    // add/or/xor are good to go, but the more complicated prod/min/max/and
    // need semi-ring lowering when converting to linalg.
    Region& region = reduce.getBody();
    Block& block = region.emplaceBlock();
    block.addArgument(blockArgumentType, loc);
    block.addArgument(blockArgumentType, loc);
    auto* firstArgument = block.args_begin();
    auto secondArgument = block.args_rbegin();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      Value red =
          rewriter.create<ReduceOp>(loc, *firstArgument, *secondArgument);
      rewriter.create<mhlo::ReturnOp>(loc, red);
    }
    rewriter.replaceOp(op, reduce.getResults());
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

struct SparseSelectRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 3 && "Need three input tensors");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    // Reconstruct the operation.
    rewriter.replaceOpWithNewOp<mhlo::SelectOp>(op, op.getResults().getTypes(),
                                                op.getInputs());
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
        ctx, srcEnc.getLvlTypes(), srcEnc.getDimToLvl(), srcEnc.getPosWidth(),
        srcEnc.getCrdWidth(), slice_attrs);
    auto sliceTp = RankedTensorType::get(retTp.getShape(),
                                         retTp.getElementType(), sliceEnc);
    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTp, op.getInputs()[0], ValueRange(), ValueRange(),
        ValueRange(), static_offsets, static_sizes, static_strides);
    // TODO(peiming): This weakens the performance benefit we get from the
    // sparse compiler by forcing every slice to be materialized while the
    // sparse compiler supports view-based slice.
    rewriter.replaceOpWithNewOp<sparse_tensor::ConvertOp>(op, retTp, slice);
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

struct SparseUnpackCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    // TODO(peiming): Canonicalizes these two cases. The old bridge that uses
    // jax.BCOO/BCSR does not require buffer lengths.
    unsigned unpack_bufs_num = op.getInputs().size() - 1;
    assert(op.getResults().size() == unpack_bufs_num ||
           op.getResults().size() == unpack_bufs_num * 2);
    SmallVector<Type> unpack_ret_tp(
        op.getResults().take_front(unpack_bufs_num).getTypes());
    // Extra lengths for each buffer returned.
    unpack_ret_tp.append(unpack_bufs_num, rewriter.getIndexType());
    Value tensor = op.getInputs()[0];
    Value out_vals = op.getInputs()[1];
    ValueRange out_lvls = op.getInputs().drop_front(2);
    // Constructs the UnpackOp.
    auto unpack_op = rewriter.create<sparse_tensor::UnpackOp>(
        op.getLoc(), unpack_ret_tp, tensor, out_vals, out_lvls);
    assert(unpack_op.getResults().size() == unpack_bufs_num * 2);
    ValueRange bufs = unpack_op.getResults().take_front(unpack_bufs_num);
    ValueRange lens = unpack_op.getResults().take_back(unpack_bufs_num);

    // Wraps the scalar value into a "scalar tensor", i.e., tensor<i64>
    SmallVector<Value> rets(bufs.begin(), bufs.end());
    if (op.getResults().size() == unpack_bufs_num * 2) {
      ValueRange ret_lens = op.getResults().take_back(unpack_bufs_num);
      for (auto [len, tensor_len] : llvm::zip(lens, ret_lens)) {
        auto ret_t_len = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), tensor_len.getType(), ValueRange{});
        auto int_len = rewriter.create<arith::IndexCastUIOp>(
            op.getLoc(), ret_t_len.getType().getElementType(), len);
        auto ret_len = rewriter.create<tensor::InsertOp>(
            op.getLoc(), ret_t_len.getType(), int_len, ret_t_len, ValueRange{});
        rets.push_back(ret_len);
      }
    }
    rewriter.replaceOp(op, rets);
    return success();
  }
};

struct SparseSDDMMCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 3 && "Need S, A, B matrices");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    Location loc = op.getLoc();
    Value matS = op.getInputs()[0];
    Value matA = op.getInputs()[1];
    Value matB = op.getInputs()[2];
    auto etp = matS.getType().dyn_cast<RankedTensorType>().getElementType();
    // Build the enveloping generic op with the following trait:
    //   indexing_maps = [
    //     affine_map<(i,j,k) -> (i,k)>,  // A
    //     affine_map<(i,j,k) -> (k,j)>,  // B
    //     affine_map<(i,j,k) -> (i,j)>   // S
    //   ],
    //   iterator_types = ["parallel", "parallel", "reduction"],
    //   doc = "S(i,j) += spy[S(i,j)] x SUM_k A(i,k) B(k,j)"
    SmallVector<utils::IteratorType, 3> iteratorTypes;
    iteratorTypes.push_back(utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction);
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr i, j, k;
    bindDims(op.getContext(), i, j, k);
    auto indexingMaps = infer({{i, k}, {k, j}, {i, j}});
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{matS.getType()}, ValueRange{matA, matB},
        ValueRange{matS}, indexingMaps, iteratorTypes);
    // Construct semi-ring op.
    Block* main = rewriter.createBlock(&genericOp.getRegion(), {},
                                       {etp, etp, etp}, {loc, loc, loc});
    Value argS = main->getArgument(2);
    rewriter.setInsertionPointToStart(&genericOp.getRegion().front());
    auto semiring = rewriter.create<sparse_tensor::UnaryOp>(loc, etp, argS);
    rewriter.createBlock(&semiring.getPresentRegion(), {}, etp, loc);
    rewriter.setInsertionPointToStart(&semiring.getPresentRegion().front());
    auto mul = rewriter.create<arith::MulFOp>(loc, main->getArgument(0),
                                              main->getArgument(1));
    rewriter.create<sparse_tensor::YieldOp>(loc, mul.getResult());
    rewriter.setInsertionPointAfter(semiring);
    // Construct reduction op.
    auto identity =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(etp));
    auto custom = rewriter.create<sparse_tensor::ReduceOp>(
        loc, etp, argS, semiring.getResult(), identity);
    Block* red =
        rewriter.createBlock(&custom.getRegion(), {}, {etp, etp}, {loc, loc});
    rewriter.setInsertionPointToStart(&custom.getRegion().front());
    auto add = rewriter.create<arith::AddFOp>(loc, red->getArgument(0),
                                              red->getArgument(1));
    rewriter.create<sparse_tensor::YieldOp>(loc, add.getResult());
    rewriter.setInsertionPointAfter(custom);
    rewriter.create<linalg::YieldOp>(loc, custom.getResult());
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

// This rewriter rewrites 2:4 SpMM custom op to linalg.generic operator that
// carries the DENSE24 trait and does multiplication.
struct Sparse2To4SpMMCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 3 && "Need C, A, B matrices");
    assert(op.getResults().size() == 1 && "Need one output tensor");
    Location loc = op.getLoc();
    Value mat_c = op.getInputs()[0];
    Value mat_a = op.getInputs()[1];
    Value mat_b = op.getInputs()[2];

    auto etp = mat_c.getType().dyn_cast<RankedTensorType>().getElementType();
    // Build the enveloping generic op with the following trait:
    //   indexing_maps = [
    //     affine_map<(i,j,k) -> (i,k)>,  // A
    //     affine_map<(i,j,k) -> (k,j)>,  // B
    //     affine_map<(i,j,k) -> (i,j)>   // S
    //   ],
    //   iterator_types = ["parallel", "parallel", "reduction"],
    //   doc = "C(i,j) += SUM_k A(i,k) B(k,j)"
    SmallVector<utils::IteratorType, 3> iteratorTypes;
    iteratorTypes.push_back(utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction);
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr i, j, k;
    bindDims(op.getContext(), i, j, k);
    auto indexing_maps = infer({{i, k}, {k, j}, {i, j}});
    auto generic_op = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{mat_c.getType()}, ValueRange{mat_a, mat_b},
        ValueRange{mat_c}, indexing_maps, iteratorTypes);
    // Set DENSE24 attribute.
    generic_op->setAttr("DENSE24", rewriter.getI32IntegerAttr(1));
    // Construct operations in the linalg.generic block.
    Block* main = rewriter.createBlock(&generic_op.getRegion(), {},
                                       {etp, etp, etp}, {loc, loc, loc});
    Value arg_c = main->getArgument(2);
    rewriter.setInsertionPointToStart(&generic_op.getRegion().front());
    auto mul = rewriter.create<arith::MulFOp>(loc, main->getArgument(0),
                                              main->getArgument(1));
    auto add = rewriter.create<arith::AddFOp>(loc, mul.getResult(), arg_c);
    rewriter.create<linalg::YieldOp>(loc, add.getResult());
    rewriter.replaceOp(op, generic_op.getResults());
    return success();
  }
};

class SparseCustomCallRewriter : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;
  using SparseCustomTargetRewriter = std::function<LogicalResult(
      mhlo::CustomCallOp op, PatternRewriter& rewriter)>;

  const llvm::StringMap<SparseCustomTargetRewriter> rewriter_map_{
      // Internal custom ops that need rewriting.
      std::make_pair("sparse_tensor_add",
                     SparseBinaryCallRewriter<mhlo::AddOp>()),
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
      std::make_pair("sparse_tensor_broadcast_in_dim",
                     SparseBroadcastInDimCallRewriter()),
      std::make_pair("sparse_tensor_concatenate",
                     SparseConcatenateCallRewriter()),
      std::make_pair("sparse_tensor_conv_general_dilated",
                     SparseConvCallRewriter()),
      std::make_pair("sparse_tensor_convert", SparseConvertCallRewriter()),
      std::make_pair("sparse_tensor_dot_general", SparseDotCallRewriter()),
      std::make_pair("sparse_tensor_dynamic_slice",
                     SparseDynSliceCallRewriter()),
      std::make_pair(
          "sparse_tensor_gt_SIGNED",
          SparseCmpNoEqualCallRewriter<mhlo::ComparisonDirection::GT,
                                       mhlo::ComparisonType::SIGNED>()),
      std::make_pair(
          "sparse_tensor_gt_FLOAT",
          SparseCmpNoEqualCallRewriter<mhlo::ComparisonDirection::GT,
                                       mhlo::ComparisonType::FLOAT>()),
      std::make_pair(
          "sparse_tensor_gt_UNSIGNED",
          SparseCmpNoEqualCallRewriter<mhlo::ComparisonDirection::GT,
                                       mhlo::ComparisonType::UNSIGNED>()),
      std::make_pair(
          "sparse_tensor_lt_SIGNED",
          SparseCmpNoEqualCallRewriter<mhlo::ComparisonDirection::LT,
                                       mhlo::ComparisonType::SIGNED>()),
      std::make_pair(
          "sparse_tensor_lt_FLOAT",
          SparseCmpNoEqualCallRewriter<mhlo::ComparisonDirection::LT,
                                       mhlo::ComparisonType::FLOAT>()),
      std::make_pair(
          "sparse_tensor_lt_UNSIGNED",
          SparseCmpNoEqualCallRewriter<mhlo::ComparisonDirection::LT,
                                       mhlo::ComparisonType::UNSIGNED>()),
      std::make_pair("sparse_tensor_mul",
                     SparseBinaryCallRewriter<mhlo::MulOp>()),
      std::make_pair(
          "sparse_tensor_ne_SIGNED",
          SparseCmpNoEqualCallRewriter<mhlo::ComparisonDirection::NE,
                                       mhlo::ComparisonType::SIGNED>()),
      std::make_pair(
          "sparse_tensor_ne_FLOAT",
          SparseCmpNoEqualCallRewriter<mhlo::ComparisonDirection::NE,
                                       mhlo::ComparisonType::FLOAT>()),
      std::make_pair(
          "sparse_tensor_ne_UNSIGNED",
          SparseCmpNoEqualCallRewriter<mhlo::ComparisonDirection::NE,
                                       mhlo::ComparisonType::UNSIGNED>()),
      std::make_pair("sparse_tensor_reduce_and",
                     SparseReduceCallRewriter<mhlo::AndOp>()),
      std::make_pair("sparse_tensor_reduce_max",
                     SparseReduceCallRewriter<mhlo::MaxOp>()),
      std::make_pair("sparse_tensor_reduce_min",
                     SparseReduceCallRewriter<mhlo::MinOp>()),
      std::make_pair("sparse_tensor_reduce_or",
                     SparseReduceCallRewriter<mhlo::OrOp>()),
      std::make_pair("sparse_tensor_reduce_prod",
                     SparseReduceCallRewriter<mhlo::MulOp>()),
      std::make_pair("sparse_tensor_reduce_sum",
                     SparseReduceCallRewriter<mhlo::AddOp>()),
      std::make_pair("sparse_tensor_reduce_xor",
                     SparseReduceCallRewriter<mhlo::XorOp>()),
      std::make_pair("sparse_tensor_reshape", SparseReshapeCallRewriter()),
      std::make_pair("sparse_tensor_select_n", SparseSelectRewriter()),
      std::make_pair("sparse_tensor_sinh",
                     SparseUnaryChloCallRewriter<chlo::SinhOp>()),
      std::make_pair("sparse_tensor_slice", SparseSliceCallRewriter()),
      std::make_pair("sparse_tensor_sparse_pack",
                     SparseBatchedPackCallRewriter()),
      std::make_pair("sparse_tensor_sparse_unpack", SparseUnpackCallRewriter()),
      std::make_pair("sparse_tensor_sub",
                     SparseBinaryCallRewriter<mhlo::SubtractOp>()),
      std::make_pair("sparse_tensor_tan",
                     SparseUnaryChloCallRewriter<chlo::TanOp>()),
      std::make_pair("sparse_tensor_transpose", SparseTransposeCallRewriter()),
      // User custom ops that need rewriting.
      std::make_pair("sparse_jax_sddmm", SparseSDDMMCallRewriter()),
      std::make_pair("sparse_jax_2to4_spmm", Sparse2To4SpMMCallRewriter()),
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
