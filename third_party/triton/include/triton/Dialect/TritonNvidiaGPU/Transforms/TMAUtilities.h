#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::nvidia_gpu {

constexpr inline int TMA_SIZE_BYTES = 128;
constexpr inline int TMA_ALIGN = 128;

inline bool isFp4Padded(Attribute encoding) {
  auto mmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return mmaEnc && mmaEnc.getFp4Padded();
}

template <typename BuilderT>
inline SmallVector<Value> translateTMAIndices(BuilderT &builder, Location loc,
                                              Attribute encoding,
                                              SmallVector<Value> indices) {
  if (isFp4Padded(encoding)) {
    auto two = builder.template create<arith::ConstantIntOp>(loc, 2, 32);
    indices.back() =
        builder.template create<arith::MulIOp>(loc, indices.back(), two);
  }
  return indices;
}

inline gpu::CTALayoutAttr updateCTALayoutForShape(gpu::CTALayoutAttr ctaLayout,
                                                  ArrayRef<int64_t> shape) {
  auto rank = shape.size();
  if (ctaLayout.getRank() == rank)
    return ctaLayout;

  auto ctx = ctaLayout.getContext();
  if (ctaLayout.getRank() > rank) {
    unsigned rankDiff = ctaLayout.getRank() - rank;
    return gpu::CTALayoutAttr::get(
        ctx, ctaLayout.getCTAsPerCGA().drop_front(rankDiff),
        ctaLayout.getCTASplitNum().drop_front(rankDiff),
        ctaLayout.getCTAOrder().drop_front(rankDiff));
  }
  // For rank-reducing loads, we need to rank-increase the CTA Layout
  auto rankDiff = rank - ctaLayout.getRank();
  for (unsigned i = 0; i < rankDiff; ++i) {
    assert(shape[i] == 1 && "Should only happen for rank-reducing loads");
  }
  SmallVector<unsigned> CTAsPerCGA(rank, 1);
  SmallVector<unsigned> CTASplitNum(rank, 1);
  SmallVector<unsigned> CTAOrder(rank, 1);

  llvm::copy(ctaLayout.getCTAsPerCGA(), CTAsPerCGA.begin() + rankDiff);
  llvm::copy(ctaLayout.getCTASplitNum(), CTASplitNum.begin() + rankDiff);
  for (unsigned i = 0; i < rankDiff; ++i) {
    CTAOrder[i] = rank - i;
  }
  llvm::copy(ctaLayout.getCTAOrder(), CTAOrder.begin() + rankDiff);
  return gpu::CTALayoutAttr::get(ctx, CTAsPerCGA, CTASplitNum, CTAOrder);
}

inline gpu::SharedEncodingTrait
updateEncodingForShape(Operation *op, gpu::SharedEncodingTrait encoding,
                       RankedTensorType tensorType) {
  auto ctx = encoding.getContext();
  auto ctaLayout = gpu::getCTALayout(encoding);
  if (auto nvmmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding)) {
    auto existingCta = nvmmaEnc.getCTALayout();
    if (!existingCta)
      return nvmmaEnc;

    auto newCtaEnc = updateCTALayoutForShape(ctaLayout, tensorType.getShape());
    return gpu::NVMMASharedEncodingAttr::get(
        ctx, nvmmaEnc.getSwizzlingByteWidth(), nvmmaEnc.getTransposed(),
        nvmmaEnc.getElementBitWidth(), nvmmaEnc.getFp4Padded(), newCtaEnc);
  }
  if (auto swizEnc = dyn_cast<gpu::SwizzledSharedEncodingAttr>(encoding)) {
    auto existingCta = swizEnc.getCTALayout();
    if (!existingCta)
      return swizEnc;

    auto rank = tensorType.getRank();
    SmallVector<unsigned> order(
        swizEnc.getOrder().drop_front(swizEnc.getOrder().size() - rank));
    auto newCtaEnc = updateCTALayoutForShape(ctaLayout, tensorType.getShape());
    return gpu::SwizzledSharedEncodingAttr::get(
        ctx, swizEnc.getVec(), swizEnc.getPerPhase(), swizEnc.getMaxPhase(),
        order, newCtaEnc);
  }

  constexpr auto msg = "Internal Error: Unhandled tensor descriptor encoding";
  if (op)
    op->emitError() << msg;
  llvm::report_fatal_error(msg);
}

inline triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc) {
  auto descBlockType = cast<TensorDescType>(desc.getType()).getBlockType();
  Attribute encoding = descBlockType.getEncoding();
  if (!encoding) {
    constexpr auto msg =
        "Internal Error: Tensor descriptor should have encoding set";
    if (op)
      op->emitError() << msg;
    llvm::report_fatal_error(msg);
  }
  auto sharedEnc = cast<gpu::SharedEncodingTrait>(encoding);
  if (descBlockType.getShape() == tensorType.getShape())
    return sharedEnc;

  return updateEncodingForShape(op, sharedEnc, tensorType);
}

template <typename BuilderT>
mlir::LogicalResult createTMADesc(mlir::Value tmaPtr,
                                  mlir::triton::MakeTensorDescOp op,
                                  BuilderT &builder) {
  using namespace mlir;
  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto mkI32Constant = [&](int32_t val) {
    return builder.template create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(val));
  };

  auto elemType = op.getBase().getType().getPointeeType();
  auto elemSize = elemType.getIntOrFloatBitWidth() / 8;
  auto mmaEncoding = llvm::dyn_cast_or_null<gpu::NVMMASharedEncodingAttr>(
      op.getType().getBlockType().getEncoding());
  bool fp4Padded = mmaEncoding && mmaEncoding.getFp4Padded();

  int paddingScale = fp4Padded ? 2 : 1;
  int32_t contig_dim_size = op.getTensorShape().back() * paddingScale;
  int32_t contig_dim_size_in_bytes = contig_dim_size * elemSize;
  if (contig_dim_size_in_bytes > 128) {
    contig_dim_size = 128 / elemSize;
  }
  llvm::SmallVector<Value> boxDim;
  if (fp4Padded && contig_dim_size != 128) {
    op->emitError(
        "FP4 padded loads require 128 elements or more in the last dim");
  }
  boxDim.push_back(mkI32Constant(contig_dim_size));
  for (int k = op.getTensorShape().size() - 2; k >= 0; --k) {
    boxDim.push_back(mkI32Constant(op.getTensorShape()[k]));
  }

  unsigned swizzleBytes = 0;
  if (mmaEncoding) {
    swizzleBytes = mmaEncoding.getSwizzlingByteWidth();
    if (fp4Padded) {
      assert(swizzleBytes == 128 &&
             "elem type .b4x16_p64 supports only 128B swizzling");
    }
  } else {
    auto swizzledEnc = dyn_cast<gpu::SwizzledSharedEncodingAttr>(
        op.getType().getBlockType().getEncoding());
    if (!swizzledEnc || swizzledEnc.getVec() != 1 ||
        swizzledEnc.getPerPhase() != 1 || swizzledEnc.getMaxPhase() != 1) {
      op->emitError() << "Unhandled encoding type";
      return failure();
    }
  }

  int32_t swizzle_mode = 0;
  if (swizzleBytes == 128) {
    swizzle_mode = 3;
  } else if (swizzleBytes == 64) {
    swizzle_mode = 2;
  } else if (swizzleBytes == 32) {
    swizzle_mode = 1;
  }

  Value elemSizeVal = builder.template create<arith::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(elemSize));

  SmallVector<Value> globalDim(llvm::reverse(op.getShape()));
  SmallVector<Value> globalStride;
  for (int k = op.getStrides().size() - 2; k >= 0; --k) {
    globalStride.push_back(op.getStrides()[k]);
  }

  if (fp4Padded) {
    // Convert number of bytes to number of mxfp4 elements
    globalDim[0] = builder.template create<arith::MulIOp>(loc, globalDim[0],
                                                          mkI32Constant(2));
  }

  SmallVector<Value> elementStride(globalDim.size(), mkI32Constant(1));

  for (int i = 0; i < globalStride.size(); ++i)
    globalStride[i] = builder.template create<arith::MulIOp>(
        loc, globalStride[i], elemSizeVal);

  int elemTypeEnum;

  if (fp4Padded) {
    elemTypeEnum = 14; // .b4x16_p64
  } else {
    switch (elemSize) {
    case 1: {
      elemTypeEnum = 0;
      break;
    }
    case 2: {
      elemTypeEnum = 1;
      break;
    }
    case 4: {
      elemTypeEnum = 2;
      break;
    }
    default: {
      op->emitError()
          << "Tensor descriptor element type must have size 1, 2, or 4 but got "
          << elemSize;
      return failure();
    }
    }
  }

  builder.template create<triton::ExperimentalTensormapCreateOp>(
      loc,
      /*desc_ptr=*/tmaPtr,
      /*global_address=*/op.getBase(),
      /*box_dim=*/boxDim,
      /*global_dim=*/globalDim,
      /*global_stride=*/globalStride,
      /*element_strides=*/elementStride,
      /*elem_type*/ builder.getI32IntegerAttr(elemTypeEnum),
      /*interleave_layout*/ builder.getI32IntegerAttr(0),
      /*swizzle_mode=*/builder.getI32IntegerAttr(swizzle_mode),
      /*fill_mode=*/builder.getI32IntegerAttr(0));
  return success();
}

} // namespace mlir::triton::nvidia_gpu
