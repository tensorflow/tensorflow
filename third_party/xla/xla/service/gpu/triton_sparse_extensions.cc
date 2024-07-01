/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/triton_sparse_extensions.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Dialect/NVGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "llvm/Support/ErrorHandling.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

using namespace mlir;  // NOLINT(build/namespaces)

// The functions below are defined in AccelerateMatmul.cpp.
namespace mlir::triton::gpu {
SmallVector<unsigned, 3> getWarpsPerTile(
    Operation *dotOp, ArrayRef<int64_t> shape, int version, int numWarps,
    const SmallVector<unsigned, 3> &instrShape);
int computeOrigBitWidth(Value x);
Value getSharedMemMMAOperand(Value v, mlir::PatternRewriter &rewriter,
                             int opIdx, bool allowTranspose);
}  // namespace mlir::triton::gpu

// The functions below are defined in WGMMA.cpp.
Value createDescriptor(ConversionPatternRewriter &rewriter, Location loc,
                       int64_t swizzling, uint32_t stride);
int64_t getSwizzlingFromLayout(const triton::gpu::SharedEncodingAttr &layout,
                               uint32_t widthInByte);
triton::nvgpu::WGMMAEltType getMmaRetType(Value);
triton::nvgpu::WGMMAEltType getMmaOperandType(Value, bool);

namespace {

struct TritonSparseDotPattern
    : public OpConversionPattern<triton::gpu::SparseDotOp> {
  using OpConversionPattern<triton::gpu::SparseDotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::SparseDotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = cast<RankedTensorType>(op.getType());
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    int numCTAs = typeConverter->getNumCTAs();

    auto rank = origShape.size();
    auto numElements = product<int64_t>(origShape);
    SmallVector<unsigned> retSizePerThread(rank, 1);
    if (numElements / (numWarps * threadsPerWarp) >= 4) {
      retSizePerThread[rank - 1] = 2;
      retSizePerThread[rank - 2] = 2;
    }
    if (numElements / (numWarps * threadsPerWarp) >= 16) {
      retSizePerThread[rank - 1] = 4;
      retSizePerThread[rank - 2] = 4;
    }
    SmallVector<unsigned> retOrder(rank);
    for (unsigned i = 0; i < rank; ++i) retOrder[i] = rank - 1 - i;
    Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder, numWarps,
        threadsPerWarp, numCTAs);
    RankedTensorType retType =
        RankedTensorType::get(origShape, origType.getElementType(), dEncoding);

    // a & b must be of smem layout
    auto aType = cast<RankedTensorType>(adaptor.getA().getType());
    auto bType = cast<RankedTensorType>(adaptor.getB().getType());
    Type aEltType = aType.getElementType();
    Type bEltType = bType.getElementType();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding) return failure();
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!isa<triton::gpu::DotOperandEncodingAttr>(aEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 0, dEncoding, aEltType);
      auto dstType =
          RankedTensorType::get(aType.getShape(), aEltType, encoding);
      a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), dstType, a);
    }
    if (!isa<triton::gpu::DotOperandEncodingAttr>(bEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 1, dEncoding, bEltType);
      auto dstType =
          RankedTensorType::get(bType.getShape(), bEltType, encoding);
      b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), dstType, b);
    }
    c = rewriter.create<triton::gpu::ConvertLayoutOp>(c.getLoc(), retType, c);

    // aMeta must be of smem layout
    auto aMetaType = cast<RankedTensorType>(adaptor.getAMeta().getType());
    Attribute aMetaEncoding = aMetaType.getEncoding();
    if (!aMetaEncoding) return failure();
    Value aMeta = adaptor.getAMeta();
    if (!isa<triton::gpu::SparseDotMetaEncodingAttr>(aMetaEncoding)) {
      Attribute encoding =
          triton::gpu::SparseDotMetaEncodingAttr::get(getContext(), dEncoding);
      auto dstType = RankedTensorType::get(
          aMetaType.getShape(), aMetaType.getElementType(), encoding);
      aMeta = rewriter.create<triton::gpu::ConvertLayoutOp>(aMeta.getLoc(),
                                                            dstType, aMeta);
    }

    auto new_op = rewriter.replaceOpWithNewOp<triton::gpu::SparseDotOp>(
        op, retType, a, b, c, aMeta);
    for (const NamedAttribute attr : op->getAttrs()) {
      if (!new_op->hasAttr(attr.getName()))
        new_op->setAttr(attr.getName(), attr.getValue());
    }

    return success();
  }
};

class AddSparseDotEncodingPass
    : public PassWrapper<AddSparseDotEncodingPass, OperationPass<ModuleOp>> {
 public:
  AddSparseDotEncodingPass() = default;
  AddSparseDotEncodingPass(int32_t num_warps, int32_t threads_per_warp,
                           int32_t num_ctas) {
    num_warps_ = num_warps;
    threads_per_warp_ = threads_per_warp;
    num_ctas_ = num_ctas;
  }
  AddSparseDotEncodingPass(const AddSparseDotEncodingPass &other) {
    num_warps_ = other.num_warps_;
    threads_per_warp_ = other.threads_per_warp_;
    num_ctas_ = other.num_ctas_;
  };

  StringRef getArgument() const override { return "add-sparse-encoding"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    TritonGPUTypeConverter typeConverter(context, num_warps_, threads_per_warp_,
                                         num_ctas_);
    auto pattern =
        std::make_unique<TritonSparseDotPattern>(typeConverter, context);
    RewritePatternSet patterns(context, std::move(pattern));
    TritonGPUConversionTarget target(*context, typeConverter);
    target.addDynamicallyLegalOp<triton::gpu::SparseDotOp>(
        [](triton::gpu::SparseDotOp op) {
          return op.getAMeta().getType().getEncoding() != nullptr;
        });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddSparseDotEncodingPass)

 private:
  Option<int32_t> num_warps_{
      *this, "num-warps", llvm::cl::desc("number of warps"), llvm::cl::init(4)};
  Option<int32_t> threads_per_warp_{
      *this, "threads-per-warp", llvm::cl::desc("number of threads per warp"),
      llvm::cl::init(32)};
  Option<int32_t> num_ctas_{*this, "num-ctas",
                            llvm::cl::desc("number of ctas in a cga"),
                            llvm::cl::init(1)};
};

class SparseBlockedToMMA : public RewritePattern {
  using ConvertLayoutOp = triton::gpu::ConvertLayoutOp;
  using SparseDotOp = triton::gpu::SparseDotOp;
  using SparseDotMetaEncodingAttr = triton::gpu::SparseDotMetaEncodingAttr;
  using NvidiaMmaEncodingAttr = triton::gpu::NvidiaMmaEncodingAttr;

 public:
  SparseBlockedToMMA(MLIRContext *context, int compute_capability)
      : RewritePattern(SparseDotOp::getOperationName(), 2, context),
        compute_capability_(compute_capability) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto dotOp = cast<SparseDotOp>(op);
    auto ctx = op->getContext();
    Value a = dotOp.getA();
    Value b = dotOp.getB();

    // Check data-types and SM compatibility
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    assert(compute_capability_ >= 80 &&
           "SparseDot is supported on Ampere and higher");
    bool allowV3 = !triton::tools::getBoolEnv("DISABLE_MMA_V3");
    int versionMajor = compute_capability_ >= 90 && allowV3 ? 3 : 2;

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = triton::gpu::getShapePerCTA(oldRetType);
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    auto CTALayout = triton::gpu::getCTALayout(oldRetType.getEncoding());

    auto instrShape =
        mmaVersionToInstrShape(versionMajor, retShapePerCTA,
                               cast<RankedTensorType>(a.getType()), numWarps);
    auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                        numWarps, instrShape);
    NvidiaMmaEncodingAttr mmaEnc =
        NvidiaMmaEncodingAttr::get(ctx, versionMajor, /*versionMinor=*/0,
                                   warpsPerTile, CTALayout, instrShape);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);

    if (versionMajor == 2) {
      int minBitwidth = std::min(triton::gpu::computeOrigBitWidth(a),
                                 triton::gpu::computeOrigBitWidth(b));
      int kWidth = 32 / minBitwidth;

      // convert A operand
      auto oldAType = cast<RankedTensorType>(a.getType());
      auto newAEncoding = DotOperandEncodingAttr::get(ctx, 0, mmaEnc, kWidth);
      auto newAType = RankedTensorType::get(
          oldAType.getShape(), oldAType.getElementType(), newAEncoding);
      a = rewriter.create<ConvertLayoutOp>(a.getLoc(), newAType, a);

      // convert B operand
      auto oldBType = cast<RankedTensorType>(b.getType());
      auto newBEncoding = DotOperandEncodingAttr::get(ctx, 1, mmaEnc, kWidth);
      auto newBType = RankedTensorType::get(
          oldBType.getShape(), oldBType.getElementType(), newBEncoding);
      b = rewriter.create<ConvertLayoutOp>(b.getLoc(), newBType, b);
    } else {
      auto eltType = dotOp.getA().getType().getElementType();
      // In MMAV3 transpose is only supported for f16 and bf16.
      bool allowTranspose = eltType.isF16() || eltType.isBF16();
      a = triton::gpu::getSharedMemMMAOperand(a, rewriter, 0, allowTranspose);
      b = triton::gpu::getSharedMemMMAOperand(b, rewriter, 1, allowTranspose);
    }

    // convert metadata
    Value meta = dotOp.getAMeta();
    auto oldMetaType = cast<RankedTensorType>(meta.getType());
    auto newMetaType = RankedTensorType::get(
        oldMetaType.getShape(), oldMetaType.getElementType(),
        SparseDotMetaEncodingAttr::get(ctx, mmaEnc));
    meta = rewriter.create<ConvertLayoutOp>(meta.getLoc(), newMetaType, meta);

    // convert dot instruction
    auto newDot = rewriter.create<SparseDotOp>(dotOp.getLoc(), newRetType, a, b,
                                               newAcc, meta);

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, oldRetType,
                                                 newDot.getResult());
    return success();
  }

 private:
  int compute_capability_;
};

class SparseBlockedToMMAPass
    : public PassWrapper<SparseBlockedToMMAPass, OperationPass<ModuleOp>> {
 public:
  SparseBlockedToMMAPass() = default;

  StringRef getArgument() const override { return "sparse-blocked-to-mma"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    auto compute_capability = getNVIDIAComputeCapability(module);
    auto pattern =
        std::make_unique<SparseBlockedToMMA>(context, compute_capability);
    RewritePatternSet patterns(context, std::move(pattern));
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseBlockedToMMAPass)
};

namespace SharedToSparseDotOperand {
namespace {
constexpr int kThreadsPerWarp = 32;

// Each 16x16 original sparse matrix tile requires 16 metadata values of 16-bit
// size, where the first thread (T0) in each 4-thread group holds two such
// values in a register (32-bit).
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#sparse-matrix-storage
constexpr int kTileSize = 16;
constexpr int kThreadsInGroup = 4;
constexpr int kMetadataElementsPerPackedValue = 8;  // 8 x 2-bit = 16-bit
constexpr int kMetadataLineOffset = kThreadsPerWarp / kThreadsInGroup;
}  // namespace

Value convertLayout(ConversionPatternRewriter &rewriter, Location loc,
                    Value tensor,
                    triton::gpu::SparseDotMetaEncodingAttr sparseEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread) {
  // Calculate tile size as number of mask elements (4xi4).
  NvidiaMmaEncodingAttr mmaLayout =
      cast<NvidiaMmaEncodingAttr>(sparseEncoding.getParent());
  SmallVector<unsigned> warpsPerCTA = mmaLayout.getWarpsPerCTA();
  SmallVector<unsigned> shapePerCTATile = {
      kTileSize * warpsPerCTA[0], kTileSize / kMetadataElementsPerPackedValue};
  Value strideM = smemObj.strides[0];
  Value strideK = smemObj.strides[1];

  // Calculate offset in the tile for the current thread.
  Value threadsPerWarp = i32_val(kThreadsPerWarp);
  Value warpId = udiv(thread, threadsPerWarp);
  Value warpGroupId;
  if (mmaLayout.isHopper()) {
    warpGroupId = urem(warpId, i32_val(warpsPerCTA[0]));
  } else {
    assert(mmaLayout.isAmpere());
    warpGroupId = udiv(warpId, i32_val(warpsPerCTA[1]));
  }
  Value laneId = urem(thread, threadsPerWarp);
  Value laneGroupId = udiv(laneId, i32_val(kThreadsInGroup));
  Value columnId = urem(laneId, i32_val(shapePerCTATile[1]));
  Value rowId = add(mul(warpGroupId, i32_val(kTileSize)), laneGroupId);

  // Calculate number of tile repetitions.
  auto shape = cast<MemDescType>(tensor.getType()).getShape();
  int repM = shape[0] / shapePerCTATile[0];
  int repK = shape[1] / shapePerCTATile[1];
  assert(repM > 0 && repK > 0);

  // Load sparse metadata from shared memory.
  MLIRContext *ctx = tensor.getContext();
  Type ptrTy = ptr_ty(ctx, 3);
  Value base = gep(ptrTy, i16_ty, smemObj.base, i32_val(0));
  SmallVector<Value> values;

  for (int k = 0; k < repK; ++k) {
    for (int m = 0; m < repM; ++m) {
      Value row = add(rowId, i32_val(m * shapePerCTATile[0]));
      Value column = add(columnId, i32_val(k * shapePerCTATile[1]));
      Value offset1 = add(mul(row, strideM), mul(column, strideK));
      Value offset2 = add(offset1, mul(i32_val(kMetadataLineOffset), strideM));
      Value lower = load(i16_ty, gep(ptrTy, i16_ty, base, offset1));
      Value upper = load(i16_ty, gep(ptrTy, i16_ty, base, offset2));
      values.push_back(lower);
      values.push_back(upper);
    }
  }

  // Pack resulting values as LLVM struct.
  Type structTy = struct_ty(SmallVector<Type>(values.size(), i16_ty));
  return packLLElements(loc, typeConverter, values, rewriter, structTy);
}
}  // namespace SharedToSparseDotOperand

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
 public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<triton::gpu::SharedEncodingAttr>(srcLayout) &&
        isa<triton::gpu::SparseDotMetaEncodingAttr>(dstLayout)) {
      return lowerSharedToSparseMeta(op, adaptor, rewriter);
    }
    return failure();
  }

 private:
  // shared -> sparse dot meta
  LogicalResult lowerSharedToSparseMeta(
      triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto sparseEncoding = cast<triton::gpu::SparseDotMetaEncodingAttr>(
        cast<RankedTensorType>(op.getResult().getType()).getEncoding());
    auto llvmElemTy = getTypeConverter()->convertType(
        cast<MemDescType>(op.getSrc().getType()).getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    Value res = SharedToSparseDotOperand::convertLayout(
        rewriter, loc, op.getSrc(), sparseEncoding, smemObj, getTypeConverter(),
        getThreadId(rewriter, loc));

    rewriter.replaceOp(op, res);
    return success();
  }
};

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::SharedEncodingAttr;

using ValueTableV2 = std::map<std::pair<unsigned, unsigned>, Value>;

constexpr int kContractingFactor = 2;  // implied by N:M (2:4)
constexpr int kCore = 2;               // number of core matrices per batch
constexpr int kCoreTile = kCore * kContractingFactor;

// ----- Ampere implementation.

ValueTableV2 getValuesFromDotOperandLayoutStruct(SmallVector<Value> elems,
                                                 int n0, int n1) {
  int offset = 0;
  ValueTableV2 vals;
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      vals[{kCore * i, kCore * j}] = elems[offset++];
      vals[{kCore * i, kCore * j + 1}] = elems[offset++];
      vals[{kCore * i + 1, kCore * j}] = elems[offset++];
      vals[{kCore * i + 1, kCore * j + 1}] = elems[offset++];
    }
  }
  return vals;
}

std::string getMmaSpPtxInstruction(Type type) {
  if (type.isF16()) {
    return "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32";
  } else if (type.isBF16()) {
    return "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32";
  }
  llvm::report_fatal_error("Unsupported SparseDotOp operand type");
}

LogicalResult convertSparseMMA(triton::gpu::SparseDotOp op,
                               triton::gpu::SparseDotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter) {
  // Get number of repetitions across the dimensions.
  auto aTensorTy = cast<RankedTensorType>(op.getA().getType());
  auto bTensorTy = cast<RankedTensorType>(op.getB().getType());

  auto layoutA = dyn_cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto layoutB = dyn_cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  assert(layoutA != nullptr && layoutB != nullptr);

  int bitwidth = aTensorTy.getElementType().getIntOrFloatBitWidth();
  auto mmaEnc = cast<NvidiaMmaEncodingAttr>(layoutA.getParent());
  auto repA = mmaEnc.getMMAv2Rep(triton::gpu::getShapePerCTA(aTensorTy),
                                 bitwidth, layoutA.getOpIdx());
  auto repB = mmaEnc.getMMAv2Rep(triton::gpu::getShapePerCTA(bTensorTy),
                                 bitwidth, layoutB.getOpIdx());

  assert(repA[0] == 1 && repB[0] == 1);  // batch size
  assert(repB[1] == repA[2] * kContractingFactor);
  int repM = repA[1], repN = repB[2], repK = repB[1];

  // Arrange loaded values into positions.
  Location loc = op.getLoc();
  auto ha = getValuesFromDotOperandLayoutStruct(
      unpackLLElements(loc, adaptor.getA(), rewriter), repM,
      repK / kContractingFactor);
  auto hb = getValuesFromDotOperandLayoutStruct(
      unpackLLElements(loc, adaptor.getB(), rewriter),
      std::max(repN / kCore, 1), repK);

  // Combine loaded metadata values.
  auto hMeta = unpackLLElements(loc, adaptor.getAMeta(), rewriter);
  SmallVector<Value> hMetaPacked;
  for (int i = 0; i < hMeta.size(); i += kCore) {
    Value lower = zext(i32_ty, hMeta[i]);
    Value upper = zext(i32_ty, hMeta[i + 1]);
    Value packed = or_(shl(upper, i32_val(16)), lower);
    hMetaPacked.push_back(packed);
  }

  // Flatten accumulator values.
  auto fc = unpackLLElements(loc, adaptor.getC(), rewriter);

  // Create `mma.sp` instruction for 4/8 core matrices.
  auto callMma = [&](unsigned m, unsigned n, unsigned k) {
    triton::PTXBuilder builder;
    auto &mma =
        *builder.create(getMmaSpPtxInstruction(aTensorTy.getElementType()));

    auto retArgs = builder.newListOperand(kCoreTile, "=f");
    auto cArgs = builder.newListOperand();
    int baseIdx = m * repN * kCore + n * kCoreTile;
    for (int i = 0; i < kCoreTile; ++i) {
      cArgs->listAppend(builder.newOperand(fc[baseIdx + i], std::to_string(i)));
    }
    int i = k / kContractingFactor;
    auto aArgs = builder.newListOperand({
        {ha[{m, i}], "r"},
        {ha[{m + 1, i}], "r"},
        {ha[{m, i + 1}], "r"},
        {ha[{m + 1, i + 1}], "r"},
    });
    auto bArgs = builder.newListOperand({
        {hb[{n, k}], "r"},
        {hb[{n, k + 1}], "r"},
        {hb[{n, k + 2}], "r"},
        {hb[{n, k + 3}], "r"},
    });
    auto metaArg =
        builder.newOperand(hMetaPacked[k / kCoreTile * repM + m / kCore], "r");
    auto selector = builder.newConstantOperand(0);
    mma(retArgs, aArgs, bArgs, cArgs, metaArg, selector);

    Type fp32x4Ty = LLVM::LLVMStructType::getLiteral(
        op.getContext(), SmallVector<Type>(kCoreTile, f32_ty));
    Value mmaOut = builder.launch(rewriter, loc, fp32x4Ty);
    for (int i = 0; i < kCoreTile; ++i) {
      fc[baseIdx + i] = extract_val(f32_ty, mmaOut, i);
    }
  };

  for (int k = 0; k < repK; k += kContractingFactor)
    for (int m = 0; m < repM; ++m)
      for (int n = 0; n < repN; ++n) callMma(kCore * m, n, kCore * k);

  // Replace with new packed result.
  Type structTy = LLVM::LLVMStructType::getLiteral(
      op.getContext(), SmallVector<Type>(fc.size(), f32_ty));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);
  rewriter.replaceOp(op, res);

  return success();
}

// ----- Hopper implementation.

constexpr int kThreadsPerWarp = 32;
constexpr int kWarpsInGroup = 4;
constexpr int kMmaAccumulatorCount = 2;
constexpr int kMmaLineSize = 128;
constexpr int kMmaAlignment = 16;

// Shared memory descriptor builder for WGMMA.
Value smemDescriptor(int a, int b, ConversionPatternRewriter &rewriter,
                     Location loc, std::vector<unsigned int> instrShape,
                     bool trans, int dimWpt, Value warpId, MemDescType tensorTy,
                     Value baseDesc, int minor) {
  auto sharedLayout = cast<SharedEncodingAttr>(tensorTy.getEncoding());
  int elemBytes = tensorTy.getElementTypeBitWidth() / 8;
  int elemsPerSwizzlingRow =
      kMmaLineSize / sharedLayout.getPerPhase() / elemBytes;
  Value elemsPerSwizzlingRowVal = i32_val(elemsPerSwizzlingRow);

  Value k = i32_val(b * instrShape[1]);
  Value m = add(i32_val(a * dimWpt * instrShape[0]),
                mul(warpId, i32_val(instrShape[0])));
  if (trans) {
    std::swap(k, m);
  }
  Value leading_offset = mul(udiv(k, elemsPerSwizzlingRowVal),
                             i32_val(minor * elemsPerSwizzlingRow));
  Value stride_offset = mul(m, elemsPerSwizzlingRowVal);
  Value offset =
      add(add(leading_offset, stride_offset), urem(k, elemsPerSwizzlingRowVal));
  Value off1 = mul(i32_val(elemBytes), offset);
  Value off_ = zext(i64_ty, udiv(off1, i32_val(kMmaAlignment)));

  return add(baseDesc, off_);
}

LogicalResult convertSparseWGMMA(triton::gpu::SparseDotOp op,
                                 triton::gpu::SparseDotOp::Adaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter,
                                 Value thread) {
  // Get number of repetitions across the dimensions.
  auto aTensorTy = cast<MemDescType>(op.getA().getType());
  auto bTensorTy = cast<MemDescType>(op.getB().getType());
  auto dTensorTy = cast<RankedTensorType>(op.getD().getType());
  auto mmaEnc = cast<NvidiaMmaEncodingAttr>(dTensorTy.getEncoding());

  auto shapePerCTA = getShapePerCTA(dTensorTy);
  auto shapePerCTATile = getShapePerCTATile(mmaEnc);
  auto instrShape = mmaEnc.getInstrShape();
  int repM = ceil<unsigned>(shapePerCTA[0], shapePerCTATile[0]);
  int repN = ceil<unsigned>(shapePerCTA[1], shapePerCTATile[1]);
  int repK = ceil<unsigned>(bTensorTy.getShape()[0],
                            instrShape[2] * kContractingFactor);

  // Flatten accumulator values.
  auto loc = op.getLoc();
  auto fc = unpackLLElements(loc, adaptor.getC(), rewriter);
  int accSize = kMmaAccumulatorCount * (instrShape[1] / kWarpsInGroup);
  assert(fc.size() == repM * repN * accSize);

  // Get warp ID.
  auto wpt = mmaEnc.getWarpsPerCTA();
  Value warp =
      and_(udiv(thread, i32_val(kThreadsPerWarp)), i32_val(0xFFFFFFFC));
  Value warpM = urem(warp, i32_val(wpt[0]));
  Value warpMN = udiv(warp, i32_val(wpt[0]));
  Value warpN = urem(warpMN, i32_val(wpt[1]));

  // Create descriptor.
  auto getSharedData = [&](Value arg, MemDescType tensorTy) {
    auto sharedObj = getSharedMemoryObjectFromStruct(
        loc, arg, typeConverter->convertType(tensorTy.getElementType()),
        rewriter);
    auto sharedLayout = cast<SharedEncodingAttr>(tensorTy.getEncoding());
    auto shape = getShapePerCTA(tensorTy);
    auto ord = sharedLayout.getOrder();
    int byteSize = aTensorTy.getElementTypeBitWidth() / 8;
    int64_t swizzling =
        getSwizzlingFromLayout(sharedLayout, shape[ord[0]] * byteSize);
    Value baseDesc = createDescriptor(rewriter, loc, swizzling, shape[ord[1]]);
    baseDesc =
        add(baseDesc, lshr(ptrtoint(i64_ty, sharedObj.base), int_val(64, 4)));
    return std::make_tuple(shape, ord, baseDesc);
  };

  // Create descriptor for loading A from shared memory.
  auto tA = getSharedData(adaptor.getA(), aTensorTy);
  Value warpA = urem(warpM, i32_val(std::get<0>(tA)[0] / instrShape[0]));
  bool transA = std::get<1>(tA)[0] == 0;
  auto loadA = [&](int m, int k) {
    return smemDescriptor(m, k, rewriter, loc, {instrShape[0], instrShape[2]},
                          transA, wpt[0], warpA, aTensorTy, std::get<2>(tA),
                          std::get<0>(tA)[std::get<1>(tA)[1]]);
  };

  // Create descriptor for loading B from shared memory.
  auto tB = getSharedData(adaptor.getB(), bTensorTy);
  Value warpB = urem(warpN, i32_val(std::get<0>(tB)[1] / instrShape[1]));
  bool transB = std::get<1>(tB)[0] == 1;
  auto loadB = [&](int n, int k) {
    return smemDescriptor(n, k, rewriter, loc,
                          {instrShape[1], instrShape[2] * kContractingFactor},
                          transB, wpt[1], warpB, bTensorTy, std::get<2>(tB),
                          std::get<0>(tB)[std::get<1>(tB)[1]]);
  };

  // Load metadata from shared memory.
  auto hMeta = unpackLLElements(loc, adaptor.getAMeta(), rewriter);
  SmallVector<Value> hMetaPacked;
  for (int i = 0; i < hMeta.size(); i += kCore) {
    Value lower = zext(i32_ty, hMeta[i]);
    Value upper = zext(i32_ty, hMeta[i + 1]);
    Value packed = or_(shl(upper, i32_val(16)), lower);
    hMetaPacked.push_back(packed);
  }
  assert(hMetaPacked.size() == repM * repK);

  // Generate prologue.
  triton::nvgpu::WGMMAEltType eltTypeA = getMmaOperandType(op.getA(), false);
  triton::nvgpu::WGMMAEltType eltTypeB = getMmaOperandType(op.getB(), false);
  triton::nvgpu::WGMMAEltType eltTypeC = getMmaRetType(op.getD());

  triton::nvgpu::WGMMALayout layoutA = transA ? triton::nvgpu::WGMMALayout::col
                                              : triton::nvgpu::WGMMALayout::row;
  triton::nvgpu::WGMMALayout layoutB = transB ? triton::nvgpu::WGMMALayout::row
                                              : triton::nvgpu::WGMMALayout::col;

  rewriter.create<triton::nvgpu::FenceAsyncSharedOp>(loc, 0);
  rewriter.create<triton::nvgpu::WGMMAFenceOp>(loc);

  // Generate main loop.
  for (int m = 0; m < repM; ++m) {
    for (int n = 0; n < repN; ++n) {
      llvm::MutableArrayRef acc(&fc[(m * repN + n) * accSize], accSize);
      auto accTy = LLVM::LLVMStructType::getLiteral(
          op.getContext(), SmallVector<Type>(accSize, f32_ty));
      Value d = packLLElements(loc, typeConverter, acc, rewriter, accTy);
      for (int k = 0; k < repK; ++k) {
        Value a = loadA(m, k);
        Value b = loadB(n, k);
        Value meta = hMetaPacked[k * repM + m];
        d = rewriter.create<triton::nvgpu::SparseWGMMAOp>(
            loc, accTy, a, meta, b, d, kWarpsInGroup * instrShape[0],
            instrShape[1], kContractingFactor * instrShape[2], eltTypeC,
            eltTypeA, eltTypeB, layoutA, layoutB);
      }
      auto res = unpackLLElements(loc, d, rewriter);
      for (int i = 0; i < res.size(); ++i) {
        acc[i] = res[i];
      }
    }
  }

  // Replace with new packed result.
  Type structTy = LLVM::LLVMStructType::getLiteral(
      op.getContext(), SmallVector<Type>(fc.size(), f32_ty));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

  rewriter.create<triton::nvgpu::WGMMACommitGroupOp>(loc);
  res = rewriter.create<triton::nvgpu::WGMMAWaitGroupOp>(loc, res, 0);
  rewriter.replaceOp(op, res);

  return success();
}

// ----- Dispatch based on architecture.

LogicalResult rewriteSparseDotOp(triton::gpu::SparseDotOp op,
                                 triton::gpu::SparseDotOp::Adaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) {
  auto resultTy = cast<RankedTensorType>(op.getResult().getType());
  NvidiaMmaEncodingAttr mmaLayout =
      cast<NvidiaMmaEncodingAttr>(resultTy.getEncoding());

  if (mmaLayout.isAmpere()) {
    return convertSparseMMA(op, adaptor, typeConverter, rewriter);
  }
  if (mmaLayout.isHopper()) {
    return convertSparseWGMMA(op, adaptor, typeConverter, rewriter,
                              getThreadId(rewriter, op.getLoc()));
  }

  llvm::report_fatal_error(
      "Unsupported SparseDotOp found when converting TritonGPU to LLVM.");
}

struct SparseDotOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::SparseDotOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::SparseDotOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::SparseDotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return rewriteSparseDotOp(op, adaptor, getTypeConverter(), rewriter);
  }
};

class SparseLocalLoadOpToLLVMPass
    : public PassWrapper<SparseLocalLoadOpToLLVMPass, OperationPass<ModuleOp>> {
 public:
  SparseLocalLoadOpToLLVMPass() = default;

  StringRef getArgument() const override { return "sparse-local-load-to-llvm"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, mlir::gpu::GPUDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect, mlir::gpu::GPUDialect,
                           arith::ArithDialect>();
    target.addDynamicallyLegalOp<triton::gpu::LocalLoadOp>(
        [](triton::gpu::LocalLoadOp op) {
          return !isa<triton::gpu::SparseDotMetaEncodingAttr>(
              op.getType().getEncoding());
        });
    mlir::LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    auto pattern = std::make_unique<LocalLoadOpConversion>(typeConverter);
    RewritePatternSet patterns(context, std::move(pattern));
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseLocalLoadOpToLLVMPass)
};

class SparseDotOpToLLVMPass
    : public PassWrapper<SparseDotOpToLLVMPass, OperationPass<ModuleOp>> {
 public:
  SparseDotOpToLLVMPass() = default;

  StringRef getArgument() const override { return "sparse-dot-to-llvm"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, mlir::gpu::GPUDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect, mlir::gpu::GPUDialect,
                           arith::ArithDialect, triton::nvgpu::NVGPUDialect>();
    target.addIllegalOp<triton::gpu::SparseDotOp>();
    mlir::LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    auto pattern = std::make_unique<SparseDotOpConversion>(typeConverter);
    RewritePatternSet patterns(context, std::move(pattern));
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseLocalLoadOpToLLVMPass)
};

}  // namespace

std::unique_ptr<Pass> xla::gpu::createAddSparseDotEncodingPass(
    int32_t num_warps, int32_t threads_per_warp, int32_t num_ctas) {
  return std::make_unique<AddSparseDotEncodingPass>(num_warps, threads_per_warp,
                                                    num_ctas);
}

std::unique_ptr<mlir::Pass> xla::gpu::createSparseBlockedToMMAPass() {
  return std::make_unique<SparseBlockedToMMAPass>();
}

std::unique_ptr<mlir::Pass> xla::gpu::createSparseLocalLoadOpToLLVMPass() {
  return std::make_unique<SparseLocalLoadOpToLLVMPass>();
}

std::unique_ptr<mlir::Pass> xla::gpu::createSparseDotOpToLLVMPass() {
  return std::make_unique<SparseDotOpToLLVMPass>();
}

void xla::gpu::registerSparsePasses() {
  registerPass([] { return std::make_unique<AddSparseDotEncodingPass>(); });
  registerPass(createSparseBlockedToMMAPass);
  registerPass(createSparseLocalLoadOpToLLVMPass);
  registerPass(createSparseDotOpToLLVMPass);
}
