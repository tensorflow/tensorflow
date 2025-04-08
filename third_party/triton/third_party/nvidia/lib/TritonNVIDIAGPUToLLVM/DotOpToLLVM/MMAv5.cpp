#include "Dialect/NVGPU/IR/Dialect.h"
#include "MMAHelpers.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::NVIDIA;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::NVMMASharedEncodingAttr;

mlir::triton::NVIDIA::DotOpMmaV5TmemLoader::DotOpMmaV5TmemLoader(
    Value tensor, Value base, SmallVector<unsigned int> instrShape,
    bool interleaved, bool trans)
    : base(base), instrShape(instrShape), interleaved(interleaved),
      trans(trans) {
  auto ty = cast<MemDescType>(tensor.getType());
  auto tmemEncoding =
      cast<nvidia_gpu::TensorMemoryEncodingAttr>(ty.getEncoding());
  unpacked = tmemEncoding.getUnpacked();
  int elTyWidth = ty.getElementTypeBitWidth();
  numElementsPer32b = unpacked ? 1 : 32 / elTyWidth;
  auto shapePerCTA = triton::gpu::getShapePerCTA(ty);
  numRepM = ceil<unsigned>(shapePerCTA[0], instrShape[0]);
}

Value mlir::triton::NVIDIA::DotOpMmaV5TmemLoader::tmemLoad(
    int a, int b, ConversionPatternRewriter &rewriter, Location loc) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  int numRows = 64;
  if (interleaved || instrShape[0] >= 128)
    numRows = 128;
  int numColPerBlock =
      ((instrShape[0] * instrShape[1]) / numRows) / numElementsPer32b;
  Value address = base;
  int blockId = a + b * numRepM;
  address = tb.ptrtoint(i32_ty, address);
  if (!interleaved) {
    address = tb.add(address, tb.i32_val(numColPerBlock * blockId));
  } else {
    int blockIdIsOdd = blockId & 1;
    int blockIdPrevEven = blockId - blockIdIsOdd;
    Value offset = tb.i32_val(numColPerBlock * blockIdPrevEven +
                              ((16 * blockIdIsOdd) << 16));
    address = tb.add(address, offset);
  }
  return address;
}

namespace {

enum class mxfpKind { mxf8f6f4 = 0, mxf4 = 1, mxf4nvf4 = 2 };

inline mxfpKind getMXFPKind(ScaleDotElemType typeA, ScaleDotElemType typeB,
                            Type scaleAType, Type scaleBType) {
  if (typeA == ScaleDotElemType::E2M1 && typeB == ScaleDotElemType::E2M1) {
    if (llvm::isa<Float8E4M3FNType>(scaleAType) &&
        llvm::isa<Float8E4M3FNType>(scaleBType)) {
      return mxfpKind::mxf4nvf4;
    }
    return mxfpKind::mxf4;
  }
  return mxfpKind::mxf8f6f4;
};

static Value createInstDescriptor(ConversionPatternRewriter &rewriter,
                                  triton::nvidia_gpu::TCGen5MMAOp op, int M,
                                  int N, bool transposeA, bool transposeB) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  union TCGen5InstructionDescriptor {
    uint32_t descriptor;
    struct {
      uint32_t sparsitySelector : 2;
      uint32_t sparsity : 1;
      uint32_t : 1;
      uint32_t dType : 2;
      uint32_t : 1;
      uint32_t aType : 3;
      uint32_t bType : 3;
      uint32_t negateA : 1;
      uint32_t negateB : 1;
      uint32_t transposeA : 1;
      uint32_t transposeB : 1;
      uint32_t N : 6;
      uint32_t : 1;
      uint32_t M : 5;
      uint32_t : 1;
      uint32_t shift : 2;
    };
  };
  auto getTypeEncoding = [](Type type) {
    if (type.isF16())
      return 0;
    if (type.isBF16())
      return 1;
    if (type.isF32())
      return 2;
    if (llvm::isa<Float8E4M3FNType>(type))
      return 0;
    if (llvm::isa<Float8E5M2Type>(type))
      return 1;
    llvm_unreachable("Unsupported type.");
  };
  static_assert(sizeof(TCGen5InstructionDescriptor) == 4,
                "instruction descriptor size should be 32 bits.");
  TCGen5InstructionDescriptor desc;
  desc.descriptor = 0;
  desc.transposeA = transposeA;
  desc.transposeB = transposeB;
  desc.M = M >> 4;
  desc.N = N >> 3;
  desc.aType = getTypeEncoding(op.getA().getType().getElementType());
  desc.bType = getTypeEncoding(op.getB().getType().getElementType());
  Type dstElType = op.getD().getType().getElementType();
  assert(dstElType.isF16() || dstElType.isF32());
  desc.dType = dstElType.isF16() ? 0 : 1;
  return b.int_val(32, desc.descriptor);
}

static Value createScaleInstDescriptor(ConversionPatternRewriter &rewriter,
                                       triton::nvidia_gpu::TCGen5MMAScaledOp op,
                                       int M, int N, bool transposeA,
                                       bool transposeB, int scaleFactorsubIdxA,
                                       int scaleFactorsubIdxB,
                                       mxfpKind mxfpInstKind) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  union TCGen5InstructionDescriptor {
    uint32_t descriptor;
    struct {
      uint32_t sparsitySelector : 2;
      uint32_t sparsity : 1;
      uint32_t : 1;
      uint32_t BScaleFactor : 2;
      uint32_t : 1;
      uint32_t aType : 3;
      uint32_t bType : 3;
      uint32_t negateA : 1;
      uint32_t negateB : 1;
      uint32_t transposeA : 1;
      uint32_t transposeB : 1;
      uint32_t N : 6;
      uint32_t scaleType : 1;
      uint32_t M : 5;
      uint32_t AScaleFactor : 2;
      uint32_t : 1;
    };
  };
  auto getTypeEncoding = [](ScaleDotElemType type, bool isMXF4) {
    switch (type) {
    case ScaleDotElemType::E4M3:
      return 0;
    case ScaleDotElemType::E5M2:
      return 1;
    case ScaleDotElemType::E2M3:
      return 3;
    case ScaleDotElemType::E3M2:
      return 4;
    case ScaleDotElemType::E2M1:
      return !isMXF4 ? 5 : 1;
    default:
      break;
    }
    llvm_unreachable("Unsupported type.");
  };
  static_assert(sizeof(TCGen5InstructionDescriptor) == 4,
                "instruction descriptor size should be 32 bits.");
  TCGen5InstructionDescriptor desc;
  desc.descriptor = 0;
  desc.transposeA = transposeA;
  desc.transposeB = transposeB;
  desc.M = M >> 4;
  desc.N = N >> 3;
  desc.aType =
      getTypeEncoding(op.getAType(), mxfpInstKind != mxfpKind::mxf8f6f4);
  desc.bType =
      getTypeEncoding(op.getBType(), mxfpInstKind != mxfpKind::mxf8f6f4);
  desc.AScaleFactor = scaleFactorsubIdxA;
  desc.BScaleFactor = scaleFactorsubIdxB;
  // Hardcoded UE8M0 scale type.
  desc.scaleType = 1;

  if (mxfpInstKind != mxfpKind::mxf8f6f4) {
    assert(desc.aType == 1 && desc.bType == 1);
    assert(desc.AScaleFactor <= 1 && desc.BScaleFactor <= 1);
    assert(desc.transposeA == 0 &&
           "MMAv5 with kind=mxf4 does not support transpose");
    assert(desc.transposeB == 0 &&
           "MMAv5 with kind=mxf4 does not support transpose");
    if (mxfpInstKind == mxfpKind::mxf4) {
      desc.AScaleFactor *= 2;
      desc.BScaleFactor *= 2;
      assert(desc.AScaleFactor == 0 ||
             desc.AScaleFactor == 2 &&
                 "MMAv5 with kind=mxf4 only supports SFA_ID 0 or 2");
      assert(desc.BScaleFactor == 0 ||
             desc.BScaleFactor == 2 &&
                 "MMAv5 with kind=mxf4 only supports SFB_ID 0 or 2");
    } else if (mxfpInstKind == mxfpKind::mxf4nvf4) {
      desc.scaleType = 0; // UE4M3
      assert(desc.AScaleFactor == 0 &&
             "MMAv5 with kind=mxf4nvf4 currently only supports SFA_ID 0");
      assert(desc.BScaleFactor == 0 &&
             "MMAv5 with kind=mxf4nvf4 currently only supports SFB_ID 0");
    }
  }

  return b.int_val(32, desc.descriptor);
}

static void createGen5MMA(ConversionPatternRewriter &rewriter, Location loc,
                          triton::nvidia_gpu::TCGen5MMAOp op, Value a, Value b,
                          Value d, Value pred, Value instDescriptor,
                          Value useInitAcc, bool aInTMem, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  std::string opcode =
      "tcgen05.mma.cta_group::" + std::to_string(twoCTAs ? 2 : 1) + ".kind::";
  Type srcElementTy = op.getA().getType().getElementType();
  if (srcElementTy.isF16() || srcElementTy.isBF16())
    opcode += "f16";
  else if (srcElementTy.isF32())
    opcode += "tf32";
  else if (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(srcElementTy))
    opcode += "f8f6f4";
  else
    assert(0 && "Unsupported type.");
  auto *accOp = ptxBuilder.newAddrOperand(d, "r");
  auto *aOp = aInTMem ? ptxBuilder.newAddrOperand(a, "r")
                      : ptxBuilder.newOperand(a, "l");
  auto *bOp = ptxBuilder.newOperand(b, "l");
  auto *instDescOp = ptxBuilder.newOperand(instDescriptor, "r");
  auto *useInitAccOp = ptxBuilder.newOperand(useInitAcc, "b");
  auto &mmaOp = *ptxBuilder.create<PTXInstr>(opcode);
  mmaOp({accOp, aOp, bOp, instDescOp, useInitAccOp}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createScaledGen5MMA(ConversionPatternRewriter &rewriter,
                                Location loc,
                                triton::nvidia_gpu::TCGen5MMAScaledOp op,
                                Value a, Value b, Value d, Value scaleA,
                                Value scaleB, Value pred, Value instDescriptor,
                                Value useInitAcc, bool aInTmem,
                                mxfpKind mxfpInstKind) {
  PTXBuilder ptxBuilder;
  std::string opcode;
  if (mxfpInstKind == mxfpKind::mxf8f6f4) {
    opcode =
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X";
  } else if (mxfpInstKind == mxfpKind::mxf4) {
    opcode = "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X";
  } else if (mxfpInstKind == mxfpKind::mxf4nvf4) {
    opcode =
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X";
  } else {
    assert(0 && "Unsupported mxfp kind.");
  }
  auto *accOp = ptxBuilder.newAddrOperand(d, "r");
  auto *aOp = aInTmem ? ptxBuilder.newAddrOperand(a, "r")
                      : ptxBuilder.newOperand(a, "l");
  auto *bOp = ptxBuilder.newOperand(b, "l");
  auto *instDescOp = ptxBuilder.newOperand(instDescriptor, "r");
  auto *scaleAOp = ptxBuilder.newAddrOperand(scaleA, "r");
  auto *scaleBOp = ptxBuilder.newAddrOperand(scaleB, "r");
  auto *useInitAccOp = ptxBuilder.newOperand(useInitAcc, "b");
  auto &mmaOp = *ptxBuilder.create<PTXInstr>(opcode);
  mmaOp({accOp, aOp, bOp, instDescOp, scaleAOp, scaleBOp, useInitAccOp})
      .predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createMMACommit(ConversionPatternRewriter &rewriter, Location loc,
                            Value barrier, Value pred, bool twoCTAs = false) {
  PTXBuilder ptxBuilder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<PTXBuilder::Operand *> ptxOperands;
  auto *predOperand = ptxBuilder.newOperand(pred, "b");
  ptxOperands.push_back(predOperand);
  auto *barrierOperand = ptxBuilder.newOperand(barrier, "l");
  ptxOperands.push_back(barrierOperand);
  std::string opcode;
  if (twoCTAs) {
    // .multicast::cluster and mask 0x3 means the completion of UTCMMA.2CTA will
    // be boardcasted into CTAid 0 and 1
    auto *ctaMask = ptxBuilder.newOperand(b.int_val(16, 0x3), "h");
    ptxOperands.push_back(ctaMask);
    opcode = "@$0 "
             "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::"
             "cluster.multicast::cluster.b64 [$1], $2;";
  } else {
    opcode = "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];";
  }
  auto &barrierOp = *ptxBuilder.create<PTXInstr>(opcode);
  barrierOp(ptxOperands, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

void convertDot(const LLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, Location loc,
                triton::nvidia_gpu::TCGen5MMAOp op, Value a, Value b, Value d,
                Value loadedA, Value loadedB, Value loadedD, Value useDFlag,
                Value pred, Value barrier) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  bool twoCTAs = op.getTwoCtas().has_value();
  // Only run mma on one thread. We currently use elect as ptxas is not able to
  // detect that tid.x == 0 is true only for 1 thread.
  Value warpId = rewriter.create<nvgpu::WarpIdOp>(loc);
  Value wapr0 = tb.icmp_eq(warpId, tb.i32_val(0));
  if (twoCTAs) {
    // TODO: we have to sync the two CTAs because we currently don't use remove
    // barriers for the copies.
    rewriter.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::ClusterWaitOp>(loc);

    Value clusterId = rewriter.create<nvgpu::ClusterCTAIdOp>(loc);
    Value cluster0 = tb.icmp_eq(clusterId, tb.i32_val(0));
    pred = tb.and_(pred, cluster0);
  }
  pred = tb.and_(pred, wapr0);

  // Wrap the whole mma code sequence within a IF block.
  auto *curBlock = rewriter.getInsertionBlock();
  auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  auto *mmaBlock = rewriter.createBlock(curBlock->getParent(),
                                        std::next(Region::iterator(curBlock)));
  rewriter.setInsertionPointToEnd(curBlock);
  rewriter.create<LLVM::CondBrOp>(loc, pred, mmaBlock, endBlock);
  // Emit the rest in mmaBlock
  rewriter.setInsertionPointToEnd(mmaBlock);

  pred = LLVM::NVIDIA::createElectPredicate(loc, rewriter);

  auto aTensorTy = cast<MemDescType>(a.getType());
  auto bTensorTy = cast<MemDescType>(b.getType());
  auto dTensorTy = cast<MemDescType>(d.getType());
  bool aInTmem = true;
  bool transA = false;
  if (auto aSharedLayout =
          dyn_cast<NVMMASharedEncodingAttr>(aTensorTy.getEncoding())) {
    transA = aSharedLayout.getTransposed();
    aInTmem = false;
  }
  auto bSharedLayout = cast<NVMMASharedEncodingAttr>(bTensorTy.getEncoding());
  bool transB = !bSharedLayout.getTransposed();
  Value baseA;
  if (aInTmem) {
    baseA = loadedA;
  } else {
    baseA =
        getSharedMemoryObjectFromStruct(
            loc, loadedA,
            typeConverter->convertType(aTensorTy.getElementType()), rewriter)
            .getBase();
  }
  Value baseB =
      getSharedMemoryObjectFromStruct(
          loc, loadedB, typeConverter->convertType(bTensorTy.getElementType()),
          rewriter)
          .getBase();

  SmallVector<int64_t> dstPerCTA = triton::gpu::getShapePerCTA(dTensorTy);
  unsigned int M = dstPerCTA[0];
  unsigned int N = dstPerCTA[1];
  unsigned int K = aTensorTy.getDimSize(1);
  // Get MMA size based on acc layout.
  auto tensorMemAttr = cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      dTensorTy.getEncoding());
  int mmaSizeM = tensorMemAttr.getBlockM();
  int mmaSizeN = tensorMemAttr.getBlockN();
  int mmaSizeK = 256 / aTensorTy.getElementTypeBitWidth();
  int numRepM = ceil<unsigned>(M, mmaSizeM);
  int numRepN = ceil<unsigned>(N, mmaSizeN);
  int numRepK = ceil<unsigned>(K, mmaSizeK);
  assert((!aTensorTy.getElementType().isF32() || !(transA || transB)) &&
         "Currently don't support transpose for F32.");
  bool interleaved = (mmaSizeM == 64 && (numRepM > 1 || numRepN > 1));
  Value instDescriptor =
      createInstDescriptor(rewriter, op, twoCTAs ? mmaSizeM * 2 : mmaSizeM,
                           mmaSizeN, transA, transB);
  Value zero = tb.i32_val(0);
  SmallVector<int64_t> shapeA(triton::gpu::getShapePerCTA(aTensorTy));
  SmallVector<int64_t> shapeB(triton::gpu::getShapePerCTA(bTensorTy));
  SmallVector<unsigned> aOperandShape = {(unsigned)mmaSizeM,
                                         (unsigned)mmaSizeK};
  std::unique_ptr<DotOpMmaMemLoader> aLoader;
  if (aInTmem) {
    aLoader = std::make_unique<DotOpMmaV5TmemLoader>(a, baseA, aOperandShape,
                                                     interleaved, transA);
  } else {
    aLoader = std::make_unique<DotOpMmaV3SmemLoader>(
        a, baseA, shapeA, shapeA, zero, 1, transA, aOperandShape,
        aTensorTy.getElementTypeBitWidth(), rewriter, loc);
  }
  DotOpMmaV3SmemLoader bLoader =
      DotOpMmaV3SmemLoader(b, baseB, shapeB, shapeB, zero, 1, transB,
                           {(unsigned)mmaSizeN, (unsigned)mmaSizeK},
                           bTensorTy.getElementTypeBitWidth(), rewriter, loc);
  DotOpMmaV5TmemLoader dLoader = DotOpMmaV5TmemLoader(
      d, loadedD, {(unsigned)mmaSizeM, (unsigned)mmaSizeN}, interleaved, false);
  for (int m = 0; m < numRepM; m++) {
    for (int n = 0; n < numRepN; n++) {
      Value useInitAcc = useDFlag;
      Value accAddress = dLoader.tmemLoad(m, n, rewriter, loc);
      for (int k = 0; k < numRepK; k++) {
        a = aLoader->memLoad(m, k, rewriter, loc);
        b = bLoader.smemLoad(n, k, rewriter, loc);
        createGen5MMA(rewriter, loc, op, a, b, accAddress, pred, instDescriptor,
                      useInitAcc, aInTmem, twoCTAs);
        useInitAcc = tb.i1_val(1);
      }
    }
  }
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, barrier, i64_ty, rewriter);
  createMMACommit(rewriter, loc, smemObj.getBase(), pred, twoCTAs);
  rewriter.create<LLVM::BrOp>(loc, endBlock);
}

struct TCGen5MMAOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TCGen5MMAOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TCGen5MMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto AEnc = op.getA().getType().getEncoding();
    auto BEnc = op.getB().getType().getEncoding();
    assert(mlir::isa<NVMMASharedEncodingAttr>(AEnc) ||
           mlir::isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(AEnc) &&
               "Operand A should use Shared or Tensor memory layout.");
    assert(mlir::isa<NVMMASharedEncodingAttr>(BEnc) &&
           "Operand B should use Shared layout.");
    assert(op.getBarrier() &&
           "tensorcore op should have a barrier at this point.");
    auto typeConverter = getTypeConverter();
    convertDot(typeConverter, rewriter, op.getLoc(), op, //
               op.getA(), op.getB(), op.getD(),          //
               adaptor.getA(), adaptor.getB(), adaptor.getD(),
               adaptor.getUseD(), adaptor.getPred(), adaptor.getBarrier());
    rewriter.eraseOp(op);
    return success();
  }
};

static int64_t getFormatBitSize(ScaleDotElemType type) {
  switch (type) {
  case ScaleDotElemType::E4M3:
    return 8;
  case ScaleDotElemType::E5M2:
    return 8;
  case ScaleDotElemType::E2M3:
    return 6;
  case ScaleDotElemType::E3M2:
    return 6;
  case ScaleDotElemType::E2M1:
    return 4;
  default:
    llvm_unreachable("Unsupported type.");
  }
}

struct TCGen5MMAScaledOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TCGen5MMAScaledOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TCGen5MMAScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op.getBarrier() &&
           "tensorcore op should have a barrier at this point.");
    auto typeConverter = getTypeConverter();
    Location loc = op.getLoc();
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto aTensorTy = cast<MemDescType>(op.getA().getType());
    auto bTensorTy = cast<MemDescType>(op.getB().getType());
    auto dTensorTy = cast<MemDescType>(op.getD().getType());
    mxfpKind mxfpInstKind = getMXFPKind(
        op.getAType(), op.getBType(), op.getAScale().getType().getElementType(),
        op.getBScale().getType().getElementType());
    bool opKindIsMXFP4 = mxfpInstKind != mxfpKind::mxf8f6f4;
    bool aInTmem = true;
    bool transA = false;
    if (auto aSharedLayout =
            dyn_cast<NVMMASharedEncodingAttr>(aTensorTy.getEncoding())) {
      transA = aSharedLayout.getTransposed();
      aInTmem = false;
    }
    auto bSharedLayout = cast<NVMMASharedEncodingAttr>(bTensorTy.getEncoding());
    bool transB = !bSharedLayout.getTransposed();
    Value baseA =
        getSharedMemoryObjectFromStruct(
            loc, adaptor.getA(),
            typeConverter->convertType(aTensorTy.getElementType()), rewriter)
            .getBase();
    Value baseB =
        getSharedMemoryObjectFromStruct(
            loc, adaptor.getB(),
            typeConverter->convertType(bTensorTy.getElementType()), rewriter)
            .getBase();
    Value baseD = adaptor.getD();
    baseD = tb.ptrtoint(i32_ty, baseD);
    Value baseScaleA = adaptor.getAScale();
    Value baseScaleB = adaptor.getBScale();
    baseScaleA = tb.ptrtoint(i32_ty, baseScaleA);
    baseScaleB = tb.ptrtoint(i32_ty, baseScaleB);

    unsigned int M = dTensorTy.getDimSize(0);
    unsigned int N = dTensorTy.getDimSize(1);
    int numBitsUnpackedPerElementA = opKindIsMXFP4
                                         ? getFormatBitSize(op.getAType())
                                         : aTensorTy.getElementTypeBitWidth();
    int numBitsUnpackedPerElementB = opKindIsMXFP4
                                         ? getFormatBitSize(op.getBType())
                                         : bTensorTy.getElementTypeBitWidth();
    unsigned int K =
        (aTensorTy.getDimSize(1) * 8) / getFormatBitSize(op.getAType());

    // Get MMA size based on acc layout.
    auto tensorMemAttr = cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
        dTensorTy.getEncoding());
    int mmaSizeM = tensorMemAttr.getBlockM();
    int mmaSizeN = tensorMemAttr.getBlockN();
    int mmaSizeK = !opKindIsMXFP4 ? 32 : 64;
    int numRepM = ceil<unsigned>(M, mmaSizeM);
    int numRepN = ceil<unsigned>(N, mmaSizeN);
    int numRepK = ceil<unsigned>(K, mmaSizeK);
    bool interleaved = (mmaSizeM == 64 && (numRepM > 1 || numRepN > 1));

    Value zero = tb.i32_val(0);
    SmallVector<int64_t> shapeA(
        triton::gpu::getAllocationShapePerCTA(aTensorTy));
    SmallVector<int64_t> shapeB(
        triton::gpu::getAllocationShapePerCTA(bTensorTy));
    if (opKindIsMXFP4) {
      shapeA[1] *= 2;
      shapeB[0] *= 2;
    }
    SmallVector<unsigned> aOperandShape = {(unsigned)mmaSizeM,
                                           (unsigned)mmaSizeK};
    std::unique_ptr<DotOpMmaMemLoader> aLoader;
    if (aInTmem) {
      aLoader = std::make_unique<DotOpMmaV5TmemLoader>(
          op.getA(), baseA, aOperandShape, interleaved, transA);
    } else {
      aLoader = std::make_unique<DotOpMmaV3SmemLoader>(
          op.getA(), baseA, shapeA, shapeA, zero, 1, transA, aOperandShape,
          numBitsUnpackedPerElementA, rewriter, loc);
    }
    DotOpMmaV3SmemLoader bLoader =
        DotOpMmaV3SmemLoader(op.getB(), baseB, shapeB, shapeB, zero, 1, transB,
                             {(unsigned)mmaSizeN, (unsigned)mmaSizeK},
                             numBitsUnpackedPerElementB, rewriter, loc);

    // Only run mma on one thread. We currently use elect as ptxas is not able
    // to detect that tid.x == 0 is true only for 1 thread.
    Value pred =
        tb.and_(adaptor.getPred(),
                LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter));
    int numRows = 128;
    int colSizeInBits = 32;
    int numColPerBlock =
        ceil<int>((mmaSizeM * mmaSizeN * dTensorTy.getElementTypeBitWidth()),
                  (numRows * colSizeInBits));

    int scaleFactorColsPerSet = [](mxfpKind kind) {
      switch (kind) {
      case mxfpKind::mxf8f6f4:
        return 1;
      case mxfpKind::mxf4:
        return 2;
      case mxfpKind::mxf4nvf4:
        return 4;
      default:
        llvm_unreachable("Unsupported mxfp kind.");
      }
    }(mxfpInstKind);
    int numColPerScaleBlockA =
        ceil<int>(triton::nvidia_gpu::getTmemAllocSizes(
                      cast<MemDescType>(op.getAScale().getType()))
                      .numCols,
                  numRepM * (ceil<int>(numRepK, 4 / scaleFactorColsPerSet)));
    int numColPerScaleBlockB =
        ceil<int>(triton::nvidia_gpu::getTmemAllocSizes(
                      cast<MemDescType>(op.getBScale().getType()))
                      .numCols,
                  numRepN * (ceil<int>(numRepK, 4 / scaleFactorColsPerSet)));
    for (int m = 0; m < numRepM; m++) {
      for (int n = 0; n < numRepN; n++) {
        // Blocks are laid out along M first then N as described in
        // `TensorMemorySpace` definition.
        int blockId = m + n * numRepM;
        Value accAddress = tb.add(baseD, tb.i32_val(numColPerBlock * blockId));
        Value useInitAcc = op.getUseD();
        for (int k = 0; k < numRepK; k++) {
          Value a = aLoader->memLoad(m, k, rewriter, loc);
          Value b = bLoader.smemLoad(n, k, rewriter, loc);
          int subWordIdx = k % (4 / scaleFactorColsPerSet);
          int wordIdx = k / (4 / scaleFactorColsPerSet);
          Value scaleA = tb.add(baseScaleA, tb.i32_val((m + wordIdx * numRepM) *
                                                       numColPerScaleBlockA));
          Value scaleB = tb.add(baseScaleB, tb.i32_val((n + wordIdx * numRepN) *
                                                       numColPerScaleBlockB));
          Value instDescriptor = createScaleInstDescriptor(
              rewriter, op, mmaSizeM, mmaSizeN, transA, transB, subWordIdx,
              subWordIdx, mxfpInstKind);
          createScaledGen5MMA(rewriter, loc, op, a, b, accAddress, scaleA,
                              scaleB, pred, instDescriptor, useInitAcc, aInTmem,
                              mxfpInstKind);
          useInitAcc = tb.i1_val(1);
        }
      }
    }
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getBarrier(), i64_ty, rewriter);
    createMMACommit(rewriter, loc, smemObj.getBase(), pred);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace mlir {
namespace triton {
namespace NVIDIA {

void populateTCGen5MMAOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit) {
  patterns.add<TCGen5MMAOpConversion, TCGen5MMAScaledOpConversion>(
      typeConverter, benefit);
}

} // namespace NVIDIA
} // namespace triton
} // namespace mlir
