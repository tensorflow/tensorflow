#include "NVGPUToLLVM/NVGPUToLLVMPass.h"

#include "Dialect/NVGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "NVGPUToLLVM/Passes.h.inc"

namespace ttn = mlir::triton::nvgpu;
using ttn::Constraints;
using ttn::OperandsAndConstraints;

namespace {

const std::string kWgmmaFenceOp = "wgmma.fence.sync.aligned;";
const std::string kWgmmaCommitGroupOp = "wgmma.commit_group.sync.aligned;";
const std::string kClusterWaitOp = "barrier.cluster.wait.aligned;";
const std::string kFenceMbarrierInitOp = "fence.mbarrier_init.release.cluster;";
const std::string kClusterCtaIdOp = "{\n"
                                    ".reg .u32 a<5>;              \n"
                                    "mov.u32 a0, %cluster_ctaid.x;\n"  // x
                                    "mov.u32 a1, %cluster_ctaid.y;\n"  // y
                                    "mov.u32 a2, %cluster_ctaid.z;\n"  // z
                                    "mov.u32 a3, %cluster_nctaid.x;\n" // nx
                                    "mov.u32 a4, %cluster_nctaid.y;\n" // ny
                                    "mad.lo.u32 a1, a2, a4, a1;     \n"
                                    "mad.lo.u32 $0, a1, a3, a0;     \n"
                                    "}";

bool isNumber(const std::string &s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

Type getTypeFromConstraint(char constraint, PatternRewriter &rewriter) {
  Type ty;
  if (constraint == 'b')
    ty = IntegerType::get(rewriter.getContext(), 1);
  else if (constraint == 'h')
    ty = IntegerType::get(rewriter.getContext(), 16);
  else if (constraint == 'r')
    ty = IntegerType::get(rewriter.getContext(), 32);
  else if (constraint == 'l')
    ty = IntegerType::get(rewriter.getContext(), 64);
  else if (constraint == 'f')
    ty = Float32Type::get(rewriter.getContext());
  else if (constraint == 'd')
    ty = Float64Type::get(rewriter.getContext());
  else {
    assert(false && "Unsupported constraint");
  }
  return ty;
}

// Converts the given value to the type represented by the constraint
// E.g. if val is of type llvmptr and constraint is 'r', then we convert
// val to i32 using ptrtoint(i32_ty, val)
Value convertToType(Value val, std::string constraint, Location loc,
                    PatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto isConstraintNumber = isNumber(constraint);
  if (!isConstraintNumber) {
    auto ty = getTypeFromConstraint(constraint[0], rewriter);
    if (isa<LLVM::LLVMPointerType>(val.getType())) {
      return b.ptrtoint(ty, val);
    } else {
      assert(val.getType().getIntOrFloatBitWidth() <=
                 ty.getIntOrFloatBitWidth() &&
             "Cannot convert to a smaller type");
      if (val.getType().getIntOrFloatBitWidth() < ty.getIntOrFloatBitWidth())
        return b.zext(ty, val);
    }
  }
  return val;
}

SmallVector<PTXBuilder::Operand *>
getPtxOutputs(const nvgpu::Constraints &outputConstraints,
              PTXBuilder &ptxBuilder) {
  SmallVector<PTXBuilder::Operand *> ptxOutputs;
  for (unsigned i = 0; i < outputConstraints.size(); i++) {
    auto *ptxOutput = ptxBuilder.newOperand(outputConstraints[i]);
    ptxOutputs.push_back(ptxOutput);
  }
  return ptxOutputs;
}

OperandsAndConstraints
unpackOperands(const OperandsAndConstraints &operandsAndConstraints,
               PTXBuilder &ptxBuilder, Location loc,
               PatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  OperandsAndConstraints unpackedOperands;
  for (const auto &[operand, constraint] : operandsAndConstraints) {
    auto llvmStruct = llvm::dyn_cast<LLVM::LLVMStructType>(operand.getType());
    // if a constraint is a number, then we are doing input/output tying
    // if the operand is a struct, then we need to unpack it, and
    // add the constraint to each of the unpacked operands uses the constraint
    // as an offset
    auto isConstraintNumber = isNumber(constraint);
    if (llvmStruct) {
      for (unsigned i = 0; i < llvmStruct.getBody().size(); i++) {
        if (isConstraintNumber) {
          auto constraintInt = std::stoi(constraint) + i;
          unpackedOperands.push_back(
              {b.extract_val(llvmStruct.getBody()[i], operand, i),
               std::to_string(constraintInt)});
        } else {
          unpackedOperands.push_back(
              {b.extract_val(llvmStruct.getBody()[i], operand, i), constraint});
        }
      }
    } else {
      unpackedOperands.push_back({operand, constraint});
    }
  }
  return unpackedOperands;
}

SmallVector<PTXBuilder::Operand *>
getPtxOperands(const OperandsAndConstraints &operandsAndConstraints,
               PTXBuilder &ptxBuilder, Location loc,
               PatternRewriter &rewriter) {
  SmallVector<PTXBuilder::Operand *> ptxOperands;
  auto unpackedOperandsAndConstraints =
      unpackOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
  for (auto &[operand, constraint] : unpackedOperandsAndConstraints) {
    auto convertedOperand = convertToType(operand, constraint, loc, rewriter);
    auto *ptxOperand = ptxBuilder.newOperand(convertedOperand, constraint);
    ptxOperands.push_back(ptxOperand);
  }
  return ptxOperands;
}

std::string patchPtxAsm(Operation *op, std::string ptxAsm) {
  std::vector<std::pair<int, int>> patchLocations;
  std::vector<std::string> patchValues;
  auto start = ptxAsm.find("#", 0);
  while (start != std::string::npos) {
    auto endIterator =
        std::find_if(ptxAsm.begin() + start + 1, ptxAsm.end(),
                     [](unsigned char c) { return !std::isalnum(c); });

    assert(endIterator != ptxAsm.end() && "unexpected asm format");

    auto end = std::distance(ptxAsm.begin(), endIterator);
    auto patchLocation = std::make_pair(start, end);
    patchLocations.push_back(patchLocation);
    auto patchValue = ptxAsm.substr(start + 1, end - start - 1);
    patchValues.push_back(patchValue);
    start = ptxAsm.find("#", end);
  }
  assert(patchLocations.size() == patchValues.size() &&
         "patchLocations and patchValues should have the same size");
  if (patchLocations.size() == 0) {
    return ptxAsm;
  }
  std::string res = "";
  size_t prevStart = 0;
  unsigned i = 0;
  for (auto &[start, end] : patchLocations) {
    res += ptxAsm.substr(prevStart, start - prevStart);
    auto integerAttr = op->getAttrOfType<IntegerAttr>(patchValues[i]);
    auto attr = integerAttr.getInt();
    res += std::to_string(attr);
    prevStart = end;
    i++;
  }
  if (prevStart < ptxAsm.size())
    res += ptxAsm.substr(prevStart, ptxAsm.size() - prevStart);
  return res;
}

template <typename SourceOp>
class NVGPUOpGenericPattern : public OpRewritePattern<SourceOp> {
public:
  explicit NVGPUOpGenericPattern(MLIRContext *context, std::string ptxAsm,
                                 Constraints outputConstraints,
                                 Constraints inputConstraints)
      : OpRewritePattern<SourceOp>(context), ptxAsm(std::move(ptxAsm)),
        outputConstraints(outputConstraints),
        inputConstraints(inputConstraints) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    OperandsAndConstraints operandsAndConstraints;
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
      operandsAndConstraints.push_back(
          {op->getOperand(i), inputConstraints[i]});
    }
    return rewriteAsPtxAsm(op, rewriter, ptxAsm, operandsAndConstraints,
                           outputConstraints);
  }

private:
  std::string ptxAsm;
  Constraints outputConstraints;
  Constraints inputConstraints;
};

class FenceAsyncSharedOpPattern
    : public OpRewritePattern<ttn::FenceAsyncSharedOp> {
public:
  using OpRewritePattern<ttn::FenceAsyncSharedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::FenceAsyncSharedOp op,
                                PatternRewriter &rewriter) const override {
    std::string ptxAsm = op.getBCluster() ? "fence.proxy.async.shared::cluster;"
                                          : "fence.proxy.async.shared::cta;";
    return rewriteAsPtxAsm(op, rewriter, std::move(ptxAsm));
  }
};

class WarpIdOpPattern : public OpRewritePattern<ttn::WarpIdOp> {
public:
  using OpRewritePattern<ttn::WarpIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::WarpIdOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // If this is inside a warp specialize op, compute the relative thread ID
    // within the warp group.
    Value tid = rewriter.create<NVVM::ThreadIdXOp>(loc, i32_ty);
    if (std::optional<int> startId =
            getWarpGroupStartThreadId(rewriter.getInsertionBlock()))
      tid = rewriter.create<LLVM::SubOp>(loc, tid, b.i32_val(*startId));

    Value warpId = b.udiv(tid, b.i32_val(32));
    // This indicates to PTXAS that the result and its derived values are
    // uniform across the warp. For example, if a branch condition derives from
    // this value, it can be proven to be non-divergent.
    warpId = LLVM::NVIDIA::shuffleIdx(loc, rewriter, warpId, 0);
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

class ClusterArriveOpPattern : public OpRewritePattern<ttn::ClusterArriveOp> {
public:
  using OpRewritePattern<ttn::ClusterArriveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::ClusterArriveOp op,
                                PatternRewriter &rewriter) const override {
    std::string ptxAsm = op.getRelaxed()
                             ? "barrier.cluster.arrive.relaxed.aligned;"
                             : "barrier.cluster.arrive.aligned;";
    return rewriteAsPtxAsm(op, rewriter, std::move(ptxAsm));
  }
};

// Base class for Matrix Operation Patterns
template <typename MatrixOpType, typename ConcreteMatrixOpPattern>
class MatrixOpPattern : public OpRewritePattern<MatrixOpType> {
public:
  using OpRewritePattern<MatrixOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatrixOpType op,
                                PatternRewriter &rewriter) const override {
    unsigned vecSize = getVectorSize(op);
    bool trans = op.getTrans();
    // Template method for PTX assembly generation
    std::string ptxAsm =
        (llvm::Twine(ConcreteMatrixOpPattern::kOpCode) +
         getPtxModifiers(vecSize, trans) + " " + getOperands(op, vecSize) + ";")
            .str();

    OperandsAndConstraints operandAndConstraints =
        getOperandsAndConstraints(op, vecSize);
    Constraints outputConstraints = getOutputConstraints(op, vecSize);

    return rewriteAsPtxAsm(op, rewriter, ptxAsm, operandAndConstraints,
                           outputConstraints);
  }

protected:
  // Shared helper methods
  std::string getPtxModifiers(unsigned vecSize, bool trans) const {
    auto ptxAsmBase = llvm::Twine(".sync.aligned.m8n8");
    const std::string suffix = trans ? ".trans.shared.b16" : ".shared.b16";
    switch (vecSize) {
    case 1:
      return (ptxAsmBase + ".x1" + suffix).str();
    case 2:
      return (ptxAsmBase + ".x2" + suffix).str();
    case 4:
      return (ptxAsmBase + ".x4" + suffix).str();
    default:
      llvm_unreachable("Invalid vector size");
    }
  }

  std::string getPtxRegOperands(unsigned startIdx, unsigned count) const {
    llvm::SmallString<20> regOperands;
    llvm::raw_svector_ostream stream(regOperands);
    stream << "{";
    for (unsigned i = 0; i < count; i++) {
      stream << "$" + llvm::utostr(startIdx + i);
      if (i != count - 1)
        stream << ", ";
    }
    stream << "}";
    return std::string(regOperands.str());
  }

  std::string getPtxAddrOperand(unsigned idx) const {
    return (llvm::Twine("[$") + llvm::utostr(idx) + "]").str();
  }

  virtual std::string getOperands(MatrixOpType op, unsigned vecSize) const = 0;
  virtual OperandsAndConstraints
  getOperandsAndConstraints(MatrixOpType op, unsigned vecSize) const = 0;
  virtual Constraints getOutputConstraints(MatrixOpType op,
                                           unsigned vecSize) const = 0;
  virtual unsigned getVectorSize(MatrixOpType op) const = 0;
};

// StoreMatrixOp Pattern
class StoreMatrixOpPattern
    : public MatrixOpPattern<ttn::StoreMatrixOp, StoreMatrixOpPattern> {
public:
  using MatrixOpPattern<ttn::StoreMatrixOp,
                        StoreMatrixOpPattern>::MatrixOpPattern;
  static constexpr const char *kOpCode = "stmatrix";

protected:
  unsigned getVectorSize(ttn::StoreMatrixOp op) const override {
    return op.getVals().size();
  }

  std::string getOperands(ttn::StoreMatrixOp op,
                          unsigned vecSize) const override {
    return (llvm::Twine(getPtxAddrOperand(0)) + ", " +
            getPtxRegOperands(1, vecSize))
        .str();
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::StoreMatrixOp op,
                            unsigned vecSize) const override {
    OperandsAndConstraints constraints = {{op.getAddr(), "r"}};
    for (unsigned i = 0; i < vecSize; i++) {
      constraints.push_back({op.getVals()[i], "r"});
    }
    return constraints;
  }

  Constraints getOutputConstraints(ttn::StoreMatrixOp op,
                                   unsigned vecSize) const override {
    return {}; // No output constraints for StoreMatrixOp
  }
};

// LoadMatrixOp Pattern
class LoadMatrixOpPattern
    : public MatrixOpPattern<ttn::LoadMatrixOp, LoadMatrixOpPattern> {
public:
  using MatrixOpPattern<ttn::LoadMatrixOp,
                        LoadMatrixOpPattern>::MatrixOpPattern;
  static constexpr const char *kOpCode = "ldmatrix";

protected:
  unsigned getVectorSize(ttn::LoadMatrixOp op) const override {
    auto resultType = cast<LLVM::LLVMStructType>(op.getType());
    return resultType.getBody().size();
  }

  std::string getOperands(ttn::LoadMatrixOp op,
                          unsigned vecSize) const override {
    return (llvm::Twine(getPtxRegOperands(0, vecSize)) + ", " +
            getPtxAddrOperand(vecSize))
        .str();
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::LoadMatrixOp op,
                            unsigned vecSize) const override {
    return {{op.getAddr(), "r"}};
  }

  Constraints getOutputConstraints(ttn::LoadMatrixOp op,
                                   unsigned vecSize) const override {
    return Constraints(vecSize, "=r");
  }
};

class LoadAcquireOpPattern : public OpRewritePattern<ttn::LoadAcquireOp> {
public:
  using OpRewritePattern<ttn::LoadAcquireOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::LoadAcquireOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type valueTy = op.getType();
    const unsigned valueNBits = std::max(8u, valueTy.getIntOrFloatBitWidth());
    const size_t maxWordWidth = std::max<size_t>(32, valueNBits);
    const size_t width = std::min((size_t)valueNBits, maxWordWidth);

    const std::string writeConstraint =
        (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");
    PTXBuilder ptxBuilder;
    bool init = true;
    auto *dstOpr = ptxBuilder.newOperand(writeConstraint, init); // =r operation
    auto *addrOpr =
        ptxBuilder.newAddrOperand(op.getAddr(), "l", 0 /* in_off */);
    auto &ld =
        ptxBuilder.create<>("ld")
            ->global()
            .o("cta", op.getScope() == triton::nvgpu::MemSyncScope::CTA)
            .o("gpu", op.getScope() == triton::nvgpu::MemSyncScope::GPU)
            .o("sys", op.getScope() == triton::nvgpu::MemSyncScope::SYSTEM)
            .o("acquire", op.getSem() == triton::nvgpu::MemSemantic::ACQUIRE)
            .o("relaxed", op.getSem() == triton::nvgpu::MemSemantic::RELAXED)
            .b(width);
    ld(dstOpr, addrOpr).maybePredicate(op.getMask(), "b");

    // Create inline ASM signature
    Type retTy = IntegerType::get(getContext(), width);
    Value ret = ptxBuilder.launch(rewriter, loc, retTy);
    ret = b.bitcast(ret, op.getType());

    rewriter.replaceOp(op, {ret});
    return success();
  }
};

class WGMMAWaitGroupOpPattern : public OpRewritePattern<ttn::WGMMAWaitGroupOp> {
public:
  using OpRewritePattern<ttn::WGMMAWaitGroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::WGMMAWaitGroupOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteAsPtxAsm(op, rewriter, getPtxAsm(op),
                           getOperandsAndConstraints(op),
                           getOutputConstraints(op));
  }

  Constraints getOutputConstraints(ttn::WGMMAWaitGroupOp op) const {
    auto outputStructType = cast<LLVM::LLVMStructType>(op.getType());
    std::vector<std::string> outputConstraints;
    outputConstraints.reserve(outputStructType.getBody().size());
    for (mlir::Type type : outputStructType.getBody()) {
      if (type.isF32()) {
        outputConstraints.push_back("=f");
        continue;
      } else if (type.isF64()) {
        outputConstraints.push_back("=d");
        continue;
      }
      unsigned bitwidth = isa<LLVM::LLVMPointerType>(type) ?
          64 : type.getIntOrFloatBitWidth();
      switch (bitwidth) {
        case 1:
          outputConstraints.push_back("=b");
          break;
        case 16:
          outputConstraints.push_back("=h");
          break;
        case 32:
          outputConstraints.push_back("=r");
          break;
        case 64:
          outputConstraints.push_back("=l");
          break;
        default:
          assert(false && "unsupported bitwidth");
      }
    }
    return outputConstraints;
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::WGMMAWaitGroupOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto input = op.getInput();
    operandsAndConstraints.push_back({input, "0"});
    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::WGMMAWaitGroupOp op) const {
    auto outputStructType = dyn_cast<LLVM::LLVMStructType>(op.getType());
    uint32_t numCRegs = outputStructType.getBody().size();
    std::string args = "";
    uint32_t asmOpIdx = 0;
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    auto ptxAsm = "// wait for regs: " + args + "\n\t" +
                  "wgmma.wait_group.sync.aligned #pendings;";
    return ptxAsm;
  }
};

class WGMMAOpPattern : public OpRewritePattern<ttn::WGMMAOp> {
public:
  using OpRewritePattern<ttn::WGMMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::WGMMAOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteAsPtxAsm(op, rewriter, getPtxAsm(op),
                           getOperandsAndConstraints(op),
                           getOutputConstraints(op));
  }

  std::vector<std::string> getOutputConstraints(ttn::WGMMAOp op) const {
    // TODO (zahi): Return type must always be a struct for wgmma, currently
    // we rely on the size of output constraints vector to determine whether
    // the output is a struct or not. We should find a way to pass this info
    auto resultType = op.getType();

    auto outputStructType = dyn_cast<LLVM::LLVMStructType>(resultType);
    uint32_t numOutputRegs = outputStructType.getBody().size();
    std::string output =
        outputStructType.getBody().front().isF32() ? "=f" : "=r";
    return std::vector<std::string>(numOutputRegs, output);
  }

  OperandsAndConstraints getOperandsAndConstraints(ttn::WGMMAOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto opA = op.getOpA();
    auto opB = op.getOpB();
    auto opC = op.getOpC();
    auto opScaleD = op.getUseC();
    auto typeA = opA.getType();

    auto structTypeA = dyn_cast<LLVM::LLVMStructType>(typeA);

    // TODO (zahi): is this the best way to tie inputs/outputs ?
    if (opC)
      operandsAndConstraints.push_back({opC, "0"});

    if (structTypeA) {
      operandsAndConstraints.push_back({opA, "r"});
    } else {
      operandsAndConstraints.push_back({opA, "l"});
    }

    // Operand B (must be `desc`)
    operandsAndConstraints.push_back({opB, "l"});

    // `scale-d`
    if (op.getOpC())
      operandsAndConstraints.push_back({opScaleD, "b"});

    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::WGMMAOp op) const {
    using namespace ttn;
    auto opA = op.getOpA();
    auto opB = op.getOpB();
    auto m = op.getM();
    auto n = op.getN();
    auto k = op.getK();
    auto eltTypeC = op.getEltTypeC();
    auto eltTypeA = op.getEltTypeA();
    auto eltTypeB = op.getEltTypeB();
    auto layoutA = op.getLayoutA();
    auto layoutB = op.getLayoutB();

    // Register checks
    auto typeA = opA.getType();
    auto typeB = opB.getType();
    auto typeOutput = op.getType();
    auto structTypeA = dyn_cast<LLVM::LLVMStructType>(typeA);
    auto structTypeB = dyn_cast<LLVM::LLVMStructType>(typeB);
    auto structTypeOutput = dyn_cast<LLVM::LLVMStructType>(typeOutput);
    assert(!structTypeB && "Operand B can not be registers");
    assert(structTypeOutput && "Output and C operand must be registers");

    // Element type, MNK shape and transposing support check
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
    bool transA = layoutA == WGMMALayout::col;
    bool transB = layoutB == WGMMALayout::row;
    bool supported = false, needTransArgs = false, floatTypeWGMMA = false;
    assert(m % 8 == 0 && n % 8 == 0 && k % 8 == 0);
    // Below instructions do support transposing, must pass `trans` arguments
    supported |=
        (eltTypeA == WGMMAEltType::f16) && (eltTypeB == WGMMAEltType::f16) &&
        (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
        (m == 64 && 8 <= n && n <= 256 && k == 16);
    supported |= (eltTypeA == WGMMAEltType::bf16) &&
                 (eltTypeB == WGMMAEltType::bf16) &&
                 (eltTypeC == WGMMAEltType::f32) &&
                 (m == 64 && 8 <= n && n <= 256 && k == 16);
    needTransArgs = supported;
    floatTypeWGMMA = supported;
    // Below instructions do not support transposing
    if (!supported && !transA && !transB) {
      supported |= (eltTypeA == WGMMAEltType::tf32) &&
                   (eltTypeB == WGMMAEltType::tf32) &&
                   (eltTypeC == WGMMAEltType::f32) &&
                   (m == 64 && 8 <= n && n <= 256 && k == 8);
      supported |=
          (eltTypeA == WGMMAEltType::e4m3 || eltTypeA == WGMMAEltType::e5m2) &&
          (eltTypeB == WGMMAEltType::e4m3 || eltTypeB == WGMMAEltType::e5m2) &&
          (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
          (m == 64 && 8 <= n && n <= 256 && k == 32);
      floatTypeWGMMA = supported;
      // Below instructions are integer-based
      supported |= (eltTypeA == WGMMAEltType::s8) &&
                   (eltTypeB == WGMMAEltType::s8) &&
                   (eltTypeC == WGMMAEltType::s32) &&
                   (m == 64 && 8 <= n && n <= 224 && k == 32);
    }
    assert(supported && "WGMMA type or shape is not supported");

    // Operands
    uint32_t asmOpIdx = 0;
    std::string args = "";

    // Output and operand C
    uint32_t numCRegs = structTypeOutput.getBody().size();

    args += "{";
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    args += "}, ";

    if (op.getOpC())
      asmOpIdx += numCRegs;

    // Operand A
    if (structTypeA) {
      uint32_t numARegs = structTypeA.getBody().size();
      args += "{";
      for (uint32_t i = 0; i < numARegs; ++i) {
        args +=
            "$" + std::to_string(asmOpIdx++) + (i == numARegs - 1 ? "" : ",");
      }
      args += "}, ";
    } else {
      args += "$" + std::to_string(asmOpIdx++) + ", ";
    }

    // Operand B (must be `desc`)
    args += "$" + std::to_string(asmOpIdx++) + ", ";

    // `scale-d`
    if (op.getOpC())
      args += "$" + std::to_string(asmOpIdx++);
    else
      args += "0";

    // `imm-scale-a`, and `imm-scale-b` are 1 by default only for float-based
    // WGMMA
    if (floatTypeWGMMA)
      args += ", 1, 1";

    // Push `trans-a` and `trans-b` args if needed (determined as constant)
    if (needTransArgs) {
      if (!structTypeA)
        args += ", " + std::to_string(transA);
      args += ", " + std::to_string(transB);
    }

    auto ptxAsm = "wgmma.mma_async.sync.aligned"
                  ".m" +
                  std::to_string(m) + "n" + std::to_string(n) + "k" +
                  std::to_string(k) + "." + stringifyEnum(eltTypeC).str() +
                  "." + stringifyEnum(eltTypeA).str() + "." +
                  stringifyEnum(eltTypeB).str() + " " + args + ";";
    return ptxAsm;
  }
};

static Value createTMAlloc(IRRewriter &rewriter, LLVM::LLVMFuncOp func,
                           size_t size, Value pred, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  Location loc = func.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value sharedMem = mlir::LLVM::getStackPointer(rewriter, func);
  std::string ptxString =
      "@$0 tcgen05.alloc.cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
      ".sync.aligned.shared::cta.b32 [$1], " + std::to_string(size) + ";";

  auto &allocOp = *ptxBuilder.create<>(ptxString);
  allocOp(
      {ptxBuilder.newOperand(pred, "b"), ptxBuilder.newOperand(sharedMem, "r")},
      /*onlyAttachMLIRArgs=*/true);
  auto voidTy = void_ty(func->getContext());
  ptxBuilder.launch(rewriter, loc, void_ty(func->getContext()));
  rewriter.create<NVVM::Barrier0Op>(loc);
  Value address = b.load(i32_ty, sharedMem);
  rewriter.create<NVVM::Barrier0Op>(loc);
  address = b.inttoptr(ptr_ty(func.getContext(), 6), address);
  return address;
}

static void createRelinquishAlloc(IRRewriter &rewriter, Location loc,
                                  Value pred, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  std::string ptxString = "@$0 tcgen05.relinquish_alloc_permit.cta_group::" +
                          std::to_string(twoCTAs ? 2 : 1) + ".sync.aligned;";
  auto &f = *ptxBuilder.create<>(ptxString);
  f({ptxBuilder.newOperand(pred, "b")}, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

void freeTMAlloc(LLVM::LLVMFuncOp func, Value alloc, size_t size, Value pred,
                 bool twoCTAs) {
  func.walk([&](LLVM::ReturnOp ret) {
    OpBuilder b(ret);
    auto ctx = ret->getContext();
    auto loc = ret.getLoc();
    auto voidTy = void_ty(ctx);
    PTXBuilder ptxBuilder;
    // Calculate the predicate in the inline asm to avoid creating long
    // liveranges.
    std::string ptxString =
        "@$0 tcgen05.dealloc.cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
        ".sync.aligned.b32 $1, " + std::to_string(size) + ";";
    auto &dealloc = *ptxBuilder.create<>(ptxString);
    dealloc(
        {ptxBuilder.newOperand(pred, "b"), ptxBuilder.newOperand(alloc, "r")},
        /*onlyAttachMLIRArgs=*/true);
    ptxBuilder.launch(b, loc, void_ty(ctx));
  });
}

static Value initTensorMemory(LLVM::LLVMFuncOp func) {
  auto mod = func->getParentOfType<ModuleOp>();
  assert(mod->hasAttr("ttg.tensor_memory_size"));
  size_t size = cast<IntegerAttr>(mod->getAttr("ttg.tensor_memory_size"))
                    .getValue()
                    .getZExtValue();
  if (size == 0)
    return Value();
  IRRewriter rewriter(func.getContext());
  rewriter.setInsertionPointToStart(&func.front());
  auto ctx = mod.getContext();
  auto loc = func.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // A proper error will be raised by the frontend, but to allow compilation to
  // continue we emit a trap.
  if (size > 512) {
    rewriter.create<LLVM::Trap>(loc);
    return rewriter.create<LLVM::UndefOp>(loc, ptr_ty(ctx, 6));
  }

  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
  // Assume that 2CTAs is used if we have two CTAs this is pessimistic but
  // should be fine for now.
  bool useTwoCTAs = numCTAs == 2;
  // This code is only executed by the default warp group.
  Value threadId = rewriter.create<NVVM::ThreadIdXOp>(loc, i32_ty);
  Value pred = b.icmp_ult(threadId, b.i32_val(32));
  Value alloc = createTMAlloc(rewriter, func, size, pred, useTwoCTAs);
  createRelinquishAlloc(rewriter, loc, pred, useTwoCTAs);
  // TODO: pred will have a long liverange, we need to check if this is a
  // problem and how it can be fixed.
  freeTMAlloc(func, alloc, size, pred, useTwoCTAs);
  return alloc;
}

static void lowerTensorMemoryAlloc(ModuleOp mod) {
  SmallVector<Operation *> baseOps;
  LLVM::LLVMFuncOp kernel = nullptr;
  mod.walk([&](ttn::TensorMemoryBaseAddress baseOp) {
    baseOps.push_back(baseOp);
    if (!kernel)
      kernel = baseOp->getParentOfType<LLVM::LLVMFuncOp>();
    assert(kernel == baseOp->getParentOfType<LLVM::LLVMFuncOp>() &&
           "TODO: add support for function calls using tmem.");
  });
  if (baseOps.empty())
    return;
  // TODO: Handle cases of matmul used in noinline functions.
  assert(LLVM::isKernel(kernel));
  Value newBase = initTensorMemory(kernel);
  if (!newBase)
    return;
  for (auto baseOp : baseOps) {
    baseOp->getResult(0).replaceAllUsesWith(newBase);
    baseOp->erase();
  }
}

class ConvertNVGPUToLLVM : public ConvertNVGPUToLLVMBase<ConvertNVGPUToLLVM> {

public:
  explicit ConvertNVGPUToLLVM() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

#define POPULATE_NVGPU_OP(SRC_OP, ASM)                                         \
  patterns.add<NVGPUOpGenericPattern<SRC_OP>>(context, ASM, Constraints(),     \
                                              Constraints());
    POPULATE_NVGPU_OP(ttn::WGMMAFenceOp, kWgmmaFenceOp)
    POPULATE_NVGPU_OP(ttn::WGMMACommitGroupOp, kWgmmaCommitGroupOp)
    POPULATE_NVGPU_OP(ttn::ClusterWaitOp, kClusterWaitOp)
#undef POPULATE_NVGPU_OP
    patterns.add<NVGPUOpGenericPattern<ttn::ClusterCTAIdOp>>(
        context, kClusterCtaIdOp, Constraints({"=r"}), Constraints());

    patterns
        .add<FenceAsyncSharedOpPattern, LoadMatrixOpPattern,
             StoreMatrixOpPattern, ClusterArriveOpPattern, WGMMAOpPattern,
             LoadAcquireOpPattern, WGMMAWaitGroupOpPattern, WarpIdOpPattern>(
            context);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();

    lowerTensorMemoryAlloc(mod);
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

LogicalResult
nvgpu::rewriteAsPtxAsm(Operation *op, PatternRewriter &rewriter,
                       std::string ptxAsm,
                       const OperandsAndConstraints &operandsAndConstraints,
                       const Constraints &outputConstraints) {
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  ptxAsm = patchPtxAsm(op, std::move(ptxAsm));
  auto hasSideEffects = !isMemoryEffectFree(op);

  PTXBuilder ptxBuilder;
  auto ptxOutputs = getPtxOutputs(outputConstraints, ptxBuilder);
  auto ptxOperands =
      getPtxOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
  SmallVector<PTXBuilder::Operand *> outputsAndOperands = ptxOutputs;
  outputsAndOperands.append(ptxOperands.begin(), ptxOperands.end());
  auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
  ptxInstr(outputsAndOperands, /*onlyAttachMLIRArgs=*/true);
  auto retTy =
      op->getNumResults() == 0 ? void_ty(ctx) : op->getResult(0).getType();
  auto res = ptxBuilder.launch(rewriter, loc, retTy,
                               /*hasSideEffects*/ hasSideEffects);
  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
  } else {
    rewriter.replaceOp(op, res);
  }

  return success();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertNVGPUToLLVMPass() {
  return std::make_unique<::ConvertNVGPUToLLVM>();
}

} // namespace triton
} // namespace mlir
