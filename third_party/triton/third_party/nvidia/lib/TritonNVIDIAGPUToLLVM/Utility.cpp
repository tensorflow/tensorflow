#include "Utility.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir {
namespace LLVM {
namespace NVIDIA {
using namespace mlir::triton;

static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter, Value val,
                               Value i, NVVM::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl(loc, rewriter, val0, i, mode, clamp);
    val1 = shuffleCommonImpl(loc, rewriter, val1, i, mode, clamp);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }
  Type type = val.getType();
  if (type != i32_ty) {
    val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.zext(i32_ty, val);
  }
  Value mask = b.i32_val(0xFFFFFFFF);
  Value result = rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, val, i, clamp,
                                               mode, UnitAttr());
  if (type != i32_ty) {
    if (bits < 32)
      result = b.trunc(int_ty(bits), result);
    result = b.bitcast(result, type);
  }
  return result;
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, NVVM::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, val, i, mode, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), NVVM::ShflKind::bfly,
                       b.i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), NVVM::ShflKind::up,
                       b.i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, i, NVVM::ShflKind::idx,
                       b.i32_val(0x1f));
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);

  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  std::string sreg = numCTAs == 1 ? "ctaid." : "clusterid.";
  sreg.append(1, 'x' + axis); // 0 -> 'x', 1 -> 'y', 2 -> 'z'
  return getSRegValue(rewriter, loc, sreg);
}

Value getSRegValue(OpBuilder &rewriter, Location loc, StringRef sRegStr) {
  ValueRange args;
  auto intrName = Twine("llvm.nvvm.read.ptx.sreg.") + sRegStr;
  auto callOp =
      createLLVMIntrinsicCallOp(rewriter, loc, intrName.str(), i32_ty, args);
  return callOp.getResult(0);
}

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value mask) {
  Value args[] = {a, b, mask};
  auto op =
      createLLVMIntrinsicCallOp(rewriter, loc, "llvm.nvvm.prmt", i32_ty, args);
  return op.getResult(0);
}

/// Create a predicate with just single active thread.
Value createElectPredicate(Location loc, RewriterBase &rewriter) {
  return rewriter.create<NVVM::ElectSyncOp>(loc, i1_ty);
}

void createSyncWarp(Location loc, OpBuilder &rewriter) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Type resultTy = void_ty(rewriter.getContext());
  Value args[] = {b.i32_val(0xffffffff)};
  createLLVMIntrinsicCallOp(rewriter, loc, "llvm.nvvm.bar.warp.sync", resultTy,
                            args);
}

Value createElectPredicateWarp0(Location loc, RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value threadId = getThreadId(rewriter, loc);
  Value warp0 = b.icmp_ult(threadId, b.i32_val(32));
  return b.and_(warp0, createElectPredicate(loc, rewriter));
}

} // namespace NVIDIA
} // namespace LLVM
} // namespace mlir
