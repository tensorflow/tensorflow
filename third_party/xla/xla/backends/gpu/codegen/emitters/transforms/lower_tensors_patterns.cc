/* Copyright 2026 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/emitters/transforms/lower_tensors_patterns.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAggregateOps.h"
#include "mlir/Dialect/LLVMIR/LLVMArithOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMCallOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConversionOps.h"
#include "mlir/Dialect/LLVMIR/LLVMMemoryOps.h"
#include "mlir/Dialect/LLVMIR/LLVMModuleOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialectDecl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/atomic_rmw_utils.h"
#include "xla/codegen/emitters/transforms/lower_tensors_common.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

using ::mlir::failure;
using ::mlir::Location;
using ::mlir::LogicalResult;
using ::mlir::OpBuilder;
using ::mlir::OpRewritePattern;
using ::mlir::SmallVector;
using ::mlir::success;
using ::mlir::Type;
using ::mlir::TypeRange;
using ::mlir::UnrealizedConversionCastOp;
using ::mlir::Value;
using ::mlir::ValueRange;

namespace arith = ::mlir::arith;
namespace scf = ::mlir::scf;
namespace ml = ::mlir::LLVM;
namespace se = ::stream_executor;
namespace vector = ::mlir::vector;

constexpr int kGPUSharedMemoryAddrSpace = 3;

Value GetPtr(Value value) {
  if (!mlir::isa<ml::LLVMPointerType>(value.getType())) {
    if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast.getNumOperands() == 1 &&
          mlir::isa<ml::LLVMPointerType>(cast.getOperand(0).getType())) {
        return cast.getOperand(0);
      }
    }
    return nullptr;
  }
  return value;
}

bool IsSharedMemoryBuffer(Value dest) {
  Value root = emitters::TraceDestinationBuffer(dest);
  if (!root) {
    return false;
  }
  if (root.getDefiningOp<AllocateSharedOp>() != nullptr) {
    return true;
  }
  Value ptr = GetPtr(root);
  llvm::SmallPtrSet<Value, 8> visited_ptrs;
  while (ptr) {
    if (!visited_ptrs.insert(ptr).second) {
      break;
    }
    auto ptr_ty = mlir::dyn_cast<ml::LLVMPointerType>(ptr.getType());
    if (ptr_ty && ptr_ty.getAddressSpace() == kGPUSharedMemoryAddrSpace) {
      return true;
    }
    if (auto cast = ptr.getDefiningOp<ml::AddrSpaceCastOp>()) {
      ptr = cast.getArg();
      continue;
    }
    if (auto gep = ptr.getDefiningOp<ml::GEPOp>()) {
      ptr = gep.getBase();
      continue;
    }
    break;
  }
  return false;
}

struct RewriteAllocateShared : OpRewritePattern<AllocateSharedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AllocateSharedOp op, mlir::PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto shaped_ty = mlir::cast<mlir::ShapedType>(op.getResult().getType());
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ml::GlobalOp global = emitters::CreateGlobalOp(
        mlir::Attribute{}, "shared_", shaped_ty, module,
        /*is_constant=*/false, kGPUSharedMemoryAddrSpace, b);
    // Shared memory exists only on GPU, where an atomic update of an element
    // narrower than 32 bits (every insert of an element narrower than a byte,
    // and any narrow AtomicRMWOp) becomes a 32 bit CAS on the containing word,
    // found with ptrmask. The masked pointer must stay inside this object, so
    // the allocation is word aligned and padded to a whole number of words.
    // This also keeps the CAS from touching bytes of a neighboring allocation.
    // Wider types keep their natural alignment.
    Type element_type = shaped_ty.getElementType();
    if (element_type.isIntOrFloat() &&
        element_type.getIntOrFloatBitWidth() < 32) {
      auto array_ty = mlir::cast<ml::LLVMArrayType>(global.getGlobalType());
      int64_t elem_bytes = CeilOfRatio<int64_t>(
          array_ty.getElementType().getIntOrFloatBitWidth(), 8);
      int64_t padded =
          RoundUpTo<int64_t>(array_ty.getNumElements() * elem_bytes, 4) /
          elem_bytes;
      global.setGlobalTypeAttr(mlir::TypeAttr::get(
          ml::LLVMArrayType::get(array_ty.getElementType(), padded)));
      global.setAlignment(4);
    }

    rewriter.setInsertionPoint(op);
    auto addr = ml::AddressOfOp::create(rewriter, op.getLoc(), global);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getResult().getType(),
        ml::AddrSpaceCastOp::create(rewriter, op.getLoc(),
                                    ml::LLVMPointerType::get(op.getContext()),
                                    addr)
            .getResult());
    return success();
  }
};

struct RewriteSyncThreads : OpRewritePattern<SyncThreadsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      SyncThreadsOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::gpu::BarrierOp::create(rewriter, op.getLoc());
    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};

llvm::StringRef GetAtomicSyncScope(const se::GpuComputeCapability& gpu_cc) {
  if (gpu_cc.IsRocm()) {
    return llvm::StringRef("agent-one-as");
  }
  return llvm::StringRef();
}

// AMDGPU SIMemoryLegalizer only applies agent scope L1 cache bypass when
// ordering is at least monotonic. CUDA and Intel GPU use unordered so atomic
// loads remain Ref only in LLVM alias analysis.
ml::AtomicOrdering GetAtomicLoadStoreOrdering(
    const se::GpuComputeCapability& gpu_cc) {
  return gpu_cc.IsRocm() ? ml::AtomicOrdering::monotonic
                         : ml::AtomicOrdering::unordered;
}

void SetAmdAtomicRMWAttrs(mlir::OpBuilder& b, ml::AtomicRMWOp rmw_op,
                          const se::GpuComputeCapability& gpu_cc) {
  if (!gpu_cc.IsRocm()) {
    return;
  }
  mlir::UnitAttr unit_attr = b.getUnitAttr();
  rmw_op->setDiscardableAttr(
      mlir::ROCDL::ROCDLDialect::NoRemoteMemoryAttrHelper::getNameStr(),
      unit_attr);
  rmw_op->setDiscardableAttr(
      mlir::ROCDL::ROCDLDialect::NoFineGrainedMemoryAttrHelper::getNameStr(),
      unit_attr);
}

bool IsGpuAtomicIntegral(Type element_type) {
  if (!element_type.isSignlessInteger()) {
    return false;
  }
  unsigned bit_width = element_type.getIntOrFloatBitWidth();
  return bit_width == 32 || bit_width == 64;
}

class RewriteGpuDirectAtomicStore : public OpRewritePattern<AtomicStoreOp> {
 public:
  RewriteGpuDirectAtomicStore(mlir::MLIRContext* context,
                              const se::GpuComputeCapability& gpu_cc)
      : OpRewritePattern<AtomicStoreOp>(context, /*benefit=*/2),
        gpu_cc_(gpu_cc) {}

  LogicalResult matchAndRewrite(
      AtomicStoreOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getIndices().size() > 1 || op.getDest() == op.getResult()) {
      return rewriter.notifyMatchFailure(
          op, "only acyclic 0D and 1D tensors are supported");
    }
    if (!emitters::TraceDestinationBuffer(op.getDest())) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot trace the destination buffer");
    }

    Location loc = op.getLoc();
    Value value = op.getValue();
    Type element_type = value.getType();
    llvm::StringRef sync_scope = GetAtomicSyncScope(gpu_cc_);
    ml::AtomicOrdering direct_ordering = GetAtomicLoadStoreOrdering(gpu_cc_);

    if (auto complex_ty = mlir::dyn_cast<mlir::ComplexType>(element_type)) {
      // The 64 bit store needs 8 byte alignment, but complex<f32> only has the
      // 4 byte ABI alignment of f32.
      if (complex_ty.getElementType().isF32() &&
          emitters::GetGuaranteedByteAlignment(op.getDest(), op.getIndices(),
                                               op) >= 8) {
        mlir::ImplicitLocOpBuilder b(loc, rewriter);
        Value addr =
            emitters::CreateAtomicGep(op.getDest(), op.getIndices(), b);
        Value store_value = emitters::CreateBitcast(b, value, b.getI64Type());
        ml::StoreOp::create(rewriter, loc, store_value, addr,
                            /*alignment=*/8,
                            /*isVolatile=*/false, /*isNonTemporal=*/false,
                            /*isInvariantGroup=*/false, direct_ordering,
                            sync_scope);
        rewriter.replaceOp(op, op.getDest());
        return success();
      }
      return rewriter.notifyMatchFailure(
          op, "complex atomic_store requires CAS fallback");
    }

    if (element_type.isF32() || element_type.isF64() ||
        IsGpuAtomicIntegral(element_type)) {
      mlir::ImplicitLocOpBuilder b(loc, rewriter);
      Value addr = emitters::CreateAtomicGep(op.getDest(), op.getIndices(), b);
      Value store_value = value;
      // LLVM's SPIRV backend only declares Int64Atomics on OpAtomicStore for
      // 64 bit integer operands, not 64 bit floats.
      if (gpu_cc_.IsOneAPI() && element_type.isF64()) {
        store_value = arith::BitcastOp::create(b, b.getI64Type(), store_value);
      }
      ml::StoreOp::create(
          rewriter, loc, store_value, addr,
          /*alignment=*/element_type.getIntOrFloatBitWidth() / 8,
          /*isVolatile=*/false, /*isNonTemporal=*/false,
          /*isInvariantGroup=*/false, direct_ordering, sync_scope);
      rewriter.replaceOp(op, op.getDest());
      return success();
    }

    return rewriter.notifyMatchFailure(
        op, "element type requires shared atomicCAS fallback");
  }

 private:
  se::GpuComputeCapability gpu_cc_;
};

// Certain computations, such as float addition and integer maximization, can
// be implemented directly with an LLVM atomic instruction.
class RewriteGpuDirectAtomicRMW : public OpRewritePattern<AtomicRMWOp> {
 public:
  // Uses `benefit = 1` so `SimplifyAtomicRMWBody` (3) and
  // `ConvertInvariantAtomicRMWToStore` (2) run first, while shared vector
  // scalarization / CAS fallback (`PatternBenefit(0)`) runs only when direct
  // hardware atomic emission declines.
  RewriteGpuDirectAtomicRMW(mlir::MLIRContext* context,
                            const se::GpuComputeCapability& gpu_cc)
      : OpRewritePattern<AtomicRMWOp>(context, /*benefit=*/1),
        gpu_cc_(gpu_cc) {}

  LogicalResult matchAndRewrite(
      AtomicRMWOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getIndices().size() > 1 || op.getInput() == op.getResult()) {
      return rewriter.notifyMatchFailure(
          op, "only acyclic 0D and 1D tensors are supported");
    }
    if (!emitters::TraceDestinationBuffer(op.getInput())) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot trace the destination buffer");
    }
    std::optional<emitters::AtomicModifierParameters> modifier_parameters =
        emitters::GetAtomicModifierParameters(op);
    if (!modifier_parameters.has_value()) {
      return rewriter.notifyMatchFailure(
          op, "atomic_rmw body is not a direct hardware modifier");
    }
    if (failed(rewriteAsDirectAtomicRMW(op, *modifier_parameters, rewriter))) {
      return rewriter.notifyMatchFailure(
          op, "target does not support direct hardware atomic_rmw");
    }
    rewriter.replaceOp(op, op.getInput());
    return success();
  }

 private:
  LogicalResult rewriteAsDirectAtomicRMW(
      AtomicRMWOp op,
      const emitters::AtomicModifierParameters& modifier_parameters,
      mlir::PatternRewriter& rewriter) const {
    Type element_type = op.getCurrentValue().getType();
    Location loc = op.getLoc();
    llvm::StringRef sync_scope = GetAtomicSyncScope(gpu_cc_);
    mlir::ImplicitLocOpBuilder b(loc, rewriter);

    auto materialize_addr_and_modifier = [&]() -> std::pair<Value, Value> {
      Value addr = emitters::CreateAtomicGep(op.getInput(), op.getIndices(), b);
      Value modifier_arg = emitters::MaterializeAtomicModifier(
          rewriter, loc, modifier_parameters.modifier);
      return {addr, modifier_arg};
    };

    if (std::optional<ml::AtomicBinOp> atomic_bin_op =
            emitters::GetIntegralAtomicBinOp(modifier_parameters.kind)) {
      if (!IsGpuAtomicIntegral(element_type)) {
        return failure();
      }
      auto [addr, modifier_arg] = materialize_addr_and_modifier();
      auto rmw_op = ml::AtomicRMWOp::create(
          rewriter, loc, *atomic_bin_op, addr, modifier_arg,
          ml::AtomicOrdering::monotonic, sync_scope);
      SetAmdAtomicRMWAttrs(rewriter, rmw_op, gpu_cc_);
      return success();
    }

    switch (modifier_parameters.kind) {
      case emitters::AtomicRMWKind::kFAdd:
      case emitters::AtomicRMWKind::kFSub: {
        bool negated =
            (modifier_parameters.kind == emitters::AtomicRMWKind::kFSub);
        if (const auto* rocm_cc = gpu_cc_.rocm_compute_capability()) {
          return emitAMDAtomicFAdd(loc, op, modifier_parameters.modifier,
                                   element_type, *rocm_cc, rewriter, negated);
        }
        if (gpu_cc_.IsOneAPI()) {
          return emitSPIRVAtomicFAdd(loc, op, modifier_parameters.modifier,
                                     element_type, rewriter, negated);
        }
        return emitNVidiaAtomicFAdd(
            loc, op, modifier_parameters.modifier, element_type,
            *gpu_cc_.cuda_compute_capability(), rewriter, negated);
      }
      case emitters::AtomicRMWKind::kFMax: {
        if (!element_type.isF32()) {
          return failure();
        }
        auto [addr, modifier_arg] = materialize_addr_and_modifier();
        rewriteAtomicFMaxAsIntAtomics(loc, modifier_arg, addr, sync_scope,
                                      rewriter);
        return success();
      }
      default:
        return failure();
    }
  }

  LogicalResult emitNVidiaAtomicFAdd(
      Location loc, AtomicRMWOp op, mlir::OpFoldResult modifier,
      Type element_type,
      const se::CudaComputeCapability& cuda_compute_capability, OpBuilder& b,
      bool negated) const {
    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(element_type)) {
      return emitNvidiaVectorizedAtomicFAdd(
          loc, op, modifier, vector_type, cuda_compute_capability, b, negated);
    }
    // "atom.add.f64 requires sm_60 or higher."
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
    bool is_supported_f16_atomic =
        element_type.isF16() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::kVolta);
    bool is_supported_bf16_atomic =
        element_type.isBF16() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::kHopper);
    bool is_supported_f64_atomic =
        element_type.isF64() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::kPascal);
    if (!element_type.isF32() && !is_supported_f16_atomic &&
        !is_supported_bf16_atomic && !is_supported_f64_atomic) {
      return failure();
    }
    mlir::ImplicitLocOpBuilder implicit_b(loc, b);
    Value addr =
        emitters::CreateAtomicGep(op.getInput(), op.getIndices(), implicit_b);
    Value modifier_arg = emitters::MaterializeAtomicModifier(b, loc, modifier);
    if (negated) {
      modifier_arg = arith::NegFOp::create(b, loc, modifier_arg);
    }

    auto rmw_op =
        ml::AtomicRMWOp::create(b, loc, ml::AtomicBinOp::fadd, addr,
                                modifier_arg, ml::AtomicOrdering::monotonic);
    rmw_op.setIgnoreDenormalMode(true);
    return success();
  }

  LogicalResult emitNvidiaVectorizedAtomicFAdd(
      Location loc, AtomicRMWOp op, mlir::OpFoldResult modifier,
      mlir::VectorType vector_type,
      const se::CudaComputeCapability& cuda_compute_capability, OpBuilder& b,
      bool negated) const {
    // Hopper supports 2 element and 4 element vectors of f32 in global memory.
    if (!(vector_type.getElementType().isF32() &&
          (vector_type.getNumElements() == 2 ||
           vector_type.getNumElements() == 4) &&
          cuda_compute_capability.IsAtLeast(
              se::CudaComputeCapability::kHopper)) ||
        IsSharedMemoryBuffer(op.getInput())) {
      return failure();
    }
    mlir::ImplicitLocOpBuilder implicit_b(loc, b);
    Value addr =
        emitters::CreateAtomicGep(op.getInput(), op.getIndices(), implicit_b);
    Value modifier_arg = emitters::MaterializeAtomicModifier(b, loc, modifier);
    if (negated) {
      modifier_arg = arith::NegFOp::create(b, loc, modifier_arg);
    }

    // TODO(https://github.com/llvm/llvm-project/issues/122760): Switch to
    // AtomicRMWOp once the bug is fixed. Currently vectorized AtomicRMWOp
    // lowers incorrectly to PTX due to a bug in NVPTX.  We use inline asm to
    // work around this.
    auto asm_dialect_attr =
        ml::AsmDialectAttr::get(b.getContext(), ml::AsmDialect::AD_ATT);
    std::string asm_string;
    std::string constraints;
    if (vector_type.getNumElements() == 2) {
      asm_string = "atom.global.v2.f32.add {$0, $1}, [$2], {$3, $4};";
      constraints = "=f,=f,l,f,f";
    } else if (vector_type.getNumElements() == 4) {
      asm_string =
          "atom.global.v4.f32.add {$0, $1, $2, $3}, [$4], {$5, $6, $7, $8};";
      constraints = "=f,=f,=f,=f,l,f,f,f,f";
    }
    SmallVector<Value> asm_operands{addr};
    for (int i = 0; i < vector_type.getNumElements(); ++i) {
      asm_operands.push_back(vector::ExtractOp::create(
          b, loc, modifier_arg,
          arith::ConstantIndexOp::create(b, loc, i).getResult()));
    }
    SmallVector<Type> output_types(vector_type.getNumElements(),
                                   vector_type.getElementType());
    auto output_type =
        ml::LLVMStructType::getLiteral(b.getContext(), output_types);
    ml::InlineAsmOp::create(b, loc, output_type, asm_operands, asm_string,
                            constraints,
                            /*has_side_effects=*/true,
                            /*is_align_stack=*/true, ml::TailCallKind::None,
                            /*convergent=*/false, asm_dialect_attr,
                            /*operand_attrs=*/mlir::ArrayAttr());
    return success();
  }

  LogicalResult emitAMDAtomicFAdd(
      Location loc, AtomicRMWOp op, mlir::OpFoldResult modifier,
      Type element_type,
      const se::RocmComputeCapability& rocm_compute_capability, OpBuilder& b,
      bool negated) const {
    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(element_type)) {
      bool supported_v2f16 =
          vector_type.getElementType().isF16() &&
          rocm_compute_capability.has_packed_fp16_atomics_support();
      bool supported_v2bf16 =
          vector_type.getElementType().isBF16() &&
          rocm_compute_capability.has_packed_bf16_atomics_support();
      if (vector_type.getNumElements() != 2 ||
          (!supported_v2f16 && !supported_v2bf16) ||
          IsSharedMemoryBuffer(op.getInput()) ||
          emitters::GetGuaranteedByteAlignment(op.getInput(), op.getIndices(),
                                               op) < 4) {
        return failure();
      }
    } else if (!element_type.isF32() && !element_type.isF16() &&
               !element_type.isBF16() && !element_type.isF64()) {
      return failure();
    } else if ((element_type.isF16() || element_type.isBF16()) &&
               IsSharedMemoryBuffer(op.getInput())) {
      return failure();
    }
    mlir::ImplicitLocOpBuilder implicit_b(loc, b);
    Value addr =
        emitters::CreateAtomicGep(op.getInput(), op.getIndices(), implicit_b);
    Value modifier_arg = emitters::MaterializeAtomicModifier(b, loc, modifier);
    if (negated) {
      modifier_arg = arith::NegFOp::create(b, loc, modifier_arg);
    }

    if ((element_type.isF16() &&
         rocm_compute_capability.has_packed_fp16_atomics_support()) ||
        (element_type.isBF16() &&
         rocm_compute_capability.has_packed_bf16_atomics_support())) {
      auto packed_type = mlir::VectorType::get({2}, element_type);
      auto i64_type = b.getI64Type();
      auto i32_type = b.getI32Type();
      auto i16_type = b.getI16Type();
      Value addr_int = ml::PtrToIntOp::create(b, loc, i64_type, addr);
      Value addr_masked = ml::AndOp::create(
          b, loc, addr_int, ml::ConstantOp::create(b, loc, i64_type, -4));

      Value offset = ml::AndOp::create(
          b, loc, ml::TruncOp::create(b, loc, i32_type, addr_int),
          ml::ConstantOp::create(b, loc, i32_type, 2));

      Value shift = ml::MulOp::create(
          b, loc, offset, ml::ConstantOp::create(b, loc, i32_type, 8));

      Value modifier_int =
          ml::BitcastOp::create(b, loc, i16_type, modifier_arg);

      Value modifier_masked = ml::ShlOp::create(
          b, loc, ml::ZExtOp::create(b, loc, i32_type, modifier_int), shift);

      constexpr int kGlobalMemory = 1;
      addr = ml::IntToPtrOp::create(
          b, loc, ml::LLVMPointerType::get(b.getContext(), kGlobalMemory),
          addr_masked);

      modifier_arg =
          ml::BitcastOp::create(b, loc, packed_type, modifier_masked);
    }

    auto rmw_op = ml::AtomicRMWOp::create(
        b, loc, ml::AtomicBinOp::fadd, addr, modifier_arg,
        ml::AtomicOrdering::monotonic, GetAtomicSyncScope(gpu_cc_));
    rmw_op.setIgnoreDenormalMode(true);
    SetAmdAtomicRMWAttrs(b, rmw_op, gpu_cc_);

    return success();
  }

  LogicalResult emitSPIRVAtomicFAdd(Location loc, AtomicRMWOp op,
                                    mlir::OpFoldResult modifier,
                                    Type element_type, OpBuilder& b,
                                    bool negated) const {
    if (mlir::isa<mlir::VectorType>(element_type)) {
      return failure();
    }
    // TODO(intel-tf): Expand support to 16 bit floating precisions on supported
    // platforms.
    if (!element_type.isF32() && !element_type.isF64()) {
      return failure();
    }
    mlir::ImplicitLocOpBuilder implicit_b(loc, b);
    Value addr =
        emitters::CreateAtomicGep(op.getInput(), op.getIndices(), implicit_b);
    Value modifier_arg = emitters::MaterializeAtomicModifier(b, loc, modifier);
    if (negated) {
      modifier_arg = arith::NegFOp::create(b, loc, modifier_arg);
    }

    auto rmw_op =
        ml::AtomicRMWOp::create(b, loc, ml::AtomicBinOp::fadd, addr,
                                modifier_arg, ml::AtomicOrdering::monotonic);
    rmw_op.setIgnoreDenormalMode(true);
    return success();
  }

  void rewriteAtomicFMaxAsIntAtomics(Location loc, Value modifier_arg,
                                     Value addr, llvm::StringRef sync_scope,
                                     OpBuilder& b) const {
    Type element_type = modifier_arg.getType();
    CHECK(element_type.isF32());
    // Evaluating floating max using integer atomics has the limitation of not
    // propagating -NaNs. To handle this, we check if the update value is -NaN
    // and convert it to a positive one by dropping the sign-bit.
    Value current = emitters::CreateAtomicLoad(
        b, loc, element_type, addr, GetAtomicLoadStoreOrdering(gpu_cc_),
        sync_scope);

    Value current_is_nan =
        ml::FCmpOp::create(b, loc, ml::FCmpPredicate::uno, current, current);
    auto is_current_nan =
        scf::IfOp::create(b, loc, /*resultTypes=*/TypeRange{}, current_is_nan,
                          /*addThenBlock=*/true, /*addElseBlock=*/true);
    auto if_current_nan_then_builder =
        OpBuilder::atBlockEnd(is_current_nan.thenBlock(), b.getListener());
    scf::YieldOp::create(if_current_nan_then_builder, loc);

    auto if_current_nan_else_builder =
        OpBuilder::atBlockEnd(is_current_nan.elseBlock(), b.getListener());
    Value is_modifier_nan =
        ml::FCmpOp::create(if_current_nan_else_builder, loc,
                           ml::FCmpPredicate::uno, modifier_arg, modifier_arg);
    auto f32_nan = mlir::APFloat::getNaN(mlir::APFloat::IEEEsingle());
    Value nan = ml::ConstantOp::create(if_current_nan_else_builder, loc,
                                       b.getF32Type(), f32_nan);
    Value no_negative_nan_source = ml::SelectOp::create(
        if_current_nan_else_builder, loc, is_modifier_nan, nan, modifier_arg);
    Value current_less_than_modifier = ml::FCmpOp::create(
        if_current_nan_else_builder, loc, ml::FCmpPredicate::ult, current,
        no_negative_nan_source);

    // This check allows us to skip the atomic update all-together at the
    // expense of reading the value in memory for every update.
    auto if_need_update = scf::IfOp::create(if_current_nan_else_builder, loc,
                                            /*resultTypes=*/TypeRange{},
                                            current_less_than_modifier,
                                            /*withElseRegion=*/true,
                                            /*addElseBlock=*/false);
    scf::YieldOp::create(if_current_nan_else_builder, loc);

    auto then_builder =
        OpBuilder::atBlockEnd(if_need_update.thenBlock(), b.getListener());
    Value source_float_as_int = arith::BitcastOp::create(
        then_builder, loc, then_builder.getI32Type(), no_negative_nan_source);
    Value c0 = ml::ConstantOp::create(then_builder, loc, b.getI32Type(), 0);
    Value is_not_negative = ml::ICmpOp::create(
        then_builder, loc, ml::ICmpPredicate::sge, source_float_as_int, c0);
    scf::IfOp::create(
        then_builder, loc, is_not_negative,
        [&](OpBuilder& nested_b, Location nested_loc) {
          // atomicMax((int *)address, __float_as_int(val))
          auto rmw_op = ml::AtomicRMWOp::create(
              nested_b, loc, ml::AtomicBinOp::max, addr, source_float_as_int,
              ml::AtomicOrdering::monotonic, sync_scope);
          SetAmdAtomicRMWAttrs(nested_b, rmw_op, gpu_cc_);
          scf::YieldOp::create(nested_b, nested_loc);
        },
        [&](OpBuilder& nested_b, Location nested_loc) {
          // atomicMin((unsigned int *)address, __float_as_uint(val))
          auto rmw_op = ml::AtomicRMWOp::create(
              nested_b, loc, ml::AtomicBinOp::umin, addr, source_float_as_int,
              ml::AtomicOrdering::monotonic, sync_scope);
          SetAmdAtomicRMWAttrs(nested_b, rmw_op, gpu_cc_);
          scf::YieldOp::create(nested_b, nested_loc);
        });
    scf::YieldOp::create(then_builder, loc);
  }

  se::GpuComputeCapability gpu_cc_;
};

class RewriteGpuSetDynamicDimSize
    : public OpRewritePattern<SetDynamicDimSizeOp> {
 public:
  RewriteGpuSetDynamicDimSize(mlir::MLIRContext* context,
                              const se::GpuComputeCapability& gpu_cc)
      : OpRewritePattern<SetDynamicDimSizeOp>(context, /*benefit=*/2),
        load_ordering_(GetAtomicLoadStoreOrdering(gpu_cc)),
        sync_scope_(GetAtomicSyncScope(gpu_cc)) {}

  LogicalResult matchAndRewrite(
      SetDynamicDimSizeOp op, mlir::PatternRewriter& rewriter) const override {
    if (static_cast<int64_t>(op.getDim()) != 0) {
      return rewriter.notifyMatchFailure(op, "only dim 0 can start mid-word");
    }
    auto tensor_type =
        mlir::cast<mlir::RankedTensorType>(op.getTensor().getType());
    std::optional<int> sub_byte_width =
        emitters::GetSubByteBitWidth(tensor_type.getElementType());
    if (!sub_byte_width) {
      return rewriter.notifyMatchFailure(op, "element type is byte-sized");
    }
    int64_t packed_bytes =
        CeilOfRatio<int64_t>(tensor_type.getNumElements(), 8 / *sub_byte_width);
    int64_t misalign = packed_bytes % 4;
    if (misalign == 0) {
      return rewriter.notifyMatchFailure(op, "dim 0 metadata is word-aligned");
    }
    Value base_tensor = emitters::TraceDestinationBuffer(op.getTensor());
    if (!base_tensor) {
      return rewriter.notifyMatchFailure(
          op, "deferring untraceable buffer error to fallback pattern");
    }

    // On GPU, sub-byte writes use 32-bit cmpxchg on the enclosing aligned
    // word. When dim 0 starts mid-word, update the overlapping aligned word
    // with cmpxchg so concurrent tail writes cannot clobber the low bytes of
    // the size, then write only the remaining bytes in the next word.
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value tensor_ptr =
        UnrealizedConversionCastOp::create(
            b, ml::LLVMPointerType::get(b.getContext()), base_tensor)
            .getResult(0);
    Type i32_type = b.getI32Type();
    int64_t aligned_offset = packed_bytes - misalign;
    int low_bits = static_cast<int>(misalign * 8);
    uint32_t low_mask_val = (uint32_t{1} << low_bits) - 1;

    Value aligned_addr =
        emitters::AddByteOffsetToPointer(b, tensor_ptr, aligned_offset);
    Value low_mask =
        ml::ConstantOp::create(b, i32_type, static_cast<int32_t>(low_mask_val));
    Value low_shift = ml::ConstantOp::create(b, i32_type, low_bits);
    Value shifted_size = ml::ShlOp::create(b, op.getSize(), low_shift);
    Value initial = emitters::CreateAtomicLoad(
        b, op.getLoc(), i32_type, aligned_addr, load_ordering_, sync_scope_);

    scf::WhileOp::create(
        b, TypeRange{i32_type}, ValueRange{initial},
        [&](mlir::OpBuilder& builder, Location loc, ValueRange values) {
          mlir::ImplicitLocOpBuilder loop_b(loc, builder);
          Value old_value = values[0];
          Value preserved = ml::AndOp::create(loop_b, old_value, low_mask);
          Value new_value = ml::OrOp::create(loop_b, preserved, shifted_size);
          Value cmpxchg = ml::AtomicCmpXchgOp::create(
              loop_b, loc, aligned_addr, old_value, new_value,
              /*success_ordering=*/ml::AtomicOrdering::monotonic,
              /*failure_ordering=*/ml::AtomicOrdering::monotonic, sync_scope_);
          Value next = ml::ExtractValueOp::create(loop_b, cmpxchg, 0);
          Value ok = ml::ExtractValueOp::create(loop_b, cmpxchg, 1);
          Value low_bit = ml::ConstantOp::create(
              loop_b, loop_b.getOneAttr(loop_b.getI1Type()));
          Value not_ok = ml::XOrOp::create(loop_b, ok, low_bit);
          scf::ConditionOp::create(loop_b, not_ok, ValueRange{next});
        },
        [&](mlir::OpBuilder& builder, Location loc, ValueRange values) {
          scf::YieldOp::create(builder, loc, values);
        });

    Value tail_addr =
        emitters::AddByteOffsetToPointer(b, tensor_ptr, aligned_offset + 4);
    int high_shift_bits = 32 - low_bits;
    Value high_shifted = ml::LShrOp::create(
        b, op.getSize(), ml::ConstantOp::create(b, i32_type, high_shift_bits));
    if (misalign == 1) {
      Value byte_val = ml::TruncOp::create(b, b.getI8Type(), high_shifted);
      ml::StoreOp::create(b, byte_val, tail_addr, /*alignment=*/1);
    } else if (misalign == 2) {
      Value i16_val = ml::TruncOp::create(b, b.getI16Type(), high_shifted);
      ml::StoreOp::create(b, i16_val, tail_addr, /*alignment=*/2);
    } else {
      Value low_i16 = ml::TruncOp::create(b, b.getI16Type(), high_shifted);
      ml::StoreOp::create(b, low_i16, tail_addr, /*alignment=*/2);
      Value top_byte = ml::TruncOp::create(
          b, b.getI8Type(),
          ml::LShrOp::create(b, high_shifted,
                             ml::ConstantOp::create(b, i32_type, 16)));
      Value top_addr =
          emitters::AddByteOffsetToPointer(b, tensor_ptr, aligned_offset + 6);
      ml::StoreOp::create(b, top_byte, top_addr, /*alignment=*/1);
    }

    rewriter.replaceOp(op, op.getTensor());
    return success();
  }

 private:
  ml::AtomicOrdering load_ordering_;
  llvm::StringRef sync_scope_;
};

}  // namespace

LogicalResult LowerGpuTensors(mlir::ModuleOp module,
                              const se::DeviceDescription& device_info) {
  mlir::MLIRContext* context = module.getContext();
  mlir::RewritePatternSet alloc_patterns(context);
  alloc_patterns.add<RewriteAllocateShared>(context);
  mlir::walkAndApplyPatterns(module, std::move(alloc_patterns));

  emitters::LowerTensorsConfig config;
  config.allow_mixed_tensor_results = false;
  config.atomic_sub_byte_inserts = true;
  config.min_atomic_bit_width = 32;
  config.max_cas_bit_width =
      device_info.cuda_compute_capability().IsAtLeastHopper() ? 128 : 64;
  config.atomic_sync_scope =
      std::string(GetAtomicSyncScope(device_info.gpu_compute_capability()));
  config.atomic_load_store_ordering =
      GetAtomicLoadStoreOrdering(device_info.gpu_compute_capability());

  return emitters::LowerTensors(
      module, config, [&](mlir::RewritePatternSet& patterns) {
        patterns.add<RewriteSyncThreads>(context);
        patterns.add<RewriteGpuDirectAtomicRMW, RewriteGpuDirectAtomicStore,
                     RewriteGpuSetDynamicDimSize>(
            context, device_info.gpu_compute_capability());
      });
}

}  // namespace xla::gpu
