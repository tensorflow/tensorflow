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
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/layout_util.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_LOWERTENSORSPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

using mlir::failure;
using mlir::success;
using mlir::Value;
using mlir::ValueRange;

struct RewriteFunctionSignatures : mlir::OpRewritePattern<mlir::func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::func::FuncOp op, mlir::PatternRewriter& rewriter) const override {
    auto is_tensor = [](mlir::Type ty) {
      return ty.isa<mlir::RankedTensorType>();
    };
    if (!llvm::any_of(op.getFunctionType().getInputs(), is_tensor)) {
      return rewriter.notifyMatchFailure(op,
                                         "the function has no input tensors");
    }

    bool some_tensor_result =
        llvm::any_of(op.getFunctionType().getResults(), is_tensor);
    bool all_tensor_results =
        llvm::all_of(op.getFunctionType().getResults(), is_tensor);
    if (some_tensor_result && !all_tensor_results) {
      op->emitOpError("function has a mix of tensor and non-tensor results");
      return failure();
    }

    mlir::TypeRange new_results = op.getFunctionType().getResults();
    if (some_tensor_result) {
      new_results = {};
      auto terminator = op.getFunctionBody().front().getTerminator();
      rewriter.setInsertionPoint(terminator);
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(terminator);
    }

    llvm::SmallVector<mlir::Type> new_operands(
        op.getFunctionType().getInputs());
    for (auto&& [index, operand] : llvm::enumerate(new_operands)) {
      if (is_tensor(operand)) {
        rewriter.setInsertionPointToStart(&op.getBody().front());
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
            op.getLoc(), operand, op.getArgument(index));
        op.getArgument(index).replaceAllUsesExcept(cast.getResult(0), cast);
        operand = mlir::LLVM::LLVMPointerType::get(op.getContext());
      }
    }

    op.setFunctionType(rewriter.getFunctionType(new_operands, new_results));
    auto& entry = op->getRegion(0).front();
    for (auto [arg, arg_type] : llvm::zip(entry.getArguments(), new_operands)) {
      arg.setType(arg_type);
    }

    return success();
  }
};

mlir::LLVM::GEPOp CreateGep(mlir::Operation* op,
                            mlir::TypedValue<mlir::RankedTensorType> tensor,
                            ValueRange indices,
                            mlir::PatternRewriter& rewriter) {
  auto ptr = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto byte_shape = ShapeUtil::MakeShape(U8, tensor.getType().getShape());
  if (auto encoding = tensor.getType().getEncoding()) {
    *byte_shape.mutable_layout() = LayoutUtil::MakeLayout(llvm::to_vector(
        encoding.cast<mlir::DenseElementsAttr>().getValues<int64_t>()));
  }
  auto linearize_map = mlir::getAffineConstantExpr(0, rewriter.getContext());
  for (auto [dim, stride] :
       llvm::enumerate(*ShapeUtil::ByteStrides(byte_shape))) {
    linearize_map = linearize_map +
                    mlir::getAffineDimExpr(dim, rewriter.getContext()) * stride;
  }

  rewriter.setInsertionPoint(op);
  Value index = rewriter.create<mlir::affine::AffineApplyOp>(
      tensor.getLoc(), linearize_map, indices);
  auto index_ty =
      ShapeUtil::ElementsIn(byte_shape) < std::numeric_limits<int32_t>::max()
          ? rewriter.getI32Type()
          : rewriter.getI64Type();
  index = rewriter.create<mlir::arith::IndexCastUIOp>(tensor.getLoc(), index_ty,
                                                      index);

  auto tensor_ptr = rewriter
                        .create<mlir::UnrealizedConversionCastOp>(
                            tensor.getLoc(), ptr, tensor)
                        .getResult(0);
  mlir::LLVMTypeConverter converter(rewriter.getContext());
  auto llvm_element_type =
      converter.convertType(tensor.getType().getElementType());
  auto gep = rewriter.create<mlir::LLVM::GEPOp>(
      tensor.getLoc(), ptr, llvm_element_type, tensor_ptr, index);
  gep.setInbounds(true);
  return gep;
}

struct RewriteTensorExtract : mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto gep = CreateGep(op, op.getTensor(), op.getIndices(), rewriter);
    auto load =
        rewriter
            .create<mlir::LLVM::LoadOp>(gep.getLoc(), gep.getElemType(), gep)
            .getResult();
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getType(), load);
    return success();
  }
};

struct RewriteTensorInsert : mlir::OpRewritePattern<mlir::tensor::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op,
      mlir::PatternRewriter& rewriter) const override {
    Value dest = op.getDest();
    while (dest.getDefiningOp()) {
      int result_number = dest.cast<mlir::OpResult>().getResultNumber();
      if (auto insert = dest.getDefiningOp<mlir::tensor::InsertOp>()) {
        dest = insert.getDest();
      } else if (auto scf_if = dest.getDefiningOp<mlir::scf::IfOp>()) {
        // Pick one of the branches, they're required to yield the same buffers.
        dest = scf_if.getThenRegion().front().getTerminator()->getOperand(
            result_number);
      } else if (auto scf_for = dest.getDefiningOp<mlir::scf::ForOp>()) {
        dest = scf_for.getInitArgs()[result_number];
      } else if (dest.getDefiningOp<mlir::UnrealizedConversionCastOp>() ||
                 dest.getDefiningOp<AllocateSharedOp>()) {
        break;
      } else {
        return op.emitOpError("unsupported dest type");
      }
    }

    auto gep =
        CreateGep(op, dest.cast<mlir::TypedValue<mlir::RankedTensorType>>(),
                  op.getIndices(), rewriter);
    auto scalar_value = op.getScalar();
    mlir::LLVMTypeConverter converter(getContext());
    auto llvm_type = converter.convertType(scalar_value.getType());
    scalar_value = rewriter
                       .create<mlir::UnrealizedConversionCastOp>(
                           gep.getLoc(), llvm_type, scalar_value)
                       .getResult(0);
    rewriter.create<mlir::LLVM::StoreOp>(gep.getLoc(), scalar_value, gep);

    op.replaceAllUsesWith(op.getDest());
    op.erase();
    return success();
  }
};

struct RewriteCall : mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::func::CallOp op, mlir::PatternRewriter& rewriter) const override {
    if (!llvm::any_of(op->getOperandTypes(), [](mlir::Type ty) {
          return ty.isa<mlir::RankedTensorType>();
        })) {
      return rewriter.notifyMatchFailure(op, "the call has no input tensors");
    }

    for (const auto&& [index, arg] : llvm::enumerate(op.getOperands())) {
      if (arg.getType().isa<mlir::RankedTensorType>()) {
        op.setOperand(
            index,
            rewriter
                .create<mlir::UnrealizedConversionCastOp>(
                    op.getLoc(),
                    mlir::LLVM::LLVMPointerType::get(op.getContext()), arg)
                .getResult(0));
      }
    }
    return success();
  }
};

struct RewriteAllocateShared : mlir::OpRewritePattern<AllocateSharedOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      AllocateSharedOp op, mlir::PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto shaped_ty = op.getResult().getType().cast<mlir::ShapedType>();
    constexpr int kGPUSharedMemoryAddrSpace = 3;
    auto array_ty = mlir::LLVM::LLVMArrayType::get(shaped_ty.getElementType(),
                                                   shaped_ty.getNumElements());

    std::string name;
    int index = 0;
    do {
      name = absl::StrCat("shared_", index);
      ++index;
    } while (module.lookupSymbol(name));

    rewriter.setInsertionPointToStart(module.getBody());
    auto global = rewriter.create<mlir::LLVM::GlobalOp>(
        op.getLoc(), array_ty, /*isConstant=*/false,
        /*linkage=*/mlir::LLVM::Linkage::Private, name,
        /*value=*/mlir::Attribute{},
        /*alignment=*/0, kGPUSharedMemoryAddrSpace);

    rewriter.setInsertionPoint(op);
    auto addr = rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), global);
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getResult().getType(),
        rewriter
            .create<mlir::LLVM::AddrSpaceCastOp>(
                op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
                addr)
            .getResult());
    return success();
  }
};

struct RewriteSyncThreads : mlir::OpRewritePattern<SyncThreadsOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncThreadsOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.create<mlir::gpu::BarrierOp>(op.getLoc());
    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};

// Implements atomic binary operations using atomic compare-and-swap
// (atomicCAS) as follows:
//   1. Reads the value from the memory pointed to by output_address and
//     records it as old_output.
//   2. Uses old_output as one of the source operand to perform the binary
//     operation and stores the result in new_output.
//   3. Calls atomicCAS which implements compare-and-swap as an atomic
//     operation. In particular, atomicCAS reads the value from the memory
//     pointed to by output_address, and compares the value with old_output. If
//     the two values equal, new_output is written to the same memory location
//     and true is returned to indicate that the atomic operation succeeds.
//     Otherwise, the new value read from the memory is returned. In this case,
//     the new value is copied to old_output, and steps 2. and 3. are repeated
//     until atomicCAS succeeds.
//
// On Nvidia GPUs, atomicCAS can only operate on 32 bit and 64 bit integers. If
// the element type of the binary operation is 32 bits or 64 bits, the integer
// type of the same size is used for the atomicCAS operation. On the other hand,
// if the element type is smaller than 32 bits, int32_t is used for the
// atomicCAS operation. In this case, atomicCAS reads and writes 32 bit values
// from the memory, which is larger than the memory size required by the
// original atomic binary operation. We mask off the last two bits of the
// output_address and use the result as an address to read the 32 bit values
// from the memory. This can avoid out of bound memory accesses if tensor
// buffers are 4 byte aligned and have a size of 4N, an assumption that the
// runtime can guarantee.
struct RewriteAtomicRMW : mlir::OpRewritePattern<AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      AtomicRMWOp op, mlir::PatternRewriter& rewriter) const override {
    namespace ml = mlir::LLVM;
    mlir::Location loc = op.getLoc();
    auto input = op.getInput();

    // Use 32-bit atomic type for small input types.
    mlir::Type result_ty = op.getResult().getType().getElementType();
    unsigned int result_size = result_ty.getIntOrFloatBitWidth();
    bool small_type = result_size < 32;
    mlir::Type atomic_ty =
        mlir::IntegerType::get(op.getContext(), small_type ? 32 : result_size);

    // Calculate load address for the input.
    Value addr = CreateGep(op, input, op.getIndices(), rewriter);
    Value shift, mask;
    if (small_type) {
      // Update input pointer by discarding the last two bits - i.e. align to
      // 32-bit boundary for small input types (will not result in OOB, as the
      // input alignment is at least 32 bits).
      mlir::Type addr_int_ty = rewriter.getI64Type();
      Value addr_int = rewriter.create<ml::PtrToIntOp>(loc, addr_int_ty, addr);
      Value addr_offset = rewriter.create<ml::AndOp>(
          loc, addr_int, rewriter.create<ml::ConstantOp>(loc, addr_int_ty, 3));
      Value index = rewriter.create<ml::MulOp>(
          loc, addr_offset,
          rewriter.create<ml::ConstantOp>(loc, addr_int_ty, -1));
      addr =
          rewriter.create<ml::GEPOp>(loc, addr.getType(), rewriter.getI8Type(),
                                     addr, index, /*inbounds=*/true);

      // Calculate the bit shift (assume little-endianness).
      Value offset = rewriter.create<ml::TruncOp>(loc, atomic_ty, addr_offset);
      shift = rewriter.create<ml::MulOp>(
          loc, offset,
          rewriter.create<ml::ConstantOp>(loc, offset.getType(), 8));

      // Compose the update mask.
      Value bits_long = rewriter.create<ml::ConstantOp>(loc, atomic_ty, -1);
      Value bits_short = rewriter.create<ml::ZExtOp>(
          loc, atomic_ty,
          rewriter.create<ml::ConstantOp>(
              loc, rewriter.getIntegerType(result_size), -1));
      mask = rewriter.create<ml::XOrOp>(
          loc, bits_long, rewriter.create<ml::ShlOp>(loc, bits_short, shift));
    }

    // Load initial atomic value and create the loop.
    Value initial = rewriter.create<ml::LoadOp>(loc, atomic_ty, addr);
    rewriter.create<mlir::scf::WhileOp>(
        loc, mlir::TypeRange{atomic_ty}, ValueRange{initial},
        [&](mlir::OpBuilder& b, mlir::Location loc, ValueRange values) {
          Value old_value = values[0];

          // Convert atomic value to input value.
          Value input_value;
          if (small_type) {
            Value short_value = b.create<ml::TruncOp>(
                loc, b.getIntegerType(result_size),
                b.create<ml::LShrOp>(loc, old_value, shift));
            input_value = b.create<ml::BitcastOp>(loc, result_ty, short_value);
          } else {
            input_value = b.create<ml::BitcastOp>(loc, result_ty, old_value);
          }

          // Perform computation on the loaded input value.
          rewriter.mergeBlocks(&op.getComputation().front(), b.getBlock(),
                               {input_value});
          auto yield_op = b.getBlock()->getTerminator();
          Value result = yield_op->getOperand(0);
          rewriter.eraseOp(yield_op);

          // Convert resulting value to atomic value.
          Value new_value;
          if (small_type) {
            Value cast_value = rewriter.create<ml::ZExtOp>(
                loc, atomic_ty,
                rewriter.create<ml::BitcastOp>(
                    loc, rewriter.getIntegerType(result_size), result));
            new_value = rewriter.create<ml::OrOp>(
                loc, rewriter.create<ml::AndOp>(loc, old_value, mask),
                rewriter.create<ml::ShlOp>(loc, cast_value, shift));
          } else {
            new_value = b.create<ml::BitcastOp>(loc, atomic_ty, result);
          }

          // Try saving the result atomically, retry if failed.
          Value cmpxchg = b.create<ml::AtomicCmpXchgOp>(
              loc, addr, old_value, new_value,
              /*success_ordering=*/ml::AtomicOrdering::seq_cst,
              /*failure_ordering=*/ml::AtomicOrdering::seq_cst);
          Value next = b.create<ml::ExtractValueOp>(loc, cmpxchg, 0);
          Value ok = b.create<ml::ExtractValueOp>(loc, cmpxchg, 1);
          Value low_bit =
              b.create<ml::ConstantOp>(loc, b.getOneAttr(b.getI1Type()));
          Value not_ok = b.create<ml::XOrOp>(loc, ok, low_bit);
          b.create<mlir::scf::ConditionOp>(loc, not_ok, ValueRange{next});
        },
        [&](mlir::OpBuilder& b, mlir::Location loc, ValueRange values) {
          b.create<mlir::scf::YieldOp>(loc, values);
        });
    rewriter.replaceOp(op, input);
    return success();
  }
};

class LowerTensorsPass : public impl::LowerTensorsPassBase<LowerTensorsPass> {
 public:
  void runOnOperation() override {
    mlir::RewritePatternSet tensor_patterns(&getContext());
    tensor_patterns
        .add<RewriteAllocateShared, RewriteSyncThreads, RewriteTensorExtract,
             RewriteTensorInsert, RewriteAtomicRMW>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(tensor_patterns)))) {
      signalPassFailure();
    }

    mlir::RewritePatternSet function_patterns(&getContext());
    function_patterns.add<RewriteFunctionSignatures, RewriteCall>(
        &getContext());
    mlir::scf::ForOp::getCanonicalizationPatterns(function_patterns,
                                                  &getContext());
    mlir::scf::IfOp::getCanonicalizationPatterns(function_patterns,
                                                 &getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(function_patterns)))) {
      signalPassFailure();
    }

    getOperation()->walk([this](mlir::LLVM::LoadOp load) {
      Value addr = load.getAddr();
      while (auto gep = addr.getDefiningOp<mlir::LLVM::GEPOp>()) {
        addr = gep.getBase();
      }
      if (addr.getDefiningOp<mlir::LLVM::AddrSpaceCastOp>()) {
        // Shared memory - no need to annotate anything.
        return;
      }
      if (auto base = mlir::dyn_cast<mlir::BlockArgument>(addr)) {
        if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(
                base.getOwner()->getParentOp())) {
          if (func.getArgAttr(base.getArgNumber(), "xla.invariant")) {
            load.setInvariant(true);
          }
          return;
        }
      }
      load.emitOpError("load op address is not (a GEP of) a function argument");
      signalPassFailure();
    });
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerTensorsPass() {
  return std::make_unique<LowerTensorsPass>();
}

}  // namespace gpu
}  // namespace xla
