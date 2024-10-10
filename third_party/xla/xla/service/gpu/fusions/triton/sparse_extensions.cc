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
#include "nvidia/include/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/service/gpu/fusions/triton/passes.h"
#include "xla/service/gpu/fusions/triton/xla_triton_ops.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
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

namespace ttn = triton::nvgpu;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ttn::OperandsAndConstraints;

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
ttn::WGMMAEltType getMmaRetType(Value);
ttn::WGMMAEltType getMmaOperandType(Value, bool);

namespace xla::gpu {
namespace {

#define GEN_PASS_DEF_SPARSEADDENCODINGPASS
#define GEN_PASS_DEF_SPARSEBLOCKEDTOMMAPASS
#define GEN_PASS_DEF_SPARSEDOTOPTOLLVMPASS
#define GEN_PASS_DEF_SPARSELOCALLOADTOLLVMPASS
#define GEN_PASS_DEF_SPARSEREMOVELAYOUTCONVERSIONPASS
#define GEN_PASS_DEF_SPARSEWGMMAOPTOLLVMPASS
#include "xla/service/gpu/fusions/triton/passes.h.inc"

constexpr int kThreadsPerWarp = 32;
// Each 16x16 original sparse matrix tile requires 16 metadata values of
// 16-bit size, where the first thread (T0) in each 4-thread group holds two
// such values in a register (32-bit).
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#sparse-matrix-storage
constexpr int kTileSize = 16;
constexpr int kMetaElementsBitSize = 2;
// Metadata elements are packed into 16-bits values.
constexpr int kMetaElementsPerPackedValue = 16 / kMetaElementsBitSize;
constexpr int kColumnsPerCtaTile = kTileSize / kMetaElementsPerPackedValue;

struct SparseAddEncoding
    : public OpConversionPattern<mlir::triton::xla::SparseDotOp> {
  using OpConversionPattern<
      mlir::triton::xla::SparseDotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::triton::xla::SparseDotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    RankedTensorType op_type = cast<RankedTensorType>(op.getType());

    auto op_shape = op_type.getShape();
    auto type_converter = getTypeConverter<TritonGPUTypeConverter>();
    int num_warps = type_converter->getNumWarps();
    int threads_per_warp = type_converter->getThreadsPerWarp();
    int num_ctas = type_converter->getNumCTAs();

    auto rank = op_shape.size();
    auto num_elements = product<int64_t>(op_shape);
    SmallVector<unsigned> ret_size_per_thread(rank, 1);
    if (num_elements / (num_warps * threads_per_warp) >= 4) {
      ret_size_per_thread[rank - 1] = 2;
      ret_size_per_thread[rank - 2] = 2;
    }
    if (num_elements / (num_warps * threads_per_warp) >= 16) {
      ret_size_per_thread[rank - 1] = 4;
      ret_size_per_thread[rank - 2] = 4;
    }
    SmallVector<unsigned> ret_order(rank);
    for (unsigned i = 0; i < rank; ++i) ret_order[i] = rank - 1 - i;

    Attribute d_encoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), op_shape, ret_size_per_thread, ret_order, num_warps,
        threads_per_warp, num_ctas);
    RankedTensorType return_type =
        RankedTensorType::get(op_shape, op_type.getElementType(), d_encoding);

    // a must be of smem layout
    auto a_type = cast<RankedTensorType>(adaptor.getA().getType());
    Type a_element_type = a_type.getElementType();
    Attribute a_encoding = a_type.getEncoding();
    if (!a_encoding) return failure();
    Value a = adaptor.getA();
    if (!isa<triton::gpu::DotOperandEncodingAttr>(a_encoding)) {
      Attribute new_encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 0, d_encoding, a_element_type);
      auto tensor_type = RankedTensorType::get(a_type.getShape(),
                                               a_element_type, new_encoding);
      a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), tensor_type,
                                                        a);
    }

    // b must be of smem layout
    auto b_type = cast<RankedTensorType>(adaptor.getB().getType());
    Type b_element_type = b_type.getElementType();
    Attribute b_encoding = b_type.getEncoding();
    if (!b_encoding) return failure();
    Value b = adaptor.getB();
    if (!isa<triton::gpu::DotOperandEncodingAttr>(b_encoding)) {
      Attribute new_encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 1, d_encoding, b_element_type);
      auto tensor_type = RankedTensorType::get(b_type.getShape(),
                                               b_element_type, new_encoding);
      b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), tensor_type,
                                                        b);
    }
    Value c = adaptor.getC();
    c = rewriter.create<triton::gpu::ConvertLayoutOp>(c.getLoc(), return_type,
                                                      c);

    // aMeta must be of smem layout
    auto a_meta_type = cast<RankedTensorType>(adaptor.getAMeta().getType());
    Attribute a_meta_encoding = a_meta_type.getEncoding();
    if (!a_meta_encoding) return failure();
    Value a_meta = adaptor.getAMeta();
    if (!isa<triton::gpu::SparseDotMetaEncodingAttr>(a_meta_encoding)) {
      Attribute new_encoding =
          triton::gpu::SparseDotMetaEncodingAttr::get(getContext(), d_encoding);
      auto tensor_type = RankedTensorType::get(
          a_meta_type.getShape(), a_meta_type.getElementType(), new_encoding);
      a_meta = rewriter.create<triton::gpu::ConvertLayoutOp>(
          a_meta.getLoc(), tensor_type, a_meta);
    }

    auto new_op = rewriter.replaceOpWithNewOp<mlir::triton::xla::SparseDotOp>(
        op, return_type, a, b, c, a_meta);
    for (const NamedAttribute attr : op->getAttrs()) {
      if (!new_op->hasAttr(attr.getName()))
        new_op->setAttr(attr.getName(), attr.getValue());
    }

    return success();
  }
};

struct SparseAddEncodingPass
    : public impl::SparseAddEncodingPassBase<SparseAddEncodingPass> {
  using impl::SparseAddEncodingPassBase<
      SparseAddEncodingPass>::SparseAddEncodingPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    TritonGPUTypeConverter type_converter(context, num_warps_,
                                          threads_per_warp_, num_ctas_);
    auto pattern = std::make_unique<SparseAddEncoding>(type_converter, context);
    RewritePatternSet patterns(context, std::move(pattern));
    TritonGPUConversionTarget target(*context, type_converter);
    target.addDynamicallyLegalOp<mlir::triton::xla::SparseDotOp>(
        [](mlir::triton::xla::SparseDotOp op) {
          return op.getAMeta().getType().getEncoding() != nullptr;
        });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

class SparseBlockedToMMA : public RewritePattern {
  using ConvertLayoutOp = triton::gpu::ConvertLayoutOp;
  using SparseDotOp = mlir::triton::xla::SparseDotOp;
  using NvidiaMmaEncodingAttr = triton::gpu::NvidiaMmaEncodingAttr;

 public:
  SparseBlockedToMMA(MLIRContext *context, int compute_capability)
      : RewritePattern(SparseDotOp::getOperationName(), 2, context),
        compute_capability_(compute_capability) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto dot_op = cast<SparseDotOp>(op);
    auto context = op->getContext();
    Value a = dot_op.getA();
    Value b = dot_op.getB();

    // Check data-types and SM compatibility
    RankedTensorType ret_type = dot_op.getType();
    if (!ret_type.getEncoding() ||
        isa<NvidiaMmaEncodingAttr>(ret_type.getEncoding()))
      return failure();

    assert(compute_capability_ >= 80 &&
           "SparseDot is only supported on Ampere or higher");
    bool allow_v3 = !triton::tools::getBoolEnv("DISABLE_MMA_V3");
    int version_major = compute_capability_ >= 90 && allow_v3 ? 3 : 2;

    // get MMA encoding and new return type given the number of warps
    auto ret_shape_per_cta = triton::gpu::getShapePerCTA(ret_type);
    auto mod = op->getParentOfType<ModuleOp>();
    int num_warps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    auto cta_layout = triton::gpu::getCTALayout(ret_type.getEncoding());

    auto instr_shape =
        mmaVersionToInstrShape(version_major, ret_shape_per_cta,
                               cast<RankedTensorType>(a.getType()), num_warps);
    auto warps_per_tile = mlir::triton::gpu::getWarpsPerTile(
        dot_op, ret_shape_per_cta, version_major, num_warps, instr_shape);
    NvidiaMmaEncodingAttr mma_enc =
        NvidiaMmaEncodingAttr::get(context, version_major, /*versionMinor=*/0,
                                   warps_per_tile, cta_layout, instr_shape);
    auto new_ret_type = RankedTensorType::get(
        ret_type.getShape(), ret_type.getElementType(), mma_enc);

    // convert accumulator
    auto acc = dot_op.getOperand(2);
    auto new_acc =
        rewriter.create<ConvertLayoutOp>(acc.getLoc(), new_ret_type, acc);

    if (version_major == 2) {  // MMAV2
      int min_bit_width = std::min(triton::gpu::computeOrigBitWidth(a),
                                   triton::gpu::computeOrigBitWidth(b));
      int k_width = 32 / min_bit_width;

      // convert A operand
      auto new_a_encoding =
          DotOperandEncodingAttr::get(context, 0, mma_enc, k_width);
      auto a_type = cast<RankedTensorType>(a.getType());
      a_type = RankedTensorType::get(a_type.getShape(), a_type.getElementType(),
                                     new_a_encoding);
      a = rewriter.create<ConvertLayoutOp>(a.getLoc(), a_type, a);

      // convert B operand
      auto new_b_encoding =
          DotOperandEncodingAttr::get(context, 1, mma_enc, k_width);
      auto b_type = cast<RankedTensorType>(b.getType());
      b_type = RankedTensorType::get(b_type.getShape(), b_type.getElementType(),
                                     new_b_encoding);
      b = rewriter.create<ConvertLayoutOp>(b.getLoc(), b_type, b);

    } else {  // MMAV3
      assert(version_major == 3 &&
             "Sparsity is only supported with MMAV2 or higher");
      auto elt_type = dot_op.getA().getType().getElementType();
      // In MMAV3 transpose is only supported for f16 and bf16.
      bool allow_transpose = elt_type.isF16() || elt_type.isBF16();
      // Shared memory allocations that will be used by the dot op.
      a = triton::gpu::getSharedMemMMAOperand(a, rewriter, 0, allow_transpose);
      b = triton::gpu::getSharedMemMMAOperand(b, rewriter, 1, allow_transpose);
    }

    // convert metadata
    Value meta = dot_op.getAMeta();
    auto meta_type = cast<RankedTensorType>(meta.getType());
    meta_type = RankedTensorType::get(
        meta_type.getShape(), meta_type.getElementType(),
        triton::gpu::SparseDotMetaEncodingAttr::get(context, mma_enc));
    meta = rewriter.create<ConvertLayoutOp>(meta.getLoc(), meta_type, meta);

    // convert dot instruction
    auto new_dot = rewriter.create<SparseDotOp>(dot_op.getLoc(), new_ret_type,
                                                a, b, new_acc, meta);

    // convert back to return type
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, ret_type,
                                                 new_dot.getResult());
    return success();
  }

 private:
  int compute_capability_;
};

struct SparseBlockedToMMAPass
    : public impl::SparseBlockedToMMAPassBase<SparseBlockedToMMAPass> {
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
};

struct SparseRemoveLayoutConversionPass
    : public impl::SparseRemoveLayoutConversionPassBase<
          SparseRemoveLayoutConversionPass> {
  void runOnOperation() override {
    getOperation().walk([&](triton::gpu::ConvertLayoutOp op) {
      ImplicitLocOpBuilder builder(op.getLoc(), op);
      // Skip if the source is already in shared memory.
      auto src_encoding =
          cast<RankedTensorType>(op.getSrc().getType()).getEncoding();
      if (isa<triton::gpu::SharedEncodingAttr>(src_encoding)) {
        return;
      }
      auto dst_type = cast<RankedTensorType>(op.getType());
      // Skip if the destination is not a sparse dot meta.
      if (!isa<triton::gpu::SparseDotMetaEncodingAttr>(
              dst_type.getEncoding())) {
        return;
      }

      auto shared_layout = builder.getAttr<triton::gpu::SharedEncodingAttr>(
          // Packing metadata elements together. No swizzling.
          /*vec=*/kMetaElementsPerPackedValue, /*perPhase=*/1, /*maxPhase=*/1,
          triton::gpu::getOrder(src_encoding),
          triton::gpu::getCTALayout(src_encoding));
      auto mem_type = triton::MemDescType::get(
          dst_type.getShape(), dst_type.getElementType(), shared_layout,
          builder.getAttr<triton::gpu::SharedMemorySpaceAttr>());
      Value alloc =
          builder.create<triton::gpu::LocalAllocOp>(mem_type, op.getSrc());
      Value convert = builder.create<triton::gpu::LocalLoadOp>(dst_type, alloc);
      op.replaceAllUsesWith(convert);
      op.erase();
    });
  }
};

class SparseLocalLoadToLLVM
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
 public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MemDescType src_ty = op.getSrc().getType();
    if (!isa<triton::gpu::SharedEncodingAttr>(src_ty.getEncoding()))
      return failure();
    RankedTensorType dst_ty = op.getType();
    if (!isa<triton::gpu::SparseDotMetaEncodingAttr>(dst_ty.getEncoding()))
      return failure();
    return lowerSharedToSparseMeta(op, adaptor, rewriter);
  }

 private:
  // lowering metadata (local_load: shared -> sparse dot meta) to LLVM
  LogicalResult lowerSharedToSparseMeta(
      triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto load_sparse_encoding = cast<triton::gpu::SparseDotMetaEncodingAttr>(
        cast<RankedTensorType>(op.getResult().getType()).getEncoding());

    // Calculate tile size as number of mask elements (4xi4).
    NvidiaMmaEncodingAttr mma_layout =
        cast<NvidiaMmaEncodingAttr>(load_sparse_encoding.getParent());
    SmallVector<unsigned> warps_per_cta = mma_layout.getWarpsPerCTA();

    // Calculate offset in the tile for the current thread.
    Value threads_per_warp = i32_val(kThreadsPerWarp);
    Value thread_id = getThreadId(rewriter, loc);
    Value warp_id = udiv(thread_id, threads_per_warp);
    Value warp_group_id;
    if (mma_layout.isHopper()) {
      // Hopper MMA instructions force a warp order of [0, 1]. See docs:
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-wgmma-mma-async-m64nnk8
      warp_group_id = urem(warp_id, i32_val(warps_per_cta[0]));
    } else {
      assert(mma_layout.isAmpere() &&
             "SparseDot is only supported on Ampere and Hopper");
      warp_group_id = udiv(warp_id, i32_val(warps_per_cta[1]));
    }
    // Calculate row and column id, based on mma.sp.sync.aligned.m16n8k32:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#sparse-mma-metadata-16832-f16bf16.
    // column-id takes into consideration that we pack elements for metadata.
    constexpr int kThreadsInGroup = 4;
    constexpr int kMetadataLineOffset = kThreadsPerWarp / kThreadsInGroup;
    Value lane_id = urem(thread_id, threads_per_warp);
    Value lane_group_id = udiv(lane_id, i32_val(kThreadsInGroup));
    Value row_id = add(mul(warp_group_id, i32_val(kTileSize)), lane_group_id);
    SmallVector<unsigned> shape_per_cta_tile = {kTileSize * warps_per_cta[0],
                                                kColumnsPerCtaTile};
    Value column_id = urem(lane_id, i32_val(shape_per_cta_tile[1]));

    // Calculate number of tile repetitions.
    Value tensor = op.getSrc();
    auto shape = cast<MemDescType>(tensor.getType()).getShape();
    int rep_m = shape[0] / shape_per_cta_tile[0];
    int rep_k = shape[1] / shape_per_cta_tile[1];
    assert(rep_m > 0 && rep_k > 0);

    // Load sparse metadata from shared memory.
    auto elem_ty = getTypeConverter()->convertType(
        cast<MemDescType>(tensor.getType()).getElementType());
    auto s_mem_obj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), elem_ty, rewriter);
    Value stride_m = s_mem_obj.strides[0];
    Value stride_k = s_mem_obj.strides[1];
    MLIRContext *ctx = tensor.getContext();
    Type ptr_ty = ptr_ty(ctx, 3);
    Value base = gep(ptr_ty, i16_ty, s_mem_obj.base, i32_val(0));
    SmallVector<Value> values;

    for (int k = 0; k < rep_k; ++k) {
      for (int m = 0; m < rep_m; ++m) {
        // Each thread processes two different rows.
        Value row_lower = add(row_id, i32_val(m * shape_per_cta_tile[0]));
        Value row_upper = add(row_lower, i32_val(kMetadataLineOffset));
        Value column = add(column_id, i32_val(k * shape_per_cta_tile[1]));
        Value offset_lower =
            add(mul(row_lower, stride_m), mul(column, stride_k));
        Value offset_upper =
            add(mul(row_upper, stride_m), mul(column, stride_k));
        Value lower = load(i16_ty, gep(ptr_ty, i16_ty, base, offset_lower));
        Value upper = load(i16_ty, gep(ptr_ty, i16_ty, base, offset_upper));
        values.push_back(lower);
        values.push_back(upper);
      }
    }

    // Pack resulting values as LLVM struct.
    Type struct_ty = struct_ty(SmallVector<Type>(values.size(), i16_ty));
    Value res =
        packLLElements(loc, getTypeConverter(), values, rewriter, struct_ty);

    rewriter.replaceOp(op, res);
    return success();
  }
};


bool IsLocalLoadWithSparseEncoding(Operation *op) {
  auto local_load = mlir::dyn_cast<triton::gpu::LocalLoadOp>(op);
  if (!local_load) return false;
  return isa<triton::gpu::SparseDotMetaEncodingAttr>(
      local_load.getType().getEncoding());
}

struct SparseLocalLoadToLLVMPass
    : public impl::SparseLocalLoadToLLVMPassBase<SparseLocalLoadToLLVMPass> {
  void runOnOperation() override {
    // Exit early if there are no sparse ops.
    ModuleOp mod = getOperation();
    if (!ContainsOp(mod, IsLocalLoadWithSparseEncoding)) return;

    // Allocate shared memory and set barrier
    // This is also done in the TritonGPUToLLVMPass but we need to do it before
    // we write the local load op to LLVM to have barriers in the right place.
    // See b/358375493.
    ModuleAllocation allocation(getOperation());
    ModuleMembarAnalysis membar_pass(&allocation);
    membar_pass.run();

    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect, mlir::gpu::GPUDialect,
                           arith::ArithDialect>();
    target.addDynamicallyLegalOp<triton::gpu::LocalLoadOp>(
        [](triton::gpu::LocalLoadOp op) {
          return !isa<triton::gpu::SparseDotMetaEncodingAttr>(
              op.getType().getEncoding());
        });
    LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    auto pattern = std::make_unique<SparseLocalLoadToLLVM>(typeConverter);
    RewritePatternSet patterns(context, std::move(pattern));
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

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

LogicalResult convertSparseMMA(mlir::triton::xla::SparseDotOp op,
                               mlir::triton::xla::SparseDotOp::Adaptor adaptor,
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

LogicalResult convertSparseWGMMA(
    mlir::triton::xla::SparseDotOp op,
    mlir::triton::xla::SparseDotOp::Adaptor adaptor,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
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
  ttn::WGMMAEltType eltTypeA = getMmaOperandType(op.getA(), false);
  ttn::WGMMAEltType eltTypeB = getMmaOperandType(op.getB(), false);
  ttn::WGMMAEltType eltTypeC = getMmaRetType(op.getD());

  ttn::WGMMALayout layoutA =
      transA ? ttn::WGMMALayout::col : ttn::WGMMALayout::row;
  ttn::WGMMALayout layoutB =
      transB ? ttn::WGMMALayout::row : ttn::WGMMALayout::col;

  rewriter.create<ttn::FenceAsyncSharedOp>(loc, 0);
  rewriter.create<ttn::WGMMAFenceOp>(loc);

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
        d = rewriter.create<ttn::SparseWGMMAOp>(
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

  rewriter.create<ttn::WGMMACommitGroupOp>(loc);
  res = rewriter.create<ttn::WGMMAWaitGroupOp>(loc, res, 0);
  rewriter.replaceOp(op, res);

  return success();
}

// ----- Dispatch based on architecture.

LogicalResult rewriteSparseDotOp(
    mlir::triton::xla::SparseDotOp op,
    mlir::triton::xla::SparseDotOp::Adaptor adaptor,
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
    : public ConvertOpToLLVMPattern<mlir::triton::xla::SparseDotOp> {
  using ConvertOpToLLVMPattern<
      mlir::triton::xla::SparseDotOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      mlir::triton::xla::SparseDotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return rewriteSparseDotOp(op, adaptor, getTypeConverter(), rewriter);
  }
};

struct SparseDotOpToLLVMPass
    : public impl::SparseDotOpToLLVMPassBase<SparseDotOpToLLVMPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect, NVVM::NVVMDialect,
                           arith::ArithDialect, ttn::NVGPUDialect>();
    target.addIllegalOp<mlir::triton::xla::SparseDotOp>();
    target.addIllegalDialect<mlir::gpu::GPUDialect>();
    LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    RewritePatternSet patterns(context);
    patterns.add<SparseDotOpConversion>(typeConverter);
    // TODO(b/358375493): Remove this once TritonGPUToLLVMTypeConverter is
    // splitted into smaller passes.
    populateGpuToNVVMConversionPatterns(typeConverter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

class SparseWGMMAOpPattern : public OpRewritePattern<ttn::SparseWGMMAOp> {
 public:
  using OpRewritePattern<ttn::SparseWGMMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::SparseWGMMAOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteAsPtxAsm(op, rewriter, getPtxAsm(op),
                           getOperandsAndConstraints(op),
                           getOutputConstraints(op));
  }

  std::vector<std::string> getOutputConstraints(ttn::SparseWGMMAOp op) const {
    auto outputStructType = cast<LLVM::LLVMStructType>(op.getType());
    uint32_t numOutputRegs = outputStructType.getBody().size();
    std::string output =
        outputStructType.getBody().front().isF32() ? "=f" : "=r";
    return std::vector<std::string>(numOutputRegs, output);
  }

  OperandsAndConstraints getOperandsAndConstraints(
      ttn::SparseWGMMAOp op) const {
    return {{op.getOpC(), "0"},
            {op.getOpA(), "l"},
            {op.getOpB(), "l"},
            {op.getMetaA(), "r"}};
  }

  std::string getPtxAsm(ttn::SparseWGMMAOp op) const {
    auto m = op.getM();
    auto n = op.getN();
    auto k = op.getK();
    auto eltTypeC = op.getEltTypeC();
    auto eltTypeA = op.getEltTypeA();
    auto eltTypeB = op.getEltTypeB();
    auto layoutA = op.getLayoutA();
    auto layoutB = op.getLayoutB();

    // Only f16/bf16 variant is supported.
    using WGMMAEltType = ttn::WGMMAEltType;
    [[maybe_unused]] bool supported =
        eltTypeC == WGMMAEltType::f32 &&
        ((eltTypeA == WGMMAEltType::f16 && eltTypeB == WGMMAEltType::f16) ||
         (eltTypeA == WGMMAEltType::bf16 && eltTypeB == WGMMAEltType::bf16)) &&
        (m == 64 && 8 <= n && n <= 256 && n % 8 == 0 && k == 32);
    assert(supported && "Sparse WGMMA type or shape is not supported");

    // Operands
    uint32_t asmOpIdx = 0;
    std::string args = "";

    // Output and operand C
    uint32_t numCRegs =
        cast<LLVM::LLVMStructType>(op.getType()).getBody().size();
    args += "{";
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    args += "}, ";
    asmOpIdx += numCRegs;

    // Operands A and B (must be `desc`)
    args += "$" + std::to_string(asmOpIdx++) + ", ";
    args += "$" + std::to_string(asmOpIdx++) + ", ";

    // Metadata for A
    args += "$" + std::to_string(asmOpIdx++) + ", 0, ";

    // `scale-d`, `imm-scale-a`, and `imm-scale-b` are 1 by default
    args += "1, 1, 1";

    // `trans-a` and `trans-b`
    using WGMMALayout = ttn::WGMMALayout;
    args += ", " + std::to_string(layoutA == WGMMALayout::col);
    args += ", " + std::to_string(layoutB == WGMMALayout::row);

    auto ptxAsm =
        "wgmma.mma_async.sp.sync.aligned"
        ".m" +
        std::to_string(m) + "n" + std::to_string(n) + "k" + std::to_string(k) +
        "." + stringifyEnum(eltTypeC).str() + "." +
        stringifyEnum(eltTypeA).str() + "." + stringifyEnum(eltTypeB).str() +
        " " + args + ";";
    return ptxAsm;
  }
};

struct SparseWGMMAOpToLLVMPass
    : public impl::SparseWGMMAOpToLLVMPassBase<SparseWGMMAOpToLLVMPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto pattern = std::make_unique<SparseWGMMAOpPattern>(context);
    RewritePatternSet patterns(context, std::move(pattern));
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateSparseAddEncodingPass(int32_t num_warps,
                                                  int32_t threads_per_warp,
                                                  int32_t num_ctas) {
  SparseAddEncodingPassOptions options;
  options.num_warps_ = num_warps;
  options.threads_per_warp_ = threads_per_warp;
  options.num_ctas_ = num_ctas;
  return std::make_unique<SparseAddEncodingPass>(options);
}

std::unique_ptr<Pass> CreateSparseBlockedToMMAPass() {
  return std::make_unique<SparseBlockedToMMAPass>();
}

std::unique_ptr<Pass> CreateSparseRemoveLayoutConversionPass() {
  return std::make_unique<SparseRemoveLayoutConversionPass>();
}

std::unique_ptr<Pass> CreateSparseLocalLoadToLLVMPass() {
  return std::make_unique<SparseLocalLoadToLLVMPass>();
}

std::unique_ptr<Pass> CreateSparseDotOpToLLVMPass() {
  return std::make_unique<SparseDotOpToLLVMPass>();
}

std::unique_ptr<Pass> CreateSparseWGMMAOpToLLVMPass() {
  return std::make_unique<SparseWGMMAOpToLLVMPass>();
}

}  // namespace xla::gpu
