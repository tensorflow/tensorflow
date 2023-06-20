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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_triton.h"

#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT(build/c++11): required to interface with LLVM
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/tensor_float_32_utils.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"

namespace xla {
namespace gpu {

namespace ma = ::mlir::arith;
namespace ml = ::mlir::LLVM;
namespace mn = ::mlir::NVVM;
namespace mt = ::mlir::triton;

using ::llvm::SmallVector;
using ::mlir::Type;
using ::mlir::Value;

namespace {

// XLA -> Triton type conversions.
Type TritonType(mlir::OpBuilder b, PrimitiveType t) {
  switch (t) {
    case F64:
      return b.getF64Type();
    case F32:
      return b.getF32Type();
    case F16:
      return b.getF16Type();
    case BF16:
      return b.getBF16Type();
    case S64:
      return b.getI64Type();
    case S32:
      return b.getI32Type();
    case S16:
      return b.getI16Type();
    case PRED:
      // Treat PRED as S8.
    case S8:
      return b.getI8Type();
    default:
      LOG(FATAL) << "This type is not supported yet: "
                 << primitive_util::LowercasePrimitiveTypeName(t);
  }
}

// Triton type conversions.
Value Cast(mlir::ImplicitLocOpBuilder& b, Value value, Type dst_element_ty) {
  Type src_ty = value.getType();
  Type src_element_ty = src_ty;
  Type fp32_ty = b.getF32Type();
  Type dst_ty = dst_element_ty;
  if (auto src_shaped_ty = src_ty.dyn_cast<mlir::ShapedType>()) {
    src_element_ty = src_shaped_ty.getElementType();
    dst_ty = src_shaped_ty.clone(src_shaped_ty.getShape(), dst_element_ty);
    fp32_ty = src_shaped_ty.clone(src_shaped_ty.getShape(), b.getF32Type());
  }
  if (src_ty == dst_ty) {
    return value;
  }

  // Float <=> float
  auto src_fp_element_ty = src_element_ty.dyn_cast<mlir::FloatType>();
  auto dst_fp_element_ty = dst_element_ty.dyn_cast<mlir::FloatType>();
  if (src_fp_element_ty && dst_fp_element_ty) {
    // f16 <=> bf16 is a bit special, since we can neither extend, nor truncate
    // one into the other. Instead, we first extend src to f32, and then
    // truncate to dst.
    if ((src_element_ty.isF16() && dst_element_ty.isBF16()) ||
        (src_element_ty.isBF16() && dst_element_ty.isF16())) {
      return b.create<ma::TruncFOp>(dst_ty,
                                    b.create<ma::ExtFOp>(fp32_ty, value));
    } else if (src_fp_element_ty.getFPMantissaWidth() >
               dst_fp_element_ty.getFPMantissaWidth()) {
      return b.create<ma::TruncFOp>(dst_ty, value);
    } else {
      return b.create<ma::ExtFOp>(dst_ty, value);
    }
  }
  // int => float
  if (src_element_ty.isa<mlir::IntegerType>() && dst_fp_element_ty) {
    // TODO(b/266862493): Support unsigned integer types.
    return b.create<ma::SIToFPOp>(dst_ty, value);
  }
  // float => int
  if (src_fp_element_ty && dst_element_ty.isa<mlir::IntegerType>()) {
    // TODO(b/266862493): Support unsigned integer types.
    return b.create<ma::FPToSIOp>(dst_ty, value);
  }

  LOG(FATAL) << "Type conversion not supported: "
             << llvm_ir::DumpToString(src_element_ty) << " -> "
             << llvm_ir::DumpToString(dst_element_ty);
}

// Create a scalar constant.
template <typename T>
ma::ConstantOp CreateConst(mlir::ImplicitLocOpBuilder b, Type type, T value) {
  if (type.isa<mlir::IntegerType>()) {
    return b.create<ma::ConstantOp>(b.getIntegerAttr(type, value));
  }
  if (type.isa<mlir::FloatType>()) {
    return b.create<ma::ConstantOp>(
        b.getFloatAttr(type, static_cast<double>(value)));
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

// Create a tensor constant.
template <typename T>
ma::ConstantOp CreateConst(mlir::ImplicitLocOpBuilder& b, Type type, T value,
                           mlir::ArrayRef<int64_t> shape) {
  auto tensor_type = mlir::RankedTensorType::get(shape, type);
  if (auto int_type = type.dyn_cast<mlir::IntegerType>()) {
    return b.create<ma::ConstantOp>(mlir::DenseElementsAttr::get(
        tensor_type, mlir::APInt(int_type.getIntOrFloatBitWidth(), value)));
  }
  if (auto float_type = type.dyn_cast<mlir::FloatType>()) {
    return b.create<ma::ConstantOp>(mlir::DenseElementsAttr::get(
        tensor_type, b.getFloatAttr(type, static_cast<double>(value))));
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

// TODO(b/269489810): Contribute nicer builders to Triton, so we don't need to
// define these utilities.
Value Splat(mlir::ImplicitLocOpBuilder& b, Value value,
            mlir::ArrayRef<int64_t> shape) {
  auto type = mlir::RankedTensorType::get(shape, value.getType());
  return b.create<mt::SplatOp>(type, value);
}

using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;

Value Broadcast(mlir::ImplicitLocOpBuilder& b, TensorValue value,
                mlir::ArrayRef<int64_t> shape) {
  auto type =
      mlir::RankedTensorType::get(shape, value.getType().getElementType());
  return b.create<mt::BroadcastOp>(type, value);
}

Value Range(mlir::ImplicitLocOpBuilder& b, int32_t limit) {
  auto type = mlir::RankedTensorType::get(limit, b.getI32Type());
  return b.create<mt::MakeRangeOp>(type, 0, limit);
}

Value AddPtr(mlir::ImplicitLocOpBuilder& b, Value ptr, Value offset) {
  return b.create<mt::AddPtrOp>(ptr.getType(), ptr, offset);
}

void CreateTritonPipeline(mlir::OpPassManager& pm,
                          const se::CudaComputeCapability& cc, int num_warps,
                          int num_stages) {
  const int ccAsInt = cc.major * 10 + cc.minor;
  // Based on optimize_ttir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mt::createCombineOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Based on ttir_to_ttgir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(mt::createConvertTritonToTritonGPUPass(num_warps));
  // Based on optimize_ttgir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(mlir::createTritonGPUCoalescePass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUAccelerateMatmulPass(ccAsInt));
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createTritonGPUPipelinePass(num_stages));
  pm.addPass(mlir::createTritonGPUPrefetchPass());
  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUDecomposeConversionsPass());
  pm.addPass(mlir::createTritonGPUReorderInstructionsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Based on translateTritonGPUToLLVMIR() in
  // @triton//:lib/Target/LLVMIR/LLVMIRTranslation.cpp
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mt::createConvertTritonGPUToLLVMPass(ccAsInt));
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

// Extract additional attributes from an LLVM function that are not passed
// to the builder directly.
SmallVector<mlir::NamedAttribute> GetExtraAttrs(ml::LLVMFuncOp func) {
  llvm::StringSet<> registered_attr_names{
      func.getSymNameAttrName().getValue(),
      func.getFunctionTypeAttrName().getValue(),
      func.getLinkageAttrName().getValue(),
      func.getDsoLocalAttrName().getValue(),
      func.getCConvAttrName().getValue(),
      func.getArgAttrsAttrName().getValue(),
      func.getFunctionEntryCountAttrName().getValue()};
  return llvm::to_vector(
      llvm::make_filter_range(func->getAttrs(), [&](mlir::NamedAttribute attr) {
        return !registered_attr_names.contains(attr.getName().getValue());
      }));
}

// Strip address spaces from function parameters.
void StripParameterAddressSpaces(mlir::RewriterBase& rewriter,
                                 ml::LLVMFuncOp func) {
  // Figure out what the new signature should be.
  ml::LLVMFunctionType func_ty = func.getFunctionType();
  SmallVector<Type, 3> generic_func_params(
      llvm::map_range(func_ty.getParams(), [](Type type) -> Type {
        auto ptr_ty = type.dyn_cast<ml::LLVMPointerType>();
        if (!ptr_ty) return type;
        if (ptr_ty.getAddressSpace() != mn::kGlobalMemorySpace) return type;
        return ml::LLVMPointerType::get(ptr_ty.getElementType());
      }));
  ml::LLVMFunctionType generic_func_ty =
      func_ty.clone(generic_func_params, func_ty.getReturnTypes());

  // Create a function with the new signature.
  SmallVector<mlir::DictionaryAttr, 3> arg_attrs(llvm::map_range(
      func.getArgAttrsAttr().getValue(),
      [](mlir::Attribute attr) { return attr.cast<mlir::DictionaryAttr>(); }));
  auto generic_func = rewriter.create<ml::LLVMFuncOp>(
      func.getLoc(), func.getSymName(), generic_func_ty, func.getLinkage(),
      func.getDsoLocal(), func.getCConv(), GetExtraAttrs(func), arg_attrs,
      func.getFunctionEntryCount());

  // Convert generic address spaces back to original ones within the function
  // body.
  mlir::Block* entry = generic_func.addEntryBlock();
  rewriter.setInsertionPointToEnd(entry);
  SmallVector<Value, 3> converted_args;
  for (auto [arg, type] :
       llvm::zip(generic_func.getArguments(), func_ty.getParams())) {
    Value converted = arg;
    if (arg.getType() != type) {
      converted = rewriter.create<ml::AddrSpaceCastOp>(arg.getLoc(), type, arg);
    }
    converted_args.push_back(converted);
  }

  // Move the rest of function body from the original function.
  rewriter.cloneRegionBefore(func.getBody(), generic_func.getBody(),
                             generic_func.getBody().end());
  rewriter.eraseOp(func);
  rewriter.mergeBlocks(entry->getNextNode(), entry, converted_args);
}

// Rewrite signatures of kernel functions to use generic data pointers and
// cast them to global ones within the kernel.
struct GeneralizeKernelSignaturePass
    : mlir::PassWrapper<GeneralizeKernelSignaturePass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GeneralizeKernelSignaturePass);
  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    getOperation()->walk([&](ml::LLVMFuncOp func) {
      if (!func->hasAttr(mn::NVVMDialect::getKernelFuncAttrName())) {
        return;
      }
      rewriter.setInsertionPointAfter(func);
      StripParameterAddressSpaces(rewriter, func);
    });
  }
};

// Variable naming: lhs [m, k] x rhs [k, n] -> out [m, n].
// TODO(b/270937368): Split this up into smaller functions.
template <typename IndexT>
StatusOr<LaunchDimensions> MatMulImpl(
    mlir::OpBuilder builder, const HloDotInstruction* dot_instr,
    mlir::triton::FuncOp fn,
    const tensorflow::AutotuneResult::TritonGemmKey& config, int shmem_budget) {
  const HloInstruction* root = dot_instr->parent()->root_instruction();
  CHECK(!root->shape().IsTuple());

  // We'll be creating a lot of instructions from a single dot, use an
  // implicit loc builder so we don't have to pass around the location all the
  // time.
  auto loc = mlir::NameLoc::get(builder.getStringAttr(dot_instr->name()));
  mlir::ImplicitLocOpBuilder b(loc, builder);
  Type i32_ty = b.getI32Type();
  Type int_ty;
  if constexpr (std::is_same_v<IndexT, int64_t>) {
    int_ty = b.getI64Type();
  } else {
    int_ty = b.getI32Type();
  }
  const DotDimensionNumbers& dims = dot_instr->dot_dimension_numbers();
  const DotFusionAnalysis analysis(dot_instr->parent(), config.split_k());
  const HloInstruction* lhs_param0 =
      *analysis.ScopeParameters(DotFusionAnalysis::Scope::LHS).cbegin();
  const HloInstruction* rhs_param0 =
      *analysis.ScopeParameters(DotFusionAnalysis::Scope::RHS).cbegin();

  Type lhs_ty = TritonType(b, lhs_param0->shape().element_type());
  Type rhs_ty = TritonType(b, rhs_param0->shape().element_type());

  // Rely on dot decomposer: there is just one contracting and one
  // non-contracting dimension on each side + batch ones optionally.
  CHECK_EQ(dims.lhs_contracting_dimensions_size(), 1);
  CHECK_EQ(dims.rhs_contracting_dimensions_size(), 1);

  const bool have_split_k = config.split_k() > 1;
  if (have_split_k) {
    // Split-K dimension has to be the first batch one and have an index
    // just before the contracting one.
    // Size of this dimension has to match the split_k value.
    CHECK_EQ(dims.lhs_batch_dimensions(0),
             dims.lhs_contracting_dimensions(0) - 1);
    CHECK_EQ(dims.rhs_batch_dimensions(0),
             dims.rhs_contracting_dimensions(0) - 1);
    CHECK_EQ(config.split_k(), dot_instr->operand(0)->shape().dimensions(
                                   dims.lhs_contracting_dimensions(0) - 1));
    CHECK_EQ(config.split_k(), dot_instr->operand(1)->shape().dimensions(
                                   dims.rhs_contracting_dimensions(0) - 1));
  }

  CHECK_LE(dims.lhs_batch_dimensions_size(), 1 + have_split_k);
  const bool have_batch = dims.lhs_batch_dimensions_size() - have_split_k;
  CHECK_EQ(dot_instr->operand(0)->shape().rank(),
           2 + have_split_k + have_batch);
  const int64_t lhs_noncontracting_dim_idx =
      GetNonContractingDims(dot_instr->operand(0)->shape(),
                            dims.lhs_batch_dimensions(),
                            dims.lhs_contracting_dimensions())
          .value()[0];
  const int64_t rhs_noncontracting_dim_idx =
      GetNonContractingDims(dot_instr->operand(1)->shape(),
                            dims.rhs_batch_dimensions(),
                            dims.rhs_contracting_dimensions())
          .value()[0];

  // Logical output dimensions are always ordered as:
  //   split-K, batch, non-contracting LHS, non-contracting RHS,
  // where split-K and batch are optional.
  const int rhs_nc_out_idx = dot_instr->shape().rank() - 1;
  const int lhs_nc_out_idx = dot_instr->shape().rank() - 2;
  const int split_k_out_idx = have_split_k ? 0 : -1;
  const int batch_out_idx = have_batch ? (have_split_k ? 1 : 0) : -1;

  // LHS non-contracting dimension length.
  // LHS non-contracting can be split, so this holds its full size unlike the
  // m_minor.
  int m =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, lhs_nc_out_idx)
          ->at(0)
          .count;

  // Contracting dimension length.
  const int k = dot_instr->operand(0)->shape().dimensions(
                    dims.lhs_contracting_dimensions(0)) *
                config.split_k();

  // LHS non-contracting can be split into two.
  bool lhs_nc_split = false;
  // Either batch size or upper part of the length of a split nc dimension.
  int batch_size = 1;
  IndexT stride_lhs_m = 0;
  IndexT stride_lhs_k = 0;
  IndexT stride_lhs_batch = 0;
  IndexT stride_rhs_batch = 0;
  const DotFusionAnalysis::DimIterationSpec* lhs_nc_iter_spec =
      analysis.IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                        lhs_noncontracting_dim_idx);
  lhs_nc_split = lhs_nc_iter_spec->size() > 1;
  // For now split non-contracting and batch are not supported simultaneously
  // because they are implemented via same mechanism.
  CHECK_LE(have_batch + lhs_nc_split, 1);
  if (lhs_nc_split) {
    batch_size = lhs_nc_iter_spec->at(1).count;
    CHECK_GE(batch_size, 1);
    stride_lhs_batch = lhs_nc_iter_spec->at(1).stride;
    CHECK_GE(stride_lhs_batch, 1);
  } else if (have_batch) {
    const int64_t lhs_batch_dim_idx = *(dims.lhs_batch_dimensions().cend() - 1);
    batch_size = analysis
                     .IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                               lhs_batch_dim_idx)
                     ->at(0)
                     .count;
    CHECK_GE(batch_size, 1);
    stride_lhs_batch = analysis
                           .IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                                     lhs_batch_dim_idx)
                           ->at(0)
                           .stride;
    CHECK_GE(stride_lhs_batch, 1);
  }

  CHECK_EQ(lhs_nc_iter_spec->size(), 1 + lhs_nc_split);
  CHECK_EQ(analysis
               .IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                         dims.lhs_contracting_dimensions(0))
               ->size(),
           1);
  stride_lhs_m = lhs_nc_iter_spec->at(0).stride;
  stride_lhs_k = analysis
                     .IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                               dims.lhs_contracting_dimensions(0))
                     ->at(0)
                     .stride;
  // Just the fastest-varying part of it if the dimension is split.
  m = lhs_nc_iter_spec->at(0).count;

  CHECK_GE(m, 1);

  IndexT stride_rhs_k = 0;
  IndexT stride_rhs_n = 0;
  // Splitting of RHS non-contracting is not supported yet.
  CHECK_EQ(analysis
               .IterSpec(DotFusionAnalysis::Scope::RHS, rhs_param0,
                         rhs_noncontracting_dim_idx)
               ->size(),
           1);
  stride_rhs_k = analysis
                     .IterSpec(DotFusionAnalysis::Scope::RHS, rhs_param0,
                               dims.rhs_contracting_dimensions(0))
                     ->at(0)
                     .stride;
  stride_rhs_n = analysis
                     .IterSpec(DotFusionAnalysis::Scope::RHS, rhs_param0,
                               rhs_noncontracting_dim_idx)
                     ->at(0)
                     .stride;
  if (have_batch) {
    const int64_t rhs_batch_dim_idx = *(dims.rhs_batch_dimensions().cend() - 1);
    stride_rhs_batch = analysis
                           .IterSpec(DotFusionAnalysis::Scope::RHS, rhs_param0,
                                     rhs_batch_dim_idx)
                           ->at(0)
                           .stride;
    CHECK_GE(stride_rhs_batch, 1);
  }

  constexpr int group_m = 8;

  IndexT stride_out_m =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, lhs_nc_out_idx)
          ->at(0)
          .stride;
  const int64_t n =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, rhs_nc_out_idx)
          ->at(0)
          .count;
  CHECK_GE(n, 1);
  IndexT stride_out_n =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, rhs_nc_out_idx)
          ->at(0)
          .stride;
  IndexT stride_out_split_k = 0;
  if (have_split_k) {
    stride_out_split_k =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, split_k_out_idx)
            ->at(0)
            .stride;
    CHECK_GE(stride_out_split_k, 1);
  }
  IndexT stride_out_batch = 0;
  if (have_batch) {
    stride_out_batch =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, batch_out_idx)
            ->at(0)
            .stride;
    CHECK_GE(stride_out_batch, 1);
  } else if (lhs_nc_split) {
    // Dimension of the output produced by the non-contracting LHS one
    // is physically contiguous even if the producing LHS one is split.
    // Because the major part of the split is implemented using the batch
    // logic stride_out_batch is populated here as the stride of the minor
    // part times its size.
    stride_out_batch = stride_out_m * m;
  }

  const int block_m = config.block_m();
  const int block_k = config.block_k();
  const int block_n = config.block_n();

  CHECK_GE(block_m, 16);
  CHECK_GE(block_k, 16);
  CHECK_GE(block_n, 16);

  VLOG(3) << block_m << " " << block_k << " " << block_n << " "
          << config.num_warps() << " " << config.num_stages();

  const int grid_m = ceil(1.0 * m / block_m);
  const int grid_n = ceil(1.0 * n / block_n);
  const int width = group_m * grid_n;

  Type root_ty = TritonType(b, dot_instr->shape().element_type());
  // Data type to which dot() inputs are converted.
  Type dot_ty = b.getF32Type();
  if (lhs_ty.isF32() || rhs_ty.isF32()) {
    dot_ty = b.getF32Type();
  } else if (lhs_ty.isBF16() || rhs_ty.isBF16()) {
    dot_ty = b.getBF16Type();
  } else if (lhs_ty.isF16() || rhs_ty.isF16()) {
    dot_ty = b.getF16Type();
  }

  const int required_shmem_size = (block_m * lhs_ty.getIntOrFloatBitWidth() +
                                   block_n * rhs_ty.getIntOrFloatBitWidth()) *
                                  block_k * config.num_stages() / 8;
  if (required_shmem_size > shmem_budget) {
    return ResourceExhausted("Requires too much shared memory: %d > %d",
                             required_shmem_size, shmem_budget);
  }

  // TODO(b/266862493): Accumulator can be integer too.
  // Otherwise only f64 x f64 -> f64 uses f64 accumulator.
  mlir::FloatType acc_ty =
      (root_ty.isF64() && dot_ty.isF64()) ? b.getF64Type() : b.getF32Type();

  // X block size is 32-bit, Y and Z are 16-bit. Use X for large dimensions.
  constexpr int64_t kBlockCountYZLimit = 65536;
  const bool large_batch = batch_size >= kBlockCountYZLimit;
  auto pid_batch = b.create<mt::GetProgramIdOp>(
      large_batch ? mt::ProgramIDDim::X : mt::ProgramIDDim::Y);
  auto pid_nc = b.create<mt::GetProgramIdOp>(large_batch ? mt::ProgramIDDim::Y
                                                         : mt::ProgramIDDim::X);
  auto pid_k = b.create<mt::GetProgramIdOp>(mt::ProgramIDDim::Z);

  // In the imaginary situation where both batch size and grid_m * grid_n
  // are over 65535 we have to give up. Given the minimal m, n block sizes of 16
  // this requires at least 256 GB of output.
  CHECK_LT(batch_size * grid_m * grid_n,
           kBlockCountYZLimit * kBlockCountYZLimit);

  const LaunchDimensions launch_dimensions{
      {large_batch ? batch_size : grid_m * grid_n,
       large_batch ? grid_m * grid_n : batch_size, config.split_k()},
      {config.num_warps() * WarpSize(), 1, 1}};

  auto group_id = b.create<ma::DivSIOp>(pid_nc, CreateConst(b, i32_ty, width));
  ma::ConstantOp group_m_op = CreateConst(b, i32_ty, group_m);
  auto first_pid_m = b.create<ma::MulIOp>(group_id, group_m_op);
  auto sub0 = b.create<ma::SubIOp>(CreateConst(b, i32_ty, grid_m), first_pid_m);
  auto group_size = b.create<ma::SelectOp>(
      b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, sub0, group_m_op), sub0,
      group_m_op);

  // TODO(b/269489810): Contribute nicer builders to Triton, so we don't need to
  // define these utilities.

  // Extend int32 indexes to int64, if necessary.
  auto convert_scalar = [&](Value value) -> Value {
    if constexpr (std::is_same_v<IndexT, int64_t>) {
      return b.create<ma::ExtSIOp>(int_ty, value);
    }
    return value;
  };
  auto convert_range = [&](Value value) -> Value {
    if constexpr (std::is_same_v<IndexT, int64_t>) {
      auto type = mlir::RankedTensorType::get(
          value.dyn_cast<TensorValue>().getType().getShape(), int_ty);
      return b.create<ma::ExtSIOp>(type, value);
    }
    return value;
  };

  auto pid_m = b.create<ma::AddIOp>(first_pid_m,
                                    b.create<ma::RemSIOp>(pid_nc, group_size));
  auto pid_m_stride =
      b.create<ma::MulIOp>(pid_m, CreateConst(b, i32_ty, block_m));
  // TODO(b/270351731): Consider regenerating range_m to reduce register
  // pressure if we figure out how to make this optimization survive CSE.
  auto range_m =
      b.create<ma::AddIOp>(Splat(b, pid_m_stride, block_m), Range(b, block_m));

  auto pid_n = b.create<ma::DivSIOp>(
      b.create<ma::RemSIOp>(pid_nc, CreateConst(b, i32_ty, width)), group_size);
  auto pid_n_stride =
      b.create<ma::MulIOp>(pid_n, CreateConst(b, i32_ty, block_n));
  auto range_n =
      b.create<ma::AddIOp>(Splat(b, pid_n_stride, block_n), Range(b, block_n));

  auto range_k = b.create<ma::AddIOp>(
      Splat(b, b.create<ma::MulIOp>(pid_k, CreateConst(b, i32_ty, block_k)),
            block_k),
      Range(b, block_k));

  SmallVector<int64_t, 2> shape_m_1{block_m, 1};
  auto range_lhs_m = convert_range(
      b.create<ma::RemSIOp>(range_m, CreateConst(b, i32_ty, m, block_m)));
  auto lhs_offsets_m =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_lhs_m, 1),
                           CreateConst(b, int_ty, stride_lhs_m, shape_m_1));
  SmallVector<int64_t, 2> shape_1_k{1, block_k};
  auto lhs_offsets_k = b.create<ma::MulIOp>(
      b.create<mt::ExpandDimsOp>(convert_range(range_k), 0),
      CreateConst(b, int_ty, stride_lhs_k, shape_1_k));
  SmallVector<int64_t, 2> shape_m_k{block_m, block_k};
  auto lhs_offset_batch = b.create<ma::MulIOp>(
      convert_scalar(pid_batch), CreateConst(b, int_ty, stride_lhs_batch));
  auto lhs_offsets_init = b.create<ma::AddIOp>(
      Broadcast(b, lhs_offsets_m.getResult().template cast<TensorValue>(),
                shape_m_k),
      Broadcast(b, lhs_offsets_k.getResult().template cast<TensorValue>(),
                shape_m_k));
  lhs_offsets_init = b.create<ma::AddIOp>(
      lhs_offsets_init, Splat(b, lhs_offset_batch, shape_m_k));

  SmallVector<int64_t, 2> shape_k_1{block_k, 1};
  auto rhs_offsets_k = b.create<ma::MulIOp>(
      b.create<mt::ExpandDimsOp>(convert_range(range_k), 1),
      CreateConst(b, int_ty, stride_rhs_k, shape_k_1));
  SmallVector<int64_t, 2> shape_1_n{1, block_n};
  auto range_rhs_n = convert_range(
      b.create<ma::RemSIOp>(range_n, CreateConst(b, i32_ty, n, block_n)));
  auto rhs_offsets_n =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_rhs_n, 0),
                           CreateConst(b, int_ty, stride_rhs_n, shape_1_n));
  SmallVector<int64_t, 2> shape_k_n{block_k, block_n};
  auto rhs_offset_batch = b.create<ma::MulIOp>(
      convert_scalar(pid_batch), CreateConst(b, int_ty, stride_rhs_batch));
  auto rhs_offsets_init = b.create<ma::AddIOp>(
      Broadcast(b, rhs_offsets_k.getResult().template cast<TensorValue>(),
                shape_k_n),
      Broadcast(b, rhs_offsets_n.getResult().template cast<TensorValue>(),
                shape_k_n));
  rhs_offsets_init = b.create<ma::AddIOp>(
      rhs_offsets_init, Splat(b, rhs_offset_batch, shape_k_n));
  SmallVector<int64_t, 2> shape_m_n{block_m, block_n};
  ma::ConstantOp accumulator_init = CreateConst(b, acc_ty, 0, shape_m_n);

  auto body_builder = [&](mlir::OpBuilder&, mlir::Location, Value ki,
                          mlir::ValueRange iterArgs) {
    Value lhs_offsets = iterArgs[0];
    Value rhs_offsets = iterArgs[1];
    Value accumulator = iterArgs[2];
    Value lhs_mask = nullptr;
    Value rhs_mask = nullptr;
    Value zeros_like_lhs = nullptr;
    Value zeros_like_rhs = nullptr;
    // TODO(b/269726484): Peel the loop instead of inserting a masked load in
    // every iteration, even the ones that do not need it.
    if (k % (block_k * config.split_k()) > 0) {
      zeros_like_lhs = CreateConst(b, lhs_ty, 0, shape_m_k);
      zeros_like_rhs = CreateConst(b, rhs_ty, 0, shape_k_n);
      auto elements_in_tile =
          b.create<ma::SubIOp>(CreateConst(b, i32_ty, k), ki);
      lhs_mask =
          Broadcast(b,
                    b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                         b.create<mt::ExpandDimsOp>(range_k, 0),
                                         Splat(b, elements_in_tile, shape_1_k))
                        .getResult()
                        .template cast<TensorValue>(),
                    shape_m_k);
      rhs_mask =
          Broadcast(b,
                    b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                         b.create<mt::ExpandDimsOp>(range_k, 1),
                                         Splat(b, elements_in_tile, shape_k_1))
                        .getResult()
                        .template cast<TensorValue>(),
                    shape_k_n);
    }
    auto lhs_tile = b.create<mt::LoadOp>(
        AddPtr(
            b,
            Splat(b, fn.getArgument(lhs_param0->parameter_number()), shape_m_k),
            lhs_offsets),
        lhs_mask, zeros_like_lhs, mt::CacheModifier::NONE,
        mt::EvictionPolicy::NORMAL,
        /*isVolatile=*/false);
    auto rhs_tile = b.create<mt::LoadOp>(
        AddPtr(
            b,
            Splat(b, fn.getArgument(rhs_param0->parameter_number()), shape_k_n),
            rhs_offsets),
        rhs_mask, zeros_like_rhs, mt::CacheModifier::NONE,
        mt::EvictionPolicy::NORMAL,
        /*isVolatile=*/false);

    Value casted_lhs_tile = Cast(b, lhs_tile, dot_ty);
    Value casted_rhs_tile = Cast(b, rhs_tile, dot_ty);

    auto accumulator_next = b.create<mt::DotOp>(
        casted_lhs_tile, casted_rhs_tile, accumulator,
        /*allowTF32=*/tsl::tensor_float_32_execution_enabled());

    Value lhs_offsets_next = b.create<ma::AddIOp>(
        lhs_offsets,
        CreateConst(b, int_ty, block_k * config.split_k() * stride_lhs_k,
                    shape_m_k));
    Value rhs_offsets_next = b.create<ma::AddIOp>(
        rhs_offsets,
        CreateConst(b, int_ty, block_k * config.split_k() * stride_rhs_k,
                    shape_k_n));

    b.create<mlir::scf::YieldOp>(
        mlir::ValueRange{lhs_offsets_next, rhs_offsets_next, accumulator_next});
  };
  Value acc_final =
      b.create<mlir::scf::ForOp>(
           /*lowerBound=*/b.create<ma::ConstantIntOp>(0, /*width=*/32),
           /*upperBound=*/b.create<ma::ConstantIntOp>(k, /*width=*/32),
           /*step=*/
           b.create<ma::ConstantIntOp>(block_k * config.split_k(),
                                       /*width=*/32),
           /*iterArgs=*/
           mlir::ValueRange{lhs_offsets_init, rhs_offsets_init,
                            accumulator_init},
           body_builder)
          .getResult(2);

  // Output tile offsets.
  auto out_offset_batch = b.create<ma::MulIOp>(
      convert_scalar(pid_batch), CreateConst(b, int_ty, stride_out_batch));
  auto out_offset_split_k = b.create<ma::MulIOp>(
      convert_scalar(pid_k), CreateConst(b, int_ty, stride_out_split_k));
  auto out_offsets_m = b.create<ma::MulIOp>(
      b.create<mt::ExpandDimsOp>(convert_range(range_m), 1),
      CreateConst(b, int_ty, stride_out_m, shape_m_1));

  auto out_offsets_n = b.create<ma::MulIOp>(
      b.create<mt::ExpandDimsOp>(convert_range(range_n), 0),
      CreateConst(b, int_ty, stride_out_n, shape_1_n));
  auto out_offsets = b.create<ma::AddIOp>(Splat(b, out_offset_batch, shape_m_1),
                                          out_offsets_m);
  out_offsets = b.create<ma::AddIOp>(
      Broadcast(b, out_offsets.getResult().template cast<TensorValue>(),
                shape_m_n),
      Broadcast(b, out_offsets_n.getResult().template cast<TensorValue>(),
                shape_m_n));
  out_offsets = b.create<ma::AddIOp>(out_offsets,
                                     Splat(b, out_offset_split_k, shape_m_n));

  // Output tile store mask: check that the indices are within [M, N].
  auto rm_cmp = b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                     b.create<mt::ExpandDimsOp>(range_m, 1),
                                     CreateConst(b, i32_ty, m, shape_m_1));
  auto rn_cmp = b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                     b.create<mt::ExpandDimsOp>(range_n, 0),
                                     CreateConst(b, i32_ty, n, shape_1_n));
  auto out_mask = b.create<ma::AndIOp>(
      Broadcast(b, rm_cmp.getResult().template cast<TensorValue>(), shape_m_n),
      Broadcast(b, rn_cmp.getResult().template cast<TensorValue>(), shape_m_n));

  b.create<mt::StoreOp>(
      AddPtr(b, Splat(b, fn.getArguments().back(), shape_m_n), out_offsets),
      Cast(b, acc_final, root_ty), out_mask, mt::CacheModifier::NONE,
      mt::EvictionPolicy::NORMAL);
  return launch_dimensions;
}

}  // namespace

StatusOr<LaunchDimensions> MatMul(
    mlir::OpBuilder builder, const HloComputation* computation,
    mlir::triton::FuncOp fn,
    const tensorflow::AutotuneResult::TritonGemmKey& config, int shmem_budget) {
  const HloDotInstruction* dot_instr = DynCast<HloDotInstruction>(
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot));
  // Use 32-bit indexing if addressing any of the inputs or the output (which
  // could grow if split_k is set) does not cross the INT_MAX boundary.
  // Otherwise, fall back to 64-bit indexing, which is slower.
  bool use_64bit_indexing =
      ShapeUtil::ElementsIn(dot_instr->operand(0)->shape()) > INT_MAX ||
      ShapeUtil::ElementsIn(dot_instr->operand(1)->shape()) > INT_MAX ||
      ShapeUtil::ElementsIn(dot_instr->shape()) * config.split_k() > INT_MAX;
  if (use_64bit_indexing) {
    return MatMulImpl<int64_t>(builder, dot_instr, fn, config, shmem_budget);
  } else {
    return MatMulImpl<int32_t>(builder, dot_instr, fn, config, shmem_budget);
  }
}

StatusOr<LaunchDimensions> TritonWrapper(
    absl::string_view fn_name, const HloComputation* hlo_computation,
    const se::CudaComputeCapability& cc, const GpuDeviceInfo& device_info,
    const AutotuneResult::TritonGemmKey& config, llvm::Module* llvm_module,
    LaunchDimensionsGenerator generator, mlir::MLIRContext& mlir_context) {
  // This is a heuristic that serves as a proxy for register usage and code
  // size.
  //
  // We have noticed that tilings with very long LLVM IR code are both slow to
  // compile and slow to run. This can be for example due to register spills.
  // So we should skip these tilings to save time. But it's better to skip them
  // before the LLVM IR is generated. To do that, we came up with a formula that
  // strongly correlates with the LLVM IR size.
  // The formula is the size of the
  // two input and the output thread block tiles divided by the number of warps.
  // We read https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/ as a
  // reference, and found the formula by trial and error.
  //
  // To regenerate the limit, we have to run an exhaustive search on all
  // tilings for a few different HLOs, printing the runtimes and the heuristic
  // values.
  // From that, we can find a limit, such that all tilings within alpha *
  // optimal_runtime have a heuristic value less than or equal to the limit.
  //
  // In our measurements, all tilings which were within 1.13 * optimal_runtime
  // had a complexity_heuristic_value <= kComplexityHeuristicLimit.
  //
  // See go/tiling-heuristic for more details.
  constexpr int64_t kComplexityHeuristicLimit = 9000;
  int64_t complexity_heuristic_value =
      (config.block_m() * config.block_n() +
       (config.block_m() + config.block_n()) * config.block_k()) /
      config.num_warps();
  VLOG(2) << "Complexity heuristic: " << complexity_heuristic_value;
  if (complexity_heuristic_value > kComplexityHeuristicLimit) {
    return ResourceExhausted("Tiling complexity heuristic exceeded: %d > %d",
                             complexity_heuristic_value,
                             kComplexityHeuristicLimit);
  }

  mlir_context.loadDialect<mt::TritonDialect>();
  mlir::OpBuilder b(&mlir_context);
  auto loc = mlir::NameLoc::get(b.getStringAttr(hlo_computation->name()));
  auto triton_module = mlir::ModuleOp::create(loc);
  b.setInsertionPointToEnd(triton_module.getBody());

  VLOG(3) << hlo_computation->ToString();
  VLOG(2) << config.DebugString();

  // Build Triton kernel.
  Type root_ty = TritonType(
      b, hlo_computation->root_instruction()->shape().element_type());
  SmallVector<Type, 2> fn_arg_types;
  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    fn_arg_types.push_back(mt::PointerType::get(
        TritonType(b, p->shape().element_type()), mn::kGlobalMemorySpace));
  }

  fn_arg_types.push_back(mt::PointerType::get(root_ty, mn::kGlobalMemorySpace));

  auto fn = b.create<mt::FuncOp>(loc, fn_name,
                                 b.getFunctionType(fn_arg_types, std::nullopt));
  for (int i = 0; i < fn.getNumArguments(); ++i) {
    fn.setArgAttr(i, "tt.divisibility", b.getIntegerAttr(b.getI32Type(), 16));
  }
  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      generator(b, hlo_computation, fn, config,
                                device_info.shared_memory_per_block_optin));

  b.create<mt::ReturnOp>(loc);
  CHECK(mlir::succeeded(mlir::verify(triton_module)));

  VLOG(4) << llvm_ir::DumpToString(triton_module);

  // Compile Triton kernel to LLVM.
  mlir::PassManager pm(&mlir_context);

  std::optional<llvm::raw_fd_ostream> log_stream;
  const HloModule* hlo_module = hlo_computation->parent();
  if (hlo_module->config().debug_options().xla_gpu_dump_llvmir()) {
    const std::string basename =
        absl::StrCat(absl::string_view(tsl::io::Basename(hlo_module->name())),
                     ".triton-passes.log");
    std::string outputs_dir;
    if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
      outputs_dir = hlo_module->config().debug_options().xla_dump_to();
    }
    if (!outputs_dir.empty()) {
      std::string path = tsl::io::JoinPath(outputs_dir, basename);
      std::error_code err;
      log_stream.emplace(path, err, llvm::sys::fs::OF_None);
      if (err) {
        log_stream.reset();
      }
      auto print_before = [](mlir::Pass*, mlir::Operation*) { return true; };
      auto print_after = [](mlir::Pass*, mlir::Operation*) { return false; };
      pm.getContext()->disableMultithreading();
      pm.enableIRPrinting(print_before, print_after, /*printModuleScope=*/true,
                          /*printAfterOnlyOnChange=*/true,
                          /*printAfterOnlyOnFailure=*/false, *log_stream,
                          /*opPrintingFlags=*/{});
    } else {
      LOG(ERROR) << "--xla_gpu_dump_llvmir is set, but neither the environment "
                 << "variable TEST_UNDECLARED_OUTPUTS_DIR nor the flag "
                 << "--xla_dump_to is set, so the llvm dumps are disabled.";
    }
  }

  CreateTritonPipeline(pm, cc, config.num_warps(), config.num_stages());
  // Triton generates pointers to the global address space, while XLA needs a
  // kernel signature with pointers to the generic address space.
  pm.addPass(std::make_unique<GeneralizeKernelSignaturePass>());
  // llvm::Linker::linkModules() segfaults if we don't strip locations.
  pm.addPass(mlir::createStripDebugInfoPass());

  CHECK(mlir::succeeded(pm.run(triton_module)));

  if (log_stream.has_value()) {
    log_stream->flush();
  }

  // Integrate LLVM matmul kernel into XLA's LLVM module.
  const int shared_mem_bytes =
      triton_module->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared")
          .getInt();
  VLOG(2) << "Shared memory usage: " << shared_mem_bytes << " B";
  if (shared_mem_bytes > device_info.shared_memory_per_block_optin) {
    return ResourceExhausted("Shared memory size limit exceeded.");
  }
  launch_dimensions.SetSharedMemBytes(shared_mem_bytes);

  std::unique_ptr<llvm::Module> ll_triton_module = mt::translateLLVMToLLVMIR(
      &llvm_module->getContext(), triton_module, /*isROCM=*/false);
  LogAndVerify(ll_triton_module.get());
  for (auto& metadata :
       llvm::make_early_inc_range(ll_triton_module->named_metadata())) {
    ll_triton_module->eraseNamedMDNode(&metadata);
  }
  ll_triton_module->setDataLayout(llvm_module->getDataLayout());
  CHECK(!llvm::Linker::linkModules(*llvm_module, std::move(ll_triton_module)));
  LogAndVerify(llvm_module);

  return launch_dimensions;
}

}  // namespace gpu
}  // namespace xla
