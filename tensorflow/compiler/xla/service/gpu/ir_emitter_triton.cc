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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>  // NOLINT(build/c++11): required to interface with LLVM
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
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
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/path.h"
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
Value Cast(mlir::OpBuilder b, mlir::Location loc, Value value,
           Type dst_element_ty) {
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
      return b.create<ma::TruncFOp>(loc, dst_ty,
                                    b.create<ma::ExtFOp>(loc, fp32_ty, value));
    } else if (src_fp_element_ty.getFPMantissaWidth() >
               dst_fp_element_ty.getFPMantissaWidth()) {
      return b.create<ma::TruncFOp>(loc, dst_ty, value);
    } else {
      return b.create<ma::ExtFOp>(loc, dst_ty, value);
    }
  }
  // int => float
  if (src_element_ty.isa<mlir::IntegerType>() && dst_fp_element_ty) {
    // TODO(b/266862493): Support unsigned integer types.
    return b.create<ma::SIToFPOp>(loc, dst_ty, value);
  }
  // float => int
  if (src_fp_element_ty && dst_element_ty.isa<mlir::IntegerType>()) {
    // TODO(b/266862493): Support unsigned integer types.
    return b.create<ma::FPToSIOp>(loc, dst_ty, value);
  }

  LOG(FATAL) << "Type conversion not supported: "
             << llvm_ir::DumpToString(src_element_ty) << " -> "
             << llvm_ir::DumpToString(dst_element_ty);
}

// Create a scalar constant.
ma::ConstantOp CreateConst(mlir::ImplicitLocOpBuilder b, Type type, int value) {
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
ma::ConstantOp CreateConst(mlir::ImplicitLocOpBuilder b, Type type, int value,
                           mlir::ArrayRef<int64_t> shape) {
  auto tensor_type = mlir::RankedTensorType::get(shape, type);
  if (auto int_type = type.dyn_cast<mlir::IntegerType>()) {
    return b.create<ma::ConstantOp>(mlir::DenseElementsAttr::get(
        tensor_type, mlir::APInt(int_type.getIntOrFloatBitWidth(), value)));
  }
  if (auto float_type = type.dyn_cast<mlir::FloatType>()) {
    return b.create<ma::ConstantOp>(mlir::DenseElementsAttr::get(
        tensor_type, mlir::APFloat(float_type.getFloatSemantics(), value)));
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

void CreateTritonPipeline(mlir::OpPassManager& pm,
                          const se::CudaComputeCapability& cc, int num_warps,
                          int num_stages) {
  const int ccAsInt = cc.major * 10 + cc.minor;
  // Based on optimize_triton_ir() in
  // @triton//:python/triton/compiler.py
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mt::createCombineOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Based on ttir_to_ttgir() in
  // @triton//:python/triton/compiler.py
  pm.addPass(mt::createConvertTritonToTritonGPUPass(num_warps));
  // Based on optimize_ttgir() in
  // @triton//:python/triton/compiler.py
  pm.addPass(mlir::createTritonGPUCoalescePass());
  pm.addPass(mlir::createTritonGPUAccelerateMatmulPass(ccAsInt));
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUFuseTranspositionsPass());
  pm.addPass(mlir::createTritonGPUPipelinePass(num_stages));
  pm.addPass(mlir::createTritonGPUPrefetchPass());
  pm.addPass(mlir::createTritonGPUFuseTranspositionsPass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUDecomposeConversionsPass());
  if (cc.major == se::CudaComputeCapability::VOLTA) {
    pm.addPass(mlir::createTritonGPUUpdateMmaForVoltaPass());
  }
  pm.addPass(mlir::createTritonGPUReorderInstructionsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Based on translateTritonGPUToLLVMIR() in
  // @triton//:lib/Target/LLVMIR/LLVMIRTranslation.cpp
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mt::createConvertTritonGPUToLLVMPass(ccAsInt));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

// Extact additional attributes from an LLVM function that are not passed
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

}  // namespace

// Variable naming: lhs [m, k] x rhs [k, n] -> out [m, n].
// TODO(b/270937368): Split this up into smaller functions.
std::optional<LaunchDimensions> MatMul(
    mlir::OpBuilder builder, const HloDotInstruction* dot_instr,
    mlir::func::FuncOp fn,
    const tensorflow::AutotuneResult::TritonGemmKey& config, int shmem_budget) {
  // We'll be creating a lot of instructions from a single dot, use an
  // implicit loc builder so we don't have to pass around the location all the
  // time.
  auto loc = mlir::NameLoc::get(builder.getStringAttr(dot_instr->name()));
  mlir::ImplicitLocOpBuilder b(loc, builder);
  Type i32_ty = b.getI32Type();
  const DotDimensionNumbers& dims = dot_instr->dot_dimension_numbers();
  const DotFusionAnalysis analysis(dot_instr);
  const HloInstruction* hlo_lhs_param = analysis.OperandToParameter(0);
  const HloInstruction* hlo_rhs_param = analysis.OperandToParameter(1);

  Type lhs_ty = TritonType(b, hlo_lhs_param->shape().element_type());
  Type rhs_ty = TritonType(b, hlo_rhs_param->shape().element_type());

  // Rely on dot decomposer: there is just one contracting and one
  // non-contracting dimension on each side + one batch optionally.
  CHECK_EQ(dims.lhs_contracting_dimensions_size(), 1);
  CHECK_EQ(dims.rhs_contracting_dimensions_size(), 1);
  CHECK_LE(dims.lhs_batch_dimensions_size(), 1);
  const bool batch = !dims.lhs_batch_dimensions().empty();
  CHECK_EQ(dot_instr->operand(0)->shape().rank(), 2 + batch);
  const int lhs_noncontracting_dim_idx =
      NoncontractingDimensionIndex(dims.lhs_contracting_dimensions(0),
                                   batch ? dims.lhs_batch_dimensions(0) : -1);
  const int rhs_noncontracting_dim_idx =
      NoncontractingDimensionIndex(dims.rhs_contracting_dimensions(0),
                                   batch ? dims.rhs_batch_dimensions(0) : -1);

  // Non-contracting dimension lengths.
  // Just the fastest-varying part of it if the dimension is split.
  const int m = analysis.IterSpec(0, lhs_noncontracting_dim_idx)[0].count;
  const int n = analysis.IterSpec(1, rhs_noncontracting_dim_idx)[0].count;

  // Contracting dimension length.
  const int k = dot_instr->operand(0)->shape().dimensions(
      dims.lhs_contracting_dimensions(0));

  // LHS non-contracting can be split into two.
  const bool lhs_nc_split =
      (analysis.IterSpec(0, lhs_noncontracting_dim_idx).size() > 1);
  CHECK_EQ(analysis.IterSpec(0, lhs_noncontracting_dim_idx).size(),
           1 + lhs_nc_split);
  // For now split and batch are not supported simultaneously because they
  // are implemented via same mechanism.
  CHECK_LE(batch + lhs_nc_split, 1);
  // Splitting of the other ones is not supported yet.
  CHECK_EQ(analysis.IterSpec(1, rhs_noncontracting_dim_idx).size(), 1);
  CHECK_EQ(analysis.IterSpec(0, dims.lhs_contracting_dimensions(0)).size(), 1);

  const int stride_lhs_m =
      analysis.IterSpec(0, lhs_noncontracting_dim_idx)[0].stride;
  const int stride_lhs_k =
      analysis.IterSpec(0, dims.lhs_contracting_dimensions(0))[0].stride;
  const int stride_rhs_k =
      analysis.IterSpec(1, dims.rhs_contracting_dimensions(0))[0].stride;
  const int stride_rhs_n =
      analysis.IterSpec(1, rhs_noncontracting_dim_idx)[0].stride;

  // Either batch size or upper part of the length of a split nc dimension.
  int batch_size = 1;
  int stride_batch_lhs = 0;
  int stride_batch_rhs = 0;
  if (lhs_nc_split) {
    batch_size = analysis.IterSpec(0, lhs_noncontracting_dim_idx)[1].count;
    stride_batch_lhs =
        analysis.IterSpec(0, lhs_noncontracting_dim_idx)[1].stride;
    stride_batch_rhs = 0;
  } else if (batch) {
    // Batch dimension should have same length left and right.
    CHECK_EQ(analysis.IterSpec(0, dims.lhs_batch_dimensions(0))[0].count,
             analysis.IterSpec(1, dims.rhs_batch_dimensions(0))[0].count);
    batch_size = analysis.IterSpec(0, dims.lhs_batch_dimensions(0))[0].count;
    stride_batch_lhs =
        analysis.IterSpec(0, dims.lhs_batch_dimensions(0))[0].stride;
    stride_batch_rhs =
        analysis.IterSpec(1, dims.rhs_batch_dimensions(0))[0].stride;
  }

  constexpr int64_t group_m = 8;

  bool transpose_output =
      !LayoutUtil::IsMonotonicWithDim0Major(dot_instr->shape().layout());
  const int stride_out_m = transpose_output ? 1 : n;
  const int stride_out_n = transpose_output ? m : 1;

  const int block_m = config.block_m();
  const int block_k = config.block_k();
  const int block_n = config.block_n();
  CHECK_GE(block_m, 32);
  CHECK_GE(block_k, 32);
  CHECK_GE(block_n, 32);

  VLOG(3) << block_m << " " << block_k << " " << block_n << " "
          << config.num_warps() << " " << config.num_stages();

  const int grid_m = ceil(1.0 * m / block_m);
  const int grid_n = ceil(1.0 * n / block_n);
  const int width = group_m * grid_n;

  // TODO(b/266863137): handle atomic add for split_k > 1.
  // This also requires output zero-init.
  const unsigned int split_k = config.split_k();
  CHECK_EQ(split_k, 1);
  const LaunchDimensions launch_dimensions{
      {grid_m * grid_n, split_k, batch_size},
      {config.num_warps() * WarpSize(), 1, 1}};
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
  // TODO(b/266857785): Add dynamic shared memory size.
  if (required_shmem_size > shmem_budget) {
    VLOG(2) << "Requires too much shared memory: " << required_shmem_size
            << " B.";
    return std::nullopt;
  }

  // TODO(b/266862493): Accumulator can be integer too.
  // Otherwise only f64 x f64 -> f64 uses f64 accumulator.
  mlir::FloatType acc_ty =
      (root_ty.isF64() && dot_ty.isF64()) ? b.getF64Type() : b.getF32Type();
  Value lhs = fn.getArgument(hlo_lhs_param->parameter_number());
  Value rhs = fn.getArgument(hlo_rhs_param->parameter_number());
  Value out = fn.getArguments().back();

  auto pid0 = b.create<mt::GetProgramIdOp>(0);
  auto pid1 = b.create<mt::GetProgramIdOp>(1);
  auto pid2 = b.create<mt::GetProgramIdOp>(2);

  auto group_id = b.create<ma::DivSIOp>(pid0, CreateConst(b, i32_ty, width));
  ma::ConstantOp group_m_op = CreateConst(b, i32_ty, group_m);
  auto first_pid_m = b.create<ma::MulIOp>(group_id, group_m_op);
  auto sub0 = b.create<ma::SubIOp>(CreateConst(b, i32_ty, grid_m), first_pid_m);
  auto group_size = b.create<ma::SelectOp>(
      b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, sub0, group_m_op), sub0,
      group_m_op);

  // TODO(b/269489810): Contribute nicer builders to Triton, so we don't need to
  // define these utilities.

  auto build_splat = [&](Value value, mlir::ArrayRef<int64_t> shape) {
    auto type = mlir::RankedTensorType::get(shape, value.getType());
    return b.create<mt::SplatOp>(type, value);
  };

  auto build_range = [&](uint32_t start, uint32_t end) {
    auto type = mlir::RankedTensorType::get(end - start, b.getI32Type());
    return b.create<mt::MakeRangeOp>(type, start, end);
  };

  using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;
  auto build_bcast = [&](TensorValue value, mlir::ArrayRef<int64_t> shape) {
    auto type =
        mlir::RankedTensorType::get(shape, value.getType().getElementType());
    return b.create<mt::BroadcastOp>(type, value);
  };

  auto build_addptr = [&](Value ptr, Value offset) {
    return b.create<mt::AddPtrOp>(ptr.getType(), ptr, offset);
  };

  auto pid_m = b.create<ma::AddIOp>(first_pid_m,
                                    b.create<ma::RemSIOp>(pid0, group_size));
  auto pid_m_stride =
      b.create<ma::MulIOp>(pid_m, CreateConst(b, i32_ty, block_m));
  // TODO(b/270351731): Consider regenerating range_m to reduce register
  // pressure if we figure out how to make this optimization survive CSE.
  auto range_m = b.create<ma::AddIOp>(build_splat(pid_m_stride, block_m),
                                      build_range(0, block_m));

  auto pid_n = b.create<ma::DivSIOp>(
      b.create<ma::RemSIOp>(pid0, CreateConst(b, i32_ty, width)), group_size);
  auto pid_n_stride =
      b.create<ma::MulIOp>(pid_n, CreateConst(b, i32_ty, block_n));
  auto range_n = b.create<ma::AddIOp>(build_splat(pid_n_stride, block_n),
                                      build_range(0, block_n));

  auto range_k = b.create<ma::AddIOp>(
      build_splat(b.create<ma::MulIOp>(pid1, CreateConst(b, i32_ty, block_k)),
                  block_k),
      build_range(0, block_k));

  SmallVector<int64_t, 2> shape_m_1{block_m, 1};
  auto range_lhs_m =
      b.create<ma::RemSIOp>(range_m, CreateConst(b, i32_ty, m, block_m));
  auto lhs_offset_m =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_lhs_m, 1),
                           CreateConst(b, i32_ty, stride_lhs_m, shape_m_1));
  SmallVector<int64_t, 2> shape_1_k{1, block_k};
  auto lhs_offset_k =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_k, 0),
                           CreateConst(b, i32_ty, stride_lhs_k, shape_1_k));
  SmallVector<int64_t, 2> shape_m_k{block_m, block_k};
  auto lhs_offset = b.create<ma::AddIOp>(
      build_bcast(lhs_offset_m.getResult().cast<TensorValue>(), shape_m_k),
      build_bcast(lhs_offset_k.getResult().cast<TensorValue>(), shape_m_k));
  auto lhs_offset_batch =
      b.create<ma::MulIOp>(pid2, CreateConst(b, i32_ty, stride_batch_lhs));
  mt::AddPtrOp lhs_ptrs_base = build_addptr(
      build_splat(build_addptr(lhs, lhs_offset_batch), shape_m_k), lhs_offset);

  SmallVector<int64_t, 2> shape_k_1{block_k, 1};
  auto rhs_off_k =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_k, 1),
                           CreateConst(b, i32_ty, stride_rhs_k, shape_k_1));
  SmallVector<int64_t, 2> shape_1_n{1, block_n};
  auto range_rhs_n =
      b.create<ma::RemSIOp>(range_n, CreateConst(b, i32_ty, n, block_n));
  auto rhs_offset_n =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_rhs_n, 0),
                           CreateConst(b, i32_ty, stride_rhs_n, shape_1_n));
  SmallVector<int64_t, 2> shape_k_n{block_k, block_n};
  auto rhs_offset = b.create<ma::AddIOp>(
      build_bcast(rhs_off_k.getResult().cast<TensorValue>(), shape_k_n),
      build_bcast(rhs_offset_n.getResult().cast<TensorValue>(), shape_k_n));
  auto rhs_offset_batch =
      b.create<ma::MulIOp>(pid2, CreateConst(b, i32_ty, stride_batch_rhs));
  mt::AddPtrOp rhs_ptrs_base = build_addptr(
      build_splat(build_addptr(rhs, rhs_offset_batch), shape_k_n), rhs_offset);
  SmallVector<int64_t, 2> shape_m_n{block_m, block_n};
  ma::ConstantOp acc_init = CreateConst(b, acc_ty, 0, shape_m_n);

  auto body_builder = [&](mlir::OpBuilder&, mlir::Location, Value ki,
                          mlir::ValueRange iterArgs) {
    Value lhs_ptrs = iterArgs[0];
    Value rhs_ptrs = iterArgs[1];
    Value acc = iterArgs[2];
    Value lhs_mask = nullptr;
    Value rhs_mask = nullptr;
    Value zeros_like_lhs = nullptr;
    Value zeros_like_rhs = nullptr;
    // TODO(b/269726484): Peel the loop instead of inserting a masked load in
    // every iteration, even the ones that do not need it.
    if (k % (block_k * split_k) > 0) {
      zeros_like_lhs = CreateConst(b, lhs_ty, 0, shape_m_k);
      zeros_like_rhs = CreateConst(b, rhs_ty, 0, shape_k_n);
      auto elements_in_tile =
          b.create<ma::SubIOp>(CreateConst(b, i32_ty, k),
                               b.create<ma::IndexCastOp>(b.getI32Type(), ki));
      lhs_mask = build_bcast(
          b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                               b.create<mt::ExpandDimsOp>(range_k, 0),
                               build_splat(elements_in_tile, shape_1_k))
              .getResult()
              .cast<TensorValue>(),
          shape_m_k);
      rhs_mask = build_bcast(
          b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                               b.create<mt::ExpandDimsOp>(range_k, 1),
                               build_splat(elements_in_tile, shape_k_1))
              .getResult()
              .cast<TensorValue>(),
          shape_k_n);
    }
    auto lhs_tile = b.create<mt::LoadOp>(lhs_ptrs, lhs_mask, zeros_like_lhs,
                                         mt::CacheModifier::NONE,
                                         mt::EvictionPolicy::NORMAL,
                                         /*isVolatile=*/false);
    auto rhs_tile = b.create<mt::LoadOp>(
        rhs_ptrs, rhs_mask, zeros_like_rhs, mt::CacheModifier::NONE,
        mt::EvictionPolicy::NORMAL, /*isVolatile=*/false);

    Value casted_lhs_tile = Cast(b, loc, lhs_tile, dot_ty);
    Value casted_rhs_tile = Cast(b, loc, rhs_tile, dot_ty);

    auto acc_next = b.create<mt::DotOp>(casted_lhs_tile, casted_rhs_tile, acc,
                                        /*allowTF32=*/true);

    mt::AddPtrOp lhs_ptrs_inc = build_addptr(
        lhs_ptrs,
        CreateConst(b, i32_ty, block_k * split_k * stride_lhs_k, shape_m_k));
    mt::AddPtrOp rhs_ptrs_inc = build_addptr(
        rhs_ptrs,
        CreateConst(b, i32_ty, block_k * split_k * stride_rhs_k, shape_k_n));

    b.create<mlir::scf::YieldOp>(
        mlir::ValueRange{lhs_ptrs_inc, rhs_ptrs_inc, acc_next});
  };
  Value acc_final =
      b.create<mlir::scf::ForOp>(
           /*lowerBound=*/b.create<ma::ConstantIndexOp>(0),
           /*upperBound=*/b.create<ma::ConstantIndexOp>(k),
           /*step=*/b.create<ma::ConstantIndexOp>(block_k * split_k),
           /*iterArgs=*/
           mlir::ValueRange{lhs_ptrs_base, rhs_ptrs_base, acc_init},
           body_builder)
          .getResult(2);

  // Output tile offsets.
  auto out_offset_batch =
      b.create<ma::MulIOp>(pid2, CreateConst(b, i32_ty, m * n));
  auto out_offset_m =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_m, 1),
                           CreateConst(b, i32_ty, stride_out_m, shape_m_1));
  mt::AddPtrOp out_ptrs_m =
      build_addptr(build_splat(build_addptr(out, out_offset_batch), shape_m_1),
                   out_offset_m);

  auto out_offset_n =
      b.create<ma::MulIOp>(b.create<mt::ExpandDimsOp>(range_n, 0),
                           CreateConst(b, i32_ty, stride_out_n, shape_1_n));
  mt::AddPtrOp out_ptrs = build_addptr(
      build_bcast(out_ptrs_m.getResult().cast<TensorValue>(), shape_m_n),
      build_bcast(out_offset_n.getResult().cast<TensorValue>(), shape_m_n));

  // Output tile store mask: check that the indices are within [M, N].
  auto rm_cmp = b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                     b.create<mt::ExpandDimsOp>(range_m, 1),
                                     CreateConst(b, i32_ty, m, shape_m_1));
  auto rn_cmp = b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                                     b.create<mt::ExpandDimsOp>(range_n, 0),
                                     CreateConst(b, i32_ty, n, shape_1_n));
  auto mask = b.create<ma::AndIOp>(
      build_bcast(rm_cmp.getResult().cast<TensorValue>(), shape_m_n),
      build_bcast(rn_cmp.getResult().cast<TensorValue>(), shape_m_n));

  b.create<mt::StoreOp>(out_ptrs, Cast(b, loc, acc_final, root_ty), mask);
  return launch_dimensions;
}

std::optional<LaunchDimensions> TritonWrapper(
    absl::string_view fn_name, const HloComputation* hlo_computation,
    const se::CudaComputeCapability& cc, const GpuDeviceInfo& device_info,
    const AutotuneResult::TritonGemmKey& config, llvm::Module* llvm_module,
    LaunchDimensionsGenerator generator) {
  // TODO(b/264317991): Pass in a context instead if this becomes to slow.
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mt::TritonDialect>();
  mlir::OpBuilder b(&mlir_context);
  auto loc = mlir::NameLoc::get(b.getStringAttr(hlo_computation->name()));
  auto triton_module = mlir::ModuleOp::create(loc);
  b.setInsertionPointToEnd(triton_module.getBody());

  const HloInstruction* root =
      (hlo_computation->root_instruction()->opcode() == HloOpcode::kBitcast)
          ? hlo_computation->root_instruction()->operand(0)
          : hlo_computation->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kDot);
  VLOG(3) << root->parent()->ToString();

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

  auto fn = b.create<mlir::func::FuncOp>(
      loc, fn_name, b.getFunctionType(fn_arg_types, std::nullopt));
  for (int i = 0; i < fn.getNumArguments(); ++i) {
    fn.setArgAttr(i, "tt.divisibility", b.getIntegerAttr(b.getI32Type(), 16));
  }
  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  std::optional<LaunchDimensions> launch_dimensions =
      generator(b, ::xla::Cast<HloDotInstruction>(root), fn, config,
                device_info.shared_memory_per_block);
  if (!launch_dimensions.has_value()) {
    return std::nullopt;
  }
  b.create<mlir::func::ReturnOp>(loc);
  CHECK(mlir::succeeded(mlir::verify(triton_module)));

  VLOG(4) << triton_module;

  // Compile Triton kernel to LLVM.
  mlir::PassManager pm(&mlir_context);

  std::optional<llvm::raw_fd_ostream> log_stream;
  const HloModule* hlo_module = hlo_computation->parent();
  if (hlo_module->config().debug_options().xla_gpu_dump_llvmir()) {
    const std::string basename =
        absl::StrCat(absl::string_view(tsl::io::Basename(hlo_module->name())),
                     ".triton-passes.log");
    std::string outputs_dir;
    tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir);
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
  int shared_mem_bytes =
      triton_module->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared")
          .getInt();
  VLOG(2) << "Shared memory usage: " << shared_mem_bytes << " B";
  // TODO(b/266857785): Add dynamic shared memory size.
  if (shared_mem_bytes > device_info.shared_memory_per_block) {
    LOG(WARNING) << "Shared memory size limit exceeded.";
    return std::nullopt;
  }
  launch_dimensions->SetSharedMemBytes(shared_mem_bytes);

  std::unique_ptr<llvm::Module> ll_triton_module =
      mt::translateLLVMToLLVMIR(&llvm_module->getContext(), triton_module);
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
