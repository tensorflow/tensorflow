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

#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWERING_UTILS_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWERING_UTILS_H_

#include "absl/strings/str_format.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Operation.h"

namespace xla {
namespace emitters {

// Ensure AMDGPU allocas use address space 5 (private).
// AMDGPU requires allocas in AS5, but MLIR lowering creates them in AS0.
void EnsureAMDGPUAllocasUseAS5(mlir::Operation* operation);

namespace spirv {
namespace mm = ::mlir::math;
namespace ml = ::mlir::LLVM;
template <typename... Ops>
struct SPIRVMathOps {};

inline auto getSPIRVMathOps() {
  return SPIRVMathOps<
      mm::AcosOp, mm::AcoshOp, mm::AsinOp, mm::AsinhOp, mm::Atan2Op, mm::AtanOp,
      mm::AtanhOp, mm::CosOp, mm::CoshOp, mm::ExpM1Op, mm::ExpOp, mm::Log1pOp,
      mm::LogOp, mm::SinOp, mm::SinhOp, mm::TanOp, mm::TanhOp>{};
}

template <typename Op>
struct OpToSPVFuncCallLowering : public mlir::ConvertOpToLLVMPattern<Op> {
 public:
  explicit OpToSPVFuncCallLowering(const mlir::LLVMTypeConverter& converter,
                                   mlir::StringRef base)
      : mlir::ConvertOpToLLVMPattern<Op>(converter, 1), base(base) {}

  mlir::LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type = op->getResultTypes().front();
    mlir::SmallVector<mlir::Type> arg_types(adaptor.getOperands().getTypes());

    bool can_lower =
        mlir::isa<mlir::Float32Type, mlir::Float64Type>(result_type);
    for (mlir::Type type : arg_types) {
      can_lower = can_lower && (type == result_type);
    }

    if (!can_lower) {
      return rewriter.notifyMatchFailure(
          op, " expected F32 or F64 result and operand types");
    }

    std::string name =
        base + (mlir::isa<mlir::Float32Type>(result_type) ? "f" : "d");

    auto symbol_table =
        op->template getParentWithTrait<mlir::OpTrait::SymbolTable>();
    auto func = mlir::dyn_cast_or_null<ml::LLVMFuncOp>(
        mlir::SymbolTable::lookupSymbolIn(symbol_table, name));
    if (!func) {
      mlir::OpBuilder b(symbol_table->getRegion(0));
      func = ml::LLVMFuncOp::create(
          b, symbol_table->getLoc(), name,
          ml::LLVMFunctionType::get(result_type, arg_types));
      func.setCConv(ml::cconv::CConv::SPIR_FUNC);
    }
    auto call =
        ml::CallOp::create(rewriter, op->getLoc(), func, adaptor.getOperands());
    call.setCConv(func.getCConv());

    rewriter.replaceOp(op, {call.getResult()});
    return mlir::success();
  }

  const std::string base;
};

// Lowers single/double precision math ops to SPIR-V OCL driver intrinsics to
// preserve accuracy, particularly for small-magnitude inputs where generic MLIR
// lowering may lose precision.
template <typename Op>
inline void populateLLVMSPVMathOpPatterns(mlir::LLVMTypeConverter& converter,
                                          mlir::RewritePatternSet& patterns) {
  auto op_name =
      mlir::OperationName(Op::getOperationName(), &converter.getContext())
          .stripDialect();
  auto base =
      absl::StrFormat("_Z%u__spirv_ocl_%s", op_name.size() + 12, op_name.str());
  patterns.add<OpToSPVFuncCallLowering<Op>>(converter, base);
}

template <typename... Ops>
void populateMathToLLVMSPVConversionPatterns(
    spirv::SPIRVMathOps<Ops...>, mlir::LLVMTypeConverter& converter,
    mlir::RewritePatternSet& patterns) {
  (spirv::populateLLVMSPVMathOpPatterns<Ops>(converter, patterns), ...);
}

}  // namespace spirv
}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWERING_UTILS_H_
