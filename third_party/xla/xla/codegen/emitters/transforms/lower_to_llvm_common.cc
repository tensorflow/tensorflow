/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/codegen/emitters/transforms/lower_to_llvm_common.h"

#include <utility>

#include "absl/functional/function_ref.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xla {
namespace emitters {

mlir::LogicalResult LowerToLLVM(
    mlir::ModuleOp op,
    absl::FunctionRef<mlir::LogicalResult(mlir::LLVMTypeConverter&,
                                          mlir::RewritePatternSet&,
                                          mlir::ConversionTarget&)>
        populate_platform_patterns,
    bool lower_math_log1p) {
  // Populate type conversions.
  mlir::LowerToLLVMOptions llvm_opts(op.getContext(), mlir::DataLayout(op));
  mlir::LLVMTypeConverter type_converter(op.getContext(), llvm_opts);
  mlir::LLVMConversionTarget target(*op.getContext());

  // Populate patterns.
  mlir::RewritePatternSet patterns(op.getContext());
  mlir::arith::populateArithExpandOpsPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(type_converter, patterns);
  if (mlir::failed(
          populate_platform_patterns(type_converter, patterns, target))) {
    return mlir::failure();
  }
  mlir::populateFuncToLLVMConversionPatterns(type_converter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(type_converter,
                                                       patterns);
  mlir::ub::populateUBToLLVMConversionPatterns(type_converter, patterns);
  mlir::populateVectorToLLVMConversionPatterns(type_converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(type_converter,
                                                        patterns);
  mlir::populateComplexToLLVMConversionPatterns(type_converter, patterns);

  //  Set up target.
  target.addIllegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                           mlir::complex::ComplexDialect>();
  target.addLegalOp<mlir::ModuleOp>();

  if (mlir::failed(applyPartialConversion(op, target, std::move(patterns)))) {
    return mlir::failure();
  }

  // Clean up any leftover math ops.
  mlir::RewritePatternSet mathPatterns(op.getContext());
  mlir::populateMathToLLVMConversionPatterns(type_converter, mathPatterns,
                                             lower_math_log1p);
  target.addIllegalDialect<mlir::math::MathDialect>();

  if (mlir::failed(applyFullConversion(op, target, std::move(mathPatterns)))) {
    return mlir::failure();
  }
  return mlir::success();
}

}  // namespace emitters
}  // namespace xla
