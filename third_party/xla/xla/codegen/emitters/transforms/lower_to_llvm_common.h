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
#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWER_TO_LLVM_COMMON_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWER_TO_LLVM_COMMON_H_

#include "absl/functional/function_ref.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xla {
namespace emitters {

mlir::LogicalResult LowerToLLVM(
    mlir::ModuleOp op,
    absl::FunctionRef<mlir::LogicalResult(mlir::LLVMTypeConverter&,
                                          mlir::RewritePatternSet&,
                                          mlir::ConversionTarget&)>
        populate_platform_patterns =
            [](mlir::LLVMTypeConverter&, mlir::RewritePatternSet&,
               mlir::ConversionTarget&) { return mlir::success(); },
    bool lower_math_log1p = false);

}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWER_TO_LLVM_COMMON_H_
