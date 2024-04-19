/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/mlir/runtime/transforms/calling_convention.h"

#include <iterator>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "xla/mlir/runtime/ir/rt_dialect.h"

namespace xla {
namespace runtime {

CallingConvention DefaultCallingConvention() {
  return [](mlir::FunctionType func) {
    mlir::MLIRContext* ctx = func.getContext();

    llvm::SmallVector<mlir::Type> inputs = {ExecutionContextType::get(ctx)};
    inputs.reserve(1 + func.getNumInputs());
    llvm::append_range(inputs, func.getInputs());

    return mlir::FunctionType::get(ctx, inputs, func.getResults());
  };
}

CallingConvention DefaultCallingConvention(mlir::TypeConverter type_converter) {
  return [c = std::move(type_converter)](mlir::FunctionType func) mutable {
    mlir::MLIRContext* ctx = func.getContext();

    // Track if all type conversions were successful.
    bool failed_conversion = false;

    auto convert = [&](llvm::ArrayRef<mlir::Type> types,
                       llvm::SmallVector<mlir::Type>& converted) {
      llvm::for_each(types, [&](mlir::Type type) {
        mlir::LogicalResult result = c.convertType(type, converted);
        if (result.failed()) failed_conversion = true;
      });
    };
    // Add execution context as the first argument.
    llvm::SmallVector<mlir::Type> inputs = {ExecutionContextType::get(ctx)};
    inputs.reserve(1 + func.getNumInputs());
    convert(func.getInputs(), inputs);

    // Apply type conversion to all results types.
    llvm::SmallVector<mlir::Type> results;
    results.reserve(func.getNumResults());
    convert(func.getResults(), results);

    // Return null if any of the type conversions failed.
    if (failed_conversion) return mlir::FunctionType();

    return mlir::FunctionType::get(ctx, inputs, results);
  };
}

CallingConvention ResultsToOutsCallingConvention(
    mlir::TypeConverter type_converter) {
  return [c = std::move(type_converter)](mlir::FunctionType func) mutable {
    mlir::MLIRContext* ctx = func.getContext();

    // Track if all type conversions were successful.
    bool failed_conversion = false;

    auto convert = [&](llvm::ArrayRef<mlir::Type> types,
                       llvm::SmallVector<mlir::Type>& converted) {
      llvm::for_each(types, [&](mlir::Type type) {
        mlir::LogicalResult result = c.convertType(type, converted);
        if (result.failed()) failed_conversion = true;
      });
    };

    llvm::SmallVector<mlir::Type> inputs;
    inputs.reserve(1 + func.getNumInputs() + func.getNumResults());
    inputs.push_back(ExecutionContextType::get(ctx));
    convert(func.getInputs(), inputs);
    convert(func.getResults(), inputs);

    // Return null if any of the type conversions failed.
    if (failed_conversion) return mlir::FunctionType();

    return mlir::FunctionType::get(ctx, inputs, {});
  };
}

}  // namespace runtime
}  // namespace xla
