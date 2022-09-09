/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/transforms/runtime/calling_convention.h"

#include <iterator>
#include <utility>

#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.h"

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
    auto convert = [&](mlir::Type type) -> mlir::Type {
      auto converted = c.convertType(type);
      if (!converted) failed_conversion = true;
      return converted;
    };

    // Add execution context as the first argument.
    llvm::SmallVector<mlir::Type> inputs = {ExecutionContextType::get(ctx)};
    inputs.reserve(1 + func.getNumInputs());
    llvm::transform(func.getInputs(), std::back_inserter(inputs), convert);

    // Apply type conversion to all results types.
    llvm::SmallVector<mlir::Type> results;
    results.reserve(func.getNumResults());
    llvm::transform(func.getResults(), std::back_inserter(results), convert);

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

    auto convert = [&](mlir::Type type) -> mlir::Type {
      auto converted = c.convertType(type);
      if (!converted) failed_conversion = true;
      return converted;
    };

    llvm::SmallVector<mlir::Type> inputs;
    inputs.reserve(1 + func.getNumInputs() + func.getNumResults());
    inputs.push_back(ExecutionContextType::get(ctx));
    llvm::transform(func.getInputs(), std::back_inserter(inputs), convert);
    llvm::transform(func.getResults(), std::back_inserter(inputs), convert);

    // Return null if any of the type conversions failed.
    if (failed_conversion) return mlir::FunctionType();

    return mlir::FunctionType::get(ctx, inputs, {});
  };
}

}  // namespace runtime
}  // namespace xla
