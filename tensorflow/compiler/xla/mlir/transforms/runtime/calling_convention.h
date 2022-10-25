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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_RUNTIME_CALLING_CONVENTION_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_RUNTIME_CALLING_CONVENTION_H_

#include <functional>

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace xla {
namespace runtime {

// Calling convention converts exported function types to function types with
// a well-defined ABI (e.g. tensors do not have an ABI; they must be passed
// across the function boundary as memrefs). In a nutshell it tells the XLA
// runtime how to call the compiled executable at run time, and how to return
// results back to the caller.
//
// All types in the converted function signature should have a registered
// type conversion (see `type_converter` below) to a type with defined
// argument or result ABI (see Type::ArgumentAbi and Type::ResultAbi).
//
// If conversion is not possible, calling convention must return a null value.
//
// Example: abstract executable defined in high level dialect, e.g. MHLO
//
//   ```mlir
//     func @compute(%arg0: tensor<?xf32>,
//                   %arg1: tensor<?xf32>) -> tensor<?x?xf32> { ... }
//   ```
//
//   after calling convention conversion becomes:
//
//   ```mlir
//     func @compute(%ctx: !rt.execution_context,
//                  %arg0: memref<?xf32>,
//                  %arg1: memref<?xf32>) -> memref<?x?xf32> { ... }
//   ```
//
// Calling convention function type is not the same as the exported function
// type produced by the compilation pipeline for several reasons:
//
// 1) Compilation pipeline produces LLVM functions with LLVM types, and high
//    level information is lost, e.g. all memrefs are deconstructed into
//    primitive fields when passed as inputs.
//
// 2) Exported function always returns void, and uses runtime API to return
//    results back to the caller (see `xla-rt-export-functions` pass).
//
// Calling convention function type is a XLA-compatible description of the
// compiled executable ABI, so that XLA runtime can correctly initialize
// CallFrame arguments, allocate memory for returned results, and then correctly
// decode results memory into the high level types (e.g. convert returned memref
// descriptor to a Tensor).
class CallingConvention
    : public std::function<mlir::FunctionType(mlir::FunctionType)> {
  using function::function;
};

// Returns a calling convention that only adds the execution context argument.
CallingConvention DefaultCallingConvention();

// Returns a calling convention that uses user-provided type converter to
// convert all inputs and results types, and adds the execution context
// argument.
CallingConvention DefaultCallingConvention(mlir::TypeConverter);

// Returns a calling convention that (1) prepends the execution context
// argument, (2) uses the user-provided type converter to convert all inputs and
// results types, and (3) converts result types into out-params by appending
// them to the arguments.
CallingConvention ResultsToOutsCallingConvention(mlir::TypeConverter);

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_RUNTIME_CALLING_CONVENTION_H_
