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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_OPENXLA_CONVERSION_XLA_GPU_API_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_OPENXLA_CONVERSION_XLA_GPU_API_H_

#include <string_view>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project

namespace xla::gpu {

// API declarations for XLA:GPU custom module implementing StreamExecutor
// integration: device kernel launches and third party libraries.
class XlaGpuApi {
 public:
  // Returns  `!iree_input.list<i32>` type.
  static mlir::Type getI32ListType(mlir::OpBuilder &b);

  // Returns `!iree_input.list<!iree_input.buffer_view>` type.
  static mlir::Type getBufferViewListType(mlir::OpBuilder &b);

  //===--------------------------------------------------------------------===//
  // XLA:GPU kernel APIs
  //===--------------------------------------------------------------------===//

  // Imports `@xla_gpu.kernel.create` into the module.
  mlir::func::FuncOp getCreateKernel(mlir::OpBuilder &b, mlir::ModuleOp module);

  // Imports `@xla_gpu.kernel.dispatch` into the module.
  mlir::func::FuncOp getDispatchKernel(mlir::OpBuilder &b,
                                       mlir::ModuleOp module);

  //===--------------------------------------------------------------------===//
  // XLA:GPU gemm (dot) APIs
  //===--------------------------------------------------------------------===//

  // Imports `@xla_gpu.dot_dimension_numbers.create` into the module.
  mlir::func::FuncOp getCreateDotDimensionsNumbers(mlir::OpBuilder &b,
                                                   mlir::ModuleOp module);

  // Imports `@xla_gpu.dot_precision.create` into the module.
  mlir::func::FuncOp getCreateDotPrecision(mlir::OpBuilder &b,
                                           mlir::ModuleOp module);

  // Imports `@xla_gpu.dot_config.create` into the module.
  mlir::func::FuncOp getCreateDotConfig(mlir::OpBuilder &b,
                                        mlir::ModuleOp module);

  // Imports `@xla_gpu.gemm.dispatch` into the module.
  mlir::func::FuncOp getDispatchGemm(mlir::OpBuilder &b, mlir::ModuleOp module);

  //===--------------------------------------------------------------------===//
  // XLA:GPU memcpy APIs
  //===--------------------------------------------------------------------===//

  // Imports `@xla_gpu.memcpy.d2d` into the module.
  mlir::func::FuncOp getD2DMemcpy(mlir::OpBuilder &b, mlir::ModuleOp module);

  // Imports `@xla_gpu.memcpy.load.i1` into the module.
  mlir::func::FuncOp getLoadI1Memcpy(mlir::OpBuilder &b, mlir::ModuleOp module);

  //===--------------------------------------------------------------------===//
  // XLA:GPU tracing APIs
  //===--------------------------------------------------------------------===//

  // Imports `@xla_gpu.trace.create` into the module.
  mlir::func::FuncOp getCreateTrace(mlir::OpBuilder &b, mlir::ModuleOp module);

 private:
  mlir::SymbolTable &symTable(mlir::ModuleOp module);

  mlir::func::FuncOp addDecl(mlir::OpBuilder &b, mlir::ModuleOp module,
                             std::string_view name,
                             mlir::FunctionType function_type);

  mlir::SymbolTableCollection sym_table_;
};

}  // namespace xla::gpu

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_OPENXLA_CONVERSION_XLA_GPU_API_H_
