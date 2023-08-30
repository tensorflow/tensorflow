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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU2_CONVERSION_XLA_GPU_API_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU2_CONVERSION_XLA_GPU_API_H_

#include <cstdint>
#include <functional>
#include <string_view>
#include <tuple>

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// API declarations for XLA:GPU custom module implementing StreamExecutor
// integration: device kernel launches and third party libraries.
//===----------------------------------------------------------------------===//

class XlaGpuApi {
 public:
  mlir::SymbolTable &symTable(mlir::ModuleOp module);

  //===--------------------------------------------------------------------===//
  // Helper functions to build XLA:GPU API arguments
  //===--------------------------------------------------------------------===//

  // Returns `!iree_input.list<i32>` type.
  static mlir::Type getI32ListType(mlir::OpBuilder &b);

  // Returns `!iree_input.list<!iree_input.buffer_view>` type.
  static mlir::Type getBufferViewListType(mlir::OpBuilder &b);

  // Returns `!iree_input.list<!xla_gpu.graph.node>` type.
  static mlir::Type getGraphNodeListType(mlir::OpBuilder &b);

  // Constructs `!iree_input.list<i32>` list from given values.
  static mlir::TypedValue<mlir::iree_compiler::IREE::Input::ListType>
  getI32List(mlir::ImplicitLocOpBuilder &b, llvm::ArrayRef<int64_t> values);

  // Exports tensor as `!iree_input.buffer_view`.
  static mlir::TypedValue<mlir::iree_compiler::IREE::Input::BufferViewType>
  getBufferView(mlir::ImplicitLocOpBuilder &b,
                mlir::TypedValue<mlir::TensorType> tensor);

  // Constructs `!iree_input.list<!iree_input.buffer_view>` list from tensors.
  static mlir::TypedValue<mlir::iree_compiler::IREE::Input::ListType>
  getBufferViewList(mlir::ImplicitLocOpBuilder &b,
                    llvm::ArrayRef<mlir::TypedValue<mlir::TensorType>> tensors);

  // Constructs `!iree_input.list<!xla_gpu.graph.node>` list from tensors.
  static mlir::TypedValue<mlir::iree_compiler::IREE::Input::ListType>
  getGraphNodeList(mlir::ImplicitLocOpBuilder &b,
                   llvm::ArrayRef<mlir::TypedValue<GraphNodeType>> nodes);

  //===---------------------------------------------------------------------===/
  // Helper functions to build globals
  //===--------------------------------------------------------------------===//

  mlir::iree_compiler::IREE::Input::GlobalOp getOrCreateGlobal(
      llvm::StringRef name, mlir::Type type, mlir::ModuleOp module,
      mlir::ImplicitLocOpBuilder &b,
      std::function<mlir::Value(mlir::ImplicitLocOpBuilder &)> initializer);

  mlir::Value loadGlobal(mlir::ImplicitLocOpBuilder &b,
                         mlir::iree_compiler::IREE::Input::GlobalOp global);

  template <typename T>
  mlir::TypedValue<T> loadGlobal(
      mlir::ImplicitLocOpBuilder &b,
      mlir::iree_compiler::IREE::Input::GlobalOp global) {
    return mlir::cast<mlir::TypedValue<T>>(loadGlobal(b, global));
  }

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
  // XLA:GPU graph construction APIs
  //===--------------------------------------------------------------------===//

  // Imports `@xla_gpu.graph.kernel_node.create` into the module.
  mlir::func::FuncOp getCreateKernelNode(mlir::OpBuilder &b,
                                         mlir::ModuleOp module);

  // Imports `@xla_gpu.graph.memcpy_node.d2d.create` into the module.
  mlir::func::FuncOp getCreateD2DMemcpyNode(mlir::OpBuilder &b,
                                            mlir::ModuleOp module);

  // Imports `@xla_gpu.graph.create` into the module.
  mlir::func::FuncOp getCreateGraph(mlir::OpBuilder &b, mlir::ModuleOp module);

  // Imports `@xla_gpu.graph.execute` into the module.
  mlir::func::FuncOp getExecuteGraph(mlir::OpBuilder &b, mlir::ModuleOp module);

  //===--------------------------------------------------------------------===//
  // XLA:GPU tracing APIs
  //===--------------------------------------------------------------------===//

  // Imports `@xla_gpu.trace.create` into the module.
  mlir::func::FuncOp getCreateTrace(mlir::OpBuilder &b, mlir::ModuleOp module);

 private:
  mlir::func::FuncOp addDecl(mlir::OpBuilder &b, mlir::ModuleOp module,
                             std::string_view name,
                             mlir::FunctionType function_type);

  mlir::SymbolTableCollection sym_table_;

  using GlobalKey = std::tuple<mlir::ModuleOp, mlir::StringAttr, mlir::Type>;
  llvm::DenseMap<GlobalKey, mlir::iree_compiler::IREE::Input::GlobalOp>
      globals_;
};

//===----------------------------------------------------------------------===//
// XLA:GPU graph building helpers
//===----------------------------------------------------------------------===//

struct XlaGpuGraphs {
  // Keep a mapping from the tensor value to the last graph node id that updated
  // the underlying storage buffer. We use this mapping to set up graph
  // dependencies inside a graph dispatch region.
  llvm::DenseMap<mlir::Block *,
                 llvm::DenseMap<mlir::TypedValue<mlir::TensorType>,
                                mlir::TypedValue<GraphNodeType>>>
      dependency;
};

}  // namespace xla::gpu

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU2_CONVERSION_XLA_GPU_API_H_
