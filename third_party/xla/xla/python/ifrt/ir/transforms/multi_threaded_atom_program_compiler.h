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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_MULTI_THREADED_ATOM_PROGRAM_COMPILER_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_MULTI_THREADED_ATOM_PROGRAM_COMPILER_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {

using CompileFuture = Future<AtomProgramCompileResult>;

// Wrapper around `AtomProgramCompiler` that offers multi-threaded dispatch
// of atom program compilations.
class MultiThreadedAtomProgramCompiler {
 public:
  explicit MultiThreadedAtomProgramCompiler(
      std::shared_ptr<AtomProgramCompiler> compiler,
      std::shared_ptr<
          absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
          compile_options_overrides,
      bool enable_sharding_propagation)
      : compiler_(std::move(compiler)),
        compile_options_overrides_(std::move(compile_options_overrides)),
        enable_sharding_propagation_{enable_sharding_propagation} {}

  // Dispatches compilation of an atom program module.
  // Depending on the type of module, a MLIR pipeline might be executed before
  // the compilation is dispatched.
  absl::StatusOr<CompileFuture> CompileModule(CallOp, mlir::ModuleOp module_op);

 private:
  // Compiles an atom XLA program.
  // Returns a future of a AtomProgramCompileResult for the compiled module.
  //
  // Note that the method runs `ifrt-compile-xla-preprocessing-pipeline`
  // before dispatching compilation.
  absl::StatusOr<CompileFuture> CompileXla(
      CallOp call_op, mlir::ModuleOp module_op,
      tsl::thread::ThreadPool* thread_pool);

  // Returns a future of a AtomProgramCompileResult for the MPMD reshard module.
  absl::StatusOr<CompileFuture> CompileMpmdReshard(mlir::ModuleOp module_op);

  // Gets the XLA compile options for the given atom program module.
  absl::StatusOr<xla::CompileOptions> GetXlaCompileOptions(
      CallOp call_op, mlir::ModuleOp module_op);

  std::shared_ptr<AtomProgramCompiler> compiler_;

  std::shared_ptr<
      absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
      compile_options_overrides_;

  // Whether to allow sharding propagation from inputs to outputs that do not
  // have sharding specified (i.e., their mhlo.sharding attribute is not set).
  bool enable_sharding_propagation_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_MULTI_THREADED_ATOM_PROGRAM_COMPILER_H_
