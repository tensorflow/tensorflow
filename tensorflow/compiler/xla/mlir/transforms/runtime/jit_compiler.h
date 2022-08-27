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

#ifndef XLA_MLIR_RUNTIME_JIT_COMPILER_H_
#define XLA_MLIR_RUNTIME_JIT_COMPILER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/runtime/calling_convention.h"
#include "tensorflow/compiler/xla/mlir/transforms/runtime/specialization.h"
#include "tensorflow/compiler/xla/mlir/transforms/runtime/type_converter.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/constraints.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/symbolic_shape.h"

namespace xla {
namespace runtime {

// JitCompiler manages parsing, specialization and compilation of a single XLA
// module to the XLA runtime executable. It owns the MLIR context where the
// module is created, and handlers to capture all compilation diagnostics
// messages.
//
// TODO(ezhulenev): Allow constructing JitCompiler (and JitExecutable) from the
// MLIR module directly without serializing it to string first.
class JitCompiler {
 public:
  using SymbolicShape = SymbolicShapesResolver::SymbolicShape;

  struct Options {
    // Register dialects that are allowed in the serialized module.
    std::function<void(mlir::DialectRegistry&)> register_dialects;

    // Create a pass pipeline that is called whenever the compiled module
    // gets specialized. This pipeline can use refined shape information and
    // symbolic shape attributes to do the shape inference and canonicalization.
    //
    // Original input module might have an undefined calling convention (e.g.
    // XLA runtime does not support unranked tensors), and specialization can be
    // required as a precondition for compilation.
    std::function<void(mlir::PassManager&)> create_specialization_pipeline;

    // Create a pass pipeline that lowers compiled module from high level
    // dialects to the LLVM dialect. XLA runtime will use the LLVM ORC compiler
    // API to compile the LLVM module at run time
    // (https://llvm.org/docs/ORCv2.html).
    //
    // This compilation pipeline must create the entrypoint function with an ABI
    // compatible with the calling convention advertised to the XLA through
    // the `calling_convention` type conversion, and for that it usually must
    // include `xla-rt-convert-to-entrypoint ` pass to convert regular functions
    // to "XLA entrypoints".
    std::function<void(mlir::PassManager&)> create_compilation_pipeline;

    // LLVM optimization level when JIT compiling a module.
    llvm::CodeGenOpt::Level jit_code_opt_level =
        llvm::CodeGenOpt::Level::Default;

    // Runtime symbols binding allows to pass user-defined bindings for symbols
    // at JIT compilation time, e.g. to bind type ids or custom calls.
    ExecutionEngine::SymbolsBinding symbols_binding;

    // Calling convention defines an ABI for XLA runtime to call an executable.
    // See `CallingConvention` documentation for details.
    CallingConvention calling_convention = DefaultCallingConvention();

    // Type converter converts MLIR types to the corresponding run time types.
    // Executable uses its own type hierarchy, parallel to MLIR's, so that it
    // doesn't depend on any parts of the MLIR after compilation produces an
    // executable artifact, because keeping MLIR context alive can be expensive
    // in terms of memory usage.
    //
    // As a side effect, it allows loading AOT compiled executables from the obj
    // files without any dependencies on MLIR.
    //
    // Default type converter knows how to convert canonical MLIR types
    // (memrefs, tensors, etc...). All user-defined types used at the compiled
    // function boundary (arguments or results) should register a custom type
    // conversion.
    //
    // When we compile the input IR, we first apply the `calling_convention` to
    // get the MLIR function type for the entrypoint, and then we convert it to
    // the corresponding run time function type.
    TypeConverter type_converter;
  };

  // Instantiates compiler from the serialized mlir source.
  static absl::StatusOr<std::unique_ptr<JitCompiler>> Instantiate(
      Options opts, std::string_view mlir_module, std::string_view entrypoint);

  // Makes an executable from an instance of the JitCompiler. This is the end of
  // life for the `JitCompiler`, it effectively converts the MLIR module
  // to the executable (function pointer) using LLVM JIT code generation.
  // Optional specialization identifier specifies if the compiled executable is
  // a default one, or a specialization.
  static absl::StatusOr<Executable> Compile(
      std::unique_ptr<JitCompiler> compiler,
      std::string_view memory_region_name,
      std::optional<size_t> specialization = std::nullopt);

  // Specialize compiled module to the arguments:
  //
  // - update all unknown dimensions according to the resolved symbolic shapes
  // - attach symbolic shape attribute to the operands
  // - sink small constants into the function body
  //
  // After entrypoint signature is updated, and all constant arguments
  // materialized in the function body, runs the user-provided specialization
  // pipeline to optimize the module based on the new information in the IR.
  //
  // Returns error if arguments are not compatible with compiled module
  // entrypoint signature.
  absl::Status Specialize(ArgumentsRef arguments,
                          llvm::ArrayRef<SymbolicShape> symbolic_shapes,
                          llvm::ArrayRef<ArgumentConstraint> constraints,
                          const SpecializationListener* listener = nullptr);

  const Options& options() const { return opts_; }

  std::string_view name() const {
    return module().getName().value_or("<unknown>");
  }

  mlir::ModuleOp module() const {
    assert(module_ && "failed to parse the mlir module");
    return *module_;
  }

  mlir::func::FuncOp entrypoint() const {
    assert(entrypoint_ && "failed to resolve entrypoint function");
    return entrypoint_;
  }

 private:
  JitCompiler(Options opts, std::string_view mlir_module,
              std::string_view entrypoint);

  absl::Status Error(std::string_view error) {
    // TODO(ezhulenev): Pass diagnstic as a status payload.
    return absl::InternalError(absl::StrCat(error, ":\n", diagnostic_));
  }

  Options opts_;
  std::unique_ptr<mlir::MLIRContext> context_;

  std::string diagnostic_;
  llvm::raw_string_ostream diagnostic_os_;

  llvm::SourceMgr source_mgr_;
  mlir::SourceMgrDiagnosticHandler handler_;

  mlir::OwningOpRef<mlir::ModuleOp> module_;  // can be null if failed to parse
  mlir::func::FuncOp entrypoint_;             // can be null if failed to parse

  bool specialized_;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_JIT_COMPILER_H_
