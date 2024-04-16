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

#ifndef XLA_MLIR_RUNTIME_TRANSFORMS_JIT_COMPILER_H_
#define XLA_MLIR_RUNTIME_TRANSFORMS_JIT_COMPILER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "xla/mlir/runtime/transforms/calling_convention.h"
#include "xla/mlir/runtime/transforms/specialization.h"
#include "xla/mlir/runtime/transforms/type_converter.h"
#include "xla/runtime/arguments.h"
#include "xla/runtime/compiler.h"
#include "xla/runtime/constraints.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/execution_engine.h"
#include "xla/runtime/symbolic_shape.h"

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
    std::function<void(DialectRegistry&)> register_dialects;

    // Create a pass pipeline that is called whenever the compiled module
    // gets specialized. This pipeline can use refined shape information and
    // symbolic shape attributes to do the shape inference and canonicalization.
    //
    // Original input module might have an undefined calling convention (e.g.
    // XLA runtime does not support unranked tensors), and specialization can be
    // required as a precondition for compilation.
    std::function<absl::Status(PassManager&)> create_specialization_pipeline;

    // Create a pass pipeline that lowers compiled module from high level
    // dialects to the LLVM dialect. XLA runtime will use the LLVM ORC compiler
    // API to compile the LLVM module at run time
    // (https://llvm.org/docs/ORCv2.html).
    //
    // This compilation pipeline must export functions invocable by the runtime
    // (convert them to an ABI compatible with the calling convention advertised
    // to XLA through the `calling_convention` type conversion), and for
    // that it usually must include `xla-rt-export-functions` pass.
    std::function<absl::Status(PassManager&)> create_compilation_pipeline;

    // LLVM optimization level when JIT compiling a module.
    llvm::CodeGenOptLevel jit_code_opt_level = llvm::CodeGenOptLevel::Default;

    // Runtime symbols binding allows to pass user-defined bindings for symbols
    // at JIT compilation time, e.g. to bind type ids or custom calls.
    ExecutionEngine::SymbolsBinding symbols_binding;

    // Calling convention defines an ABI for XLA runtime to call an executable.
    // See `CallingConvention` documentation for details.
    CallingConvention calling_convention = DefaultCallingConvention();

    // Type converter converts MLIR types to the corresponding run-time types.
    // Executable uses its own type hierarchy, parallel to MLIR's, so that it
    // doesn't depend on any parts of the MLIR after compilation produces an
    // executable artifact, because keeping MLIR context alive can be expensive
    // in terms of memory usage.
    //
    // As a side effect, it allows loading AOT compiled executables from the
    // object files without any dependencies on MLIR.
    //
    // Default type converter knows how to convert canonical MLIR types
    // (memrefs, tensors, etc...). All user-defined types used at the compiled
    // function boundary (arguments or results) should register a custom type
    // conversion.
    //
    // When we compile the input IR, we first apply the `calling_convention` to
    // get the MLIR function type for the exported function(s), and then we
    // convert it to the corresponding run-time function type.
    TypeConverter type_converter;

    // How much verification would you like to do?
    int verification_level = 0;

    // Whether to embed the LLVM IR generated in the executable
    bool embed_ir_in_executable = false;
  };

  // Instantiates compiler from the serialized mlir source.
  static absl::StatusOr<std::unique_ptr<JitCompiler>> Instantiate(
      Options opts, std::string_view mlir_module,
      absl::Span<const std::string_view> exported);

  // Instantiates compiler from the mlir module.
  static absl::StatusOr<std::unique_ptr<JitCompiler>> Instantiate(
      Options opts, mlir::ModuleOp mlir_module,
      absl::Span<const std::string_view> exported);

  // Makes an executable from an instance of the JitCompiler. This is the end of
  // life for the `JitCompiler`, it effectively converts the MLIR module
  // to the executable (function pointer) using LLVM JIT code generation.
  // Optional specialization identifier specifies if the compiled executable is
  // a default one, or a specialization.
  static absl::StatusOr<Executable> Compile(
      std::unique_ptr<JitCompiler> compiler,
      std::string_view memory_region_name,
      std::optional<size_t> specialization = std::nullopt);

  // Specialize the exported function given by 'ordinal' to the arguments:
  //
  // - update all unknown dimensions according to the resolved symbolic shapes
  // - attach symbolic shape attribute to the operands
  // - sink small constants into the function body
  //
  // After the exported function's signature is updated, and all constant
  // arguments are materialized in the function body, runs the user-provided
  // specialization pipeline to optimize the module based on the new
  // information in the IR.
  //
  // Returns an error if arguments are not compatible with the exported
  // function's signature.
  absl::Status Specialize(unsigned ordinal, ArgumentsRef arguments,
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

  size_t num_exported() const { return exported_.size(); }

  absl::Span<const mlir::FunctionOpInterface> exported() const {
    return exported_;
  }

  mlir::FunctionOpInterface exported(unsigned ordinal) const {
    assert(exported_[ordinal] && "failed to resolve exported function");
    return exported_[ordinal];
  }

 private:
  JitCompiler(Options opts, std::string_view mlir_module);
  JitCompiler(Options opts, mlir::ModuleOp mlir_module);

  absl::Status ComputeOrdinalsForExportedFunctions(
      absl::Span<const std::string_view> exported);

  absl::Status Error(std::string_view error) {
    absl::Status interr = absl::InternalError(error);
    interr.SetPayload("__jit_compiler_internal_error", absl::Cord(diagnostic_));
    return interr;
  }

  Options opts_;
  std::unique_ptr<mlir::MLIRContext> owned_context_;  // set if context is owned
  mlir::MLIRContext* context_;

  std::string diagnostic_;
  llvm::raw_string_ostream diagnostic_os_;

  llvm::SourceMgr source_mgr_;
  mlir::SourceMgrDiagnosticHandler handler_;

  mlir::OwningOpRef<mlir::ModuleOp> module_;         // null if failed to parse
  std::vector<mlir::FunctionOpInterface> exported_;  // empty if failed to parse

  bool specialized_;
};

// Adds "rt.export" with ordinal 0 to the "main" function in `module`.
// This is done by performing a run of the OrdinalAssignment pass using
// the given `mlir_context`.
absl::Status ExportMainWithOrdinal0(mlir::ModuleOp module,
                                    mlir::MLIRContext& mlir_context);

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_TRANSFORMS_JIT_COMPILER_H_
