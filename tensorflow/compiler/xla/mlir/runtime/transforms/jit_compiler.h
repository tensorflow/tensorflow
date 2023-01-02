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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_TRANSFORMS_JIT_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_TRANSFORMS_JIT_COMPILER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/transforms/calling_convention.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/specialization.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/type_converter.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/compiler.h"
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
    std::function<void(DialectRegistry&)> register_dialects;

    // Create a pass pipeline that is called whenever the compiled module
    // gets specialized. This pipeline can use refined shape information and
    // symbolic shape attributes to do the shape inference and canonicalization.
    //
    // Original input module might have an undefined calling convention (e.g.
    // XLA runtime does not support unranked tensors), and specialization can be
    // required as a precondition for compilation.
    std::function<void(PassManager&)> create_specialization_pipeline;

    // Create a pass pipeline that lowers compiled module from high level
    // dialects to the LLVM dialect. XLA runtime will use the LLVM ORC compiler
    // API to compile the LLVM module at run time
    // (https://llvm.org/docs/ORCv2.html).
    //
    // This compilation pipeline must export functions invocable by the runtime
    // (convert them to an ABI compatible with the calling convention advertised
    // to XLA through the `calling_convention` type conversion), and for
    // that it usually must include `xla-rt-export-functions` pass.
    std::function<void(PassManager&)> create_compilation_pipeline;

    // LLVM optimization level when JIT compiling a module.
    llvm::CodeGenOpt::Level jit_code_opt_level =
        llvm::CodeGenOpt::Level::Default;

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
  };

  // Instantiates compiler from the serialized mlir source.
  static absl::StatusOr<std::unique_ptr<JitCompiler>> Instantiate(
      Options opts, std::string_view mlir_module,
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

  mlir::OwningOpRef<mlir::ModuleOp> module_;         // null if failed to parse
  std::vector<mlir::FunctionOpInterface> exported_;  // empty if failed to parse

  bool specialized_;
};

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_TRANSFORMS_JIT_COMPILER_H_
