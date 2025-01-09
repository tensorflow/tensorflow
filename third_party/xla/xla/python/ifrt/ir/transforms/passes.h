/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DECL
#include "xla/python/ifrt/ir/transforms/passes.h.inc"  // IWYU pragma: export

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSpmdExpandableInterfaceVerificationPass(
    SpmdExpandableInterfaceVerificationPassOptions options = {});

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateSpmdExpansionPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtDuplicatedCalleeEliminationPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateIfrtMergeReshardsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtOutlineAtomProgramToModulePass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateIfrtVerifyDonationPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtVerifyShardingSpecifiedPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtPopulateAtomProgramMetadataPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtReshardToCopyArraysPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtLowerAtomProgramMetadataToXlaPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtRemoveIfrtAttrsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtRemoveAttrsFromOtherDialectsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtLowerMpmdReshardToCallPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtVerifyBoundExternalLoadedExecutablePass(
    std::shared_ptr<AtomExecutableMap> bound_executable_map);

// Compiles every atom program ModuleOp into LoadedExecutableOp, and
// lowers every CallOp to CallLoadedExecutableOp.
//
// This pass is not declared in td file as it doesn't have a default
// constructor. It uses an outside AtomProgramCompiler to delegate the
// compilation of atom programs.
//
// For example, the following code
// ```
// %0, %ctrl_0 = ifrt.Call @callee::@main(%arg0) on devices [0, 1]
//
// module @callee attributes {
//   func.func @main() {}
// }
// ```
//
// will be replaced by
// ```
// %0, %ctrl_0 = ifrt.CallLoadedExecutable @component__method(%arg0)
//
// ifrt.LoadedExecutable @component__method on devices [0, 1]
// ```
// }
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtCompileAtomProgramPass(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options,
    std::shared_ptr<AtomExecutableMap> atom_executable_map);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtCompileAndPropagateShardingsPass(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options,
    std::shared_ptr<AtomExecutableMap> atom_executable_map);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtPrecompileAtomProgramPreprocessingPass(
    IfrtPrecompileAtomProgramPreprocessingPassOptions options = {});

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtVerifyDeviceTypeConsistencyPass(
    IfrtVerifyDeviceTypeConsistencyPassOptions options = {});

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtAtomProgramsToVhloPass(
    tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>* atom_programs,
    std::string vhlo_target_version);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtAtomProgramsFromVhloPass(
    const tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>&
        atom_programs);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateVifrtToVersionPass(
    VifrtToVersionPassOptions options = {});

void populateIfrtToVifrtPatterns(mlir::RewritePatternSet* patterns,
                                 mlir::TypeConverter* converter,
                                 mlir::MLIRContext* context);

void populateVifrtToIfrtPatterns(mlir::RewritePatternSet* patterns,
                                 mlir::TypeConverter* converter,
                                 mlir::MLIRContext* context);

void populateVifrtToVersionPatterns(mlir::RewritePatternSet* patterns,
                                    mlir::TypeConverter* converter,
                                    mlir::MLIRContext* context);

// Generated definitions. This should be placed after all Pass creations.
#define GEN_PASS_REGISTRATION
#include "xla/python/ifrt/ir/transforms/passes.h.inc"  // IWYU pragma: export

// Registers IfrtCompileAtomProgramPass to ifrt-opt.
void RegisterIfrtCompileAtomProgramPass(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides,
    std::shared_ptr<AtomExecutableMap> atom_executable_map);

// Registers IfrtCompileAndPropagateShardingsPass to ifrt-opt.
void RegisterIfrtCompileAndPropagateShardingsPass(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides,
    std::shared_ptr<AtomExecutableMap> atom_executable_map);

// Registers IfrtVerifyBoundExternalLoadedExecutablePass to ifrt-opt.
void RegisterIfrtVerifyBoundExternalLoadedExecutablePass(
    std::shared_ptr<AtomExecutableMap> bound_executable_map);

struct IfrtToOutlinedAtomProgramsPipelineOptions
    : mlir::PassPipelineOptions<IfrtToOutlinedAtomProgramsPipelineOptions> {
  Option<bool> propagate_shardings{
      *this, "propagate_shardings",
      llvm::cl::desc("Whether to propagate shardings from executables for "
                     "unspecified shardings.")};
};

// Creates pipeline of all the IFRT IR passes that do not require
// compilation-time information (e.g., device assignments).
void CreateIfrtToOutlinedAtomProgramsPipeline(
    mlir::OpPassManager& pm,
    const IfrtToOutlinedAtomProgramsPipelineOptions& options);

// Creates a pipeline that populates metadata info for each atom program.
void CreateIfrtPopulateAtomProgramMetadataPipeline(mlir::OpPassManager& pm);

// Creates pipeline to lower an IFRT XLA program to be ready for compilation.
void CreateIfrtCompileXlaPreprocessingPipeline(mlir::OpPassManager& pm);

// Creates a pipeline that converts an IFRT IR program to a versioned IFRT IR
// program, and a versioned VHLO programs populated within `IfrtIrProgramProto`.
void CreateIfrtToVersionedPipeline(mlir::OpPassManager& pm,
                                   std::string ifrt_target_version,
                                   std::string vhlo_target_version,
                                   IfrtIrProgramProto& ifrt_ir_program);

// Creates a pipeline that converts a versioned IFRT IR program to an IFRT IR
// program.
// The pipeline runs over a versioned IFRT IR module without any atom programs
// present inside the module. The atom programs are expected to be present and
// versioned in the given `ifrt_ir_program`. The pipeline will convert these
// atom programs (i.e., from VHLO to StableHLO) and add them to the IFRT IR
// program.
void CreateIfrtFromVersionedPipeline(mlir::OpPassManager& pm,
                                     const IfrtIrProgramProto& ifrt_ir_program);

// Registers passes and pipelines to ifrt-opt.
void RegisterIfrtPassesAndPipelines(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides,
    std::shared_ptr<AtomExecutableMap> atom_executable_map,
    std::shared_ptr<AtomExecutableMap> bound_executable_map);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_
