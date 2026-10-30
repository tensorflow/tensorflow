/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_EXPORT_PASSES_H_
#define XLA_PYTHON_IFRT_IR_EXPORT_PASSES_H_

#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.pb.h"
#include "xla/python/ifrt/ir/version.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "xla/python/ifrt/ir/export/passes.h.inc"  // IWYU pragma: export

// Creates a pass that serializes outlined atom programs to VHLO inside
// `atom_programs`.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createIfrtAtomProgramsToVhloPass(
    tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>* atom_programs,
    std::string vhlo_target_version, std::string sdy_target_version);

// Creates a pass that deserializes atom programs from `atom_programs` and
// inserts them into the IFRT IR module.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createIfrtAtomProgramsFromVhloPass(
    const tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>&
        atom_programs);

// Populates conversion patterns from IFRT ops to current VIFRT ops.
void populateIfrtToVifrtPatterns(mlir::RewritePatternSet* patterns,
                                 mlir::TypeConverter* converter,
                                 mlir::MLIRContext* context);

// Populates conversion patterns from current VIFRT ops to IFRT ops.
void populateVifrtToIfrtPatterns(mlir::RewritePatternSet* patterns,
                                 mlir::TypeConverter* converter,
                                 mlir::MLIRContext* context);

// Populates conversion patterns between versions of VIFRT ops and types.
void populateVifrtToVersionPatterns(mlir::RewritePatternSet* patterns,
                                    mlir::TypeConverter* converter,
                                    Version version,
                                    mlir::MLIRContext* context);

// Creates a pipeline that converts an IFRT IR program to a versioned IFRT IR
// program, and versioned VHLO programs populated within `IfrtIrProgramProto`.
void createIfrtToVersionedPipeline(mlir::OpPassManager& pm,
                                   std::string ifrt_target_version,
                                   std::string vhlo_target_version,
                                   std::string sdy_target_version,
                                   IfrtIrProgramProto& ifrt_ir_program);

// Creates a pipeline that converts a versioned IFRT IR program to an IFRT IR
// program.
// The pipeline runs over a versioned IFRT IR module without any atom programs
// present inside the module. The atom programs are expected to be present and
// versioned in the given `ifrt_ir_program`. The pipeline will convert these
// atom programs (i.e., from VHLO to StableHLO) and add them to the IFRT IR
// program.
void createIfrtFromVersionedPipeline(mlir::OpPassManager& pm,
                                     const IfrtIrProgramProto& ifrt_ir_program);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_EXPORT_PASSES_H_
