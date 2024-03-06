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

#ifndef XLA_MLIR_RUNTIME_TRANSFORMS_PASSES_H_
#define XLA_MLIR_RUNTIME_TRANSFORMS_PASSES_H_

#include <functional>
#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "xla/mlir/runtime/ir/rt_ops.h"  // IWYU pragma: keep

namespace xla {
namespace runtime {

#define GEN_PASS_DECL_ORDINALASSIGNMENT
#define GEN_PASS_DECL_MOVEALLOCASTOENTRYBLOCK
#define GEN_PASS_DECL_EXPORTFUNCTIONS
#define GEN_PASS_DECL_CONVERTCUSTOMCALLS
#define GEN_PASS_DECL_CONVERTASSERTS
#define GEN_PASS_DECL_CONVERTRUNTIMETOLLVMPASS

#include "xla/mlir/runtime/transforms/passes.h.inc"

//===-----------------------------------------------------------------------===/
// Transformations targeting `rt` dialect.
//===-----------------------------------------------------------------------===/

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateOrdinalAssignmentPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateMoveAllocasToEntryBlockPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateExportRuntimeFunctionsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateConvertCustomCallsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateAddInitializationsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateConvertAssertsPass();

//===-----------------------------------------------------------------------===/
// Conversions targeting `rt` dialect.
//===-----------------------------------------------------------------------===/

class TypeIDNameRegistry;
class CustomCallArgEncodingSet;
class CustomCallAttrEncodingSet;
class CustomCallRetEncodingSet;

// Extension points for converting `rt` dialect to the LLVM dialect.
//
// Runtime custom calls is an extension mechanism for enabling compiled programs
// to call into the APIs provided by the user. It relies on converting
// values and attributes to the LLVM types (structs and pointers) with a
// well-defined memory layout, so that they can be passed across the function
// boundary and safely decoded (without dependency on C++ ABI).
//
// All user-defined types (values and attributes) that are passed to the custom
// calls must define the argument or attribute encoding.
struct ConvertRuntimeToLLvmOpts {
  // Register names for the TypeIDs used for encoding types of custom arguments
  // and attributes.
  std::function<void(TypeIDNameRegistry&)> populate_type_id_names;

  // Add type conversions for user-defined types to the corresponding LLVM
  // types. Conversion pass uses these extra conversions to convert arguments
  // of the entrypoint function and values passed to the custom calls. Custom
  // call argument encoding can further refine how values of LLVM types passed
  // to the custom call handlers by passing custom encoding (see below).
  std::function<void(mlir::TypeConverter&)> populate_type_conversions;

  // Add user-defined arguments encoding to the custom call lowering.
  std::function<void(CustomCallArgEncodingSet&)> populate_arg_encodings;

  // Add user-defined attributes type encoding to the custom call lowering.
  std::function<void(CustomCallRetEncodingSet&)> populate_ret_encodings;

  // Add user-defined attributes type encoding to the custom call lowering.
  std::function<void(CustomCallAttrEncodingSet&)> populate_attr_encodings;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateConvertRuntimeToLLVMPass(ConvertRuntimeToLLvmOpts opts = {});

//===-----------------------------------------------------------------------===/

#define GEN_PASS_REGISTRATION
#include "xla/mlir/runtime/transforms/passes.h.inc"

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_TRANSFORMS_PASSES_H_
