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

#ifndef XLA_PYTHON_IFRT_IR_UTILS_H_
#define XLA_PYTHON_IFRT_IR_UTILS_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

// Used for comparing CallOps without including control dependencies.
struct IfrtCallOpInfo : llvm::DenseMapInfo<CallOp> {
  static unsigned getHashValue(CallOp call_op);
  static bool isEqual(CallOp lhs, CallOp rhs);
};

// Retrieves the function named "main" from the given module, if it exists, and
// fails otherwise.
mlir::func::FuncOp GetMainFunction(mlir::ModuleOp module);

// Returns a snapshot of all top-level IFRT functions (`IsIfrtFunction`) in
// `module`. Snapshotting into a vector allows callers to safely insert or erase
// top-level callee symbols in `module` during iteration.
llvm::SmallVector<mlir::func::FuncOp> GetIfrtFunctions(mlir::ModuleOp module);

// Constructs a 1-level nested `@<callee_module>::@<callee>` `SymbolRefAttr`
// referencing `callee` inside `callee_module`.
mlir::SymbolRefAttr MakeOutlinedCalleeSymbolRef(mlir::ModuleOp callee_module,
                                                mlir::func::FuncOp callee);

// Returns the top-level symbol operation in `root_module` corresponding to
// `callee` (`callee` itself when top-level, or its enclosing child `ModuleOp`
// when outlined one module deep), or `nullptr` if `callee` has an unsupported
// parent hierarchy.
mlir::Operation* GetCalleeRootSymbolOp(mlir::func::FuncOp callee,
                                       mlir::ModuleOp root_module);

// Resolves `call_op`'s callee in `symbol_table` and verifies that it is
// outlined as `@main` inside a child `ModuleOp` of `call_op`'s enclosing
// `ModuleOp`. Emits an op error on `call_op` and returns `mlir::failure()`
// otherwise.
mlir::FailureOr<mlir::ModuleOp> GetOutlinedAtomProgramModule(
    CallOp call_op, mlir::SymbolTableCollection& symbol_table);

// Result of cloning an outlined atom-program `ModuleOp`.
struct ClonedAtomProgramModule {
  mlir::ModuleOp module;
  mlir::func::FuncOp main_func;
  mlir::SymbolRefAttr callee_ref;
};

// Clones `callee_module` with private symbol visibility, inserts it into its
// parent symbol table immediately after `callee_module`, and returns the cloned
// module, its `@main` `FuncOp`, and its `@<module>::@main` `SymbolRefAttr`.
ClonedAtomProgramModule CloneOutlinedAtomProgramModule(
    mlir::ModuleOp callee_module, mlir::SymbolTableCollection& symbol_table);

// Returns true if `attr` is a `mlir::UnitAttr` or a `mlir::BoolAttr` with
// value `true`.
bool IsTrueOrUnitAttr(mlir::Attribute attr);

// Returns true if `dict` carries any buffer donation or output-aliasing
// argument attribute (`jax.buffer_donor`, `tf.aliasing_output`, or
// `ifrt.donated`).
bool HasDonatedOrAliasedDictAttr(mlir::DictionaryAttr dict);

// Returns true if `type` is an `IfrtArrayType` whose tensor element type is
// `IfrtTokenType` (`!ifrt.array<tensor<!ifrt.token>, ...>`).
bool IsTokenArrayType(mlir::Type type);

// Returns the index of `operand_num` within `inputs`, or `std::nullopt` if
// `inputs` is empty or `operand_num` falls outside `inputs` (for example, a
// control input).
std::optional<unsigned> GetArrayInputIndex(mlir::OperandRange inputs,
                                           unsigned operand_num);

// Returns true if transferring between from and to array requires a reshard.
bool IsReshard(IfrtArrayType from, IfrtArrayType to);

// Converts a mlir::Type to a ifrt DType.
absl::StatusOr<DType> ToIfrtDType(mlir::Type type);

// Prints the MLIR operation as a string.
std::string OperationToString(mlir::Operation* op,
                              const mlir::OpPrintingFlags& flags);

// Clones a given mlir::ModuleOp into the given MLIR context.
// Note: This function is loading dialects into the context, and thus it is
// not thread-safe w.r.t. calling it with the same context from multiple
// threads.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CloneModuleIntoContext(
    mlir::ModuleOp module, mlir::MLIRContext& context);

// Returns a pretty string representation of the location.
std::string GetPrettyLocation(mlir::Location loc);

// Returns a pretty string representation of the location of the given argument
// in the given module.
//
// REQUIRES: 0 <= index < module.getNumArgOperands().
std::string GetArgPrettyLocation(int index, mlir::ModuleOp module);

// Creates a `ShardingRef` from an `IfrtArrayType`.
//
// The logical devices from the `IfrtArrayType` represent indices into the
// device_list.
absl::StatusOr<ShardingRef> ShardingFromIfrtArrayType(
    IfrtArrayType array_type, Client* client, const DeviceListRef& device_list);

// Creates an `ArraySpec` from a `mlir::Type`.
//
// Returns an error if the `array_type` cannot be converted to an `ArraySpec`.
//
// The logical devices from the `IfrtArrayType` represent indices into the
// device_list.
absl::StatusOr<ArraySpec> ArraySpecFromMlirType(
    mlir::Type array_type, Client* client, const DeviceListRef& device_list);

// Returns a DeviceList for the given device ids.
absl::StatusOr<DeviceListRef> LookUpDevices(Client* client,
                                            absl::Span<const DeviceId> ids);

// Builds and validates a `RemapPlan` from `remap_op` using `device_list` to
// resolve logical device IDs.
absl::StatusOr<RemapPlan> BuildRemapPlan(Client* absl_nonnull client,
                                         const DeviceListRef& device_list,
                                         RemapArraysOp remap_op);

// Converts `specs` shardings to `xla::OpSharding` protos, returning
// `std::nullopt` if any `ArraySpec` does not carry an `HloSharding`.
std::optional<std::vector<xla::OpSharding>> GetOpShardings(
    absl::Span<const ArraySpec> specs, absl::string_view kind);

// Extracts `PjRtLayout`s from `specs` once `layout_status` is OK.
absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>> GetLayouts(
    const absl::Status& layout_status, absl::Span<const ArraySpec> specs);

// Extracts output memory kinds from `out_specs` as a single outer entry
// representing the top-level IFRT IR program outputs. Returned
// `absl::string_view`s reference `MemoryKind`'s globally interned storage and
// remain valid after `out_specs` is destroyed.
absl::StatusOr<std::vector<std::vector<absl::string_view>>>
GetOutputMemoryKinds(absl::Span<const ArraySpec> out_specs);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_UTILS_H_
