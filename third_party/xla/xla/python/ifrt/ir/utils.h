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

#include <string>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/sharding.h"

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

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_UTILS_H_
