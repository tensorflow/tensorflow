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

#ifndef XLA_PYTHON_IFRT_IR_CONSTANTS_H_
#define XLA_PYTHON_IFRT_IR_CONSTANTS_H_

#include "llvm/ADT/StringRef.h"

namespace xla {
namespace ifrt {

// Name of UnitAttr on FuncOp to indicate it's an IFRT IR function, telling it
// apart from atom program FuncOps (callee of `ifrt.Call`).
inline constexpr llvm::StringLiteral kIfrtFunctionAttrName = "ifrt.function";

// Name of UnitAttr on FuncOp to indicate it's a VIFRT IR function, telling it
// apart from atom program FuncOps.
inline constexpr llvm::StringLiteral kVifrtFunctionAttrName = "vifrt.function";

// Name of UnitAttr on FuncOp to indicate it's an IFRT IR function that
// only reshards arrays. While functions with kIfrtFunctionAttrName attribute
// cannot be `ifrt.Call`ed, kIfrtReshardFunctionAttrName can be called.
inline constexpr llvm::StringLiteral kIfrtReshardFunctionAttrName =
    "ifrt.reshard_function";

// Name of UnitAttr on arguments of FuncOp to indicate a donated input.
// Must be used in a FuncOp with `ifrt.function` attr.
inline constexpr llvm::StringLiteral kIfrtDonatedArgAttrName = "ifrt.donated";

// Name of UnitAttr on CallOp used to indicate that the atom program is
// in "local" view (i.e., already sharded).
inline constexpr llvm::StringLiteral kIfrtLocalViewAttrName = "ifrt.local_view";

// Name of StringAttr on CallOp used to store an optional key to use into a
// mapping of user-provided compile options.
inline constexpr llvm::StringLiteral kIfrtCompileOptionsKey =
    "ifrt.compile_options_key";

inline constexpr llvm::StringLiteral kIfrtDevicesAttrName = "ifrt.devices";
inline constexpr llvm::StringLiteral kIfrtNumDevicesAttrName =
    "ifrt.num_devices";
inline constexpr llvm::StringLiteral kIfrtShardingAttrName = "ifrt.sharding";
inline constexpr llvm::StringLiteral kIfrtMemoryKindAttrName =
    "ifrt.memory_kind";
inline constexpr llvm::StringLiteral kIfrtEntryFunctionAttrName =
    "ifrt.entry_function";

// Name of UnitAttr on CallOp used to indicate that an atom program was
// partitioned by the Sdy partitioner.
inline constexpr llvm::StringLiteral kIsSdyPartitioned =
    "ifrt.is_sdy_partitioned";

inline constexpr llvm::StringLiteral kCalleeMainFuncName = "main";

// Name of StringAttr used to store the HloSharding.
inline constexpr llvm::StringLiteral kHloShardingAttrName = "mhlo.sharding";
// Name of StringAttr used to store memory kind.
inline constexpr llvm::StringLiteral kHloMemoryKindAttrName =
    "mhlo.memory_kind";
// Name of StringAttr used to store layout mode.
inline constexpr llvm::StringLiteral kHloLayoutAttrName = "mhlo.layout_mode";

inline constexpr llvm::StringLiteral kIfrtModuleTypeAttrName =
    "ifrt.module_type";

inline constexpr llvm::StringLiteral kIfrtModuleTypeXla = "xla";
inline constexpr llvm::StringLiteral kIfrtModuleTypeMpmdReshard =
    "mpmd_reshard";

// String value used as a default value for an optional `mlir::StringAttr` when
// converting to and from VIFRT.
inline constexpr llvm::StringLiteral kVifrtDefaultString = "vifrt.default";

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_CONSTANTS_H_
