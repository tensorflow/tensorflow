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

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_CONSTANTS_H_
