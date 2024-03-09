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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_CONSTANTS_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_CONSTANTS_H_

#include "llvm/ADT/StringRef.h"

namespace xla::ifrt {

inline constexpr llvm::StringLiteral kIfrtDevicesAttrName = "ifrt.devices";
inline constexpr llvm::StringLiteral kIfrtShardingAttrName = "ifrt.sharding";
inline constexpr llvm::StringLiteral kIfrtEntryFunctionAttrName =
    "ifrt.entry_function";

}  // namespace xla::ifrt

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_CONSTANTS_H_
