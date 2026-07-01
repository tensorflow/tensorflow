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

#ifndef XLA_MOSAIC_DIALECT_TPU_STRINGIFY_UTIL_H_
#define XLA_MOSAIC_DIALECT_TPU_STRINGIFY_UTIL_H_

#include <type_traits>  // IWYU pragma: keep

#include "llvm/ADT/StringRef.h"  // IWYU pragma: keep

// Defines an AbslStringify overload for MLIR enums in the enclosing namespace.
//
// IMPORTANT: This macro must be invoked inside the namespace of the enum types
// you want to stringify, because Abseil uses ADL to find AbslStringify.
#define DEFINE_ABSL_STRINGIFY_FOR_ENUMS()                                 \
  template <typename Sink, typename EnumT>                                \
  std::enable_if_t<std::is_enum_v<EnumT> &&                               \
                       std::is_convertible_v<decltype(stringifyEnum(      \
                                                 std::declval<EnumT>())), \
                                             llvm::StringRef>,            \
                   void>                                                  \
  AbslStringify(Sink& sink, EnumT value) {                                \
    sink.Append(stringifyEnum(value).str());                              \
  }

#endif  // XLA_MOSAIC_DIALECT_TPU_STRINGIFY_UTIL_H_
