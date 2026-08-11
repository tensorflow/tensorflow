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

#ifndef XLA_PYTHON_IFRT_RTTI_H_
#define XLA_PYTHON_IFRT_RTTI_H_

// TODO(hyeontaek): Remove LLVM dependencies once RTTI is reimplemented in IFRT.

// always_keep and exports are used temporarily to suppress linter warnings.

// IWYU pragma: always_keep

// IWYU pragma: begin_exports
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
// IWYU pragma: end_exports

namespace xla {
namespace ifrt {

using ::llvm::RTTIExtends;
using ::llvm::RTTIRoot;

// TODO(hyeontaek): Migrate `*_or_null` to `*_if_present`/`*_and_present`.
using ::llvm::cast;
using ::llvm::cast_if_present;
using ::llvm::cast_or_null;
using ::llvm::dyn_cast;
using ::llvm::dyn_cast_if_present;
using ::llvm::dyn_cast_or_null;
using ::llvm::isa;
using ::llvm::isa_and_nonnull;
using ::llvm::isa_and_present;

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_RTTI_H_
