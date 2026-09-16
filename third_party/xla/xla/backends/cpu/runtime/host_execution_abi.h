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

#ifndef XLA_BACKENDS_CPU_RUNTIME_HOST_EXECUTION_ABI_H_
#define XLA_BACKENDS_CPU_RUNTIME_HOST_EXECUTION_ABI_H_

#include <cstdint>

namespace xla::cpu {

// This ABI version needs to be bumped whenever we make a breaking change to the
// runtime or the compiler in such a way that AOT artifacts with the same ABI
// version get incompatible or produce different numerics.
//
// Typical examples are (Not exhaustive):
// - A breaking change to the behavior of a thunk
// - The addition or deletion of a thunk type
// - A library upgrade that can change numerics (e.g. Eigen)
constexpr int32_t kHostExecutionAbiVersion = 1;

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_HOST_EXECUTION_ABI_H_
