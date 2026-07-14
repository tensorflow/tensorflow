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

#ifndef XLA_PYTHON_INSPECT_SHARDING_H_
#define XLA_PYTHON_INSPECT_SHARDING_H_

#include <cstddef>
#include <optional>
#include <string>

#include "xla/hlo/ir/hlo_sharding.h"

// Marshalls xla::HloSharding across the .so boundary between jaxlib and a
// compiler plugin. This library must be linked into xla-based compiler plugins
// that want to support InspectSharding.

// Use "C" linkage to force the struct layouts to use the c rules.
extern "C" {

struct JAX_InspectSharding_Callback_Args {
  // Serialized xla::HloSharding.
  char* sharding_spec;
  size_t sharding_spec_size;
  const char* error_txt;  // out
  void* error_scratch;
  // Deleter for the returned error.
  void (*free_error)(JAX_InspectSharding_Callback_Args* args);
};

// Memcpy-ed into the `backend_config` field of the "InspectSharding" custom
// call. During compilation, the provided callback will be called with both
// the provided data argument and the serialized xla::HloSharding (in args).
//
// All pointers here must outlive compilation.
struct JAX_InspectSharding_Callback {
  void (*call)(void* data, JAX_InspectSharding_Callback_Args* args);
  void* data;
};

}  // extern "C"

namespace jax {

// Helpers for reading and writing to JAX_InspectSharding_Callback_Args.
void InspectShardingSetError(JAX_InspectSharding_Callback_Args* args,
                             std::string msg);
std::optional<xla::HloSharding> InspectShardingReadArgs(
    JAX_InspectSharding_Callback_Args* args);

}  // namespace jax

#endif  // XLA_PYTHON_INSPECT_SHARDING_H_
