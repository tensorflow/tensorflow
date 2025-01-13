/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_FUNCTION_LIBRARY_H_
#define XLA_BACKENDS_CPU_RUNTIME_FUNCTION_LIBRARY_H_

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

// A library of functions required by the XLA:CPU runtime to execute an XLA
// program.
//
// XLA:CPU program compiles to a collection of functions that are dispatched by
// the runtime. The most common type of compiled function is an XLA CPU Kernel,
// however some operations can be compiled to auxiliary functions that are
// invoked by operation-specific Thunks, e.g. `sort` operation comparator
// compiles to a separate function used by a SortThunk in combination with an
// `std::sort` library call.
class FunctionLibrary {
 public:
  // Compute kernel function type (corresponds to `fusion` operation).
  using Kernel = XLA_CPU_Kernel;

  // Comparator functor for `sort` operation.
  //
  // TODO(ezhulenev): We rely on legacy IrEmitter to emit comparator
  // functions, and we use legacy compute function ABI. We should emit a
  // much simpler comparator function that only takes compared values.
  using Comparator = void(bool* result, const void* run_options,
                          const void** params, const void* buffer_table,
                          const void* status, const void* prof_counters);

  virtual ~FunctionLibrary() = default;

  // We use a `TypeId` to distinguish functions of different type at run time.
  TSL_LIB_GTL_DEFINE_INT_TYPE(TypeId, int64_t);
  static constexpr TypeId kUnknownTypeId = TypeId(0);

  struct Symbol {
    TypeId type_id;
    std::string name;
  };

  template <typename F, std::enable_if_t<std::is_function_v<F>>* = nullptr>
  static Symbol Sym(std::string name) {
    return Symbol{GetTypeId<F>(), std::move(name)};
  }

  template <typename F, std::enable_if_t<std::is_function_v<F>>* = nullptr>
  absl::StatusOr<F*> ResolveFunction(absl::string_view name) {
    TF_ASSIGN_OR_RETURN(void* ptr, ResolveFunction(GetTypeId<F>(), name));
    return reinterpret_cast<F*>(ptr);
  }

 protected:
  // Returns a type-erased pointer to the function with the given name and type
  // id. Implementation might choose not to verify the type id and then it is up
  // to the caller to ensure the resolved function is of the correct type.
  virtual absl::StatusOr<void*> ResolveFunction(TypeId type_id,
                                                absl::string_view name) = 0;

 private:
  // Returns a type id for a given function type.
  template <typename F, std::enable_if_t<std::is_function_v<F>>* = nullptr>
  static TypeId GetTypeId() {
    static const TypeId id = GetNextTypeId();
    return id;
  }

  static TypeId GetNextTypeId();
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_FUNCTION_LIBRARY_H_
