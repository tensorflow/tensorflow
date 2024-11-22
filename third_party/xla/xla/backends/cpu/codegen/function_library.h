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

#ifndef XLA_BACKENDS_CPU_CODEGEN_FUNCTION_LIBRARY_H_
#define XLA_BACKENDS_CPU_CODEGEN_FUNCTION_LIBRARY_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/util.h"

namespace xla::cpu {

// A library of compiled functions required by the XLA:CPU runtime to execute
// an XLA program.
class FunctionLibrary {
 public:
  virtual ~FunctionLibrary() = default;

  using Kernel = SE_HOST_Kernel*;

  virtual absl::StatusOr<Kernel> FindKernel(std::string_view name) const {
    return Unimplemented("Kernel %s not found", name);
  }
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_FUNCTION_LIBRARY_H_
