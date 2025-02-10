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

#ifndef XLA_SERVICE_GPU_GPU_SYMBOL_REPOSITORY_H_
#define XLA_SERVICE_GPU_GPU_SYMBOL_REPOSITORY_H_

#include <optional>

#include "xla/autotune_results.pb.h"
#include "xla/service/symbol_repository.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// GPU-specific fields for SymbolRepositories.
struct GpuBackendSpecificData : public BackendSpecificData {
  std::optional<GpuCompilationEnvironment> gpu_compilation_environment;
  std::optional<AutotuneResults> autotune_results;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_SYMBOL_REPOSITORY_H_
