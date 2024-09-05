#ifndef XLA_SERVICE_CPU_CPU_SYMBOL_REPOSITORY_H_
#define XLA_SERVICE_CPU_CPU_SYMBOL_REPOSITORY_H_

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

#include "xla/service/symbol_repository.h"
#include "xla/xla.pb.h"

namespace xla::cpu {

// CPU-specific fields for SymbolRepositories.
struct CpuBackendSpecificData : public BackendSpecificData {};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_SYMBOL_REPOSITORY_H_
