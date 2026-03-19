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

#ifndef XLA_SERVICE_CPU_CPU_EXECUTABLE_RUN_OPTIONS_H_
#define XLA_SERVICE_CPU_CPU_EXECUTABLE_RUN_OPTIONS_H_

#include "xla/backends/cpu/collectives/cpu_collectives.h"

namespace xla::cpu {

// CPU-specific executable options.
// We keep these separate from ExecutableRunOptions to avoid adding
// dependencies to ExecutableRunOptions.
class CpuExecutableRunOptions {
 public:
  CpuExecutableRunOptions& set_collectives(CpuCollectives* collectives) {
    collectives_ = collectives;
    return *this;
  }
  CpuCollectives* collectives() const { return collectives_; }

 private:
  // For cross-process collectives, use this collective implementation to
  // communicate.
  CpuCollectives* collectives_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_EXECUTABLE_RUN_OPTIONS_H_
