/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_COLLECTIVES_CPU_CLIQUE_H_
#define XLA_BACKENDS_CPU_COLLECTIVES_CPU_CLIQUE_H_

#include <string>

#include "absl/status/status.h"
#include "xla/backends/cpu/collectives/cpu_clique_key.h"
#include "xla/core/collectives/clique.h"

namespace xla::cpu {

// A group of CPU communicators making up a clique.
class CpuClique final : public Clique {
 public:
  explicit CpuClique(CpuCliqueKey key);

  absl::Status HealthCheck() const final;

  std::string DebugString() const final;

 private:
  CpuCliqueKey key_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_COLLECTIVES_CPU_CLIQUE_H_
