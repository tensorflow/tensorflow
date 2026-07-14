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

#ifndef XLA_BACKENDS_CPU_COLLECTIVES_CPU_CLIQUE_KEY_H_
#define XLA_BACKENDS_CPU_COLLECTIVES_CPU_CLIQUE_KEY_H_

#include <string>

#include "absl/hash/hash.h"
#include "xla/core/collectives/clique_key.h"

namespace xla::cpu {

// Clique key for identifying a particular CPU collectives clique.
class CpuCliqueKey final : public CliqueKey {
 public:
  using CliqueKey::CliqueKey;

  bool IsSubsetOf(const CliqueKey& other) const final;
  std::string ToString() const final;

  friend bool operator==(const CpuCliqueKey& a, const CpuCliqueKey& b);
  friend bool operator<(const CpuCliqueKey& a, const CpuCliqueKey& b);
  friend bool operator>(const CpuCliqueKey& a, const CpuCliqueKey& b);

 private:
  void HashValue(absl::HashState state) const final;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_COLLECTIVES_CPU_CLIQUE_KEY_H_
