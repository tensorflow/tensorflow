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

#ifndef XLA_CORE_COLLECTIVES_CLIQUE_KEY_H_
#define XLA_CORE_COLLECTIVES_CLIQUE_KEY_H_

#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <typeindex>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/types/span.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/global_device_id.h"

namespace xla {

// Clique key for identifying a particular collective clique (set of
// communicating devices together with backend-specific information).
//
// Clique keys are backend specific and might include additional information
// that identifies a particular set of communicating devices, i.e. in GPU
// backend we distinguish between cliques that do collective operations like
// `all-reduce` and cliques that do P2P communication (`send` and `recv`), and
// these cliques launch operations (device kernels) on different device streams.
class CliqueKey {
 public:
  explicit CliqueKey(absl::Span<const GlobalDeviceId> devices);
  virtual ~CliqueKey() = default;

  CliqueKey(const CliqueKey& other) = default;
  CliqueKey& operator=(const CliqueKey& other) = default;

  CliqueKey(CliqueKey&& other) = default;
  CliqueKey& operator=(CliqueKey&& other) = default;

  // Returns the rank of the global device in the clique.
  std::optional<RankId> rank(GlobalDeviceId id) const;

  absl::Span<const GlobalDeviceId> devices() const;
  size_t num_devices() const;

  // Returns true if this clique is a subset of `other`.
  virtual bool IsSubsetOf(const CliqueKey& other) const = 0;

  virtual std::string ToString() const = 0;

  template <typename H>
  friend H AbslHashValue(H state, const CliqueKey& value);

 private:
  virtual void HashValue(absl::HashState state) const = 0;

  std::vector<GlobalDeviceId> devices_;
};

template <typename H>
H AbslHashValue(H state, const CliqueKey& value) {
  state = H::combine(std::move(state), std::type_index(typeid(&value)));
  value.HashValue(absl::HashState::Create(&state));
  return state;
}

inline std::ostream& operator<<(std::ostream& os, const CliqueKey& key) {
  return os << key.ToString();
}

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_CLIQUE_KEY_H_
