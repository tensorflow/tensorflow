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

#ifndef XLA_CORE_COLLECTIVES_CLIQUE_ID_H_
#define XLA_CORE_COLLECTIVES_CLIQUE_ID_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace xla {

// A globally unique collective clique identifier. Underlying bytes payload is
// backend specific and optional, some collectives implementations may not
// require it.
//
// Globally unique clique identifier allows multiple hosts and devices to find
// each other and agree who is a member of a clique. It is a user responsibility
// to redistribute this id to all participating hosts (i.e. JAX uses shared KV
// store for that). For single host collective operations XLA automatically
// generates a unique id for local cliques (cliques consisting of devices
// visible from a process).
class CliqueId {
 public:
  CliqueId() = default;

  explicit CliqueId(absl::string_view data);

  absl::Span<const char> data() const;
  std::string ToString() const;
  uint32_t fingerprint() const;
  size_t size() const;

  template <typename H>
  friend H AbslHashValue(H h, const CliqueId& id);

 private:
  std::vector<char> data_;
};

template <typename H>
H AbslHashValue(H h, const CliqueId& id) {
  return H::combine(std::move(h), id.data());
}

// An evenly distributed list of root ranks (cliqueIds) to spread communication
// during clique setup.
class CliqueIds {
 public:
  CliqueIds() = default;

  explicit CliqueIds(const CliqueId& id);

  void Add(const CliqueId& id);

  absl::Span<const CliqueId> data() const;

  const CliqueId& at(size_t index) const;

  uint32_t fingerprint() const;

  template <typename H>
  friend H AbslHashValue(H h, const CliqueIds& ids);

 private:
  std::vector<CliqueId> ids_;
};

template <typename H>
H AbslHashValue(H h, const CliqueIds& ids) {
  return H::combine(std::move(h), ids.data());
}

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_CLIQUE_ID_H_
