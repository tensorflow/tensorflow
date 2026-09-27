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

#include "xla/python/ifrt/memory.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/device.h"

namespace xla {
namespace ifrt {

namespace {

// Global state that keeps a stable copy of memory kind strings for `MemoryKind`
// instances.
struct MemoryKindsSet {
  absl::Mutex mu;
  absl::node_hash_set<std::string> memory_kinds_set ABSL_GUARDED_BY(mu);
  absl::string_view default_memory_kind;

  MemoryKindsSet() {
    memory_kinds_set.insert("device");
    default_memory_kind = *memory_kinds_set.begin();
  }

  static MemoryKindsSet& Get() {
    static absl::NoDestructor<MemoryKindsSet> global_set;
    return *global_set;
  }
};

absl::string_view InternMemoryKind(absl::string_view memory_kind) {
  MemoryKindsSet& global_set = MemoryKindsSet::Get();
  if ((memory_kind.data() == global_set.default_memory_kind.data() &&
       memory_kind.size() == global_set.default_memory_kind.size()) ||
      memory_kind == global_set.default_memory_kind) {
    return global_set.default_memory_kind;
  }
  absl::MutexLock lock(global_set.mu);
  auto it = global_set.memory_kinds_set.find(memory_kind);
  if (it == global_set.memory_kinds_set.end()) {
    return *global_set.memory_kinds_set.insert(std::string(memory_kind)).first;
  }
  return *it;
}

}  // namespace

MemoryKind::MemoryKind()
    : memory_kind_(MemoryKindsSet::Get().default_memory_kind) {}

MemoryKind::MemoryKind(std::optional<absl::string_view> memory_kind)
    : memory_kind_(memory_kind.has_value()
                       ? InternMemoryKind(memory_kind.value())
                       : MemoryKindsSet::Get().default_memory_kind) {}

bool MemoryKind::is_default() const {
  // Use a pointer comparison. `memory_kind_` always points to an interned
  // string. We can only check the beginning of the string because having a
  // different length will lead to a different pointer during interning.
  return memory_kind_.data() ==
         MemoryKindsSet::Get().default_memory_kind.data();
}

std::string MemoryKind::ToString() const { return std::string(memory_kind_); }

MemoryKind CanonicalizeMemoryKind(MemoryKind memory_kind,
                                  const Device* device) {
  return memory_kind;
}

char Memory::ID = 0;

}  // namespace ifrt
}  // namespace xla
