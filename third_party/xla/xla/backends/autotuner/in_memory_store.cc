/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/autotuner/in_memory_store.h"

#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/tsl/util/sorted_range.h"

namespace xla {

namespace {

struct TargetKey {
  std::string device;
  std::string explicit_version;
  std::string hlo_fingerprint;

  bool operator==(const TargetKey& other) const {
    return device == other.device &&
           explicit_version == other.explicit_version &&
           hlo_fingerprint == other.hlo_fingerprint;
  }

  bool operator!=(const TargetKey& other) const { return !(*this == other); }

  bool operator<(const TargetKey& other) const {
    if (device != other.device) {
      return device < other.device;
    }
    if (explicit_version != other.explicit_version) {
      return explicit_version < other.explicit_version;
    }
    return hlo_fingerprint < other.hlo_fingerprint;
  }

  template <typename H>
  friend H AbslHashValue(H h, const TargetKey& k) {
    return H::combine(std::move(h), k.device, k.explicit_version,
                      k.hlo_fingerprint);
  }
};

struct GlobalState {
  absl::Mutex mutex;
  absl::flat_hash_map<TargetKey, std::vector<autotuner::AutotuneEntry>> entries
      ABSL_GUARDED_BY(mutex);
};

GlobalState& GetGlobalState() {
  static absl::NoDestructor<GlobalState> state;
  return *state;
}

TargetKey ToTargetKey(const autotuner::AutotuneTargetKey& proto) {
  return TargetKey{proto.device(), proto.explicit_version(),
                   proto.hlo_fingerprint()};
}

// Returns true if the two full AutotuneKeys refer to the same cache slot (same
// target and environment).
bool IsSameCacheSlot(const autotuner::AutotuneKey& a,
                     const autotuner::AutotuneKey& b) {
  return a.target().device() == b.target().device() &&
         a.target().explicit_version() == b.target().explicit_version() &&
         a.target().hlo_fingerprint() == b.target().hlo_fingerprint() &&
         a.environment().codegen_version() ==
             b.environment().codegen_version() &&
         a.environment().codegen_options_fingerprint() ==
             b.environment().codegen_options_fingerprint();
}

}  // namespace

absl::StatusOr<std::vector<autotuner::AutotuneEntry>> InMemoryStore::Read(
    const autotuner::AutotuneTargetKey& target_key) {
  auto& state = GetGlobalState();
  absl::MutexLock lock(state.mutex);
  auto it = state.entries.find(ToTargetKey(target_key));
  if (it == state.entries.end()) {
    return std::vector<autotuner::AutotuneEntry>{};
  }
  return it->second;
}

absl::Status InMemoryStore::Write(const autotuner::AutotuneEntry& entry) {
  auto& state = GetGlobalState();
  absl::MutexLock lock(state.mutex);
  std::vector<autotuner::AutotuneEntry>& bucket =
      state.entries[ToTargetKey(entry.key().target())];
  for (autotuner::AutotuneEntry& existing : bucket) {
    if (IsSameCacheSlot(existing.key(), entry.key())) {
      *existing.mutable_value() = entry.value();
      return absl::OkStatus();
    }
  }
  bucket.push_back(entry);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<autotuner::AutotuneEntry>> InMemoryStore::ReadAll() {
  auto& state = GetGlobalState();
  absl::MutexLock lock(state.mutex);
  std::vector<autotuner::AutotuneEntry> all;
  for (const auto& [key, bucket] : tsl::KeySortedRange(state.entries)) {
    all.insert(all.end(), bucket.begin(), bucket.end());
  }
  return all;
}

void InMemoryStore::Clear() {
  auto& state = GetGlobalState();
  absl::MutexLock lock(state.mutex);
  state.entries.clear();
}

}  // namespace xla
