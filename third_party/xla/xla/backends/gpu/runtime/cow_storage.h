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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COW_STORAGE_H_
#define XLA_BACKENDS_GPU_RUNTIME_COW_STORAGE_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/device_slot.h"

namespace xla::gpu {

// Copy-on-write storage for holding per-device state slots built on first
// touch. The storage itself is thread-safe for concurrent lookups (`Find` is
// lock-free) and insertions (`GetOrCreate` serializes via a mutex and publishes
// a new immutable snapshot). However, the returned state slots are not
// internally synchronized; it is only safe to mutate them across threads if
// each thread is dedicated to a distinct device ordinal/index.
class CowStorage {
 public:
  CowStorage() = default;

  CowStorage(const CowStorage&) = delete;
  CowStorage& operator=(const CowStorage&) = delete;

  // Lock-free. Returns nullptr if `device_ordinal` is negative or has no slot
  // yet.
  DeviceSlot* Find(int device_ordinal) const;

  // Returns the slot for `device_ordinal`, building it on the first call.
  // Lock-free once the slot exists. Returns nullptr if `device_ordinal` is
  // negative.
  DeviceSlot* GetOrCreate(int device_ordinal, DeviceSlotFactoryRef factory)
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  struct Entry {
    int device_ordinal;
    DeviceSlot* slot;
  };

  // Sized for a typical 8-GPU host so snapshots stay inline in the common
  // case.
  static constexpr size_t kInlinedSnapshotSize = 8;

  // Immutable once published.
  using Snapshot = absl::InlinedVector<Entry, kInlinedSnapshotSize>;

  static DeviceSlot* FindIn(const Snapshot& snapshot, int device_ordinal);

  std::atomic<const Snapshot*> snapshot_{nullptr};
  absl::Mutex mutex_;
  // Heap slots keep their address across snapshot copies and stay cache-line
  // aligned.
  std::vector<std::unique_ptr<DeviceSlot>> slots_ ABSL_GUARDED_BY(mutex_);
  // Freed only in the destructor, so readers never see a freed snapshot.
  std::vector<std::unique_ptr<Snapshot>> snapshots_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COW_STORAGE_H_
