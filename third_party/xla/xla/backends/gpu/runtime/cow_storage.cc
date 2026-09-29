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

#include "xla/backends/gpu/runtime/cow_storage.h"

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/device_slot.h"

namespace xla::gpu {

DeviceSlot* CowStorage::Find(int device_ordinal) const {
  if (device_ordinal < 0) {
    return nullptr;
  }
  const Snapshot* snapshot = snapshot_.load(std::memory_order_acquire);
  if (snapshot == nullptr) {
    return nullptr;
  }
  return FindIn(*snapshot, device_ordinal);
}

DeviceSlot* CowStorage::GetOrCreate(int device_ordinal,
                                    DeviceSlotFactoryRef factory) {
  if (device_ordinal < 0) {
    return nullptr;
  }
  if (DeviceSlot* slot = Find(device_ordinal)) {
    return slot;
  }
  absl::MutexLock lock(mutex_);
  // Re-check under `mutex_` in case another thread concurrently created the
  // slot for `device_ordinal` after the lock-free `Find` above. Snapshots are
  // only published under `mutex_`, so relaxed is enough.
  const Snapshot* current = snapshot_.load(std::memory_order_relaxed);
  if (current != nullptr) {
    if (DeviceSlot* slot = FindIn(*current, device_ordinal)) {
      return slot;
    }
  }
  slots_.push_back(factory());
  std::unique_ptr<Snapshot> next = current == nullptr
                                       ? std::make_unique<Snapshot>()
                                       : std::make_unique<Snapshot>(*current);
  next->push_back(Entry{device_ordinal, slots_.back().get()});
  snapshots_.push_back(std::move(next));
  const Snapshot* published = snapshots_.back().get();
  snapshot_.store(published, std::memory_order_release);
  return published->back().slot;
}

DeviceSlot* CowStorage::FindIn(const Snapshot& snapshot, int device_ordinal) {
  for (const Entry& entry : snapshot) {
    if (entry.device_ordinal == device_ordinal) {
      return entry.slot;
    }
  }
  return nullptr;
}

}  // namespace xla::gpu
