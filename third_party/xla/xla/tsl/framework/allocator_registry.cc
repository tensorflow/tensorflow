/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/framework/allocator_registry.h"

#include <string>

#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {

// static
AllocatorFactoryRegistry* AllocatorFactoryRegistry::singleton() {
  static AllocatorFactoryRegistry* const singleton =
      new AllocatorFactoryRegistry;
  return singleton;
}

const AllocatorFactoryRegistry::FactoryEntry*
AllocatorFactoryRegistry::FindEntry(const string& name, int priority) const {
  for (auto& entry : factories_) {
    if (!name.compare(entry.name) && priority == entry.priority) {
      return &entry;
    }
  }
  return nullptr;
}

void AllocatorFactoryRegistry::Register(const char* source_file,
                                        int source_line, const string& name,
                                        int priority,
                                        AllocatorFactory* factory) {
  absl::MutexLock l(&mu_);
  CHECK(!first_alloc_made_) << "Attempt to register an AllocatorFactory "
                            << "after call to GetAllocator()";
  CHECK(!name.empty()) << "Need a valid name for Allocator";
  CHECK_GE(priority, 0) << "Priority needs to be non-negative";

  const FactoryEntry* existing = FindEntry(name, priority);
  if (existing != nullptr) {
    // Duplicate registration is a hard failure.
    LOG(FATAL) << "New registration for AllocatorFactory with name=" << name
               << " priority=" << priority << " at location " << source_file
               << ":" << source_line
               << " conflicts with previous registration at location "
               << existing->source_file << ":" << existing->source_line;
  }

  FactoryEntry entry;
  entry.source_file = source_file;
  entry.source_line = source_line;
  entry.name = name;
  entry.priority = priority;
  entry.factory.reset(factory);
  factories_.push_back(std::move(entry));
}

Allocator* AllocatorFactoryRegistry::GetAllocator() {
  absl::MutexLock l(&mu_);
  first_alloc_made_ = true;
  FactoryEntry* best_entry = nullptr;
  for (auto& entry : factories_) {
    if (best_entry == nullptr) {
      best_entry = &entry;
    } else if (entry.priority > best_entry->priority) {
      best_entry = &entry;
    }
  }
  if (best_entry) {
    if (!best_entry->allocator) {
      best_entry->allocator.reset(best_entry->factory->CreateAllocator());
    }
    return best_entry->allocator.get();
  } else {
    LOG(FATAL) << "No registered CPU AllocatorFactory";
    return nullptr;
  }
}

SubAllocator* AllocatorFactoryRegistry::GetSubAllocator(int numa_node) {
  absl::MutexLock l(&mu_);
  first_alloc_made_ = true;
  FactoryEntry* best_entry = nullptr;
  for (auto& entry : factories_) {
    if (best_entry == nullptr) {
      best_entry = &entry;
    } else if (best_entry->factory->NumaEnabled()) {
      if (entry.factory->NumaEnabled() &&
          (entry.priority > best_entry->priority)) {
        best_entry = &entry;
      }
    } else {
      DCHECK(!best_entry->factory->NumaEnabled());
      if (entry.factory->NumaEnabled() ||
          (entry.priority > best_entry->priority)) {
        best_entry = &entry;
      }
    }
  }
  if (best_entry) {
    int index = 0;
    if (numa_node != port::kNUMANoAffinity) {
      CHECK_LE(numa_node, port::NUMANumNodes());
      index = 1 + numa_node;
    }
    if (best_entry->sub_allocators.size() < static_cast<size_t>(index + 1)) {
      best_entry->sub_allocators.resize(index + 1);
    }
    if (!best_entry->sub_allocators[index].get()) {
      best_entry->sub_allocators[index].reset(
          best_entry->factory->CreateSubAllocator(numa_node));
    }
    return best_entry->sub_allocators[index].get();
  } else {
    LOG(FATAL) << "No registered CPU AllocatorFactory";
    return nullptr;
  }
}

}  // namespace tsl
