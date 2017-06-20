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

#include <string>

#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// static
AllocatorRegistry* AllocatorRegistry::Global() {
  static AllocatorRegistry* global_allocator_registry = new AllocatorRegistry;
  return global_allocator_registry;
}

Allocator* AllocatorRegistry::GetRegisteredAllocator(const string& name,
                                                     int priority) {
  for (auto entry : allocators_) {
    if (!name.compare(entry.name) && priority == entry.priority) {
      return entry.allocator;
    }
  }
  return nullptr;
}

void AllocatorRegistry::Register(const string& name, int priority,
                                 Allocator* allocator) {
  CHECK(!name.empty()) << "Need a valid name for Allocator";
  CHECK_GE(priority, 0) << "Priority needs to be non-negative";

  Allocator* existing = GetRegisteredAllocator(name, priority);
  if (existing != nullptr) {
    // A duplicate is if the registration name and priority match
    // but the Allocator::Name()'s don't match.
    CHECK_EQ(existing->Name(), allocator->Name())
        << "Allocator with name: [" << name << "], type [" << existing->Name()
        << "], priority: [" << priority
        << "] already registered.  Choose a different name to register "
        << "an allocator of type " << allocator->Name();

    // The allocator names match, so we can just return.
    // It should be safe to delete the allocator since the caller
    // gives up ownership of it.
    delete allocator;
    return;
  }

  AllocatorRegistryEntry tmp_entry;
  tmp_entry.name = name;
  tmp_entry.priority = priority;
  tmp_entry.allocator = allocator;

  allocators_.push_back(tmp_entry);
  int high_pri = -1;
  for (auto entry : allocators_) {
    if (high_pri < entry.priority) {
      m_curr_allocator_ = entry.allocator;
      high_pri = entry.priority;
    }
  }
}

Allocator* AllocatorRegistry::GetAllocator() {
  return CHECK_NOTNULL(m_curr_allocator_);
}

}  // namespace tensorflow
