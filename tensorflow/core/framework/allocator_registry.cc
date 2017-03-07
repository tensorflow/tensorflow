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

#include "tensorflow/core/framework/allocator_registry.h"

#include <string>

namespace tensorflow {

// static
AllocatorRegistry* AllocatorRegistry::Global() {
  static AllocatorRegistry* global_allocator_registry = new AllocatorRegistry;
  return global_allocator_registry;
}

void AllocatorRegistry::Register(const string& name, uint8_t priority,
                                 Allocator* allocator) {
  AllocatorRegistryEntry tmp_entry;
  tmp_entry.name = name;
  tmp_entry.priority = priority;
  tmp_entry.allocator = allocator;
  allocators_.push_back(tmp_entry);
  int high_pri = -1;
  for (std::vector<AllocatorRegistryEntry>::iterator it = allocators_.begin();
       it != allocators_.end(); ++it) {
    if (high_pri < it->priority) {
      m_curr_allocator_ = it->allocator;
      high_pri = it->priority;
    }
  }
}

Allocator* AllocatorRegistry::GetAllocator() {
  assert(m_curr_allocator_ != nullptr);
  return m_curr_allocator_;
}

}  // namespace tensorflow
