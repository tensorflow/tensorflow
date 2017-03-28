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

// Classes to maintain a static registry of memory allocators
#ifndef TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_REGISTRY_H_
#define TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_REGISTRY_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/allocator.h"

namespace tensorflow {

// A global AllocatorRegistry is used to hold allocators for CPU backends
class AllocatorRegistry {
 public:
  // Add an allocator to the registry.
  void Register(const string& name, int priority, Allocator* allocator);

  // Return allocator with highest priority
  // If multiple allocators have the same high priority, return one of them
  Allocator* GetAllocator();

  // Returns the global registry of allocators.
  static AllocatorRegistry* Global();

 private:
  typedef struct {
    string name;
    int priority;
    Allocator* allocator;  // not owned
  } AllocatorRegistryEntry;

  bool CheckForDuplicates(const string& name, int priority);

  std::vector<AllocatorRegistryEntry> allocators_;
  Allocator* m_curr_allocator_;  // not owned
};

namespace allocator_registration {

class AllocatorRegistration {
 public:
  AllocatorRegistration(const string& name, int priority,
                        Allocator* allocator) {
    AllocatorRegistry::Global()->Register(name, priority, allocator);
  }
};

}  // namespace allocator_registration

#define REGISTER_MEM_ALLOCATOR(name, priority, allocator) \
  REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(__COUNTER__, name, priority, allocator)

#define REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(ctr, name, priority, allocator) \
  REGISTER_MEM_ALLOCATOR_UNIQ(ctr, name, priority, allocator)

#define REGISTER_MEM_ALLOCATOR_UNIQ(ctr, name, priority, allocator) \
  static allocator_registration::AllocatorRegistration              \
      register_allocator_##ctr(name, priority, new allocator)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_REGISTRY_H_
