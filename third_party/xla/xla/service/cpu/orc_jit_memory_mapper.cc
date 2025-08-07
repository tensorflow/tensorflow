/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/orc_jit_memory_mapper.h"

#include <atomic>
#include <memory>
#include <string>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

namespace xla {
namespace cpu {
namespace orc_jit_memory_mapper {

static std::atomic<Registrar::MemoryMapperGetter*> mapper_getter_ptr{nullptr};

static absl::Mutex mapper_instances_mutex(absl::kConstInit);
static absl::flat_hash_map<std::string,
                           llvm::SectionMemoryManager::MemoryMapper*>*
    mapper_instances ABSL_GUARDED_BY(mapper_instances_mutex) = nullptr;

llvm::SectionMemoryManager::MemoryMapper* GetInstance(
    absl::string_view allocation_region_name) {
  Registrar::MemoryMapperGetter* getter =
      mapper_getter_ptr.load(std::memory_order_acquire);

  {
    if (getter == nullptr) {
      return nullptr;
    }
  }
  absl::MutexLock lock(&mapper_instances_mutex);
  auto it = mapper_instances->find(allocation_region_name);
  if (it == mapper_instances->end()) {
    it = mapper_instances
             ->insert({std::string(allocation_region_name),
                       (*getter)(allocation_region_name).release()})
             .first;
  }

  return it->second;
}

Registrar::Registrar(Registrar::MemoryMapperGetter* mapper_getter) {
  Registrar::MemoryMapperGetter* expected_nullptr = nullptr;

  CHECK(mapper_getter_ptr.compare_exchange_strong(
      expected_nullptr, mapper_getter, std::memory_order_release,
      std::memory_order_acquire));

  {
    absl::MutexLock lock(&mapper_instances_mutex);
    if (mapper_instances == nullptr) {
      mapper_instances =
          new absl::flat_hash_map<std::string,
                                  llvm::SectionMemoryManager::MemoryMapper*>();
    }
  }
}

}  // namespace orc_jit_memory_mapper
}  // namespace cpu
}  // namespace xla
