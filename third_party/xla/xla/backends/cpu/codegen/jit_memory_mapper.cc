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

#include "xla/backends/cpu/codegen/jit_memory_mapper.h"

#include <atomic>
#include <memory>
#include <string>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

namespace xla::cpu {

static absl::NoDestructor<
    std::atomic<internal::JitMemoryMapperRegistration::MemoryMapperGetter*>>
    mapper_getter(nullptr);

static absl::Mutex mapper_instances_mutex(absl::kConstInit);
static absl::NoDestructor<
    absl::flat_hash_map<std::string, llvm::SectionMemoryManager::MemoryMapper*>>
    mapper_instances ABSL_GUARDED_BY(mapper_instances_mutex);

internal::JitMemoryMapperRegistration::JitMemoryMapperRegistration(
    JitMemoryMapperRegistration::MemoryMapperGetter* getter) {
  JitMemoryMapperRegistration::MemoryMapperGetter* expected_nullptr = nullptr;
  CHECK(mapper_getter->compare_exchange_strong(expected_nullptr, getter,
                                               std::memory_order_release,
                                               std::memory_order_acquire));
}

llvm::SectionMemoryManager::MemoryMapper* GetJitMemoryMapper(
    absl::string_view allocation_region_name) {
  internal::JitMemoryMapperRegistration::MemoryMapperGetter* getter =
      mapper_getter->load(std::memory_order_acquire);

  if (getter == nullptr) {
    return nullptr;
  }

  absl::MutexLock lock(mapper_instances_mutex);
  auto it = mapper_instances->find(allocation_region_name);
  if (it == mapper_instances->end()) {
    it = mapper_instances
             ->insert({std::string(allocation_region_name),
                       (*getter)(allocation_region_name).release()})
             .first;
  }

  return it->second;
}

}  // namespace xla::cpu
