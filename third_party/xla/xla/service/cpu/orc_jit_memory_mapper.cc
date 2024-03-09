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

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace cpu {
namespace orc_jit_memory_mapper {

static absl::Mutex mapper_instance_mutex(absl::kConstInit);
static llvm::SectionMemoryManager::MemoryMapper* mapper_instance
    ABSL_GUARDED_BY(mapper_instance_mutex) = nullptr;

llvm::SectionMemoryManager::MemoryMapper* GetInstance() {
  absl::MutexLock lock(&mapper_instance_mutex);
  return mapper_instance;
}

Registrar::Registrar(
    std::unique_ptr<llvm::SectionMemoryManager::MemoryMapper> mapper) {
  absl::MutexLock lock(&mapper_instance_mutex);
  mapper_instance = mapper.release();
}
}  // namespace orc_jit_memory_mapper
}  // namespace cpu
}  // namespace xla
