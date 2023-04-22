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

#include "tensorflow/compiler/xla/service/cpu/orc_jit_memory_mapper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace xla {
namespace cpu {
namespace orc_jit_memory_mapper {

static tensorflow::mutex mapper_instance_mutex(tensorflow::LINKER_INITIALIZED);
static llvm::SectionMemoryManager::MemoryMapper* mapper_instance
    TF_GUARDED_BY(mapper_instance_mutex) = nullptr;

llvm::SectionMemoryManager::MemoryMapper* GetInstance() {
  tensorflow::mutex_lock lock(mapper_instance_mutex);
  return mapper_instance;
}

Registrar::Registrar(
    std::unique_ptr<llvm::SectionMemoryManager::MemoryMapper> mapper) {
  tensorflow::mutex_lock lock(mapper_instance_mutex);
  mapper_instance = mapper.release();
}
}  // namespace orc_jit_memory_mapper
}  // namespace cpu
}  // namespace xla
