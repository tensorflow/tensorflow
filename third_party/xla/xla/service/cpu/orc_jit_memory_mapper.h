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

#ifndef XLA_SERVICE_CPU_ORC_JIT_MEMORY_MAPPER_H_
#define XLA_SERVICE_CPU_ORC_JIT_MEMORY_MAPPER_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

namespace xla {
namespace cpu {

namespace orc_jit_memory_mapper {
// Registers (if needed) a memory mapper by name and returns it if the
// memory mapper getter has been set.  Otherwise returns nullptr.
llvm::SectionMemoryManager::MemoryMapper* GetInstance(
    absl::string_view allocation_region_name);

class Registrar {
 public:
  using MemoryMapperGetter =
      std::unique_ptr<llvm::SectionMemoryManager::MemoryMapper>(
          absl::string_view allocation_region_name);
  // Registers the `mapper_getter`.  This is a no-op if `mapper_getter` is
  // null.  Precondition:  no other memory mapper getter has been registered
  // yet.
  explicit Registrar(MemoryMapperGetter* mapper_getter);
};
}  // namespace orc_jit_memory_mapper

#define XLA_INTERNAL_REGISTER_ORC_JIT_MEMORY_MAPPER_GETTER(mapper_instance, \
                                                           ctr)             \
  static ::xla::cpu::orc_jit_memory_mapper::Registrar                       \
  XLA_INTERNAL_REGISTER_ORC_JIT_MEMORY_MAPPER_GETTER_NAME(ctr)(             \
      mapper_instance)

// __COUNTER__ must go through another macro to be properly expanded
#define XLA_INTERNAL_REGISTER_ORC_JIT_MEMORY_MAPPER_GETTER_NAME(ctr) \
  __orc_jit_memory_mapper_registrar_##ctr

// Registers the MemoryMapperGetter.
#define XLA_REGISTER_ORC_JIT_MEMORY_MAPPER_GETTER(factory) \
  XLA_INTERNAL_REGISTER_ORC_JIT_MEMORY_MAPPER_GETTER(factory, __COUNTER__)
}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_ORC_JIT_MEMORY_MAPPER_H_
