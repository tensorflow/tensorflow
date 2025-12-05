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

#ifndef XLA_BACKENDS_CPU_CODEGEN_JIT_MEMORY_MAPPER_H_
#define XLA_BACKENDS_CPU_CODEGEN_JIT_MEMORY_MAPPER_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

namespace xla::cpu {

// Registers (if needed) a memory mapper by name and returns it if the
// memory mapper getter has been set. Otherwise returns nullptr.
llvm::SectionMemoryManager::MemoryMapper* GetJitMemoryMapper(
    absl::string_view allocation_region_name);

namespace internal {
// Registers the `mapper_getter`.  This is a no-op if `mapper_getter` is
// null. Precondition:  no other memory mapper getter has been registered yet.
class JitMemoryMapperRegistration {
 public:
  using MemoryMapperGetter =
      std::unique_ptr<llvm::SectionMemoryManager::MemoryMapper>(
          absl::string_view allocation_region_name);
  explicit JitMemoryMapperRegistration(MemoryMapperGetter* mapper_getter);
};

}  // namespace internal
}  // namespace xla::cpu

#define XLA_CPU_INTERNAL_REGISTER_JIT_MEMORY_MAPPER_GETTER(INSTANCE, COUNT)  \
  static absl::NoDestructor<xla::cpu::internal::JitMemoryMapperRegistration> \
  XLA_CPU_INTERNAL_REGISTER_JIT_MEMORY_MAPPER_GETTER_NAME(COUNT)(INSTANCE)

// __COUNTER__ must go through another macro to be properly expanded
#define XLA_CPU_INTERNAL_REGISTER_JIT_MEMORY_MAPPER_GETTER_NAME(COUNT) \
  __xla_cpu_jit_memory_mapper_registration_##COUNT

// Registers the MemoryMapperGetter.
#define XLA_CPU_REGISTER_JIT_MEMORY_MAPPER_GETTER(factory) \
  XLA_CPU_INTERNAL_REGISTER_JIT_MEMORY_MAPPER_GETTER(factory, __COUNTER__)

#endif  // XLA_BACKENDS_CPU_CODEGEN_JIT_MEMORY_MAPPER_H_
