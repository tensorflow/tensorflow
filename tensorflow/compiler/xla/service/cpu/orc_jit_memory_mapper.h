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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_ORC_JIT_MEMORY_MAPPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_ORC_JIT_MEMORY_MAPPER_H_

#include <memory>

#include "llvm/ExecutionEngine/SectionMemoryManager.h"

namespace xla {
namespace cpu {

namespace orc_jit_memory_mapper {
// Returns the registered memory mapper if there is one.  Returns nullptr if no
// memory mapper is registered.
llvm::SectionMemoryManager::MemoryMapper* GetInstance();

class Registrar {
 public:
  // Registers the `mapper` as a memory mapper.  This is a no-op if `mapper` is
  // null.  Precondition:  no other memory mapper has been registered yet.
  explicit Registrar(
      std::unique_ptr<llvm::SectionMemoryManager::MemoryMapper> mapper);
};
}  // namespace orc_jit_memory_mapper

#define XLA_INTERNAL_REGISTER_ORC_JIT_MEMORY_MAPPER(mapper_instance, ctr) \
  static ::xla::cpu::orc_jit_memory_mapper::Registrar                     \
      XLA_INTERNAL_REGISTER_ORC_JIT_MEMORY_MAPPER_NAME(ctr)(mapper_instance)

// __COUNTER__ must go through another macro to be properly expanded
#define XLA_INTERNAL_REGISTER_ORC_JIT_MEMORY_MAPPER_NAME(ctr) \
  __orc_jit_memory_mapper_registrar_##ctr

// Registers the std::unique_ptr<llvm::SectionMemoryManager::MemoryMapper>
// returned by the `factory` expression.  `factory` is allowed to evaluate to
// a null unique_ptr in which case this macro does nothing.
#define XLA_REGISTER_ORC_JIT_MEMORY_MAPPER(factory) \
  XLA_INTERNAL_REGISTER_ORC_JIT_MEMORY_MAPPER(factory, __COUNTER__)
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_ORC_JIT_MEMORY_MAPPER_H_
