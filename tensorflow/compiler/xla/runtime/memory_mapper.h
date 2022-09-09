/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_MEMORY_MAPPER_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_MEMORY_MAPPER_H_

#include <memory>
#include <string>
#include <string_view>
#include <system_error>  // NOLINT

#include "tensorflow/core/platform/platform.h"

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/compiler/xla/runtime/google/memory_mapper.h"
#else
#include "tensorflow/compiler/xla/runtime/default/memory_mapper.h"
#endif

#include "llvm/ExecutionEngine/SectionMemoryManager.h"

namespace xla {
namespace runtime {

// XLA runtime memory mapper allocates memory for XLA executables (object files)
// using `memfd_create` system call, and gives the user-friendly name to the
// file descriptor, so that for example in `perf` it should be possible to
// identify input XLA programs by name.
class XlaRuntimeMemoryMapper final
    : public llvm::SectionMemoryManager::MemoryMapper {
 public:
  static std::unique_ptr<XlaRuntimeMemoryMapper> Create(std::string_view name);

  llvm::sys::MemoryBlock allocateMappedMemory(
      llvm::SectionMemoryManager::AllocationPurpose purpose, size_t len,
      const llvm::sys::MemoryBlock* const near_block, unsigned prot_flags,
      std::error_code& error_code) final;

  std::error_code protectMappedMemory(const llvm::sys::MemoryBlock& block,
                                      unsigned prot_flags) final;

  std::error_code releaseMappedMemory(llvm::sys::MemoryBlock& block) final;

 private:
  explicit XlaRuntimeMemoryMapper(std::string_view name) : name_(name) {}

  std::string name_;
};

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_MEMORY_MAPPER_H_
