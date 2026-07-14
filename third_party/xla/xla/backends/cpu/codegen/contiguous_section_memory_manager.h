/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_CONTIGUOUS_SECTION_MEMORY_MANAGER_H_
#define XLA_BACKENDS_CPU_CODEGEN_CONTIGUOUS_SECTION_MEMORY_MANAGER_H_

#include <cstdint>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Memory.h"

namespace xla::cpu {

// On Windows, LLVM may emit IMAGE_REL_AMD64_ADDR32NB COFF relocations when
// referring to read-only data, however IMAGE_REL_AMD64_ADDR32NB requires that
// the read-only data section follow within 2GB of the code. Oddly enough,
// the LLVM SectionMemoryManager does nothing to enforce this
// (https://github.com/llvm/llvm-project/issues/55386), leading to crashes on
// Windows when the sections end up in the wrong order. Since none
// of the memory managers in the LLVM tree obey the necessary ordering
// constraints, we need to roll our own.
//
// ContiguousSectionMemoryManager is an alternative to SectionMemoryManager
// that maps one large block of memory and suballocates it
// for each section, in the correct order. This is easy enough to do because of
// the llvm::RuntimeDyld::MemoryManager::reserveAllocationSpace() hook, which
// ensures that LLVM will tell us ahead of time the total sizes of all the
// relevant sections. We also know that XLA isn't going to do any more
// complicated memory management: we will allocate the sections once and we are
// done.
class ContiguousSectionMemoryManager : public llvm::RTDyldMemoryManager {
 public:
  explicit ContiguousSectionMemoryManager(
      llvm::SectionMemoryManager::MemoryMapper* mmapper);
  ~ContiguousSectionMemoryManager() override;

  bool needsToReserveAllocationSpace() override { return true; }
  void reserveAllocationSpace(uintptr_t code_size, llvm::Align code_align,
                              uintptr_t ro_data_size, llvm::Align ro_data_align,
                              uintptr_t rw_data_size,
                              llvm::Align rw_data_align) override;

  uint8_t* allocateDataSection(uintptr_t size, unsigned alignment,
                               unsigned section_id,
                               llvm::StringRef section_name,
                               bool is_read_only) override;

  uint8_t* allocateCodeSection(uintptr_t size, unsigned alignment,
                               unsigned section_id,
                               llvm::StringRef section_name) override;

  bool finalizeMemory(std::string* err_msg) override;

 private:
  llvm::SectionMemoryManager::MemoryMapper* mmapper_;
  bool mmapper_is_owned_;

  llvm::sys::MemoryBlock allocation_;

  // Sections must be in the order code < rodata < rwdata.
  llvm::sys::MemoryBlock code_block_;
  llvm::sys::MemoryBlock ro_data_block_;
  llvm::sys::MemoryBlock rw_data_block_;

  llvm::sys::MemoryBlock code_free_;
  llvm::sys::MemoryBlock ro_data_free_;
  llvm::sys::MemoryBlock rw_data_free_;

  uint8_t* Allocate(llvm::sys::MemoryBlock& free_block, std::uintptr_t size,
                    unsigned alignment);
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_CONTIGUOUS_SECTION_MEMORY_MANAGER_H_
