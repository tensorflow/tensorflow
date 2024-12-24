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

#include "xla/backends/cpu/codegen/contiguous_section_memory_manager.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>  // NOLINT

#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Process.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla::cpu {
namespace {

class DefaultMemoryMapper final
    : public llvm::SectionMemoryManager::MemoryMapper {
 public:
  llvm::sys::MemoryBlock allocateMappedMemory(
      llvm::SectionMemoryManager::AllocationPurpose purpose, size_t num_bytes,
      const llvm::sys::MemoryBlock* const near_block, unsigned flags,
      std::error_code& error_code) override {
    return llvm::sys::Memory::allocateMappedMemory(num_bytes, near_block, flags,
                                                   error_code);
  }

  std::error_code protectMappedMemory(const llvm::sys::MemoryBlock& block,
                                      unsigned flags) override {
    return llvm::sys::Memory::protectMappedMemory(block, flags);
  }

  std::error_code releaseMappedMemory(llvm::sys::MemoryBlock& m) override {
    return llvm::sys::Memory::releaseMappedMemory(m);
  }
};

}  // namespace

ContiguousSectionMemoryManager::ContiguousSectionMemoryManager(
    llvm::SectionMemoryManager::MemoryMapper* mmapper)
    : mmapper_(mmapper), mmapper_is_owned_(false) {
  if (mmapper_ == nullptr) {
    mmapper_ = new DefaultMemoryMapper();
    mmapper_is_owned_ = true;
  }
}

ContiguousSectionMemoryManager::~ContiguousSectionMemoryManager() {
  if (allocation_.allocatedSize() != 0) {
    auto ec = mmapper_->releaseMappedMemory(allocation_);
    if (ec) {
      LOG(ERROR) << "releaseMappedMemory failed with error: " << ec.message();
    }
  }
  if (mmapper_is_owned_) {
    delete mmapper_;
  }
}

void ContiguousSectionMemoryManager::reserveAllocationSpace(
    uintptr_t code_size, llvm::Align code_align, uintptr_t ro_data_size,
    llvm::Align ro_data_align, uintptr_t rw_data_size,
    llvm::Align rw_data_align) {
  CHECK_EQ(allocation_.allocatedSize(), 0);

  static const size_t page_size = llvm::sys::Process::getPageSizeEstimate();
  CHECK_LE(code_align.value(), page_size);
  CHECK_LE(ro_data_align.value(), page_size);
  CHECK_LE(rw_data_align.value(), page_size);
  code_size = RoundUpTo<uintptr_t>(code_size + code_align.value(), page_size);
  ro_data_size =
      RoundUpTo<uintptr_t>(ro_data_size + ro_data_align.value(), page_size);
  rw_data_size =
      RoundUpTo<uintptr_t>(rw_data_size + rw_data_align.value(), page_size);
  uintptr_t total_size =
      code_size + ro_data_size + rw_data_size + page_size * 3;

  std::error_code ec;
  allocation_ = mmapper_->allocateMappedMemory(
      llvm::SectionMemoryManager::AllocationPurpose::Code, total_size, nullptr,
      llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE, ec);
  if (ec) {
    LOG(ERROR) << "allocateMappedMemory failed with error: " << ec.message();
    return;
  }

  auto base = reinterpret_cast<std::uintptr_t>(allocation_.base());
  code_block_ = code_free_ =
      llvm::sys::MemoryBlock(reinterpret_cast<void*>(base), code_size);
  base += code_size;
  ro_data_block_ = ro_data_free_ =
      llvm::sys::MemoryBlock(reinterpret_cast<void*>(base), ro_data_size);
  base += ro_data_size;
  rw_data_block_ = rw_data_free_ =
      llvm::sys::MemoryBlock(reinterpret_cast<void*>(base), rw_data_size);
}

uint8_t* ContiguousSectionMemoryManager::allocateDataSection(
    uintptr_t size, unsigned alignment, unsigned section_id,
    llvm::StringRef section_name, bool is_read_only) {
  if (is_read_only) {
    return Allocate(ro_data_free_, size, alignment);
  } else {
    return Allocate(rw_data_free_, size, alignment);
  }
}

uint8_t* ContiguousSectionMemoryManager::allocateCodeSection(
    uintptr_t size, unsigned alignment, unsigned section_id,
    llvm::StringRef section_name) {
  return Allocate(code_free_, size, alignment);
}

uint8_t* ContiguousSectionMemoryManager::Allocate(
    llvm::sys::MemoryBlock& free_block, std::uintptr_t size,
    unsigned alignment) {
  auto base = reinterpret_cast<uintptr_t>(free_block.base());
  auto start = RoundUpTo<uintptr_t>(base, alignment);
  uintptr_t padded_size = (start - base) + size;
  if (padded_size > free_block.allocatedSize()) {
    LOG(ERROR) << "Failed to satisfy suballocation request for " << size;
    return nullptr;
  }
  free_block =
      llvm::sys::MemoryBlock(reinterpret_cast<void*>(base + padded_size),
                             free_block.allocatedSize() - padded_size);
  return reinterpret_cast<uint8_t*>(start);
}

bool ContiguousSectionMemoryManager::finalizeMemory(std::string* err_msg) {
  std::error_code ec;

  ec = mmapper_->protectMappedMemory(
      code_block_, llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_EXEC);
  if (ec) {
    if (err_msg) {
      *err_msg = ec.message();
    }
    return true;
  }
  ec =
      mmapper_->protectMappedMemory(ro_data_block_, llvm::sys::Memory::MF_READ);
  if (ec) {
    if (err_msg) {
      *err_msg = ec.message();
    }
    return true;
  }

  llvm::sys::Memory::InvalidateInstructionCache(code_block_.base(),
                                                code_block_.allocatedSize());
  return false;
}

}  // namespace xla::cpu
