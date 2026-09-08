/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_elf_utils.h"

#include <elf.h>

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_fatbin.h"

namespace stream_executor::cuda {

std::optional<absl::Span<const uint8_t>> ParseFatBinaryOrElf(
    const void* fat_cubin) {
  if (fat_cubin == nullptr) {
    return std::nullopt;
  }

  const auto* wrapper = static_cast<const FatbinWrapper*>(fat_cubin);
  if (wrapper->data == nullptr) {
    return std::nullopt;
  }

  const uint8_t* data_bytes = static_cast<const uint8_t*>(wrapper->data);
  size_t total_size = 0;
  const auto* header = reinterpret_cast<const FatHeader*>(data_bytes);
  if (header->magic == kFatbinMagicUncompressed ||
      header->magic == kFatbinMagicCompressed) {
    total_size = static_cast<size_t>(header->header_size) +
                 static_cast<size_t>(header->fat_size);
  } else if (header->magic == kFatbinMagicLegacy) {
    total_size =
        static_cast<size_t>(*reinterpret_cast<const uint64_t*>(data_bytes + 8));
  } else if (data_bytes[0] == 0x7f && data_bytes[1] == 'E' &&
             data_bytes[2] == 'L' && data_bytes[3] == 'F') {
    const auto* elf_header = reinterpret_cast<const Elf64_Ehdr*>(data_bytes);
    total_size = static_cast<size_t>(
        elf_header->e_shoff +
        (static_cast<uint64_t>(elf_header->e_shnum) * elf_header->e_shentsize));
  } else {
    return std::nullopt;
  }

  return absl::Span<const uint8_t>(data_bytes, total_size);
}

}  // namespace stream_executor::cuda
