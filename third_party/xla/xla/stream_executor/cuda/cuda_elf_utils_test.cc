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

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_fatbin.h"

namespace stream_executor::cuda {
namespace {

Elf64_Ehdr MakeElfHeader(uint8_t abi_version, uint32_t flags) {
  Elf64_Ehdr header{};
  header.e_ident[EI_MAG0] = 0x7f;
  header.e_ident[EI_MAG1] = 'E';
  header.e_ident[EI_MAG2] = 'L';
  header.e_ident[EI_MAG3] = 'F';
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_VERSION] = EV_CURRENT;
  header.e_ident[EI_ABIVERSION] = abi_version;
  header.e_type = ET_EXEC;
  header.e_machine = 190;  // EM_CUDA
  header.e_version = EV_CURRENT;
  header.e_flags = flags;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_shentsize = sizeof(Elf64_Shdr);
  return header;
}

std::vector<uint8_t> CreateMinimalCubin(uint8_t abi_version, uint32_t flags,
                                        size_t total_size = 128) {
  std::vector<uint8_t> cubin(total_size, 0);
  Elf64_Ehdr header = MakeElfHeader(abi_version, flags);
  header.e_shoff = total_size - sizeof(Elf64_Shdr);
  header.e_shnum = 1;
  header.e_shentsize = sizeof(Elf64_Shdr);
  std::memcpy(cubin.data(), &header, sizeof(header));
  return cubin;
}

TEST(CudaElfUtilsTest, ParseFatBinaryOrElfUncompressed) {
  std::vector<uint8_t> payload(100, 0);
  FatHeader* fhdr = reinterpret_cast<FatHeader*>(payload.data());
  fhdr->magic = kFatbinMagicUncompressed;
  fhdr->header_size = sizeof(FatHeader);
  fhdr->fat_size = payload.size() - sizeof(FatHeader);

  FatbinWrapper wrapper{0, 0, payload.data(), nullptr};
  auto span = ParseFatBinaryOrElf(&wrapper);
  ASSERT_TRUE(span.has_value());
  EXPECT_EQ(span->size(), payload.size());
}

TEST(CudaElfUtilsTest, ParseFatBinaryOrElfCompressed) {
  std::vector<uint8_t> payload(120, 0);
  FatHeader* fhdr = reinterpret_cast<FatHeader*>(payload.data());
  fhdr->magic = kFatbinMagicCompressed;
  fhdr->header_size = sizeof(FatHeader);
  fhdr->fat_size = payload.size() - sizeof(FatHeader);

  FatbinWrapper wrapper{0, 0, payload.data(), nullptr};
  auto span = ParseFatBinaryOrElf(&wrapper);
  ASSERT_TRUE(span.has_value());
  EXPECT_EQ(span->size(), payload.size());
}

TEST(CudaElfUtilsTest, ParseFatBinaryOrElfRawElf) {
  std::vector<uint8_t> elf_data = CreateMinimalCubin(7, 0x50, 256);
  FatbinWrapper wrapper{0, 0, elf_data.data(), nullptr};

  auto span = ParseFatBinaryOrElf(&wrapper);
  ASSERT_TRUE(span.has_value());
  EXPECT_EQ(span->size(), 256);
}

}  // namespace
}  // namespace stream_executor::cuda
