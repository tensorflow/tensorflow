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
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_fatbin.h"

namespace stream_executor::cuda {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

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

absl::Span<const uint8_t> AsBytes(const Elf64_Ehdr& header) {
  return absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(&header),
                                   sizeof(header));
}

TEST(CudaElfUtilsTest, IsCudaElfValid) {
  Elf64_Ehdr header = MakeElfHeader(/*abi_version=*/7, /*flags=*/0x50);
  EXPECT_TRUE(IsCudaElf(AsBytes(header)));
}

TEST(CudaElfUtilsTest, IsCudaElfInvalid) {
  uint8_t short_buf[10] = {0};
  EXPECT_FALSE(IsCudaElf(short_buf));

  Elf64_Ehdr bad_magic = MakeElfHeader(7, 0x50);
  bad_magic.e_ident[EI_MAG1] = 'X';
  EXPECT_FALSE(IsCudaElf(AsBytes(bad_magic)));

  Elf64_Ehdr elf32 = MakeElfHeader(7, 0x50);
  elf32.e_ident[EI_CLASS] = ELFCLASS32;
  EXPECT_FALSE(IsCudaElf(AsBytes(elf32)));

  Elf64_Ehdr elf_msb = MakeElfHeader(7, 0x50);
  elf_msb.e_ident[EI_DATA] = ELFDATA2MSB;
  EXPECT_FALSE(IsCudaElf(AsBytes(elf_msb)));

  Elf64_Ehdr non_cuda = MakeElfHeader(7, 0x50);
  non_cuda.e_machine = 62;  // EM_X86_64
  EXPECT_FALSE(IsCudaElf(AsBytes(non_cuda)));
}

TEST(CudaElfUtilsTest, CudaElfSizeCalculations) {
  Elf64_Ehdr header = MakeElfHeader(7, 0x50);
  header.e_shoff = 200;
  header.e_shnum = 5;
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_phoff = 64;
  header.e_phnum = 2;
  header.e_phentsize = sizeof(Elf64_Phdr);

  std::vector<uint8_t> buffer(600, 0);
  std::memcpy(buffer.data(), &header, sizeof(header));

  EXPECT_EQ(CudaElfSize(buffer), 520);

  absl::Span<const uint8_t> truncated(buffer.data(), 500);
  EXPECT_EQ(CudaElfSize(truncated), std::nullopt);

  absl::Span<const uint8_t> too_small(buffer.data(), sizeof(Elf64_Ehdr) - 1);
  EXPECT_EQ(CudaElfSize(too_small), std::nullopt);
}

TEST(CudaElfUtilsTest, CudaElfSmArchAbiV1StandardSm) {
  Elf64_Ehdr header_sm70 = MakeElfHeader(/*abi_version=*/7, /*flags=*/0x46);
  EXPECT_THAT(CudaElfSmArch(header_sm70),
              IsOkAndHolds(CudaComputeCapability(7, 0)));

  Elf64_Ehdr header_sm80 = MakeElfHeader(/*abi_version=*/7, /*flags=*/0x50);
  EXPECT_THAT(CudaElfSmArch(header_sm80),
              IsOkAndHolds(CudaComputeCapability(8, 0)));

  Elf64_Ehdr header_sm89 = MakeElfHeader(/*abi_version=*/7, /*flags=*/0x59);
  EXPECT_THAT(CudaElfSmArch(header_sm89),
              IsOkAndHolds(CudaComputeCapability(8, 9)));

  Elf64_Ehdr header_sm90 = MakeElfHeader(/*abi_version=*/7, /*flags=*/0x5a);
  EXPECT_THAT(CudaElfSmArch(header_sm90),
              IsOkAndHolds(CudaComputeCapability(9, 0)));
}

TEST(CudaElfUtilsTest, CudaElfSmArchAbiV1Accelerated) {
  Elf64_Ehdr header = MakeElfHeader(/*abi_version=*/7, /*flags=*/0x5a | 0x800);
  EXPECT_THAT(
      CudaElfSmArch(header),
      IsOkAndHolds(CudaComputeCapability(
          9, 0,
          CudaComputeCapability::FeatureExtension::kAcceleratedFeatures)));
}

TEST(CudaElfUtilsTest, CudaElfSmArchAbiV1VirtualSm) {
  Elf64_Ehdr header_virtual =
      MakeElfHeader(/*abi_version=*/7, /*flags=*/0x50 << 16);
  EXPECT_THAT(CudaElfSmArch(header_virtual),
              IsOkAndHolds(CudaComputeCapability(8, 0)));

  Elf64_Ehdr header_virtual_acc =
      MakeElfHeader(/*abi_version=*/7, /*flags=*/(0x5a << 16) | 0x800);
  EXPECT_THAT(
      CudaElfSmArch(header_virtual_acc),
      IsOkAndHolds(CudaComputeCapability(
          9, 0,
          CudaComputeCapability::FeatureExtension::kAcceleratedFeatures)));

  Elf64_Ehdr header_both =
      MakeElfHeader(/*abi_version=*/7, /*flags=*/0x50 | (0x5a << 16));
  EXPECT_THAT(CudaElfSmArch(header_both),
              IsOkAndHolds(CudaComputeCapability(8, 0)));
}

TEST(CudaElfUtilsTest, CudaElfSmArchAbiV2StandardSm) {
  Elf64_Ehdr header_sm100 =
      MakeElfHeader(/*abi_version=*/8, /*flags=*/0x64 << 8);
  EXPECT_THAT(CudaElfSmArch(header_sm100),
              IsOkAndHolds(CudaComputeCapability(10, 0)));

  Elf64_Ehdr header_sm101 =
      MakeElfHeader(/*abi_version=*/8, /*flags=*/0x65 << 8);
  EXPECT_THAT(CudaElfSmArch(header_sm101),
              IsOkAndHolds(CudaComputeCapability(10, 1)));

  Elf64_Ehdr header_sm120 =
      MakeElfHeader(/*abi_version=*/8, /*flags=*/0x78 << 8);
  EXPECT_THAT(CudaElfSmArch(header_sm120),
              IsOkAndHolds(CudaComputeCapability(12, 0)));
}

TEST(CudaElfUtilsTest, CudaElfSmArchAbiV2Accelerated) {
  Elf64_Ehdr header_sm100a =
      MakeElfHeader(/*abi_version=*/8, /*flags=*/(0x64 << 8) | 0x8);
  EXPECT_THAT(
      CudaElfSmArch(header_sm100a),
      IsOkAndHolds(CudaComputeCapability(
          10, 0,
          CudaComputeCapability::FeatureExtension::kAcceleratedFeatures)));

  Elf64_Ehdr header_sm103a =
      MakeElfHeader(/*abi_version=*/8, /*flags=*/(0x67 << 8) | 0x8);
  EXPECT_THAT(
      CudaElfSmArch(header_sm103a),
      IsOkAndHolds(CudaComputeCapability(
          10, 3,
          CudaComputeCapability::FeatureExtension::kAcceleratedFeatures)));
}

TEST(CudaElfUtilsTest, CudaElfSmArchAbiV2VirtualSm) {
  Elf64_Ehdr header_virtual =
      MakeElfHeader(/*abi_version=*/8, /*flags=*/0x64 << 16);
  EXPECT_THAT(CudaElfSmArch(header_virtual),
              IsOkAndHolds(CudaComputeCapability(10, 0)));

  Elf64_Ehdr header_virtual_acc =
      MakeElfHeader(/*abi_version=*/8, /*flags=*/(0x64 << 16) | 0x8);
  EXPECT_THAT(
      CudaElfSmArch(header_virtual_acc),
      IsOkAndHolds(CudaComputeCapability(
          10, 0,
          CudaComputeCapability::FeatureExtension::kAcceleratedFeatures)));

  Elf64_Ehdr header_both = MakeElfHeader(
      /*abi_version=*/8, /*flags=*/(0x64 << 8) | (0x64 << 16));
  EXPECT_THAT(CudaElfSmArch(header_both),
              IsOkAndHolds(CudaComputeCapability(10, 0)));
}

TEST(CudaElfUtilsTest, CudaElfSmArchErrors) {
  // Invalid ABI version
  Elf64_Ehdr header_invalid_abi0 =
      MakeElfHeader(/*abi_version=*/0, /*flags=*/0x50);
  EXPECT_THAT(CudaElfSmArch(header_invalid_abi0),
              StatusIs(absl::StatusCode::kInvalidArgument));

  Elf64_Ehdr header_invalid_abi9 =
      MakeElfHeader(/*abi_version=*/9, /*flags=*/0x50);
  EXPECT_THAT(CudaElfSmArch(header_invalid_abi9),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // No SM architecture set
  Elf64_Ehdr header_no_sm_v1 = MakeElfHeader(/*abi_version=*/7, /*flags=*/0);
  EXPECT_THAT(CudaElfSmArch(header_no_sm_v1),
              StatusIs(absl::StatusCode::kInvalidArgument));

  Elf64_Ehdr header_no_sm_v2 = MakeElfHeader(/*abi_version=*/8, /*flags=*/0);
  EXPECT_THAT(CudaElfSmArch(header_no_sm_v2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CudaElfUtilsTest, CanRunOn) {
  using FeatureExtension = CudaComputeCapability::FeatureExtension;

  CudaComputeCapability sm70{7, 0};
  CudaComputeCapability sm80{8, 0};
  CudaComputeCapability sm86{8, 6};
  CudaComputeCapability sm90{9, 0};
  CudaComputeCapability sm90a{9, 0, FeatureExtension::kAcceleratedFeatures};
  CudaComputeCapability sm100{10, 0};
  CudaComputeCapability sm100a{10, 0, FeatureExtension::kAcceleratedFeatures};
  CudaComputeCapability sm100f{10, 0,
                               FeatureExtension::kFamilyCompatibleFeatures};
  CudaComputeCapability sm103{10, 3};
  CudaComputeCapability sm107{10, 7};
  CudaComputeCapability sm120{12, 0};

  // Same major, standard minor comparison
  EXPECT_TRUE(CanRunOn(sm80, sm80));
  EXPECT_TRUE(CanRunOn(sm80, sm86));
  EXPECT_FALSE(CanRunOn(sm86, sm80));

  // Different major cannot run (binary SASS requires same major)
  EXPECT_FALSE(CanRunOn(sm80, sm90));
  EXPECT_FALSE(CanRunOn(sm70, sm80));

  // Accelerated kernels require exact match on (major, minor)
  EXPECT_TRUE(CanRunOn(sm90a, sm90));
  EXPECT_TRUE(CanRunOn(sm90a, sm90a));
  EXPECT_TRUE(CanRunOn(sm100a, sm100));
  EXPECT_TRUE(CanRunOn(sm100a, sm100a));
  EXPECT_FALSE(CanRunOn(sm100a, sm107));

  // Standard kernels on newer minor within same major
  EXPECT_TRUE(CanRunOn(sm100, sm107));
  EXPECT_TRUE(CanRunOn(sm100, sm100a));

  // Family compatible kernels within same major
  EXPECT_TRUE(CanRunOn(sm100f, sm103));
  EXPECT_FALSE(CanRunOn(sm100f, sm120));
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

TEST(CudaElfUtilsTest, FindCubinForArchSingleAndMultiArch) {
  std::vector<uint8_t> cubin_sm70 = CreateMinimalCubin(7, 0x46, 128);
  std::vector<uint8_t> cubin_sm80 = CreateMinimalCubin(7, 0x50, 128);

  std::vector<uint8_t> fatbin;
  fatbin.insert(fatbin.end(), cubin_sm70.begin(), cubin_sm70.end());
  fatbin.insert(fatbin.end(), cubin_sm80.begin(), cubin_sm80.end());

  ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> found_sm80,
                       FindCubinForArch(fatbin, CudaComputeCapability(8, 0)));
  EXPECT_EQ(found_sm80.size(), 128);
  EXPECT_EQ(found_sm80.data(), fatbin.data() + 128);

  ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> found_sm86,
                       FindCubinForArch(fatbin, CudaComputeCapability(8, 6)));
  EXPECT_EQ(found_sm86.size(), 128);
  EXPECT_EQ(found_sm86.data(), fatbin.data() + 128);

  // Incompatible target: error message contains found architectures
  auto found_sm90 = FindCubinForArch(fatbin, CudaComputeCapability(9, 0));
  EXPECT_THAT(found_sm90, StatusIs(absl::StatusCode::kNotFound,
                                   ::testing::HasSubstr("[sm_70, sm_80]")));

  // Empty fatbin: error message indicates no CUDA ELF images found
  std::vector<uint8_t> empty_buf(64, 0);
  auto found_empty = FindCubinForArch(empty_buf, CudaComputeCapability(9, 0));
  EXPECT_THAT(found_empty,
              StatusIs(absl::StatusCode::kNotFound,
                       ::testing::HasSubstr("no CUDA ELF images were found")));
}

TEST(CudaElfUtilsTest, ParseNvInfoAttributes) {
  std::vector<uint8_t> info;

  info.push_back(0x04);
  info.push_back(0x05);
  info.push_back(12);
  info.push_back(0);
  uint32_t dims[3] = {256, 1, 1};
  const uint8_t* dim_bytes = reinterpret_cast<const uint8_t*>(dims);
  info.insert(info.end(), dim_bytes, dim_bytes + 12);

  info.push_back(0x04);
  info.push_back(0x11);
  info.push_back(4);
  info.push_back(0);
  uint32_t frame_size = 128;
  const uint8_t* fs_bytes = reinterpret_cast<const uint8_t*>(&frame_size);
  info.insert(info.end(), fs_bytes, fs_bytes + 4);

  CudaKernelFuncAttributes attrs;
  ParseNvInfo(info, &attrs);
  EXPECT_EQ(attrs.max_threads_per_block, 256);
  EXPECT_EQ(attrs.local_size_bytes, 128);
}

TEST(CudaElfUtilsTest, ParseNvInfoRegCount) {
  std::vector<uint8_t> info;
  info.push_back(0x04);
  info.push_back(0x2f);
  info.push_back(8);
  info.push_back(0);
  uint32_t sym_index = 42;
  uint32_t reg_count = 64;
  const uint8_t* sym_bytes = reinterpret_cast<const uint8_t*>(&sym_index);
  const uint8_t* reg_bytes = reinterpret_cast<const uint8_t*>(&reg_count);
  info.insert(info.end(), sym_bytes, sym_bytes + 4);
  info.insert(info.end(), reg_bytes, reg_bytes + 4);

  EXPECT_EQ(ParseNvInfoRegCount(info, 42), 64);
  EXPECT_EQ(ParseNvInfoRegCount(info, 99), std::nullopt);
}

TEST(CudaElfUtilsTest, ParseFuncAttributesFromCubin) {
  const std::string kernel_name = "_Z8MyKernelv";
  const std::string text_sec = ".text." + kernel_name;
  const std::string shared_sec = ".nv.shared." + kernel_name;
  const std::string const_sec = ".nv.constant1." + kernel_name;
  const std::string info_sec = ".nv.info." + kernel_name;
  const std::string gen_info_sec = ".nv.info";

  std::string strtab;
  strtab.push_back('\0');
  auto add_str = [&](const std::string& s) -> uint32_t {
    uint32_t off = strtab.size();
    strtab += s;
    strtab.push_back('\0');
    return off;
  };
  uint32_t off_shstrtab = add_str(".shstrtab");
  uint32_t off_text = add_str(text_sec);
  uint32_t off_shared = add_str(shared_sec);
  uint32_t off_const = add_str(const_sec);
  uint32_t off_info = add_str(info_sec);
  uint32_t off_gen_info = add_str(gen_info_sec);

  std::vector<uint8_t> per_kernel_info;
  per_kernel_info.push_back(0x04);
  per_kernel_info.push_back(0x05);
  per_kernel_info.push_back(12);
  per_kernel_info.push_back(0);
  uint32_t dims[3] = {512, 1, 1};
  const uint8_t* dim_bytes = reinterpret_cast<const uint8_t*>(dims);
  per_kernel_info.insert(per_kernel_info.end(), dim_bytes, dim_bytes + 12);

  std::vector<uint8_t> generic_info;
  generic_info.push_back(0x04);
  generic_info.push_back(0x2f);
  generic_info.push_back(8);
  generic_info.push_back(0);
  uint32_t sym_index = 1;
  uint32_t reg_count = 48;
  const uint8_t* s_bytes = reinterpret_cast<const uint8_t*>(&sym_index);
  const uint8_t* r_bytes = reinterpret_cast<const uint8_t*>(&reg_count);
  generic_info.insert(generic_info.end(), s_bytes, s_bytes + 4);
  generic_info.insert(generic_info.end(), r_bytes, r_bytes + 4);

  std::vector<uint8_t> cubin(sizeof(Elf64_Ehdr), 0);

  auto append_data = [&](const void* data, size_t size,
                         size_t align = 8) -> uint64_t {
    while (cubin.size() % align != 0) {
      cubin.push_back(0);
    }
    uint64_t off = cubin.size();
    const uint8_t* p = static_cast<const uint8_t*>(data);
    cubin.insert(cubin.end(), p, p + size);
    return off;
  };

  uint64_t off_strtab_data = append_data(strtab.data(), strtab.size());
  uint64_t off_info_data =
      append_data(per_kernel_info.data(), per_kernel_info.size());
  uint64_t off_gen_info_data =
      append_data(generic_info.data(), generic_info.size());

  std::vector<Elf64_Shdr> sections(7);
  sections[1].sh_name = off_shstrtab;
  sections[1].sh_type = SHT_STRTAB;
  sections[1].sh_offset = off_strtab_data;
  sections[1].sh_size = strtab.size();

  sections[2].sh_name = off_text;
  sections[2].sh_type = SHT_PROGBITS;
  sections[2].sh_info = (32 << 24) | sym_index;
  sections[2].sh_flags = (3 << 20);

  sections[3].sh_name = off_shared;
  sections[3].sh_type = SHT_NOBITS;
  sections[3].sh_size = 2048;

  sections[4].sh_name = off_const;
  sections[4].sh_type = SHT_PROGBITS;
  sections[4].sh_size = 512;

  sections[5].sh_name = off_info;
  sections[5].sh_type = SHT_PROGBITS;
  sections[5].sh_offset = off_info_data;
  sections[5].sh_size = per_kernel_info.size();

  sections[6].sh_name = off_gen_info;
  sections[6].sh_type = SHT_PROGBITS;
  sections[6].sh_offset = off_gen_info_data;
  sections[6].sh_size = generic_info.size();

  uint64_t shoff =
      append_data(sections.data(), sections.size() * sizeof(Elf64_Shdr));

  Elf64_Ehdr header = MakeElfHeader(7, 0x5a);
  header.e_shoff = shoff;
  header.e_shnum = sections.size();
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shstrndx = 1;
  std::memcpy(cubin.data(), &header, sizeof(header));

  CudaComputeCapability cc(9, 0);
  ASSERT_OK_AND_ASSIGN(CudaKernelFuncAttributes attrs,
                       ParseFuncAttributesFromCubin(cubin, kernel_name, cc));

  EXPECT_EQ(attrs.compute_capability, cc);
  EXPECT_EQ(attrs.num_regs, 48);
  EXPECT_EQ(attrs.num_barriers, 3);
  EXPECT_EQ(attrs.static_shared_size_bytes, 2048);
  EXPECT_EQ(attrs.const_size_bytes, 512);
  EXPECT_EQ(attrs.max_threads_per_block, 512);
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
