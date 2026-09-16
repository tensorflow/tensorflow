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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_fatbin.h"

namespace stream_executor::cuda {

namespace {

// Reads a little-endian uint32_t from `data` at `offset` (bounds must be
// checked by the caller).
uint32_t ReadU32LE(absl::Span<const uint8_t> data, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, data.data() + offset, sizeof(value));
  return value;
}

// Returns the null-terminated name at byte `offset` within the section-name
// string table `strtab`.
absl::string_view NameAt(absl::Span<const uint8_t> strtab, uint32_t offset) {
  if (offset >= strtab.size()) {
    return {};
  }
  const char* start = reinterpret_cast<const char*>(strtab.data()) + offset;
  size_t max_length = strtab.size() - offset;
  size_t length = 0;
  while (length < max_length && start[length] != '\0') {
    ++length;
  }
  return absl::string_view(start, length);
}

// Parses a CUDA fatbinary container header (FatHeader).
// Returns nullopt if `data` is smaller than sizeof(FatHeader), if the magic
// does not match a known container format, or if header_size is invalid.
std::optional<FatHeader> ParseFatHeader(absl::Span<const uint8_t> data) {
  if (data.size() < sizeof(FatHeader)) {
    return std::nullopt;
  }
  FatHeader header;
  std::memcpy(&header, data.data(), sizeof(FatHeader));
  if (header.magic != kFatbinMagicUncompressed &&
      header.magic != kFatbinMagicCompressed &&
      header.magic != kFatbinMagicRelocatable) {
    return std::nullopt;
  }
  if (header.header_size < sizeof(FatHeader)) {
    return std::nullopt;
  }
  return header;
}

// Represents a parsed entry within a CUDA fatbinary container.
struct ParsedFatEntry {
  FatEntryHeader header;
  absl::Span<const uint8_t> payload;
  // Total byte size of the entry (header_size + size) used to advance to the
  // next entry.
  size_t total_size = 0;
};

// Parses a single entry from `data` (which points to the start of a
// FatEntryHeader).
// Returns nullopt if `data` is truncated, if header_size is invalid, or if the
// entry size overflows or exceeds `data.size()`.
std::optional<ParsedFatEntry> ParseFatEntry(absl::Span<const uint8_t> data) {
  if (data.size() < sizeof(FatEntryHeader)) {
    return std::nullopt;
  }
  FatEntryHeader header;
  std::memcpy(&header, data.data(), sizeof(FatEntryHeader));
  if (header.header_size < sizeof(FatEntryHeader) ||
      header.header_size > data.size()) {
    return std::nullopt;
  }
  if (header.size > data.size() - header.header_size) {
    return std::nullopt;
  }
  const size_t header_size = static_cast<size_t>(header.header_size);
  const size_t payload_size = static_cast<size_t>(header.size);
  const size_t total_size = header_size + payload_size;
  return ParsedFatEntry{header, data.subspan(header_size, payload_size),
                        total_size};
}

}  // namespace

bool IsCudaElf(absl::Span<const uint8_t> data) {
  if (data.size() < sizeof(Elf64_Ehdr)) {
    return false;
  }
  if (!(data[0] == 0x7f && data[1] == 'E' && data[2] == 'L' &&
        data[3] == 'F')) {
    return false;
  }
  Elf64_Ehdr header;
  std::memcpy(&header, data.data(), sizeof(header));
  return header.e_ident[EI_CLASS] == ELFCLASS64 &&
         header.e_ident[EI_DATA] == ELFDATA2LSB &&
         header.e_machine == kElfMachineCuda;
}

bool IsFatbin(absl::Span<const uint8_t> data) {
  return ParseFatHeader(data).has_value();
}

std::optional<size_t> CudaElfSize(absl::Span<const uint8_t> data) {
  if (data.size() < sizeof(Elf64_Ehdr)) {
    return std::nullopt;
  }
  Elf64_Ehdr header;
  std::memcpy(&header, data.data(), sizeof(header));
  size_t sections_end = 0;
  if (header.e_shnum > 0) {
    const size_t sections_bytes =
        static_cast<size_t>(header.e_shnum) * header.e_shentsize;
    if (header.e_shoff > data.size() ||
        sections_bytes > data.size() - header.e_shoff) {
      return std::nullopt;
    }
    sections_end = static_cast<size_t>(header.e_shoff) + sections_bytes;
  }
  size_t segments_end = 0;
  if (header.e_phnum > 0) {
    const size_t segments_bytes =
        static_cast<size_t>(header.e_phnum) * header.e_phentsize;
    if (header.e_phoff > data.size() ||
        segments_bytes > data.size() - header.e_phoff) {
      return std::nullopt;
    }
    segments_end = static_cast<size_t>(header.e_phoff) + segments_bytes;
  }
  size_t size = std::max(sections_end, segments_end);
  if (size == 0) {
    return std::nullopt;
  }
  return size;
}

absl::StatusOr<CudaComputeCapability> CudaElfSmArch(const Elf64_Ehdr& header) {
  const uint8_t abi_version = header.e_ident[EI_ABIVERSION];
  uint32_t sm_number = 0;
  bool is_accelerated = false;

  if (abi_version == kCudaAbiVersionV1) {
    sm_number = header.e_flags & kCudaSmMaskV1;
    is_accelerated = (header.e_flags & kCudaAcceleratorMaskV1) != 0;
    if (sm_number == 0) {
      sm_number =
          (header.e_flags & kCudaVirtualSmMaskV1) >> kCudaVirtualSmShiftV1;
    }
  } else if (abi_version == kCudaAbiVersionV2) {
    sm_number = (header.e_flags & kCudaSmMaskV2) >> kCudaSmShiftV2;
    is_accelerated = (header.e_flags & kCudaAcceleratorMaskV2) != 0;
    if (sm_number == 0) {
      sm_number =
          (header.e_flags & kCudaVirtualSmMaskV2) >> kCudaVirtualSmShiftV2;
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unsupported CUDA ELF ABI version: %u", abi_version));
  }

  if (sm_number == 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid or missing SM architecture in CUDA ELF e_flags: 0x%x "
        "(ABI version %u)",
        header.e_flags, abi_version));
  }

  const int major = static_cast<int>(sm_number / 10);
  const int minor = static_cast<int>(sm_number % 10);
  const auto feature_extension =
      is_accelerated
          ? CudaComputeCapability::FeatureExtension::kAcceleratedFeatures
          : CudaComputeCapability::FeatureExtension::kNone;

  return CudaComputeCapability{major, minor, feature_extension};
}

bool CanRunOn(const CudaComputeCapability& kernel_cc,
              const CudaComputeCapability& gpu_cc) {
  if (kernel_cc.major != gpu_cc.major) {
    return false;
  }
  switch (kernel_cc.feature_extension) {
    case CudaComputeCapability::FeatureExtension::kNone:
    case CudaComputeCapability::FeatureExtension::kFamilyCompatibleFeatures:
      return kernel_cc.minor <= gpu_cc.minor;
    case CudaComputeCapability::FeatureExtension::kAcceleratedFeatures:
      return kernel_cc.minor == gpu_cc.minor;
  }
}

namespace {

struct ParsedCudaElf {
  absl::Span<const uint8_t> bytes;
  std::optional<CudaComputeCapability> compute_capability;
  size_t size = 0;
};

std::optional<ParsedCudaElf> ParseCudaElf(absl::Span<const uint8_t> candidate) {
  std::optional<size_t> size = CudaElfSize(candidate);
  if (!size.has_value() || candidate.size() < *size ||
      candidate.size() < sizeof(Elf64_Ehdr)) {
    return std::nullopt;
  }
  ParsedCudaElf parsed;
  parsed.size = *size;
  parsed.bytes = candidate.subspan(0, *size);
  Elf64_Ehdr header;
  std::memcpy(&header, parsed.bytes.data(), sizeof(header));
  absl::StatusOr<CudaComputeCapability> elf_cc = CudaElfSmArch(header);
  if (elf_cc.ok()) {
    parsed.compute_capability = *elf_cc;
  }
  return parsed;
}

// Result of parsing the entries within a CUDA fatbinary container.
struct FatbinContainerEntries {
  std::vector<ParsedCudaElf> elfs;
  std::vector<std::string> compressed_archs;
};

// Iterates through all entries in a fatbinary container entries region,
// collecting valid CUDA ELF images and names of compressed architectures.
FatbinContainerEntries ParseContainerEntries(
    absl::Span<const uint8_t> entries) {
  FatbinContainerEntries result;
  while (!entries.empty()) {
    std::optional<ParsedFatEntry> entry = ParseFatEntry(entries);
    if (!entry.has_value() || entry->total_size == 0) {
      break;
    }

    if (entry->header.kind == kFatbinKindElf ||
        entry->header.kind == kFatbinKindElfAlt) {
      if ((entry->header.flags & kFatbinFlagCompressed) != 0 ||
          entry->header.compressed_size != 0) {
        std::string comp_arch = absl::StrFormat("sm_%d", entry->header.arch);
        if (absl::c_find(result.compressed_archs, comp_arch) ==
            result.compressed_archs.end()) {
          result.compressed_archs.push_back(comp_arch);
        }
      } else if (IsCudaElf(entry->payload)) {
        std::optional<ParsedCudaElf> elf = ParseCudaElf(entry->payload);
        if (elf.has_value()) {
          result.elfs.push_back(*elf);
        }
      }
    }

    entries = entries.subspan(entry->total_size);
  }
  return result;
}

}  // namespace
absl::StatusOr<absl::Span<const uint8_t>> FindCubinForArch(
    absl::Span<const uint8_t> fatbin, const CudaComputeCapability& cc) {
  std::optional<absl::Span<const uint8_t>> best_cubin;
  std::optional<CudaComputeCapability> best_cc;
  std::vector<std::string> found_archs;
  std::vector<std::string> compressed_archs;

  auto update_best_matching_cubin = [&](const ParsedCudaElf& elf) {
    if (!elf.compute_capability.has_value()) {
      return false;
    }
    const CudaComputeCapability& elf_cc = *elf.compute_capability;
    std::string arch_name = elf_cc.GetPtxAsTargetName();
    if (absl::c_find(found_archs, arch_name) == found_archs.end()) {
      found_archs.push_back(arch_name);
    }
    if (CanRunOn(elf_cc, cc)) {
      if (elf_cc == cc) {
        best_cc = elf_cc;
        best_cubin = elf.bytes;
        return true;
      }
      if (!best_cc.has_value() || elf_cc.minor > best_cc->minor) {
        best_cc = elf_cc;
        best_cubin = elf.bytes;
      }
    }
    return false;
  };

  if (IsFatbin(fatbin)) {
    // 1. CUDA fatbinary container. All target architectures for this
    // compilation unit are stored as entries within this single container.
    auto fat_header = ParseFatHeader(fatbin);
    if (fat_header.has_value() && fat_header->header_size <= fatbin.size()) {
      const size_t header_size = static_cast<size_t>(fat_header->header_size);
      const size_t payload_size = static_cast<size_t>(fat_header->fat_size);
      const size_t entries_size =
          std::min(payload_size, fatbin.size() - header_size);
      FatbinContainerEntries container =
          ParseContainerEntries(fatbin.subspan(header_size, entries_size));
      compressed_archs = std::move(container.compressed_archs);
      for (const ParsedCudaElf& elf : container.elfs) {
        if (update_best_matching_cubin(elf)) {
          return *best_cubin;
        }
      }
    }
  } else if (IsCudaElf(fatbin)) {
    // 2. Backwards-compatibility fallback: raw CUBIN ELF image(s).
    // Instead of loose 0x7f scanning, advance directly by each ELF's size.
    absl::Span<const uint8_t> remaining = fatbin;
    while (IsCudaElf(remaining)) {
      std::optional<ParsedCudaElf> elf = ParseCudaElf(remaining);
      if (!elf.has_value() || elf->size == 0 || elf->size > remaining.size()) {
        break;
      }
      if (update_best_matching_cubin(*elf)) {
        return *best_cubin;
      }
      remaining = remaining.subspan(elf->size);
    }
  }

  if (best_cubin.has_value()) {
    return *best_cubin;
  }

  if (found_archs.empty() && !compressed_archs.empty()) {
    return absl::NotFoundError(absl::StrFormat(
        "No uncompressed CUBIN for %s found in fatbinary. Found compressed "
        "architectures: [%s]. Static extraction requires uncompressed "
        "fatbinaries (compile with --compress=false).",
        cc.GetPtxAsTargetName(), absl::StrJoin(compressed_archs, ", ")));
  }

  if (found_archs.empty()) {
    return absl::NotFoundError(absl::StrFormat(
        "No CUBIN for %s found in fatbinary (no CUDA ELF images were found). "
        "The fatbinary may be PTX-only or compressed.",
        cc.GetPtxAsTargetName()));
  }

  return absl::NotFoundError(absl::StrFormat(
      "No CUBIN for %s found in fatbinary. Found architectures: [%s].",
      cc.GetPtxAsTargetName(), absl::StrJoin(found_archs, ", ")));
}

namespace {

// Format codes in NVIDIA's `.nv.info` TLV blob:
//   - kEifmtNval (0x01): no value
//   - kEifmtBval (0x02): 1 byte
//   - kEifmtHval (0x03): 2 bytes
//   - kEifmtSval (0x04): length-prefixed stream
constexpr uint8_t kEifmtNval = 0x01;
constexpr uint8_t kEifmtBval = 0x02;
constexpr uint8_t kEifmtHval = 0x03;
constexpr uint8_t kEifmtSval = 0x04;

// Attribute codes in NVIDIA's `.nv.info` TLV blob.
constexpr uint8_t kAttrMaxThreads =
    0x05;                                 // Launch bounds (3 x uint32: x, y, z)
constexpr uint8_t kAttrFrameSize = 0x11;  // Per-thread frame size (uint32)
constexpr uint8_t kAttrRegCount = 0x2f;   // Reg count (sym_idx: u32, regs: u32)

// Controls whether iteration in ForEachNvInfoEntry should continue or stop.
enum class IterationDecision {
  kContinue,
  kStop,
};

// Iterates over records in an NVIDIA `.nv.info` TLV blob.
//
// The .nv.info section encodes kernel attributes using a Tag-Length-Value (TLV)
// format:
//   - format (1 byte): EIFMT_NVAL (0x01, no value), EIFMT_BVAL (0x02, 1 byte),
//     EIFMT_HVAL (0x03, 2 bytes), EIFMT_SVAL (0x04, length-prefixed stream).
//   - attribute (1 byte): EIATTR_* code (e.g. EIATTR_MAX_THREADS = 0x05,
//     EIATTR_FRAME_SIZE = 0x11, EIATTR_REGCOUNT = 0x2f).
//   - length (2 bytes, little-endian, only for EIFMT_SVAL).
//   - value (variable length).
// Each record is padded to a 4-byte boundary.
//
// References:
//  - NVIDIA cuda-gdb source: https://github.com/NVIDIA/cuda-gdb (see
//    cuda-textures.c)
//  - How CUDA Binaries Work:
//      https://thesoftwarefrontier.com/how-cuda-binaries-actually-work/
//
// Calls `callback(attribute, value)` for each record.
// If `callback` returns IterationDecision::kStop, iteration stops early.
template <typename Callback>
void ForEachNvInfoEntry(absl::Span<const uint8_t> info, Callback&& callback) {
  size_t pos = 0;
  while (pos + 2 <= info.size()) {
    const uint8_t format = info[pos];
    const uint8_t attribute = info[pos + 1];
    pos += 2;

    absl::Span<const uint8_t> value;
    if (format == kEifmtNval) {
      // No value.
    } else if (format == kEifmtBval) {
      if (info.size() - pos < 1) {
        break;
      }
      value = info.subspan(pos, 1);
      pos += 1;
    } else if (format == kEifmtHval) {
      if (info.size() - pos < 2) {
        break;
      }
      value = info.subspan(pos, 2);
      pos += 2;
    } else if (format == kEifmtSval) {
      if (info.size() - pos < 2) {
        break;
      }
      const uint16_t length = static_cast<uint16_t>(info[pos]) |
                              (static_cast<uint16_t>(info[pos + 1]) << 8);
      pos += 2;
      if (length > info.size() - pos) {
        break;
      }
      value = info.subspan(pos, length);
      pos += length;
    } else {
      // Unknown record format: we can no longer reliably advance.
      break;
    }

    if (callback(attribute, value) == IterationDecision::kStop) {
      break;
    }

    // Each record is padded so the next one starts on a 4-byte boundary.
    pos = (pos + 3) & ~static_cast<size_t>(3);
  }
}

}  // namespace

void ParseNvInfo(absl::Span<const uint8_t> info,
                 CudaKernelFuncAttributes* attrs) {
  ForEachNvInfoEntry(
      info, [&](uint8_t attribute, absl::Span<const uint8_t> value) {
        if (attribute == kAttrMaxThreads && value.size() >= 12) {
          const uint32_t x = ReadU32LE(value, 0);
          const uint32_t y = ReadU32LE(value, 4);
          const uint32_t z = ReadU32LE(value, 8);
          attrs->max_threads_per_block = static_cast<int>(x * y * z);
        } else if (attribute == kAttrFrameSize && value.size() >= 4) {
          attrs->local_size_bytes = ReadU32LE(value, 0);
        }
        return IterationDecision::kContinue;
      });
}

std::optional<int> ParseNvInfoRegCount(absl::Span<const uint8_t> info,
                                       uint32_t symbol_index) {
  std::optional<int> reg_count;
  ForEachNvInfoEntry(info,
                     [&](uint8_t attribute, absl::Span<const uint8_t> value) {
                       if (attribute == kAttrRegCount && value.size() >= 8 &&
                           ReadU32LE(value, 0) == symbol_index) {
                         reg_count = static_cast<int>(ReadU32LE(value, 4));
                         return IterationDecision::kStop;
                       }
                       return IterationDecision::kContinue;
                     });
  return reg_count;
}

absl::StatusOr<CudaKernelFuncAttributes> ParseFuncAttributesFromCubin(
    absl::Span<const uint8_t> cubin, absl::string_view mangled_name,
    const CudaComputeCapability& cc) {
  CudaKernelFuncAttributes attrs;
  attrs.compute_capability = cc;

  if (cubin.size() < sizeof(Elf64_Ehdr)) {
    return absl::InvalidArgumentError("CUBIN is too small to be an ELF file");
  }
  Elf64_Ehdr header;
  std::memcpy(&header, cubin.data(), sizeof(header));

  const size_t section_table_bytes =
      static_cast<size_t>(header.e_shnum) * sizeof(Elf64_Shdr);
  if (header.e_shentsize != sizeof(Elf64_Shdr) ||
      header.e_shstrndx >= header.e_shnum || header.e_shoff > cubin.size() ||
      section_table_bytes > cubin.size() - header.e_shoff ||
      header.e_shoff % alignof(Elf64_Shdr) != 0) {
    return absl::InvalidArgumentError("CUBIN has an invalid section table");
  }

  auto read_section = [&](size_t index) {
    Elf64_Shdr shdr;
    std::memcpy(&shdr,
                cubin.data() + header.e_shoff + index * sizeof(Elf64_Shdr),
                sizeof(Elf64_Shdr));
    return shdr;
  };

  // Section-name string table.
  const Elf64_Shdr strtab_section = read_section(header.e_shstrndx);
  if (strtab_section.sh_offset > cubin.size() ||
      strtab_section.sh_size > cubin.size() - strtab_section.sh_offset) {
    return absl::InvalidArgumentError("CUBIN string table is out of bounds");
  }
  absl::Span<const uint8_t> strtab =
      cubin.subspan(strtab_section.sh_offset, strtab_section.sh_size);

  const std::string text_name = absl::StrCat(".text.", mangled_name);
  const std::string shared_name = absl::StrCat(".nv.shared.", mangled_name);
  const std::string info_name = absl::StrCat(".nv.info.", mangled_name);
  const std::string kernel_suffix = absl::StrCat(".", mangled_name);

  // Populated from the kernel's `.text` section; used to look up the register
  // count in the generic `.nv.info` section after the loop.
  bool found_text = false;
  uint32_t text_sh_info = 0;
  absl::Span<const uint8_t> generic_nv_info;

  for (int i = 0; i < header.e_shnum; ++i) {
    const Elf64_Shdr section = read_section(i);
    absl::string_view name = NameAt(strtab, section.sh_name);
    if (name == text_name) {
      found_text = true;
      text_sh_info = section.sh_info;
      // Fallback register count: on architectures <= sm_80 it is packed into
      // the high byte of sh_info. The generic `.nv.info` lookup below overrides
      // this when present (and is the only source on sm_90+).
      attrs.num_regs = static_cast<int>((section.sh_info >> 24) & 0xff);
      // The number of named barriers is packed into bits 20-24 of sh_flags
      // (best-effort).
      attrs.num_barriers = static_cast<int>((section.sh_flags >> 20) & 0x1f);
    } else if (name == ".nv.info") {
      if (section.sh_offset <= cubin.size() &&
          section.sh_size <= cubin.size() - section.sh_offset) {
        generic_nv_info = cubin.subspan(section.sh_offset, section.sh_size);
      }
    } else if (name == shared_name) {
      attrs.static_shared_size_bytes = section.sh_size;
    } else if (name == info_name) {
      if (section.sh_offset <= cubin.size() &&
          section.sh_size <= cubin.size() - section.sh_offset) {
        ParseNvInfo(cubin.subspan(section.sh_offset, section.sh_size), &attrs);
      }
    } else if (absl::StartsWith(name, ".nv.constant") &&
               !absl::StartsWith(name, ".nv.constant0.") &&
               absl::EndsWith(name, kernel_suffix)) {
      // Best-effort: sum the sizes of the kernel's constant banks, excluding
      // bank 0 which holds kernel parameters / ABI data rather than user
      // `__constant__` memory.
      attrs.const_size_bytes += section.sh_size;
    }
  }

  // Prefer the register count from the generic `.nv.info` section, keyed by the
  // kernel's symbol-table index (the low 24 bits of the `.text` section's
  // sh_info). This is the only reliable source on sm_90+.
  if (found_text && !generic_nv_info.empty()) {
    const uint32_t symbol_index = text_sh_info & 0x00ffffff;
    std::optional<int> reg_count =
        ParseNvInfoRegCount(generic_nv_info, symbol_index);
    if (reg_count.has_value()) {
      attrs.num_regs = *reg_count;
    }
  }

  return attrs;
}

// Parses CUDA Fatbinary containers or raw ELF CUBINs from a FatbinWrapper
// pointer and returns a span pointing to the binary payload.
//
// The fatbinary wrapper structure and container headers are defined in:
//  - Official CUDA Toolkit header: <fatbinary_section.h> (__fatBinC_Wrapper_t)
//  - NVIDIA CUDA Binary Utilities:
//      https://docs.nvidia.com/cuda/cuda-binary-utilities/
//  - GPU Ocelot FatBinaryContext: https://github.com/gtcasl/gpuocelot
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
  if (data_bytes[0] == 0x7f && data_bytes[1] == 'E' && data_bytes[2] == 'L' &&
      data_bytes[3] == 'F') {
    Elf64_Ehdr elf_header;
    std::memcpy(&elf_header, data_bytes, sizeof(Elf64_Ehdr));
    total_size = static_cast<size_t>(
        elf_header.e_shoff +
        (static_cast<uint64_t>(elf_header.e_shnum) * elf_header.e_shentsize));
  } else if (auto header = ParseFatHeader(
                 absl::Span<const uint8_t>(data_bytes, sizeof(FatHeader)))) {
    total_size = static_cast<size_t>(header->header_size) +
                 static_cast<size_t>(header->fat_size);
  } else {
    uint32_t magic = 0;
    std::memcpy(&magic, data_bytes, sizeof(magic));
    if (magic == kFatbinMagicLegacy) {
      uint64_t legacy_size = 0;
      std::memcpy(&legacy_size, data_bytes + 8, sizeof(legacy_size));
      total_size = static_cast<size_t>(legacy_size);
    } else {
      return std::nullopt;
    }
  }

  return absl::Span<const uint8_t>(data_bytes, total_size);
}

}  // namespace stream_executor::cuda
