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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_FATBIN_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_FATBIN_H_

#include <cstdint>

namespace stream_executor::cuda {

// Magic constants and container structures for CUDA fatbinaries and CUBINs.
// References:
//  - Official CUDA Toolkit header: <fatbinary_section.h> (defines
//    __fatBinC_Wrapper_t, FATBINC_MAGIC = 0x466243B1, .nv_fatbin section names)
//  - NVIDIA CUDA Binary Utilities Guide:
//    https://docs.nvidia.com/cuda/cuda-binary-utilities/
//  - GPU Ocelot FatBinaryContext: https://github.com/gtcasl/gpuocelot
//  - How CUDA Binaries Work:
//    https://thesoftwarefrontier.com/how-cuda-binaries-actually-work/
constexpr uint32_t kFatbinMagicUncompressed = 0xba55d10a;
constexpr uint32_t kFatbinMagicCompressed = 0xba55ed50;

// Relocatable CUDA fatbinary magic number (__fatBinC_Wrapper_t /
// FATBINC_MAGIC).
constexpr uint32_t kFatbinMagicRelocatable = 0x466243b1;

// Legacy/alternative CUDA Fatbinary payload header magic number.
constexpr uint32_t kFatbinMagicLegacy = 0x00101001;

// Entry kinds within a CUDA fatbinary container.
constexpr uint16_t kFatbinKindPtx = 0x0001;
constexpr uint16_t kFatbinKindElf = 0x0002;
constexpr uint16_t kFatbinKindElfAlt = 0x0010;

// Flags for FatEntryHeader.
constexpr uint64_t kFatbinFlag64Bit = 0x00000001;
constexpr uint64_t kFatbinFlagDebug = 0x00000002;
constexpr uint64_t kFatbinFlagCompressed = 0x00002000;

// NVIDIA CUDA ELF machine type (EM_CUDA). Not defined by every <elf.h>.
constexpr uint16_t kElfMachineCuda = 190;

// CUDA ELF ABI versions.
constexpr uint8_t kCudaAbiVersionV1 = 7;
constexpr uint8_t kCudaAbiVersionV2 = 8;

// Bitmasks for e_flags in CUDA ELF headers.
// ABI V1 (pre-Blackwell):
constexpr uint32_t kCudaSmMaskV1 = 0xff;
constexpr uint32_t kCudaAcceleratorMaskV1 = 0x800;
constexpr uint32_t kCudaVirtualSmMaskV1 = 0xff0000;
constexpr uint32_t kCudaVirtualSmShiftV1 = 16;

// ABI V2 (Blackwell and later):
constexpr uint32_t kCudaSmMaskV2 = 0xff00;
constexpr uint32_t kCudaSmShiftV2 = 8;
constexpr uint32_t kCudaAcceleratorMaskV2 = 0x8;
constexpr uint32_t kCudaVirtualSmMaskV2 = 0xff0000;
constexpr uint32_t kCudaVirtualSmShiftV2 = 16;

// CUDA fatbin wrapper structure passed to __cudaRegisterFatBinary.
//
// The fatbinary wrapper structure and container headers are defined in:
//  - Official CUDA Toolkit header: <fatbinary_section.h> (__fatBinC_Wrapper_t)
//  - NVIDIA CUDA Binary Utilities:
//    https://docs.nvidia.com/cuda/cuda-binary-utilities/
//  - GPU Ocelot FatBinaryContext: https://github.com/gtcasl/gpuocelot
struct FatbinWrapper {
  uint32_t magic;
  uint32_t version;
  const void* data;
  void* filename_or_fatbins;
};

// CUDA fatbinary header structure at the beginning of the payload.
struct FatHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t fat_size;
};

static_assert(sizeof(FatHeader) == 16, "FatHeader must be 16 bytes");

// Header for each entry inside a CUDA fatbinary container.
//
// In CUDA fatbinary containers (beginning with FatHeader), entries are laid out
// contiguously:
//   [FatHeader (16 bytes)]
//   [FatEntryHeader 1 (header_size bytes, >= 64)]
//   [Payload 1 (size bytes)]
//   [FatEntryHeader 2 (header_size bytes, >= 64)]
//   [Payload 2 (size bytes)]
//   ...
//
// The payload begins at `entry_offset + header_size`, with length `size`
// (which may include trailing alignment padding up to 8 or 16 bytes).
// The next entry begins immediately at `entry_offset + header_size + size`.
struct FatEntryHeader {
  uint16_t kind;         // Entry kind (0x01 = PTX, 0x02 = ELF, 0x10 = ELF_ALT)
  uint16_t unknown;      // Typically 0x0101
  uint32_t header_size;  // Size of this entry header in bytes (>= 64)
  uint64_t size;         // Size of data payload in bytes
  uint32_t compressed_size;    // Compressed payload size (if compressed)
  uint32_t options_offset;     // Offset to options/flags area
  uint16_t minor;              // Minor version
  uint16_t major;              // Major version
  uint32_t arch;               // Target SM architecture (e.g. 70, 80, 90, 100)
  uint32_t obj_name_offset;    // Offset to object/symbol name
  uint32_t obj_name_len;       // Length of object/symbol name
  uint64_t flags;              // Flags (bit 13 = compressed)
  uint64_t zero;               // Reserved / padding
  uint64_t decompressed_size;  // Decompressed size (if compressed)
};

static_assert(sizeof(FatEntryHeader) == 64, "FatEntryHeader must be 64 bytes");

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_FATBIN_H_
