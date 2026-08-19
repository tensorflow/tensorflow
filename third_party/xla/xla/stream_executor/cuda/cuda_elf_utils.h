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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_ELF_UTILS_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_ELF_UTILS_H_

#include <elf.h>

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace stream_executor::cuda {

// Statically-extracted CUDA kernel function attributes.
//
// These values are read directly from the compiled CUBIN (ELF) for a specific
// compute capability.
struct CudaKernelFuncAttributes {
  CudaComputeCapability compute_capability;

  // Number of registers used by each thread.
  int num_regs = 0;

  // Statically-allocated shared memory in bytes.
  size_t static_shared_size_bytes = 0;

  // User `__constant__` memory associated with the kernel in bytes.
  size_t const_size_bytes = 0;

  // Stack frame / local memory per thread in bytes.
  size_t local_size_bytes = 0;

  // Number of named barriers used by the kernel.
  int num_barriers = 0;

  // The maximum threads per block requested via `__launch_bounds__`, if any.
  std::optional<int> max_threads_per_block;
};

// Returns true if `data` begins with a little-endian 64-bit NVIDIA CUDA ELF
// header (a CUBIN).
bool IsCudaElf(absl::Span<const uint8_t> data);

// Returns the total size in bytes of the CUBIN ELF that begins at `data`
// (assuming the section header table is the last structure in the image, which
// holds for nvcc-produced cubins), or nullopt if the header is inconsistent.
std::optional<size_t> CudaElfSize(absl::Span<const uint8_t> data);

// Returns the CudaComputeCapability encoded in the given CUDA ELF (CUBIN)
// header by parsing the ABI version from e_ident and flags from e_flags
// (including accelerator and virtual SM architecture flags).
// Returns an error status if the ELF header uses an unsupported ABI version
// or does not encode a valid SM architecture.
absl::StatusOr<CudaComputeCapability> CudaElfSmArch(const Elf64_Ehdr& header);

// Returns true if a binary CUBIN compiled for `kernel_cc` can run on a GPU with
// compute capability `gpu_cc`.
bool CanRunOn(const CudaComputeCapability& kernel_cc,
              const CudaComputeCapability& gpu_cc);

// Walks the `fatbin` and returns the CUBIN ELF for the
// architecture matching `cc`, as a sub-span of `fatbin`. Returns a
// NotFoundError status if no matching CUBIN is present, listing any
// architectures found.
absl::StatusOr<absl::Span<const uint8_t>> FindCubinForArch(
    absl::Span<const uint8_t> fatbin, const CudaComputeCapability& cc);

// Parses NVIDIA's `.nv.info.<kernel>` TLV blob, filling in the fields of
// `attrs` that it encodes.
void ParseNvInfo(absl::Span<const uint8_t> info,
                 CudaKernelFuncAttributes* attrs);

// Scans the generic (non per-kernel) `.nv.info` section for the register count
// of the kernel whose symbol-table index is `symbol_index`, or nullopt if
// absent.
std::optional<int> ParseNvInfoRegCount(absl::Span<const uint8_t> info,
                                       uint32_t symbol_index);

// Extracts function attributes for `mangled_name` from the CUBIN ELF `cubin`.
absl::StatusOr<CudaKernelFuncAttributes> ParseFuncAttributesFromCubin(
    absl::Span<const uint8_t> cubin, absl::string_view mangled_name,
    const CudaComputeCapability& cc);

// Helper function to parse CUDA Fatbinary containers or raw ELF CUBINs from
// a FatbinWrapper pointer and return an absl::Span pointing to the binary
// payload.
std::optional<absl::Span<const uint8_t>> ParseFatBinaryOrElf(
    const void* fat_cubin);

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_ELF_UTILS_H_
