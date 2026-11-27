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

#include "xla/backends/gpu/libraries/cub/cub_sort_kernels.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_kernels_impl.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_elf_utils.h"
#include "xla/stream_executor/cuda/cudart_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::StatusOr<internal::CubSortTypeKernelRefs> GetKeysHostRefs(
    PrimitiveType key_type, internal::SortDirection direction) {
  switch (key_type) {
    case BF16:
      return internal::GetCubSortKeysRefs<BF16>(direction);
    case F16:
      return internal::GetCubSortKeysRefs<F16>(direction);
    case F32:
      return internal::GetCubSortKeysRefs<F32>(direction);
    case F64:
      return internal::GetCubSortKeysRefs<F64>(direction);
    case S8:
      return internal::GetCubSortKeysRefs<S8>(direction);
    case S16:
      return internal::GetCubSortKeysRefs<S16>(direction);
    case S32:
      return internal::GetCubSortKeysRefs<S32>(direction);
    case S64:
      return internal::GetCubSortKeysRefs<S64>(direction);
    case U8:
      return internal::GetCubSortKeysRefs<U8>(direction);
    case U16:
      return internal::GetCubSortKeysRefs<U16>(direction);
    case U32:
      return internal::GetCubSortKeysRefs<U32>(direction);
    case U64:
      return internal::GetCubSortKeysRefs<U64>(direction);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported key type for CUB sort: ",
                       primitive_util::LowercasePrimitiveTypeName(key_type)));
  }
}

template <PrimitiveType kKeyType>
absl::StatusOr<internal::CubSortTypeKernelRefs> GetPairsHostRefsForKey(
    int value_bit_width, internal::SortDirection direction) {
  switch (value_bit_width) {
    case 16:
      return internal::GetCubSortPairsRefs<kKeyType, 16>(direction);
    case 32:
      return internal::GetCubSortPairsRefs<kKeyType, 32>(direction);
    case 64:
      return internal::GetCubSortPairsRefs<kKeyType, 64>(direction);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported value bit width for CUB sort pairs: ", value_bit_width));
  }
}

absl::StatusOr<internal::CubSortTypeKernelRefs> GetPairsHostRefs(
    PrimitiveType key_type, int value_bit_width,
    internal::SortDirection direction) {
  switch (key_type) {
    case U8:
      return GetPairsHostRefsForKey<U8>(value_bit_width, direction);
    case U16:
      return GetPairsHostRefsForKey<U16>(value_bit_width, direction);
    case S32:
      return GetPairsHostRefsForKey<S32>(value_bit_width, direction);
    case U32:
      return GetPairsHostRefsForKey<U32>(value_bit_width, direction);
    case U64:
      return GetPairsHostRefsForKey<U64>(value_bit_width, direction);
    case F32:
      return GetPairsHostRefsForKey<F32>(value_bit_width, direction);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported key type for CUB sort pairs: ",
                       primitive_util::LowercasePrimitiveTypeName(key_type)));
  }
}

absl::StatusOr<internal::CubSortTypeKernelRefs> GetTypeKernelRefs(
    PrimitiveType key_type, std::optional<PrimitiveType> value_type,
    internal::SortDirection direction) {
  if (!value_type.has_value()) {
    return GetKeysHostRefs(key_type, direction);
  }
  if (!primitive_util::IsArrayType(*value_type)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported value type for CUB sort pairs: ",
                     primitive_util::LowercasePrimitiveTypeName(*value_type)));
  }
  return GetPairsHostRefs(key_type, primitive_util::BitWidth(*value_type),
                          direction);
}

absl::StatusOr<stream_executor::KernelLoaderSpec> LoadCubinSpec(
    internal::KernelHostRef ref,
    const stream_executor::CudaComputeCapability& compute_capability) {
  ABSL_ASSIGN_OR_RETURN(stream_executor::cuda::CudaRuntimeKernel fatbin_kernel,
                   stream_executor::cuda::FindCudaRuntimeKernel(ref.host_fun));
  absl::StatusOr<absl::Span<const uint8_t>> cubin =
      stream_executor::cuda::FindCubinForArch(fatbin_kernel.cubin,
                                              compute_capability);
  if (cubin.ok()) {
    return stream_executor::KernelLoaderSpec::CreateCudaCubinInMemorySpec(
        *cubin, std::string(fatbin_kernel.name), ref.arity);
  }
  if (!absl::IsNotFound(cubin.status())) {
    return cubin.status();
  }
  // Fall back to the full fatbinary (which embeds PTX for driver JIT) when
  // no precompiled single-arch CUBIN matches `compute_capability` (for
  // example, targeting GB300 sm_103a or sm_120 with compute_100/compute_120
  // PTX), after verifying the JIT-eligible PTX matches the host policy tier.
  ABSL_RETURN_IF_ERROR(internal::ValidateFatbinPtxPolicyTier(fatbin_kernel.cubin,
                                                        compute_capability));
  return stream_executor::KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      fatbin_kernel.cubin, std::string(fatbin_kernel.name), ref.arity);
}

}  // namespace

namespace internal {

// Verifies that `fatbin` contains a JIT-eligible PTX entry for
// `compute_capability` whose highest matching PTX arch maps to the same
// canonical SM policy tier as `CanonicalCubSortPolicySmVersion(cc.major)`.
// Because `XlaChainedPolicy` binds `ActivePolicy` at PTX compile time via
// `CUB_PTX_ARCH`, JITting PTX from a different major policy tier would mismatch
// the host planner's block geometry and cause launch failures or corruption.
absl::Status ValidateFatbinPtxPolicyTier(
    absl::Span<const uint8_t> fatbin,
    const stream_executor::CudaComputeCapability& compute_capability) {
  if (compute_capability.major < 5) {
    return absl::FailedPreconditionError(
        absl::StrCat("CUDA CUB sort requires SM >= 5.0, got ",
                     compute_capability.ToString()));
  }
  absl::StatusOr<std::vector<stream_executor::CudaComputeCapability>>
      ptx_archs = stream_executor::cuda::ParseFatbinPtxArchs(fatbin);
  if (!ptx_archs.ok()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Kernel binary for ", compute_capability.ToString(),
                     " is not a CUDA fatbinary container with PTX fallback: ",
                     ptx_archs.status().message()));
  }

  std::optional<stream_executor::CudaComputeCapability> best_ptx_cc;
  std::vector<std::string> found_ptx_archs;
  found_ptx_archs.reserve(ptx_archs->size());
  for (const stream_executor::CudaComputeCapability& ptx_cc : *ptx_archs) {
    found_ptx_archs.push_back(ptx_cc.GetPtxAsTargetName(
        stream_executor::CudaComputeCapability::CompileMode::kPtx));
    if (compute_capability.SupportsAllFeaturesOf(ptx_cc) &&
        (!best_ptx_cc.has_value() ||
         std::tie(ptx_cc.major, ptx_cc.minor) >
             std::tie(best_ptx_cc->major, best_ptx_cc->minor))) {
      best_ptx_cc = ptx_cc;
    }
  }

  if (!best_ptx_cc.has_value()) {
    return absl::FailedPreconditionError(
        absl::StrCat("No matching CUBIN or JIT-compatible PTX for ",
                     compute_capability.ToString(),
                     " in embedded CUB sort fatbinary "
                     "(embedded PTX archs: [",
                     absl::StrJoin(found_ptx_archs, ", "), "])"));
  }

  const int ptx_tier = CanonicalCubSortPolicySmVersion(best_ptx_cc->major);
  const int target_tier =
      CanonicalCubSortPolicySmVersion(compute_capability.major);
  if (ptx_tier != target_tier) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Embedded CUB sort fatbinary has no CUBIN for ",
        compute_capability.ToString(), " and its highest JIT-compatible PTX (",
        best_ptx_cc->GetPtxAsTargetName(
            stream_executor::CudaComputeCapability::CompileMode::kPtx),
        ", policy tier SM", ptx_tier, ") does not match host policy tier SM",
        target_tier));
  }
  return absl::OkStatus();
}

}  // namespace internal

absl::StatusOr<stream_executor::KernelLoaderSpec>
GetCubSortSingleTileKernelSpec(
    PrimitiveType key_type, std::optional<PrimitiveType> value_type,
    bool descending,
    const stream_executor::CudaComputeCapability& compute_capability) {
  const internal::SortDirection direction =
      descending ? internal::SortDirection::kDescending
                 : internal::SortDirection::kAscending;
  ABSL_ASSIGN_OR_RETURN(internal::CubSortTypeKernelRefs refs,
                   GetTypeKernelRefs(key_type, value_type, direction));
  return LoadCubinSpec(refs.single_tile, compute_capability);
}

}  // namespace xla::gpu
