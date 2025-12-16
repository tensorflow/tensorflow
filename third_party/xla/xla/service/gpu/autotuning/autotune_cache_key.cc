/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/autotune_cache_key.h"

#include <cmath>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

std::string AutotuneCacheKey::HloInstructionToCanonicalString(
    const HloInstruction& instr) {
  auto options = HloPrintOptions::Canonical();
  if (instr.opcode() != HloOpcode::kFusion) {
    options.set_print_backend_config(true);
    options.set_sort_backend_config(true);
    return instr.ToString(options);
  }
  options.set_print_subcomputation_mode(
      HloPrintOptions::PrintSubcomputationMode::kOff);
  options.set_print_infeed_outfeed_config(false);
  options.set_print_only_essential_constants(true);
  options.set_print_operand_shape(true);
  options.set_print_ids(false);
  options.set_canonicalize_computations(true);

  // TODO(b/266210099): This is unsound. We should probably do the fingerprint
  // of the HLO computation proto instead.
  return instr.called_computations()[0]->ToString(options);
}

std::string AutotuneCacheKey::DeviceDescriptionToCacheKey(
    const se::DeviceDescription& device_description) {
  std::string compute_capability;
  if (auto* ccc = device_description.gpu_compute_capability()
                      .cuda_compute_capability()) {
    compute_capability = absl::StrCat("CUDA: ", ccc->major, ".", ccc->minor);
  } else {
    auto* rcc =
        device_description.gpu_compute_capability().rocm_compute_capability();
    CHECK(rcc != nullptr) << "Unknown compute capability type";
    compute_capability = absl::StrCat("ROCM: ", rcc->gfx_version());
  }

  // The string below should include only as much information as is needed to
  // make it a valid key. Information that should not be included is:
  // - specs that are directly derivable from the compute capability, e.g.
  //   shared memory size. For NVIDIA GPUs, you can see what is derivable from
  //   the SM version here:
  //   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
  // - specs that are irrelevant for autotuning. E.g. the total available memory
  //   on a device is not relevant, because by itself, it does not affect the
  //   performance of single kernels.
  //
  // See b/344573710 for some discussion.

  double memory_bandwidth = device_description.memory_bandwidth() / 1e9;
  // Round the memory bandwidth to make the final string nicer to read.
  // This will also cause minute differences in bandwidth to yield the same
  // cache key, but that's fine, since the difference is inconsequential.
  memory_bandwidth = std::round(memory_bandwidth);

  constexpr double kBytesPerMegabyte = 1 << 20;
  double l2_cache_size = device_description.l2_cache_size() / kBytesPerMegabyte;

  return absl::StrCat(
      compute_capability, ", Cores: ", device_description.core_count(),
      ", GPU clock: ", device_description.clock_rate_ghz(),
      " GHz, Memory bandwidth: ", memory_bandwidth,
      " GB/s, L2 cache: ", l2_cache_size,
      " MB, DNN version: ", device_description.dnn_version().ToString());
}

AutotuneCacheKey::AutotuneCacheKey(
    const se::DeviceDescription& device_description,
    const HloInstruction& instruction, int version)
    : AutotuneCacheKey(DeviceDescriptionToCacheKey(device_description),
                       HloInstructionToCanonicalString(instruction), version) {}

}  // namespace gpu
}  // namespace xla
