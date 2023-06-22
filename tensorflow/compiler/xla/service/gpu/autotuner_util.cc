/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/autotuner_util.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>

#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"

namespace xla {
namespace gpu {

Status SerializeAutotuneResults(const AutotuneCacheMap& autotune_cache,
                                AutotuneResults* results) {
  for (const auto& [k, result] : autotune_cache) {
    auto& entry = *results->add_results();
    entry.set_device(std::string(k.GetModelStr()));
    entry.set_hlo(std::string(k.GetHlo()));
    *entry.mutable_result() = result;
  }

  // Sort the results so that they're deterministic.
  std::sort(results->mutable_results()->pointer_begin(),
            results->mutable_results()->pointer_end(),
            [](const auto* a, const auto* b) {
              return std::make_pair(absl::string_view(a->device()),
                                    absl::string_view(a->hlo())) <
                     std::make_pair(absl::string_view(b->device()),
                                    absl::string_view(b->hlo()));
            });

  return OkStatus();
}

Status LoadAutotuneResults(AutotuneCacheMap& autotune_cache,
                           const AutotuneResults& results) {
  for (const auto& result : results.results()) {
    autotune_cache[AutotuneCacheKey(result.device(), result.hlo())] =
        result.result();
  }
  return OkStatus();
}

/* static*/ StatusOr<se::DeviceMemoryBase> AutotunerUtil::CreateBuffer(
    se::RedzoneAllocator& allocator, const Shape& shape,
    const AutotuneConfig& config, int64_t& rng_state) {
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                      allocator.AllocateBytes(ShapeUtil::ByteSizeOf(shape)));
  if (config.should_init_buffers()) {
    InitializeBuffer(allocator.stream(), shape.element_type(), &rng_state,
                     buffer);
  }
  return buffer;
}

static std::string ToCanonicalString(const HloInstruction* instr) {
  auto options = HloPrintOptions::Canonical();
  if (instr->opcode() != HloOpcode::kFusion) {
    options.set_print_backend_config(true);
    return instr->ToString(options);
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
  return instr->called_computations()[0]->ToString(options);
}

AutotuneCacheKey::AutotuneCacheKey(absl::string_view model_str,
                                   const HloInstruction& instr)
    : AutotuneCacheKey(model_str, ToCanonicalString(&instr)) {}

}  // namespace gpu
}  // namespace xla
