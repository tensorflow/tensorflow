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

#include "xla/backends/gpu/autotuner/legacy_cache.h"

#include <optional>
#include <string>

#include "google/protobuf/duration.pb.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/autotuning/autotune_cache_key.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/protobuf/dnn.pb.h"

namespace xla {

namespace gpu {

using autotuner::Backend;

std::optional<LegacyCache::Config> LegacyCache::Lookup(
    const HloInstruction* instr) {
  AutotuneCacheKey key = GetAutotuneCacheKey(*instr);
  absl::StatusOr<std::optional<AutotuneResult>> result =
      AutotunerUtil::TryFindInCache(key, cache_dir_);
  if (!result.ok()) {
    LOG(ERROR) << "Failed to lookup autotune cache: " << result.status();
    return std::nullopt;
  }
  if (!result->has_value()) {
    stats_.misses++;
    return std::nullopt;
  }
  stats_.hits++;
  return GetConfig(result->value(), instr->opcode() == HloOpcode::kFusion);
}

absl::Status LegacyCache::Insert(const HloInstruction* instr,
                                 const Config& best_config) {
  AutotuneCacheKey key = GetAutotuneCacheKey(*instr);
  AutotuneResult autotune_result = GetAutotuneResult(best_config);
  absl::StatusOr<AutotunerUtil::ResultAndInserted> result_and_inserted =
      AutotunerUtil::AddResultToCaches(key, autotune_result, cache_dir_,
                                       cache_mode_);
  if (!result_and_inserted.ok()) {
    LOG(ERROR) << "Failed to insert autotune cache: "
               << result_and_inserted.status();
    return result_and_inserted.status();
  }
  return absl::OkStatus();
}

void LegacyCache::ClearCache() { AutotunerUtil::ClearAutotuneResults(); }

absl::StatusOr<std::string> LegacyCache::Serialize(
    absl::Span<const HloInstruction* const> instructions_to_serialize) {
  AutotuneCacheKeySet key_set;
  for (const HloInstruction* instr : instructions_to_serialize) {
    key_set.insert(GetAutotuneCacheKey(*instr));
  }

  std::optional<const AutotuneCacheKeySet*> keys_to_send = std::nullopt;
  if (!key_set.empty()) {
    keys_to_send = &key_set;
  }

  AutotuneResults results;
  TF_RETURN_IF_ERROR(
      AutotunerUtil::SerializeAutotuneResults(&results, keys_to_send));
  return AutotuneResultsToString(results, true);
}

absl::Status LegacyCache::Deserialize(absl::string_view serialized_cache) {
  return AutotunerUtil::LoadAutotuneResults(serialized_cache,
                                            /*as_textproto=*/true,
                                            /*allow_override=*/true);
}

AutotuneCacheKey LegacyCache::GetAutotuneCacheKey(const HloInstruction& instr) {
  AutotuneCacheKey key(device_desc_, instr);
  return key;
}

std::optional<LegacyCache::Config> LegacyCache::GetConfig(
    const AutotuneResult& result, bool is_fusion_instruction) {
  Config config;
  if (result.has_triton()) {
    config.codegen_backend = Backend::TRITON;
    config.backend_config.PackFrom(result.triton());
  } else if (result.has_gemm()) {
    config.codegen_backend = Backend::CUBLASLT;
    if (is_fusion_instruction) {
      config.codegen_backend = Backend::CUBLASLT_FISSION;
    }
    config.backend_config.PackFrom(result.gemm());
  } else if (result.has_algorithm()) {
    config.codegen_backend = Backend::CUDNN;
    config.backend_config.PackFrom(result.algorithm());
  } else if (result.has_custom_kernel_fusion()) {
    config.codegen_backend = Backend::CUSTOM_KERNEL_FISSION;
    config.backend_config.PackFrom(result.custom_kernel_fusion());
  } else if (result.has_other()) {
    if (!autotuner::Backend_Parse(result.other().name(),
                                  &config.codegen_backend)) {
      LOG(ERROR) << "Failed to parse codegen backend: "
                 << result.other().name();
      return std::nullopt;
    }
    config.backend_config = result.other().config();
  } else {
    return std::nullopt;
  }
  return config;
}

AutotuneResult LegacyCache::GetAutotuneResult(
    const LegacyCache::Config& config) {
  AutotuneResult result;
  if (config.codegen_backend == Backend::TRITON) {
    config.backend_config.UnpackTo(result.mutable_triton());
  } else if (config.codegen_backend == Backend::CUBLASLT ||
             config.codegen_backend == Backend::CUBLASLT_FISSION) {
    config.backend_config.UnpackTo(result.mutable_gemm());
  } else if (config.codegen_backend == Backend::CUDNN) {
    config.backend_config.UnpackTo(result.mutable_algorithm());
  } else if (config.codegen_backend == Backend::CUSTOM_KERNEL_FISSION) {
    config.backend_config.UnpackTo(result.mutable_custom_kernel_fusion());
  } else {
    result.mutable_other()->set_name(Backend_Name(config.codegen_backend));
    *result.mutable_other()->mutable_config() = config.backend_config;
  }
  return result;
}

}  // namespace gpu

}  // namespace xla
