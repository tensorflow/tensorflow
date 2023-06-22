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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTOTUNER_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTOTUNER_UTIL_H_

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include "tensorflow/compiler/xla/autotune_results.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/protobuf/autotuning.pb.h"

namespace xla {
namespace gpu {

struct DeviceConfig {
  se::StreamExecutor* stream_exec;  // never null

  // If the `allocator` parameter is not null, we will use it to allocate temp
  // memory while timing the various convolution algorithms.  If it's null,
  // we'll use the default allocator on the StreamExecutor.
  se::DeviceMemoryAllocator* allocator;  // may be null
};

struct DevicelessConfig {
  // The human-readable description of the device.  It can be found by using
  // stream_exec->GetDeviceDescription().model_str() when the stream executor
  // is available.
  std::string model_str;

  // A field to determine the architecture of the device. We only pick an
  // algorithm for non-Ampere architectures.
  se::CudaComputeCapability cuda_compute_capability{0, 0};
};

class AutotuneCacheKey {
 public:
  AutotuneCacheKey(absl::string_view model_str,
                   const HloInstruction& instruction);

  explicit AutotuneCacheKey(absl::string_view model_str,
                            absl::string_view hlo_canonical)
      : model_str_(model_str), hlo_canonical_(hlo_canonical) {}

  absl::string_view GetModelStr() const { return model_str_; }

  absl::string_view GetHlo() const { return hlo_canonical_; }

  template <typename H>
  friend H AbslHashValue(H h, const AutotuneCacheKey& w) {
    return H::combine(std::move(h), w.model_str_, w.hlo_canonical_);
  }

  bool operator==(const AutotuneCacheKey& w) const {
    return model_str_ == w.model_str_ && hlo_canonical_ == w.hlo_canonical_;
  }

 private:
  std::string model_str_;
  std::string hlo_canonical_;
};

class AutotuneConfig {
 public:
  bool should_init_buffers() const { return autotune_level_ >= 2; }
  bool should_reinit_output_buffer() const { return autotune_level_ >= 3; }
  bool should_check_correctness() const { return autotune_level_ >= 4; }
  bool should_crash_on_check_failure() const {
    return should_crash_on_check_failure_;
  }

  AutotuneConfig(const std::variant<DeviceConfig, DevicelessConfig>& config,
                 const DebugOptions& debug_options)
      : config_(config),
        autotune_level_(debug_options.xla_gpu_autotune_level()),
        should_crash_on_check_failure_(
            debug_options.xla_gpu_crash_on_verification_failures()),
        exhaustive_tiling_search_(
            debug_options.xla_gpu_exhaustive_tiling_search()) {}

  absl::string_view GetModelStr() const {
    if (auto deviceless_config = std::get_if<DevicelessConfig>(&config_)) {
      return deviceless_config->model_str;
    }

    const auto& device_config = std::get<DeviceConfig>(config_);
    return device_config.stream_exec->GetDeviceDescription().model_str();
  }

  se::StreamExecutor* GetExecutor() const {
    CHECK(std::holds_alternative<DeviceConfig>(config_));
    return std::get<DeviceConfig>(config_).stream_exec;
  }

  se::DeviceMemoryAllocator* GetAllocator() const {
    CHECK(std::holds_alternative<DeviceConfig>(config_));
    return std::get<DeviceConfig>(config_).allocator;
  }

  se::CudaComputeCapability GetCudaComputeCapability() const {
    if (auto c = std::get_if<DeviceConfig>(&config_)) {
      return c->stream_exec->GetDeviceDescription().cuda_compute_capability();
    }
    return std::get<DevicelessConfig>(config_).cuda_compute_capability;
  }

  bool IsDeviceless() const {
    return std::holds_alternative<DevicelessConfig>(config_);
  }

  bool ExhaustiveTilingSearch() const { return exhaustive_tiling_search_; }

 private:
  std::variant<DeviceConfig, DevicelessConfig> config_;
  int32_t autotune_level_;
  bool should_crash_on_check_failure_;
  bool exhaustive_tiling_search_;
};

using AutotuneCacheMap =
    absl::flat_hash_map<AutotuneCacheKey, tensorflow::AutotuneResult>;

Status SerializeAutotuneResults(const AutotuneCacheMap& autotune_cache,
                                AutotuneResults* results);

Status LoadAutotuneResults(AutotuneCacheMap& autotune_cache,
                           const AutotuneResults& results);

struct AutotunerUtil {
  // Create a buffer for a given operation using redzone checker, initialize
  // based on a given rng state.
  static StatusOr<se::DeviceMemoryBase> CreateBuffer(
      se::RedzoneAllocator& allocator, const Shape& shape,
      const AutotuneConfig& config, int64_t& rng_state);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTOTUNER_UTIL_H_
