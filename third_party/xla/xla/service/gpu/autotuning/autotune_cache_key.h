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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNE_CACHE_KEY_H_
#define XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNE_CACHE_KEY_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class AutotuneCacheKey {
 public:
  // Tie a version to the cache key in order to invalidate the cache when
  // necessary. This should be incremented on triton upgrades or any other
  // changes that may affect the autotuning results.
  static constexpr int kCurrentVersion = 24;

  AutotuneCacheKey(const se::DeviceDescription& device_description,
                   const HloInstruction& instruction,
                   int version = kCurrentVersion);

  absl::string_view GetModelStr() const { return model_str_; }

  absl::string_view GetHlo() const { return hlo_canonical_; }

  int GetVersion() const { return version_; }

  template <typename H>
  friend H AbslHashValue(H h, const AutotuneCacheKey& w) {
    return H::combine(std::move(h), w.model_str_, w.hlo_canonical_, w.version_);
  }

  bool operator==(const AutotuneCacheKey& w) const {
    return model_str_ == w.model_str_ && hlo_canonical_ == w.hlo_canonical_ &&
           version_ == w.version_;
  }

  std::string ToString() const {
    return absl::StrFormat("<key model='%s', hlo='%s', version=%d>", model_str_,
                           hlo_canonical_, version_);
  }

  static std::string DeviceDescriptionToCacheKey(
      const se::DeviceDescription& device_description);

  static std::string HloInstructionToCanonicalString(
      const HloInstruction& instr);

 private:
  friend class AutotunerUtil;

  explicit AutotuneCacheKey(absl::string_view model_str,
                            absl::string_view hlo_canonical)
      : model_str_(model_str), hlo_canonical_(hlo_canonical) {}

  explicit AutotuneCacheKey(absl::string_view model_str,
                            absl::string_view hlo_canonical, int version)
      : model_str_(model_str),
        hlo_canonical_(hlo_canonical),
        version_(version) {}

  std::string model_str_;
  std::string hlo_canonical_;
  int version_ = kCurrentVersion;
};

using AutotuneCacheKeySet = absl::flat_hash_set<AutotuneCacheKey>;

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNE_CACHE_KEY_H_
