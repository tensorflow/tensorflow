/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_KERNELS_CUSTOM_KERNEL_FUSION_PATTERN_H_
#define XLA_SERVICE_GPU_KERNELS_CUSTOM_KERNEL_FUSION_PATTERN_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CustomKernelFusionPattern
//===----------------------------------------------------------------------===//

// Custom kernel fusion pattern matches HLO instruction to custom kernels.
class CustomKernelFusionPattern {
 public:
  // A name of a custom call that can be added to a custom kernel fusion body to
  // allocate a workspace buffer require for the custom kernel fusion
  // implementation.
  static constexpr const char *kWorkspace = "__custom_kernel_fusion$workspace";

  virtual ~CustomKernelFusionPattern() = default;

  // Matched sequence of instructions that can be handled by a custom kernel
  // fusion.
  class Match {
   public:
    Match(CustomFusionConfig config, std::vector<HloInstruction *> instructions,
          int64_t workspace_size_bytes = 0);

    // If some of operations matched by a pattern have users outside of the
    // custom kernel fusion, pattern can optionally provide a replacement that
    // can be derived from the fusion instruction result, or from other
    // instructions in the parent computation.
    using Replacement =
        std::function<absl::StatusOr<HloInstruction *>(HloFusionInstruction *)>;

    void AddReplacement(HloInstruction *instr, Replacement replacement);
    bool HasReplacement(HloInstruction *instr) const;

    // Builds a replacement for `instr` using a `fusion` instruction constructed
    // for a pattern match.
    absl::StatusOr<HloInstruction *> BuildReplacement(
        HloInstruction *instr, HloFusionInstruction *fusion) const;

    const CustomFusionConfig &config() const { return config_; }
    absl::Span<HloInstruction *const> instructions() const {
      return instructions_;
    }

    HloInstruction *root() const { return instructions_.back(); }

    int64_t workspace_size_bytes() const { return workspace_size_bytes_; }

   private:
    CustomFusionConfig config_;
    std::vector<HloInstruction *> instructions_;
    absl::flat_hash_map<const HloInstruction *, Replacement> replacements_;
    int64_t workspace_size_bytes_;
  };

  // Returns custom fusion config and a list of instructions that matched to a
  // custom kernel fusion (one or more custom kernels). Custom kernel fusion
  // pass will outline matched instructions into a custom kernel fusion
  // operation if possible.
  //
  // TODO(ezhulenev): Today the last instruction defines custom kernel fusion
  // root (results), however we need to add support for custom kernel fusion
  // that can return intermediate result, and custom kernel fusions that require
  // an extra workspace.
  virtual std::optional<Match> TryMatch(const se::DeviceDescription &device,
                                        HloInstruction *instr) const = 0;
};

//===----------------------------------------------------------------------===//
// CustomKernelFusionPatternRegistry
//===----------------------------------------------------------------------===//

class CustomKernelFusionPatternRegistry {
 public:
  // Returns a pointer to a default custom kernel fusion pattern registry, which
  // is a global static registry.
  static CustomKernelFusionPatternRegistry *Default();

  std::vector<CustomKernelFusionPattern::Match> Match(
      const se::DeviceDescription &device, HloInstruction *instr) const;

  void Add(std::unique_ptr<CustomKernelFusionPattern> pattern);

  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void Emplace() {
    (Add(std::make_unique<Ts>()), ...);
  }

  template <typename... Ts, typename Arg,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void Emplace(Arg &&arg) {
    (Add(std::make_unique<Ts>(std::forward<Arg>(arg))), ...);
  }

 private:
  std::vector<std::unique_ptr<CustomKernelFusionPattern>> patterns_;
};

}  // namespace xla::gpu

#define XLA_REGISTER_CUSTOM_FUSION_PATTERN(PATTERN) \
  XLA_REGISTER_CUSTOM_FUSION_PATTERN_(PATTERN, __COUNTER__)

#define XLA_REGISTER_CUSTOM_FUSION_PATTERN_(PATTERN, N) \
  XLA_REGISTER_CUSTOM_FUSION_PATTERN__(PATTERN, N)

#define XLA_REGISTER_CUSTOM_FUSION_PATTERN__(PATTERN, N)         \
  ABSL_ATTRIBUTE_UNUSED static const bool                        \
      xla_custom_fusion_pattern_##N##_registered_ = [] {         \
        ::xla::gpu::CustomKernelFusionPatternRegistry::Default() \
            ->Emplace<PATTERN>();                                \
        return true;                                             \
      }()

#endif  // XLA_SERVICE_GPU_KERNELS_CUSTOM_KERNEL_FUSION_PATTERN_H_
