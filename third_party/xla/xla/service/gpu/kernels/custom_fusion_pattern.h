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

#ifndef XLA_SERVICE_GPU_KERNELS_CUSTOM_FUSION_PATTERN_H_
#define XLA_SERVICE_GPU_KERNELS_CUSTOM_FUSION_PATTERN_H_

#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "absl/base/attributes.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CustomFusionPattern
//===----------------------------------------------------------------------===//

// Custom fusion pattern matches HLO instruction to custom kernels.
class CustomFusionPattern {
 public:
  virtual ~CustomFusionPattern() = default;

  struct Match {
    CustomFusionConfig config;
    std::vector<HloInstruction *> instructions;
  };

  // Returns custom fusion config and a list of instructions that matched to a
  // custom fusion (one or more custom kernels). Custom fusion pass will outline
  // matched instructions into a custom fusion operation if possible.
  //
  // TODO(ezhulenev): Today the last instruction defines custom fusion root
  // (results), however we need to add support for custom fusion that can return
  // intermediate result, and custom fusions that require an extra workspace.
  virtual std::optional<Match> TryMatch(HloInstruction *instr) const = 0;
};

//===----------------------------------------------------------------------===//
// CustomFusionPatternRegistry
//===----------------------------------------------------------------------===//

class CustomFusionPatternRegistry {
 public:
  // Returns a pointer to a default custom fusion pattern registry, which is a
  // global static registry.
  static CustomFusionPatternRegistry *Default();

  std::vector<CustomFusionPattern::Match> Match(HloInstruction *instr) const;

  void Add(std::unique_ptr<CustomFusionPattern> pattern);

  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void Emplace() {
    (Add(std::make_unique<Ts>()), ...);
  }

 private:
  std::vector<std::unique_ptr<CustomFusionPattern>> patterns_;
};

}  // namespace xla::gpu

#define XLA_REGISTER_CUSTOM_FUSION_PATTERN(PATTERN) \
  XLA_REGISTER_CUSTOM_FUSION_PATTERN_(PATTERN, __COUNTER__)

#define XLA_REGISTER_CUSTOM_FUSION_PATTERN_(PATTERN, N) \
  XLA_REGISTER_CUSTOM_FUSION_PATTERN__(PATTERN, N)

#define XLA_REGISTER_CUSTOM_FUSION_PATTERN__(PATTERN, N)   \
  ABSL_ATTRIBUTE_UNUSED static const bool                  \
      xla_custom_fusion_pattern_##N##_registered_ = [] {   \
        ::xla::gpu::CustomFusionPatternRegistry::Default() \
            ->Emplace<PATTERN>();                          \
        return true;                                       \
      }()

#endif  // XLA_SERVICE_GPU_KERNELS_CUSTOM_FUSION_PATTERN_H_
