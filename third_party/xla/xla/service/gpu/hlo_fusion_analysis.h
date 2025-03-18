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

#ifndef XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_
#define XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class HloFusionAnalysis {
 public:
  // The type of emitted fusion.
  enum class EmitterFusionKind {
    kLoop,
    kCustomFusion,
    kTriton,
    kReduction,
    kTranspose,
    kConcatenate,
    kInputSlices,
    kScatter,
    kCuDnn,
    kDynamicMemcpy,
  };

  // Precomputed information about inputs (arguments) and outputs (roots) of the
  // fusion.
  struct InputOutputInfo {
    int smallest_input_dtype_bits;
    int smallest_output_dtype_bits;
  };

  static HloFusionAnalysis Create(FusionBackendConfig backend_config,
                                  std::unique_ptr<HloFusionAdaptor> fusion,
                                  const se::DeviceDescription* device_info);

  // Creates a HloFusionAnalysis that analyzes just instruction as a standalone
  // fusion.
  static HloFusionAnalysis Create(const HloInstruction& instruction,
                                  const se::DeviceDescription& device_info);

  // Creates a HloFusionAnalysis that analyzes a hypothetical fusion of producer
  // into consumer.
  static HloFusionAnalysis Create(const HloInstruction& producer,
                                  const HloInstruction& consumer,
                                  const se::DeviceDescription& device_info);

  const HloFusionAdaptor& fusion() const { return *fusion_; }

  const absl::InlinedVector<HloInstructionAdaptor, 2>& fusion_roots() const {
    return fusion_roots_;
  }
  HloInstructionAdaptor fusion_root(int64_t i) const {
    return fusion_roots_[i];
  }
  int64_t fusion_root_count() const { return fusion_roots_.size(); }

  const absl::InlinedVector<HloInstructionAdaptor, 2>& fusion_heroes() const {
    return fusion_heroes_;
  }
  HloInstructionAdaptor fusion_hero(int64_t i) const {
    return fusion_heroes_[i];
  }

  // Determines the fusion type for the emitter.
  EmitterFusionKind GetEmitterFusionKind() const;

  // Returns the hero reduction of the computation.
  const HloInstruction* FindHeroReduction() const;

  const se::DeviceDescription& device_info() const { return *device_info_; }

  const FusionBackendConfig& fusion_backend_config() const {
    return fusion_backend_config_;
  }

  // Returns the tiled transpose description. Requires that GetEmitterFusionKind
  // returns kTranspose.
  const TransposeDescription& tiled_transpose() const {
    CHECK(tiled_transpose_.has_value());
    return *tiled_transpose_;
  }

  const InputOutputInfo& input_output_info() const {
    return input_output_info_;
  }

 private:
  HloFusionAnalysis(FusionBackendConfig fusion_backend_config,
                    std::unique_ptr<HloFusionAdaptor> fusion,
                    absl::InlinedVector<HloInstructionAdaptor, 2> fusion_roots,
                    absl::InlinedVector<HloInstructionAdaptor, 2> fusion_heroes,
                    const se::DeviceDescription* device_info,
                    std::optional<TransposeDescription> tiled_transpose,
                    InputOutputInfo input_output_info);

  bool HasConsistentTransposeHeros() const;

  FusionBackendConfig fusion_backend_config_;

  // Owning pointer to the fusion adaptor object.
  std::unique_ptr<HloFusionAdaptor> fusion_;

  // A list of all roots of the fusion. The instruction adaptors have `fusion_`
  // as their parent and should not outlive `fusion_`.
  absl::InlinedVector<HloInstructionAdaptor, 2> fusion_roots_;

  // A list of all heroes of the fusion. The instruction adaptors have `fusion_`
  // as their parent and should not outlive `fusion_`.
  absl::InlinedVector<HloInstructionAdaptor, 2> fusion_heroes_;

  const se::DeviceDescription* device_info_;
  std::optional<TransposeDescription> tiled_transpose_;
  InputOutputInfo input_output_info_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_
