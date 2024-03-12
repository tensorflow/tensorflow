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

#include <memory>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_traversal.h"
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
  };

  // Precomputed information about inputs (arguments) and outputs (roots) of the
  // fusion.
  struct InputOutputInfo {
    bool has_4_bit_input;
    bool has_4_bit_output;
    int smallest_input_dtype_bits;
  };

  static HloFusionAnalysis Create(FusionBackendConfig backend_config,
                                  std::unique_ptr<HloFusionAdaptor> fusion,
                                  const se::DeviceDescription* device_info);
  static HloFusionAnalysis Create(const HloFusionInstruction* fusion,
                                  const se::DeviceDescription* device_info);

  const std::vector<const HloInstruction*>& fusion_roots() const {
    return fusion_roots_;
  }
  const std::vector<const HloInstruction*>& fusion_heroes() const {
    return fusion_heroes_;
  }
  const HloFusionAdaptor& fusion() const { return *fusion_; }

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
                    std::vector<const HloInstruction*> fusion_roots,
                    std::unique_ptr<HloFusionAdaptor> fusion,
                    std::vector<const HloInstruction*> fusion_heroes,
                    const se::DeviceDescription* device_info,
                    std::optional<TransposeDescription> tiled_transpose,
                    InputOutputInfo input_output_info);

  bool HasConsistentTransposeHeros() const;

  FusionBackendConfig fusion_backend_config_;
  std::vector<const HloInstruction*> fusion_roots_;
  std::unique_ptr<HloFusionAdaptor> fusion_;
  std::vector<const HloInstruction*> fusion_heroes_;
  const se::DeviceDescription* device_info_;
  std::optional<TransposeDescription> tiled_transpose_;
  InputOutputInfo input_output_info_;
};

// Creates a HloFusionAnalysis that analyzes a hypothetical fusion of producer
// into consumer.
HloFusionAnalysis AnalyzeProducerConsumerFusion(
    const HloInstruction& producer, const HloInstruction& consumer,
    const se::DeviceDescription& device_info);

// Creates a HloFusionAnalysis that analyzes just consumer as a standalone
// fusion.
HloFusionAnalysis AnalyzeFusion(const HloInstruction& consumer,
                                const se::DeviceDescription& device_info);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_
