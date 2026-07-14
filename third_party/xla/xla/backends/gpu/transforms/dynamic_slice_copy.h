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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_COPY_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_COPY_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"

namespace xla::gpu {

// Analysis result for a fusion that can be lowered as a DeviceToDeviceCopyThunk
// wrapped by DynamicSliceFusionV2Thunk. The source and destination slices are
// described by `parameters` and `results`; no HLO copy needs to exist.
struct DynamicSliceCopyFusion {
  const HloInstruction* copy_operand;
  std::vector<DynamicSliceFusion::Parameter> parameters;
  std::vector<DynamicSliceFusion::Result> results;
};

// Analysis result for a static slice fusion that can be lowered directly to a
// DeviceToDeviceCopyThunk.
struct StaticSliceCopyFusion {
  int64_t parameter_number;
  Shape slice_shape;
  int64_t source_byte_offset;
};

// Matches raw DS/DUS-root fusions that copy one contiguous slice. This does not
// match already-materialized copy-hero DynamicSliceFusionV2 custom fusions.
absl::StatusOr<std::optional<DynamicSliceCopyFusion>>
AnalyzeDynamicSliceCopyFusion(const HloInstruction* instr);

// Matches a fusion whose root is a contiguous static slice of a fusion
// parameter and can therefore be emitted as a plain D2D copy.
absl::StatusOr<std::optional<StaticSliceCopyFusion>>
AnalyzeStaticSliceCopyFusion(const HloFusionInstruction* instr);

// Returns true for a DynamicSliceFusionV2 custom fusion whose hero instruction
// is a device-to-device copy.
bool IsCopyHeroDynamicSliceFusion(const HloInstruction* instr);

// Returns true for raw DS/DUS-root copy fusions or copy-hero
// DynamicSliceFusionV2 custom fusions.
bool IsDynamicSliceCopyFusion(const HloInstruction* instr);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_COPY_H_
