/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_INDEXING_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_INDEXING_PERFORMANCE_MODEL_H_

#include <cstdint>

#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Implementation of Cost Model that uses indexing analysis to estimate amount
// of compute and memory access time.
class GpuPerformanceModelWithIndexingAnalysis : public GpuPerformanceModelBase {
 public:
  explicit GpuPerformanceModelWithIndexingAnalysis(
      const se::DeviceDescription* device_info,
      HloCostAnalysis::ShapeSizeFunction shape_size,
      mlir::MLIRContext* mlir_context)
      : hlo_op_profile_(&HloOpProfiles::Singleton().GetProfile(device_info)),
        device_info_(device_info),
        shape_size_(shape_size),
        mlir_context_(mlir_context) {}

  EstimateRunTimeData EstimateRunTimeForFusion(
      const HloFusionAnalysis& fusion_analysis, bool is_coalesced = true);

  EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* producer);

  EstimateRunTimeData EstimateRunTimeForProducerConsumer(
      const HloInstruction* producer, const HloInstruction* consumer);

  RunTimes EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<const HloInstruction* const> fused_consumers = {});

 private:
  // Returns an estimate how many FLOPs will be used to produce one element of
  // the output.
  int64_t FlopsPerElement(const HloInstruction* instr) const;

  int64_t GetShapeSizeRecursive(const Shape& shape) const;

  const HloOpProfiles::HloOpProfile* hlo_op_profile_;
  const se::DeviceDescription* device_info_;
  HloCostAnalysis::ShapeSizeFunction shape_size_;
  mlir::MLIRContext* mlir_context_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_INDEXING_PERFORMANCE_MODEL_H_
