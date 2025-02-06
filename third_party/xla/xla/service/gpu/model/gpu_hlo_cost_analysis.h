/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_HLO_COST_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_GPU_HLO_COST_ANALYSIS_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Cost analysis for GPUs.
class GpuHloCostAnalysis : public HloCostAnalysis {
  // Each instruction creating a new basic block roughly doubles the total
  // number of basic blocks and the IR code size accordingly.
  static constexpr int64_t kMaxBasicBlockSplitsPerFusion = 10;
  static constexpr int64_t kMaxIRSize = 10000;

 public:
  GpuHloCostAnalysis(
      const Options& options,
      const HloOpProfiles::HloOpProfile& hlo_elementwise_op_profile)
      : HloCostAnalysis(options),
        hlo_elementwise_op_profile_(hlo_elementwise_op_profile) {}

  explicit GpuHloCostAnalysis(const Options& options)
      : GpuHloCostAnalysis(options,
                           HloOpProfiles::Singleton().GetDefaultProfile()) {}

  GpuHloCostAnalysis(const Options& options,
                     const se::DeviceDescription& device_info)
      : GpuHloCostAnalysis(
            options, HloOpProfiles::Singleton().GetProfile(device_info)) {}

  absl::Status Preprocess(const HloInstruction* hlo) override;

  float ScalingRatio(const HloInstruction& hlo) const;
  int64_t NumOfDevices(const HloInstruction& hlo) const;
  float BytesTransferred(const HloInstruction& hlo) const;

  absl::Status HandleCustomCall(const HloInstruction* call) override;

  int64_t GetConvolutionFlops(const HloInstruction* convolution) override;

  absl::Status HandleElementwiseOp(const HloInstruction* hlo) override;

  absl::Status HandleConcatenate(const HloInstruction* hlo) override;
  absl::Status HandleAllReduce(const HloInstruction* allreduce) override;
  absl::Status HandleReduce(const HloInstruction* hlo) override;
  absl::Status HandleAllReduceStart(const HloInstruction* hlo) override;
  absl::Status HandleAllGather(const HloInstruction* hlo) override;
  absl::Status HandleAllGatherStart(const HloInstruction* hlo) override;
  absl::Status HandleAsyncStart(const HloInstruction* hlo) override;
  absl::Status HandleReduceScatter(const HloInstruction* hlo) override;

  // Estimate the total size of IR accounting for both duplication
  // of producer code by consumer and the total number of basic blocks.
  // Tell if merged IR size would be too slow to compile.
  bool ProducerConsumerMergedTooLarge(const HloInstruction& producer,
                                      const HloInstruction& consumer);

  // IR size scale of an instruction: 1 for most instructions,
  // but for fusions is the number of instructions emitted including the
  // duplication due to non-element-wise accesses.
  float IrSize(const HloInstruction& hlo) const;

  // Total common elementwise utilization of two instructions within a fusion.
  // If two parameters have several common elementwise use roots returned is
  // the sum of these utilizations. Can also be used to query if a parameter
  // is used elementwise from the fusion's root.
  float CommonElementwiseUtilization(const HloInstruction* a,
                                     const HloInstruction* b) const;

  // Returns the number of FLOPs needed to compute an element of the given
  // elementwise instruction.
  int64_t GetFlopsPerElementwiseOpElement(PrimitiveType type, HloOpcode opcode);

  // Returns the number of FLOPs needed to compute the output of the elementwise
  // instruction.
  int64_t GetFlopsForElementwiseOp(HloOpcode op_code, const Shape& shape);
  int64_t GetFlopsForElementwiseOp(const HloInstruction* instr);

 protected:
  std::unique_ptr<HloCostAnalysis> CreateNestedCostAnalysis() override;
  int64_t FusionParameterReadBytes(const HloInstruction* hlo) const override;
  absl::Status FusionCalculateUtilizations(
      const HloInstruction* fusion) override;

  size_t immediate_constant_max_elements() const override { return 8; }

  bool KeyToCopyFromSubcomputation(absl::string_view key) const override;

  // To estimate where within the computation an instruction output can be
  // reused and where it has to be recomputed again we group accesses to the
  // instruction by their origin from "element-wise use roots". All access
  // paths from such a root to the instruction are element-wise.
  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloInstruction*>>
      elementwise_use_roots_;

  // Elementwise utilization of instruction's input subtree if it is a root.
  // This is different from hlo_properties_[instr][kUtilizationKey] which
  // is the utilization of the instruction by other roots.
  absl::flat_hash_map<const HloInstruction*, float> root_utilizations_;

  // Contains a map from (opcode, element_type) to FLOPs per element estimate
  // for elementwise instructions.
  const HloOpProfiles::HloOpProfile& hlo_elementwise_op_profile_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_HLO_COST_ANALYSIS_H_
