/* Copyright 2019 Graphcore Ltd

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_NORM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_NORM_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloNormInstruction : public HloPoplarInstruction {
 public:
  explicit HloNormInstruction(const Shape& shape,
                              absl::Span<HloInstruction* const> operands,
                              absl::string_view custom_call_target,
                              int32 num_groups, float epsilon,
                              int feature_index);

  int32 num_groups() const;
  float epsilon() const;
  int feature_index() const;

 private:
  int32 num_groups_;
  float epsilon_;
  int feature_index_;

  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
};

class HloGroupNormInstruction : public HloNormInstruction {
 public:
  explicit HloGroupNormInstruction(
      const Shape& shape, HloInstruction* const operand,
      HloInstruction* const scale, HloInstruction* const offset,
      HloInstruction* const mean, HloInstruction* const variance_or_inv_std_dev,
      int32 num_groups, float epsilon, int feature_index);

  const HloInstruction* operand() const;
  const HloInstruction* scale() const;
  const HloInstruction* offset() const;
  const HloInstruction* mean() const;
  const HloInstruction* variance_or_inv_std_dev() const;

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateGroupNorm(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const scale, HloInstruction* const offset,
    HloInstruction* const mean, HloInstruction* const variance_or_inv_std_dev,
    int32 num_groups, float epsilon, int feature_index);

class HloGroupNormTrainInstruction : public HloNormInstruction {
 public:
  explicit HloGroupNormTrainInstruction(const Shape& shape,
                                        HloInstruction* const operand,
                                        HloInstruction* const scale,
                                        HloInstruction* const offset,
                                        int32 num_groups, float epsilon,
                                        int feature_index);

  const HloInstruction* operand() const;
  const HloInstruction* scale() const;
  const HloInstruction* offset() const;

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateGroupNormTrain(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const scale, HloInstruction* const offset, int32 num_groups,
    float epsilon, int feature_index);

class HloGroupNormGradInstruction : public HloNormInstruction {
 public:
  explicit HloGroupNormGradInstruction(
      const Shape& shape, HloInstruction* const operand,
      HloInstruction* const scale, HloInstruction* const offset,
      HloInstruction* const mean, HloInstruction* const variance_or_inv_std_dev,
      int32 num_groups, float epsilon, int feature_index);

  const HloInstruction* operand() const;
  const HloInstruction* scale() const;
  const HloInstruction* mean() const;
  const HloInstruction* variance_or_inv_std_dev() const;
  const HloInstruction* grad_output() const;

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateGroupNormGrad(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const scale, HloInstruction* const mean,
    HloInstruction* const variance_or_inv_std_dev,
    HloInstruction* const grad_output, int32 num_groups, float epsilon,
    int feature_index);

class HloGroupNormStatsInstruction : public HloNormInstruction {
 public:
  explicit HloGroupNormStatsInstruction(const Shape& shape,
                                        HloInstruction* const operand,
                                        int32 num_groups, float epsilon,
                                        int feature_index);

  const HloInstruction* operand() const;

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateGroupNormStats(
    const Shape& shape, HloInstruction* const operand, int32 num_groups,
    float epsilon, int feature_index);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_CUSTOM_HLO_OPS_SIMPLE_GATHER_H_
