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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_LSTM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_LSTM_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloLSTMInstruction : public HloPoplarInstruction {
 public:
  explicit HloLSTMInstruction(const Shape& shape,
                              absl::Span<HloInstruction* const> operands,
                              const std::string& custom_call_target,
                              bool is_training, int32 num_channels,
                              xla::PrimitiveType partials_type);

  bool is_training() const;
  int32 num_channels() const;
  xla::PrimitiveType partials_type() const;

 private:
  bool is_training_;
  int32 num_channels_;
  xla::PrimitiveType partials_type_;

  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
};

class HloLSTMFwdInstruction : public HloLSTMInstruction {
 public:
  explicit HloLSTMFwdInstruction(const Shape& shape,
                                 absl::Span<HloInstruction* const> operands,
                                 bool is_training, int32 num_channels,
                                 xla::PrimitiveType partials_type);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

class HloLSTMBwdInstruction : public HloLSTMInstruction {
 public:
  explicit HloLSTMBwdInstruction(const Shape& shape,
                                 absl::Span<HloInstruction* const> operands,
                                 bool is_training, int32 num_channels,
                                 xla::PrimitiveType partials_type);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateLSTMFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type);

std::unique_ptr<HloInstruction> CreateLSTMBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_CUSTOM_HLO_OPS_SIMPLE_GATHER_H_
