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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_POOLING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_POOLING_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloPoolingInstruction : public HloPoplarInstruction {
 public:
  explicit HloPoolingInstruction(const Shape& shape,
                                 absl::Span<HloInstruction* const> operands,
                                 absl::string_view custom_call_target,
                                 xla::Window window);

  const xla::Window& window() const override;

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 private:
  xla::Window window_;

  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
};

class HloMaxPoolInstruction : public HloPoolingInstruction {
 public:
  HloMaxPoolInstruction(const Shape& shape, HloInstruction* to_reduce,
                        xla::Window window);

  const HloInstruction* to_reduce() const;
  HloInstruction* mutable_to_reduce();

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateMaxPool(const Shape& shape,
                                              HloInstruction* to_reduce,
                                              xla::Window window);

class HloAvgPoolInstruction : public HloPoolingInstruction {
 public:
  HloAvgPoolInstruction(const Shape& shape, HloInstruction* to_reduce,
                        xla::Window window);

  const HloInstruction* to_reduce() const;
  HloInstruction* mutable_to_reduce();

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateAvgPool(const Shape& shape,
                                              HloInstruction* to_reduce,
                                              xla::Window window);

class HloMaxPoolGradInstruction : public HloPoolingInstruction {
 public:
  HloMaxPoolGradInstruction(const Shape& shape, HloInstruction* input,
                            HloInstruction* output, HloInstruction* output_grad,
                            xla::Window window_);

  const HloInstruction* input() const;
  const HloInstruction* output() const;
  const HloInstruction* output_grad() const;

  HloInstruction* mutable_input();
  HloInstruction* mutable_output();
  HloInstruction* mutable_output_grad();

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateMaxPoolGrad(const Shape& shape,
                                                  HloInstruction* input,
                                                  HloInstruction* output,
                                                  HloInstruction* output_grad,
                                                  xla::Window window);

class HloAvgPoolGradInstruction : public HloPoolingInstruction {
 public:
  HloAvgPoolGradInstruction(const Shape& shape, HloInstruction* output_grad,
                            xla::Window window_);

  const HloInstruction* output_grad() const;
  HloInstruction* mutable_output_grad();

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateAvgPoolGrad(const Shape& shape,
                                                  HloInstruction* output_grad,
                                                  xla::Window window);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_CUSTOM_HLO_OPS_SIMPLE_GATHER_H_
