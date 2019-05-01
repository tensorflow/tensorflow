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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ONE_HOT_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ONE_HOT_H_

#include <memory>
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloOneHotInstruction : public HloPoplarInstruction {
 public:
  explicit HloOneHotInstruction(HloInstruction* indices, int64 depth,
                                int32 axis, HloInstruction* on,
                                HloInstruction* off, const Shape outputShape);

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  uint64 NumberOfInplaceOperands() const override;

  int64 Depth() const { return depth_; }

  int32 Axis() const { return axis_; }

  bool IsPopOpsElementwise() const;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  int64 depth_;
  int32 axis_;
};

std::unique_ptr<HloInstruction> CreateOneHot(HloInstruction* indices,
                                             HloInstruction* depth,
                                             HloInstruction* on,
                                             HloInstruction* off,
                                             const Shape& shape);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ONE_HOT_H_
