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

class HloArgMinMax : public HloPoplarInstruction {
 public:
  explicit HloArgMinMax(HloInstruction* input, const Shape outputShape,
                        int64 axis, bool is_min);

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  int64 Axis() const { return axis; }

  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  int64 axis;
};

class HloArgMax : public HloArgMinMax {
 public:
  explicit HloArgMax(HloInstruction* input, const Shape outputShape, int64 axis)
      : HloArgMinMax(input, outputShape, axis, false) {}
};

class HloArgMin : public HloArgMinMax {
 public:
  explicit HloArgMin(HloInstruction* input, const Shape outputShape, int64 axis)
      : HloArgMinMax(input, outputShape, axis, true) {}
};

std::unique_ptr<HloInstruction> CreateHloArgMinMax(HloInstruction* input,
                                                   const Shape& shape,
                                                   int64 axis, bool is_min);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ONE_HOT_H_
