/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ALLOCATION_FINDER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ALLOCATION_FINDER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

#include "absl/types/optional.h"

#include <vector>

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {

struct CompilerAnnotations;

using TensorSource = std::pair<const HloInstruction*, int64>;
using DeferredAllocations = absl::flat_hash_set<TensorSource>;
using DeferredAllocationsPath = std::vector<TensorSource>;

struct TensorTarget {
  // The node in the graph which consumes the tensor
  const HloInstruction* tgt;

  // The input on the target node which consumes the tensor
  int64 input_index;

  // A node in the graph which produces a tensor that influences the
  // construction of the tensor.  Example: bias tensors should match the layout
  // of a convolution output.  'layout' points to the convolution parameter.
  absl::optional<const HloInstruction*> layout;

  // Layout can have multiple output tensors - this index identifies which
  // output tensor to use.
  absl::optional<int64> layout_output_idx;

  // A vector of operations between the source and target operations.  Sometimes
  // it is possible to allocate a tensor for consumption by a target, and then
  // transform it into the tensor as it should be allocated by the source.
  std::vector<const HloInstruction*> forward_path;

  // A path from the layout influencing operation and the target operation.
  // Sometimes it is possible to take the output of the target and transform
  // it into something that can be used to make a layout-dependent allocation
  // at the target site.
  std::vector<const HloInstruction*> backward_path;

  // A path of deferred allocations from the target operation to the
  // Parameter/Infeed input to the graph.
  DeferredAllocationsPath deferred_allocations_path;

  TensorTarget(const HloInstruction* tgt, int64 input_index,
               const HloInstruction* layout, const int64 layout_output_idx,
               const std::vector<const HloInstruction*>& forward_path,
               const std::vector<const HloInstruction*>& backward_path,
               const DeferredAllocationsPath& deferred_allocations_path)
      : tgt(tgt),
        input_index(input_index),
        layout(layout),
        layout_output_idx(layout_output_idx),
        forward_path(forward_path),
        backward_path(backward_path),
        deferred_allocations_path(deferred_allocations_path) {}

  TensorTarget(const HloInstruction* tgt, int64 input_index,
               const std::vector<const HloInstruction*>& backward_path)
      : tgt(tgt),
        input_index(input_index),
        layout(absl::nullopt),
        layout_output_idx(absl::nullopt),
        forward_path({}),
        backward_path(backward_path) {}

  TensorTarget() = default;
};

using TensorAllocationMap = std::map<TensorSource, TensorTarget>;

/**
 * This class finds all instructions that explicitly add tensors to the
 * graph.  For each one of them, it locates the downstream consumers of that
 * tensor, and if any of those instructions require a specific tensor allocation
 * method (e.g. convolution), then it notes the downstream instruction
 */
class AllocationFinder : public HloModulePass {
 public:
  AllocationFinder(CompilerAnnotations& annotations);

  ~AllocationFinder() = default;

  absl::string_view name() const override { return "allocation-finder"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  void FindConsumers(const TensorSource&, const HloInstruction* tgt, int64);

  // Should return true when target 'a' should be used over 'b'
  bool CompareTargets(const TensorTarget& a, const TensorTarget& b);

  std::set<HloInstruction*> visited;
  std::vector<const HloInstruction*> path;

  const CompilerAnnotations& annotations;
  TensorAllocationMap& tensor_allocation_map;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
