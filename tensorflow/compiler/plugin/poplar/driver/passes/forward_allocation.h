#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FORWARD_ALLOCATION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FORWARD_ALLOCATION_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

#include <fstream>
#include <queue>
#include <sstream>

namespace xla {

class HloModule;
class HloComputation;
class HloInstruction;

namespace poplarplugin {

class ForwardAllocation : public HloModulePass {
 public:
  ForwardAllocation(CompilerAnnotations& annotations);

  absl::string_view name() const override { return "forward-allocation"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  absl::optional<TensorTarget> CreateForwardAllocationTarget(
      HloReachabilityMap* reachability_map, HloInstruction* source,
      HloInstruction* target, const int64 input_index,
      HloInstruction* layout_producer, const int64 layout_output_index,
      const std::vector<HloInstruction*>& other_targets,
      const std::vector<HloInstruction*>& forward_path,
      const std::vector<HloInstruction*>& backward_path,
      const DeferredAllocationsPath& deferred_allocations_path);

  StatusOr<bool> FindLayoutSensativeTargets(
      HloComputation* comp, std::set<const HloInstruction*>& ops_with_layout);

  StatusOr<bool> FindLayoutDependentTargets(HloComputation* comp);

  absl::flat_hash_map<HloInstruction*, DeferredAllocationsPath> FindInputs(
      HloComputation* comp);

  void FlattenInputs(
      HloInstruction* inst, std::vector<const HloInstruction*> path,
      absl::flat_hash_map<HloInstruction*, DeferredAllocationsPath>&
          input_to_deferred_allocation_path);

  TensorAllocationMap& tensor_allocation_map;
  DeferredAllocations& deferred_allocations;
  const InplaceInstructions& inplace_instructions;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
