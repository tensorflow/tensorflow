#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_FORWARD_ALLOCATION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_FORWARD_ALLOCATION_H_

#include "allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

#include <fstream>
#include <queue>
#include <sstream>

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {

class ForwardAllocation : public HloModulePass {
 public:
  ForwardAllocation(CompilerAnnotations& annotations);

  absl::string_view name() const override { return "forward-allocation"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  const CompilerAnnotations& annotations;
  TensorAllocationMap& tensor_allocation_map;
  TensorAllocationMap& tensor_allocation_map_second_pass;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
