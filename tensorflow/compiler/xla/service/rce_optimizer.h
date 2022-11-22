// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_RCE_OPTIMIZER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_RCE_OPTIMIZER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Redundant code elimination (RCE).
// A pass which removes redundant instructions. The optimization
// pass removes reduce and reshape operations which add and remove
// dimensions of size one.
class RceOptimizer : public HloModulePass {
 public:
  absl::string_view name() const override { return "rce-optimizer"; }

  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_RCE_OPTIMIZER_
