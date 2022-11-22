// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DOT_ORDER_OPTIMIZER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DOT_ORDER_OPTIMIZER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which reorders nested Dots to reduce intermidiate
// memory consumption.
class DotOrderOptimizer : public HloModulePass {
 public:
  absl::string_view name() const override { return "dot-order-optimizer"; }

  // Searches for newsted dots and reorders them
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DOT_ORDER_OPTIMIZER_
