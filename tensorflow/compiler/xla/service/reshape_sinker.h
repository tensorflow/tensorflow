// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_SINKER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_SINKER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Sink reshape for furthur optimisation
class ReshapeSinker : public HloModulePass {
 public:
  absl::string_view name() const override { return "reshape-sinker"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_SINKER_
