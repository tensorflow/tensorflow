// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_EUCLIDEAN_DISTANCE_REWRITER_
#define TENSORFLOW_COMPILER_XLA_SERVICE_EUCLIDEAN_DISTANCE_REWRITER_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which applies euclidean distance optimization
class EuclideanDistanceRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "euclidean-distance-rewriter";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_EUCLIDEAN_DISTANCE_REWRITER_
