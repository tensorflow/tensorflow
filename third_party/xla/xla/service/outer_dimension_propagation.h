#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_OUTER_DIMENSION_PROPAGATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_OUTER_DIMENSION_PROPAGATION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// HLO pass that propagates "outer-dimension multiplier" information from
// parameters marked by tf_outer_marker to subsequent instructions and writes
// the relation into instruction metadata.
class OuterDimensionPropagationPass : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "outer-dimension-propagation";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_OUTER_DIMENSION_PROPAGATION_H_