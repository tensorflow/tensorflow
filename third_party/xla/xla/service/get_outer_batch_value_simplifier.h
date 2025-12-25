#ifndef XLA_SERVICE_CPU_GET_OUTER_BATCH_VALUE_SIMPLIFIER_H_
#define XLA_SERVICE_CPU_GET_OUTER_BATCH_VALUE_SIMPLIFIER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class GetOuterBatchValueSimplifier : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "get_outer_batch_value_simplifier";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads
  ) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_CPU_GET_OUTER_BATCH_VALUE_SIMPLIFIER_H_