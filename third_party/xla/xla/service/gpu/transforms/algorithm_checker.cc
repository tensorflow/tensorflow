/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/algorithm_checker.h"

#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/algorithm_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

namespace {

bool HasNonDefaultOperandPrecision(const PrecisionConfig& config) {
  return absl::c_any_of(config.operand_precision(), [](int precision) {
    return static_cast<PrecisionConfig::Precision>(precision) !=
           PrecisionConfig::DEFAULT;
  });
}

class AlgorithmCheckerVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  explicit AlgorithmCheckerVisitor(
      se::GpuComputeCapability gpu_compute_capability)
      : gpu_compute_capability_(std::move(gpu_compute_capability)) {}

  absl::Status RunOnModule(
      const HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {}) {
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      TF_RETURN_IF_ERROR(computation->Accept(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleDot(const HloInstruction* hlo) override {
    VLOG(1) << "Handling dot: " << hlo->ToString();
    const PrecisionConfig& config = hlo->precision_config();

    if (config.algorithm() != PrecisionConfig::ALG_UNSET &&
        HasNonDefaultOperandPrecision(config)) {
      LOG(WARNING)
          << "There is no need to set precisions when we set the algorithm: "
          << hlo->ToString();
    }

    if (config.algorithm() == PrecisionConfig::ALG_UNSET) {
      return absl::OkStatus();
    }

    PrimitiveType lhs_storage_type = hlo->operand(0)->shape().element_type();
    PrimitiveType rhs_storage_type = hlo->operand(1)->shape().element_type();
    PrimitiveType output_storage_type = hlo->shape().element_type();

    if (lhs_storage_type != rhs_storage_type) {
      return absl::UnimplementedError(absl::StrFormat(
          "Dot operands must have the same type when using an algorithm: %s",
          hlo->ToString()));
    }

    return algorithm_util::IsSupportedDotAlgorithmOnGpu(
               config.algorithm(), gpu_compute_capability_, lhs_storage_type,
               output_storage_type)
               ? absl::OkStatus()
               : absl::UnimplementedError(absl::StrFormat(
                     "Unsupported algorithm on the current device(s): %s",
                     PrecisionConfig::Algorithm_Name(config.algorithm())));
  }

  absl::Status DefaultAction(const HloInstruction* hlo) override {
    return absl::OkStatus();
  }

 private:
  se::GpuComputeCapability gpu_compute_capability_;
};

}  // namespace

absl::StatusOr<bool> AlgorithmChecker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(AlgorithmCheckerVisitor(gpu_compute_capability_)
                         .RunOnModule(module, execution_threads));
  // No change was made.
  return false;
}

}  // namespace gpu
}  // namespace xla
