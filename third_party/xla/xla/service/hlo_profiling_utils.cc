/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/hlo_profiling_utils.h"

#include <cstddef>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Separates the fingerprint from the instruction name in a cost name.
constexpr absl::string_view kCostNameSep = "::";

// The frontend attribute of the module that holds the fingerprint at `stage`.
absl::string_view FingerprintAttribute(HloModuleFingerprintStage stage) {
  switch (stage) {
    case HloModuleFingerprintStage::kBeforeScheduling:
      return "fingerprint_before_lhs";
  }
  LOG(FATAL) << "Unknown fingerprint stage " << static_cast<int>(stage);
}

std::optional<std::string> FindFingerprint(const FrontendAttributes& attributes,
                                           HloModuleFingerprintStage stage) {
  auto it = attributes.map().find(FingerprintAttribute(stage));
  if (it != attributes.map().end()) {
    return it->second;
  }
  return std::nullopt;
}

}  // namespace

std::string TagWithFingerprint(HloModule* module,
                               HloModuleFingerprintStage stage) {
  std::string fingerprint =
      module->GetFingerprint128(HloPrintOptions::Canonical()
                                    .set_print_backend_config(true)
                                    // The backend config can be a json string,
                                    // and the order of keys in json is not
                                    // guaranteed. So we need to sort the keys
                                    // to make the fingerprint deterministic.
                                    .set_sort_backend_config(true));
  absl::string_view attribute = FingerprintAttribute(stage);
  module->add_frontend_attribute(std::string(attribute), fingerprint);
  VLOG(1) << attribute << " for module " << module->name() << "("
          << module->unique_id() << ") = " << fingerprint;
  return fingerprint;
}

std::optional<std::string> GetHloModuleFingerprint(
    const HloModule& module, HloModuleFingerprintStage stage) {
  return FindFingerprint(module.frontend_attributes(), stage);
}

std::optional<std::string> GetHloModuleFingerprint(
    const HloModuleProto& module_proto, HloModuleFingerprintStage stage) {
  return FindFingerprint(module_proto.frontend_attributes(), stage);
}

std::string FingerprintedCostName(absl::string_view fingerprint,
                                  absl::string_view op_name) {
  return absl::StrCat(fingerprint, kCostNameSep, op_name);
}

std::optional<FingerprintedCostNameParts> SplitFingerprintedCostName(
    absl::string_view cost_name) {
  size_t sep = cost_name.find(kCostNameSep);
  if (sep == absl::string_view::npos) {
    return std::nullopt;
  }
  return FingerprintedCostNameParts{
      std::string(cost_name.substr(0, sep)),
      std::string(cost_name.substr(sep + kCostNameSep.size()))};
}

}  // namespace xla
