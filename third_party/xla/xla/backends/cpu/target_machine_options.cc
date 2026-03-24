/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/target_machine_options.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"  // IWYU pragma: keep
#include "llvm/TargetParser/Host.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace cpu {

namespace {

bool ValidateTargetMachineFeaturesString(absl::string_view features) {
  if (features.empty()) {
    return true;
  }
  for (const auto& feature : absl::StrSplit(features, ',')) {
    if ((!absl::StartsWith(feature, "+") && !absl::StartsWith(feature, "-")) ||
        feature.size() <= 1) {
      return false;
    }
  }
  return true;
}

void EnableFeaturesIfAVX512(std::vector<std::string>& features) {
  auto avx512_it = std::find_if(features.begin(), features.end(),
                                [](const std::string& feature) {
                                  return absl::StrContains(feature, "avx512");
                                });
  bool has_avx512 = avx512_it != features.end();
  if (!has_avx512) {
    return;
  }

  auto prefer_no_scatter_it = std::find_if(
      features.begin(), features.end(), [](const std::string& feature) {
        return absl::StrContains(feature, "prefer-no-scatter");
      });

  if (prefer_no_scatter_it == features.end()) {
    features.push_back("prefer-no-scatter");
  }

  auto prefer_no_gather_it = std::find_if(
      features.begin(), features.end(), [](const std::string& feature) {
        return absl::StrContains(feature, "prefer-no-gather");
      });

  if (prefer_no_gather_it == features.end()) {
    features.push_back("prefer-no-gather");
  }

  // Maintain sorted order.
  absl::c_sort(features);
}

std::pair<std::vector<std::string>, std::vector<std::string>>
GetEnabledAndDisabledFeatures(const std::vector<std::string>& features) {
  std::vector<std::string> enabled_features;
  std::vector<std::string> disabled_features;
  for (const auto& feature : features) {
    if (absl::StartsWith(feature, "+")) {
      enabled_features.push_back(feature.substr(1));
    } else if (absl::StartsWith(feature, "-")) {
      disabled_features.push_back(feature.substr(1));
    }
  }
  absl::c_sort(enabled_features);
  absl::c_sort(disabled_features);
  return std::make_pair(enabled_features, disabled_features);
}

}  // namespace

TargetMachineOptions::TargetMachineOptions() {
  triple_ = llvm::sys::getDefaultTargetTriple();
  cpu_ = llvm::sys::getHostCPUName();
}

TargetMachineOptions::TargetMachineOptions(const DebugOptions& debug_options) {
  triple_ = llvm::sys::getDefaultTargetTriple();
  auto xla_cpu_max_isa = CpuFeatureFromString(debug_options.xla_cpu_max_isa());
  auto detected_machine_attributes = DetectMachineAttributes(xla_cpu_max_isa);

  std::tie(enabled_features_, disabled_features_) =
      GetEnabledAndDisabledFeatures(detected_machine_attributes.features);

  // If `max_cpu_feature` is newer than the host CPU, we should keep the host
  // CPU name, e.g., we don't want to set the target CPU to Skylake when we
  // are on a Broadwell host.
  cpu_ = detected_machine_attributes.num_filtered_features
             ? CpuTargetFromMaxFeature(*xla_cpu_max_isa)
             : absl::string_view(llvm::sys::getHostCPUName());

  EnableFeaturesIfAVX512(enabled_features_);
}

TargetMachineOptions::TargetMachineOptions(absl::string_view triple,
                                           absl::string_view cpu,
                                           absl::string_view features)
    : triple_(triple), cpu_(cpu) {
  std::vector<std::string> features_vec = absl::StrSplit(features, ',');
  std::tie(enabled_features_, disabled_features_) =
      GetEnabledAndDisabledFeatures(features_vec);
  EnableFeaturesIfAVX512(enabled_features_);
}

bool TargetMachineOptions::operator==(const TargetMachineOptions& other) const {
  return triple_ == other.triple_ && cpu_ == other.cpu_ &&
         enabled_features_ == other.enabled_features_ &&
         disabled_features_ == other.disabled_features_;
}

std::vector<std::string> TargetMachineOptions::GetTargetMachineFeaturesVector()
    const {
  std::vector<std::string> all_features;
  all_features.reserve(enabled_features_.size() + disabled_features_.size());
  for (const auto& feature : enabled_features_) {
    all_features.push_back(absl::StrCat("+", feature));
  }
  for (const auto& feature : disabled_features_) {
    all_features.push_back(absl::StrCat("-", feature));
  }

  return all_features;
}

std::string TargetMachineOptions::GetTargetMachineFeatures() const {
  return absl::StrJoin(GetTargetMachineFeaturesVector(), ",");
}

TargetMachineOptionsProto TargetMachineOptions::ToProto() const {
  TargetMachineOptionsProto proto;
  proto.set_triple(triple_);
  proto.set_cpu(cpu_);
  proto.set_features(GetTargetMachineFeatures());
  return proto;
}

/*static*/
absl::StatusOr<TargetMachineOptions> TargetMachineOptions::FromProto(
    const TargetMachineOptionsProto& proto) {
  if (!ValidateTargetMachineFeaturesString(proto.features())) {
    return Internal("Invalid target machine features: %s",
                    std::string(proto.features()));
  }
  return TargetMachineOptions(proto.triple(), proto.cpu(), proto.features());
}

absl::Status TargetMachineOptions::SetFeatures(absl::string_view features) {
  if (!ValidateTargetMachineFeaturesString(features)) {
    return Internal("Trying to set invalid target machine features: %s",
                    std::string(features));
  }

  std::vector<std::string> features_vec = absl::StrSplit(features, ',');
  std::tie(enabled_features_, disabled_features_) =
      GetEnabledAndDisabledFeatures(features_vec);
  EnableFeaturesIfAVX512(enabled_features_);

  return absl::OkStatus();
}

}  // namespace cpu
}  // namespace xla
