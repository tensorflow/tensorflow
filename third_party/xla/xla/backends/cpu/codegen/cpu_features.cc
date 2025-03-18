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

#include "xla/backends/cpu/codegen/cpu_features.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringMap.h"  // IWYU pragma: keep
#include "llvm/TargetParser/Host.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

using tsl::port::CPUFeature;

// Returns the earliest CPU generation that supports the instruction set.
absl::string_view CpuTargetFromMaxFeature(CPUFeature max_feature) {
  switch (max_feature) {
    //===------------------------------------------------------------------===//
    // x86
    //===------------------------------------------------------------------===//
    case CPUFeature::SSE4_2:
      return "nehalem";
    case CPUFeature::AVX:
      return "sandybridge";
    case CPUFeature::AVX2:
      return "haswell";
    case CPUFeature::AVX512F:
      return "skylake-avx512";
    case CPUFeature::AVX512_VNNI:
      return "cascadelake";
    case CPUFeature::AVX512_BF16:
      return "cooperlake";
    case CPUFeature::AMX_BF16:
    case CPUFeature::AMX_INT8:
      return "sapphirerapids";
    case CPUFeature::AMX_FP16:
      return "graniterapids";

    //===------------------------------------------------------------------===//
    // AArch64
    //===------------------------------------------------------------------===//
    case CPUFeature::AARCH64_NEON:
      return "neoverse-n1";
    case CPUFeature::AARCH64_SVE:
      return "neoverse-v1";
    case CPUFeature::AARCH64_SVE2:
      return "neoverse-v2";

    default:
      LOG(FATAL) << "Unsupported max feature: " << max_feature;
  }
}

std::optional<CPUFeature> CpuFeatureFromString(absl::string_view cpu_feature) {
  if (cpu_feature.empty()) return std::nullopt;

  // Non-exhaustive list of CPU features. (Only the ones we care about.)
  static auto* x86 = [] {
    return new absl::flat_hash_map<std::string, CPUFeature>(
        //===--------------------------------------------------------------===//
        // x86
        //===--------------------------------------------------------------===//
        {{"SSE4_2", CPUFeature::SSE4_2},
         {"AVX", CPUFeature::AVX},
         {"AVX2", CPUFeature::AVX2},
         {"AVX512", CPUFeature::AVX512F},
         {"AVX512_VNNI", CPUFeature::AVX512_VNNI},
         {"AVX512_BF16", CPUFeature::AVX512_BF16},
         {"AMX", CPUFeature::AMX_BF16},  // Includes AMX_INT8.
         {"AMX_FP16", CPUFeature::AMX_FP16},
         //===-------------------------------------------------------------===//
         // AArch64
         //===-------------------------------------------------------------===//
         {"NEON", CPUFeature::AARCH64_NEON},
         {"SVE", CPUFeature::AARCH64_SVE},
         {"SVE2", CPUFeature::AARCH64_SVE2}});
  }();

  if (auto it = x86->find(absl::AsciiStrToUpper(cpu_feature));
      it != x86->end()) {
    return it->second;
  }

  LOG(WARNING) << "Unknown CPU ISA: " << cpu_feature;
  return std::nullopt;
}

// We deliberately opt-out of the cognitive complexity check because a giant
// switch statement is the most readable way to express the logic.
//
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static bool ShouldEnableX86CpuFeature(absl::string_view feature,
                                      CPUFeature max_feature) {
  // x86 CPUs have backward compatibility so newer CPUs have all features of
  // older CPUs. We go through switch cases from oldest features to newest.
  //   - Each case looks for features that are introduced in the next
  //     generation, i.e., features that should be disabled if `max_feature` is
  //     older or equal to the case's ISA.
  //   - We combine all features that needs to be disabled from all ISAs newer
  //     than `max_feature` by falling through cases.
  //
  // For example, if `max_feature` is AVX2, we start by disabling
  // AVX512-generation features in the AVX2 case, then fall through to the
  // AVX512 case to disable next-gen features (AVX512_VNNI), etc, all the way
  // down to the newest one.
  switch (max_feature) {
    case CPUFeature::SSE4_2:
      if (absl::StartsWith(feature, "avx") || feature == "f16c" ||
          feature == "vpclmulqdq" || feature == "vaes") {
        return false;
      }
      [[fallthrough]];

    case CPUFeature::AVX:
      if (absl::StartsWith(feature, "avx2") ||
          absl::StartsWith(feature, "fma")) {
        return false;
      }
      [[fallthrough]];

    case CPUFeature::AVX2:
      if (absl::StartsWith(feature, "avx512") || feature == "evex512")
        return false;
      [[fallthrough]];

    case CPUFeature::AVX512F:
      if (feature == "avx512vnni") return false;
      [[fallthrough]];

    case CPUFeature::AVX512_VNNI:
      if (feature == "avx512bf16") return false;
      [[fallthrough]];

    case CPUFeature::AVX512_BF16:
      if (absl::StartsWith(feature, "amx")) return false;
      [[fallthrough]];

    case CPUFeature::AMX_INT8:
    case CPUFeature::AMX_BF16:
      if (feature == "amx-fp16") return false;
      [[fallthrough]];

    default:
      // Leave all other features enabled.
      return true;
  }
}

static bool ShouldEnableAArch64CpuFeature(absl::string_view feature,
                                          CPUFeature max_feature) {
  // AArch64 CPUs have backward compatibility so newer CPUs have all features
  // of older CPUs. We go through switch cases from oldest features to newest.
  //   - Each case looks for features that are introduced in the next
  //     generation, i.e., features that should be disabled if `max_feature` is
  //     older or equal to the case's ISA.
  //   - We combine all features that needs to be disabled from all ISAs newer
  //     than `max_feature` by falling through cases.
  switch (max_feature) {
    case CPUFeature::AARCH64_NEON:
      if (feature == "sve") return false;
      [[fallthrough]];

    case CPUFeature::AARCH64_SVE:
      if (feature == "sve2") return false;
      [[fallthrough]];

    default:
      // Leave all other features enabled.
      return true;
  }
}

bool ShouldEnableCpuFeature(absl::string_view feature, CPUFeature max_feature) {
  if constexpr (tsl::port::IsX86CPU()) {
    return ShouldEnableX86CpuFeature(feature, max_feature);
  } else if constexpr (tsl::port::IsAarch64CPU()) {
    return ShouldEnableAArch64CpuFeature(feature, max_feature);
  }
  return true;
}

DetectedMachineAttributes DetectMachineAttributes(
    std::optional<CPUFeature> max_feature) {
  DetectedMachineAttributes result;
  // We only have x86 constraints. Skip the check if we are on non-x86 CPUs.
  bool no_feature_constraint =
      !max_feature.has_value() ||
      !(tsl::port::IsX86CPU() || tsl::port::IsAarch64CPU());
  for (const auto& [feature, enabled] : llvm::sys::getHostCPUFeatures()) {
    bool should_enable =
        enabled && (no_feature_constraint ||
                    ShouldEnableCpuFeature(feature, *max_feature));
    result.features.push_back(
        absl::StrCat(should_enable ? "+" : "-", std::string(feature)));
    result.num_filtered_features += (should_enable != enabled);
  }
  absl::c_sort(result.features);
  return result;
}

std::vector<std::string> DetectMachineAttributes() {
  return DetectMachineAttributes(std::nullopt).features;
}

}  // namespace xla::cpu
