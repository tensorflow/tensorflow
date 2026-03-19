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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_COMPUTE_CAPABILITY_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_COMPUTE_CAPABILITY_H_

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_description.pb.h"

namespace stream_executor {

// ROCm compute capability, as reported by the device description.
class RocmComputeCapability {
 public:
  // gcn_arch_name example --  gfx90a:sramecc+:xnack-
  // gfx_version is the "gfx90a" part of the gcn_arch_name
  explicit RocmComputeCapability(std::string gcn_arch_name)
      : gcn_arch_name_(std::move(gcn_arch_name)) {}

  explicit RocmComputeCapability(const RocmComputeCapabilityProto& proto)
      : gcn_arch_name_(proto.gcn_arch_name()) {}

  RocmComputeCapability() = default;

  static RocmComputeCapability EarliestCDNASupport() {
    return RocmComputeCapability{"gfx908"};
  }

  static RocmComputeCapability EarliestRDNASupport() {
    return RocmComputeCapability{"gfx1030"};
  }

  const std::string& gcn_arch_name() const { return gcn_arch_name_; }

  std::string ToString() const { return gcn_arch_name(); }

  RocmComputeCapabilityProto ToProto() const {
    RocmComputeCapabilityProto proto;
    proto.set_gcn_arch_name(gcn_arch_name_);
    return proto;
  }

  static RocmComputeCapability FromProto(
      const RocmComputeCapabilityProto& proto) {
    return RocmComputeCapability{proto.gcn_arch_name()};
  }

  bool operator==(const RocmComputeCapability& other) const {
    return gcn_arch_name_ == other.gcn_arch_name_;
  }

  bool operator!=(const RocmComputeCapability& other) const {
    return !this->operator==(other);
  }

  std::string gfx_version() const {
    //  std::strchr() is faster for the case than std::string::find()
    const char* const p_colon = std::strchr(gcn_arch_name_.c_str(), ':');
    if (nullptr == p_colon) {
      return gcn_arch_name_;  // likely it's the default invalid value
    }
    return std::string(gcn_arch_name_.c_str(), p_colon);
  }

  // note, while there's no particular reason to make the lists public, it won't
  // hurt since they are immutable, but keeping them close to methods simplifies
  // maintanance.
  static constexpr absl::string_view kSupportedGfxVersions[]{
      "gfx900",   // MI25
      "gfx906",   // MI50 / MI60
      "gfx908",   // MI100
      "gfx90a",   // MI200
      "gfx942",   // MI300
      "gfx950",   // MI350
      "gfx1030",  // RX68xx / RX69xx
      "gfx1100",  // RX7900
      "gfx1101",  // RX7700 / RX7800
      "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201"};

  bool is_supported_gfx_version() const {
    return IsThisGfxInAnyList(kSupportedGfxVersions);
  }

  std::string supported_gfx_versions_str() const {
    return absl::StrJoin(kSupportedGfxVersions, ", ");
  }

  bool gfx9_mi100() const { return gfx_version() == "gfx908"; }

  static constexpr absl::string_view kMI100Series[] = {"gfx908"};

  bool gfx9_mi200() const { return gfx_version() == "gfx90a"; }

  static constexpr absl::string_view kMI200Series[] = {"gfx90a"};

  bool gfx9_mi300() const { return gfx_version() == "gfx942"; }

  bool gfx9_mi350() const { return gfx_version() == "gfx950"; }

  static constexpr absl::string_view kMI300Series[] = {"gfx942", "gfx950"};
  bool gfx9_mi300_series() const { return IsThisGfxInAnyList(kMI300Series); }

  bool gfx9_mi100_or_later() const {
    return IsThisGfxInAnyList(kMI300Series, kMI200Series, kMI100Series);
  }

  bool gfx9_mi200_or_later() const {
    return IsThisGfxInAnyList(kMI300Series, kMI200Series);
  }

  bool gfx10_rx68xx() const { return gfx_version() == "gfx1030"; }

  bool gfx10_rx69xx() const { return gfx_version() == "gfx1030"; }

  bool gfx11() const { return absl::StartsWith(gfx_version(), "gfx11"); }

  static constexpr absl::string_view kGfx11Discrete[] = {"gfx1100", "gfx1101"};
  bool gfx11_discrete() const { return IsThisGfxInAnyList(kGfx11Discrete); }

  static constexpr absl::string_view kGfx11Apu[] = {"gfx1103", "gfx1150",
                                                    "gfx1151"};
  bool gfx11_apu() const { return IsThisGfxInAnyList(kGfx11Apu); }

  static constexpr absl::string_view kGfx11Rx7900[] = {"gfx1100", "gfx1101",
                                                       "gfx1102"};
  bool gfx11_rx7900() const {
    // TODO(AMD/TF): instead of this, other gfx11*() methods might be better
    return IsThisGfxInAnyList(kGfx11Rx7900);
  }

  bool gfx12() const { return absl::StartsWith(gfx_version(), "gfx12"); }

  static constexpr absl::string_view kGfx12Discrete[] = {"gfx1200", "gfx1201"};
  bool gfx12_discrete() const { return IsThisGfxInAnyList(kGfx12Discrete); }

  bool gfx12_rx8900() const { return gfx12_discrete(); }

  bool has_nhwc_layout_support() const { return gfx9_mi100_or_later(); }

  bool has_bf16_dtype_support() const {
    return gfx9_mi100_or_later() || gfx12() || gfx11();
  }

  bool has_fast_fp16_support() const {
    return gfx9_mi100_or_later() || gfx11() || gfx10_rx68xx() || gfx10_rx69xx();
  }

  bool has_mfma_instr_support() const { return gfx9_mi100_or_later(); }

  bool has_amd_matrix_instr() const {
    return gfx9_mi100_or_later() || gfx12() || gfx11();
  }

  bool has_packed_fp16_atomics_support() const { return gfx9_mi100_or_later(); }

  bool has_packed_bf16_atomics_support() const { return gfx9_mi300_series(); }

  bool fence_before_barrier() const {
    static constexpr absl::string_view kList[] = {"gfx900", "gfx906"};
    return !IsThisGfxInAnyList(kList);
  }

  bool has_hipblaslt() const {
    return IsThisGfxInAnyList(kMI300Series, kMI200Series, kGfx12Discrete,
                              kGfx11Discrete, kGfx11Apu);
  }

  bool has_fp8_support() const {
    return has_ocp_fp8_support() || has_nanoo_fp8_support();
  }

  bool has_ocp_fp8_support() const { return gfx9_mi350() || gfx12_discrete(); }

  bool has_nanoo_fp8_support() const { return gfx9_mi300(); }

  bool has_mx_type_support() const { return gfx9_mi350(); }

  int threads_per_warp() const { return gfx9_mi100_or_later() ? 64 : 32; }

  /// \brief Invalid gfx id for default gcn_arch_name_ value and testing
  static constexpr absl::string_view kInvalidGfx = "gfx000";

 private:
  /// \brief Takes one or more arrays of string-like objects and tests if the
  /// result of `gfx_version()` matches to any string in any of the arrays.
  template <typename... ArrayOfStrings>
  bool IsThisGfxInAnyList(ArrayOfStrings&&... arr) const {
    static_assert(sizeof...(arr) >= 1);
    const std::string gfx = gfx_version();
    return (implIsThisGfxInAnyList(std::begin(arr), std::end(arr), gfx) || ...);
  }

  /// \brief Template-less implementation of IsThisGfxInAnyList().
  /// \warning Don't use directly!
  bool implIsThisGfxInAnyList(const absl::string_view* beg,
                              const absl::string_view* end,
                              const absl::string_view gfx) const {
    return std::any_of(
        beg, end, [&gfx = gfx](const absl::string_view s) { return gfx == s; });
  }

  std::string gcn_arch_name_{kInvalidGfx};  // default to invalid arch.
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_COMPUTE_CAPABILITY_H_
