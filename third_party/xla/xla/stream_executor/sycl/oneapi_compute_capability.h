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

#ifndef XLA_STREAM_EXECUTOR_SYCL_ONEAPI_COMPUTE_CAPABILITY_H_
#define XLA_STREAM_EXECUTOR_SYCL_ONEAPI_COMPUTE_CAPABILITY_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.pb.h"

namespace stream_executor {

#define FOR_KNOWN_ONEAPI_DEVICES(EMIT)                               \
  EMIT(/*device=*/PVC, /*idx=*/0, /*gen=*/0xc, /*default_ver=*/0x3c) \
  EMIT(/*device=*/BMG, /*idx=*/1, /*gen=*/0x14, /*default_ver=*/0x1) \
  EMIT(/*device=*/DG2, /*idx=*/2, /*gen=*/0xc, /*default_ver=*/0x37)

#define DEVICE_INFO(HW, ...) {#HW, OneAPIDeviceType::k##HW},
#define DEVICE_ENTRIES(HW, I, G, V) k##HW = ((G << 22) | (V << 14) | (I)),

// Convenience utilities for each known device family. For example, PVC()
// returns the compute capability for PVC variants and IsPVC() checks whether
// this capability corresponds to a PVC variant.
#define DEVICE_UTILITIES(HW, ...)                                              \
  static OneAPIComputeCapability HW() { return OneAPIComputeCapability{#HW}; } \
  bool Is##HW() const { return device_ == OneAPIDeviceType::k##HW; }

// Each enum entry encodes the generation, version, and index of a given device
enum class OneAPIDeviceType : uint32_t {
  FOR_KNOWN_ONEAPI_DEVICES(DEVICE_ENTRIES) kUnknown
};

class OneAPIComputeCapability {
 public:
  OneAPIComputeCapability() = default;

  explicit OneAPIComputeCapability(uint32_t generation, uint32_t version);

  explicit OneAPIComputeCapability(const OneAPIComputeCapabilityProto& proto)
      : OneAPIComputeCapability(FromProto(proto)) {}

  // Device's architecture and variant are encoded in an IP version. The 10 most
  // significant bits represent the architecture, the next 8 bits represent the
  // architecture variant, and the 14 least significant bits hold the
  // architecture family index. The IP version can be queried via Level Zero
  // calls.
  explicit OneAPIComputeCapability(const uint32_t ip_version)
      : OneAPIComputeCapability((ip_version >> 22) & 0x3ff,
                                (ip_version >> 14) & 0xff) {}

  explicit OneAPIComputeCapability(absl::string_view arch,
                                   absl::string_view variant = "");

  uint32_t generation() const { return generation_; }
  uint32_t version() const { return version_; }

  std::string architecture() const;

  std::string variant() const;

  std::string ToString() const;

  FOR_KNOWN_ONEAPI_DEVICES(DEVICE_UTILITIES)

  OneAPIComputeCapabilityProto ToProto() const;

  static OneAPIComputeCapability FromProto(
      const OneAPIComputeCapabilityProto& proto);

  bool operator==(const OneAPIComputeCapability& other) const {
    return generation_ == other.generation_ && version_ == other.version_;
  }

  bool operator!=(const OneAPIComputeCapability& other) const {
    return !this->operator==(other);
  }

 private:
  struct OneAPIDeviceInfo {
    const char* name;
    OneAPIDeviceType type;
  };

  // Device-specific variant codec operations (encode, decode, validate).
  template <OneAPIDeviceType device>
  struct OneAPIDeviceVariantOps {
    static bool IsKnown(uint32_t /*version*/) { return false; }
    static std::string Decode(uint32_t /*version*/) { return ""; }
    static uint32_t Encode(absl::string_view /*variant*/) { return 0; }
  };

  static constexpr uint32_t kUnknownVariantValue = 0xff;
  static constexpr OneAPIDeviceInfo kKnownDevices[]{
      FOR_KNOWN_ONEAPI_DEVICES(DEVICE_INFO)};

  uint32_t generation_ = 0;
  uint32_t version_ = 0;
  OneAPIDeviceType device_ = OneAPIDeviceType::kUnknown;

  // Helpers to dispatch over kKnownDevices using index_sequence.
  // TODO(intel-tf): These private helpers can be abstracted into templated
  // lambdas once C++20 is available.
  template <std::size_t... I>
  OneAPIDeviceType InferDeviceType(uint32_t generation, uint32_t version,
                                   std::index_sequence<I...>);

  template <std::size_t... I>
  static uint32_t ApplyVariantToIpVersion(absl::string_view variant,
                                          uint32_t ip_version,
                                          OneAPIDeviceType type,
                                          std::index_sequence<I...>);

  template <std::size_t... I>
  std::string VariantStringForDevice(std::index_sequence<I...>) const;

  // Maps a device name string to its corresponding OneAPIDeviceType.
  static OneAPIDeviceType NameToDeviceType(absl::string_view name) {
    for (const auto& info : kKnownDevices) {
      if (absl::EqualsIgnoreCase(name, info.name)) {
        return info.type;
      }
    }
    return OneAPIDeviceType::kUnknown;
  }
};

#undef DEVICE_UTILITIES
#undef DEVICE_ENTRIES
#undef DEVICE_INFO
#undef FOR_KNOWN_ONEAPI_DEVICES

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_ONEAPI_COMPUTE_CAPABILITY_H_
