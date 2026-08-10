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

#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.pb.h"

namespace stream_executor {

// ==============================================================================
// Battlemage (BMG) variant codec
// ==============================================================================
template <>
struct OneAPIComputeCapability::OneAPIDeviceVariantOps<OneAPIDeviceType::kBMG> {
  static bool IsKnown(uint32_t version) { return (version - 1) < 2; }

  static std::string Decode(uint32_t version) {
    return absl::StrCat("G", version * 10 + 11);
  }

  static uint32_t Encode(absl::string_view variant) {
    int value;
    if (variant.length() < 2 || variant[0] != 'G' ||
        !absl::SimpleAtoi(variant.substr(1), &value)) {
      return OneAPIComputeCapability::kUnknownVariantValue;
    }
    return (value - 11) / 10;
  }
};

// ==============================================================================
// Ponte Vecchio (PVC) variant codec
// ==============================================================================
template <>
struct OneAPIComputeCapability::OneAPIDeviceVariantOps<OneAPIDeviceType::kPVC> {
  static bool IsKnown(uint32_t version) { return (version & 0xfe) == 0x3c; }

  static std::string Decode(uint32_t version) {
    return (version == 0x3d) ? "VG" : "";
  }

  static uint32_t Encode(absl::string_view variant) {
    constexpr uint32_t kDefaultVersion = 0x3c;
    return variant.empty()     ? kDefaultVersion
           : (variant == "VG") ? (kDefaultVersion + 1)
                               : OneAPIComputeCapability::kUnknownVariantValue;
  }
};

// ==============================================================================
// Alchemist (DG2) variant codec
// ==============================================================================
template <>
struct OneAPIComputeCapability::OneAPIDeviceVariantOps<OneAPIDeviceType::kDG2> {
  static bool IsKnown(uint32_t version) {
    return version == 0x37 || (version & 0xfe) == 0x38;
  }

  static std::string Decode(uint32_t version) {
    return absl::StrCat("G", version - 45);
  }

  static uint32_t Encode(absl::string_view variant) {
    int value;
    if (!absl::SimpleAtoi(variant.substr(1), &value)) {
      return OneAPIComputeCapability::kUnknownVariantValue;
    }
    return value + 45;
  }
};

template <std::size_t... I>
OneAPIDeviceType OneAPIComputeCapability::InferDeviceType(
    uint32_t generation, uint32_t version, std::index_sequence<I...>) {
  OneAPIDeviceType resolved = OneAPIDeviceType::kUnknown;
  ([&] {
    constexpr OneAPIDeviceType type = kKnownDevices[I].type;
    constexpr uint32_t arch_val = (static_cast<uint32_t>(type) >> 22) & 0x3ff;
    if (generation == arch_val &&
        OneAPIDeviceVariantOps<type>::IsKnown(version)) {
      resolved = type;
      return true;
    }
    return false;
  }() ||
   ...);  // NOLINT(whitespace/indent)
  return resolved;
}

template <std::size_t... I>
uint32_t OneAPIComputeCapability::ApplyVariantToIpVersion(
    absl::string_view variant, uint32_t ip_version, OneAPIDeviceType type,
    std::index_sequence<I...>) {
  uint32_t encoded_device = 0;
  ([&] {
    constexpr auto T = kKnownDevices[I].type;
    if (type == T) {
      encoded_device =
          (ip_version & 0xffc00000) |
          (OneAPIDeviceVariantOps<T>::Encode(absl::AsciiStrToUpper(variant))
           << 14);
      return true;
    }
    return false;
  }() ||
   ...);  // NOLINT(whitespace/indent)
  return encoded_device;
}

template <std::size_t... I>
std::string OneAPIComputeCapability::VariantStringForDevice(
    std::index_sequence<I...>) const {
  std::string result;
  ([&] {
    constexpr auto T = kKnownDevices[I].type;
    if (device_ == T) {
      result = OneAPIDeviceVariantOps<T>::Decode(version_);
      return true;
    }
    return false;
  }() ||
   ...);  // NOLINT(whitespace/indent)
  return result;
}

OneAPIComputeCapability::OneAPIComputeCapability(uint32_t generation,
                                                 uint32_t version)
    : generation_(generation),
      version_(version),
      device_(InferDeviceType(
          generation, version,
          std::make_index_sequence<std::size(kKnownDevices)>{})) {}

OneAPIComputeCapability::OneAPIComputeCapability(absl::string_view arch,
                                                 absl::string_view variant)
    : OneAPIComputeCapability([&]() -> uint32_t {
        OneAPIDeviceType type = NameToDeviceType(arch);
        auto ip_version = static_cast<uint32_t>(type);
        if ((ip_version & 0x3fff) >= std::size(kKnownDevices)) {
          return 0;
        }
        if (variant.empty()) {
          return ip_version;
        }
        return ApplyVariantToIpVersion(
            variant, ip_version, type,
            std::make_index_sequence<std::size(kKnownDevices)>{});
      }()) {}

std::string OneAPIComputeCapability::architecture() const {
  for (const auto& info : kKnownDevices) {
    if (device_ == info.type) {
      return info.name;
    }
  }
  if (generation_ == 0) {
    return "Unknown";
  }
  return absl::StrCat("Xe", generation_ / 10,
                      (generation_ % 10 == 5) ? "P" : "");
}

std::string OneAPIComputeCapability::variant() const {
  return VariantStringForDevice(
      std::make_index_sequence<std::size(kKnownDevices)>{});
}

OneAPIComputeCapability OneAPIComputeCapability::FromProto(
    const OneAPIComputeCapabilityProto& proto) {
  return OneAPIComputeCapability{proto.architecture(), proto.variant()};
}

OneAPIComputeCapabilityProto OneAPIComputeCapability::ToProto() const {
  OneAPIComputeCapabilityProto proto;
  proto.set_architecture(architecture());
  proto.set_variant(variant());
  return proto;
}

std::string OneAPIComputeCapability::ToString() const {
  const std::string& variant_string = variant();
  return absl::StrCat(architecture(), variant_string.empty() ? "" : "_",
                      variant_string);
}

}  // namespace stream_executor
