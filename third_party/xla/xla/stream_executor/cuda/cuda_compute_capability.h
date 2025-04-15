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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_COMPUTE_CAPABILITY_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_COMPUTE_CAPABILITY_H_

#include <cassert>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"

namespace stream_executor {

// CUDA compute capability, as reported by the device description.
struct CudaComputeCapability {
  int major = 0;
  int minor = 0;

  // MSVC does not like "PASCAL" symbol.
  enum CudaComputeCapabilities {
    kPascal = 6,
    kVolta = 7,
    kAmpere = 8,
    kHopper = 9,
    kBlackwell = 10
  };

  constexpr CudaComputeCapability() = default;
  constexpr CudaComputeCapability(int major, int minor) {
    this->major = major;
    this->minor = minor;
  }

  // Parses the architecture name in the format "major.minor", example: "8.6".
  static absl::StatusOr<CudaComputeCapability> FromString(
      absl::string_view cuda_arch_name);

  explicit CudaComputeCapability(const CudaComputeCapabilityProto &proto) {
    this->major = proto.major();
    this->minor = proto.minor();
  }

  static CudaComputeCapability Volta() {
    return CudaComputeCapability{kVolta, 0};
  }

  static CudaComputeCapability Ampere() {
    return CudaComputeCapability{kAmpere, 0};
  }

  static CudaComputeCapability Hopper() {
    return CudaComputeCapability{kHopper, 0};
  }

  static CudaComputeCapability Blackwell() {
    return CudaComputeCapability{kBlackwell, 0};
  }

  bool IsAtLeast(int other_major, int other_minor = 0) const {
    return IsAtLeast(CudaComputeCapability{other_major, other_minor});
  }

  bool IsAtLeast(const CudaComputeCapability &cc) const {
    return !(*this < cc);
  }

  bool IsAtLeastVolta() const {
    return major >= CudaComputeCapabilities::kVolta;
  }

  bool IsAtLeastAmpere() const {
    return major >= CudaComputeCapabilities::kAmpere;
  }

  bool IsAtLeastAda() const { return IsAtLeast(8, 9); }

  bool IsAtLeastHopper() const {
    return major >= CudaComputeCapabilities::kHopper;
  }

  bool IsAtLeastBlackwell() const {
    return major >= CudaComputeCapabilities::kBlackwell;
  }

  bool IsAmpere() const { return major == CudaComputeCapabilities::kAmpere; }

  bool IsHopper() const { return major == CudaComputeCapabilities::kHopper; }

  bool IsBlackwell() const {
    return major == CudaComputeCapabilities::kBlackwell;
  }

  bool operator<(const CudaComputeCapability &other) const {
    return ToPair() < other.ToPair();
  }

  bool operator==(const CudaComputeCapability &other) const {
    return ToPair() == other.ToPair();
  }

  bool operator!=(const CudaComputeCapability &other) const {
    return !(*this == other);
  }

  bool operator>(const CudaComputeCapability &other) const {
    return ToPair() > other.ToPair();
  }

  bool operator>=(const CudaComputeCapability &other) const {
    return ToPair() >= other.ToPair();
  }

  bool operator<=(const CudaComputeCapability &other) const {
    return ToPair() <= other.ToPair();
  }

  std::string ToString() const { return absl::StrCat(major, ".", minor); }

  std::pair<int, int> ToPair() const { return std::make_pair(major, minor); }

  CudaComputeCapabilityProto ToProto() const {
    CudaComputeCapabilityProto proto;
    proto.set_major(major);
    proto.set_minor(minor);
    return proto;
  }

  template <typename H>
  friend H AbslHashValue(H state, const CudaComputeCapability &cc) {
    return H::combine(std::move(state), cc.major, cc.minor);
  }
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMPUTE_CAPABILITY_H_
