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
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"

namespace stream_executor {

// This type represents either
// - a compute capability of a NVIDIA GPU.
// - a compilation target compute capability.
//
// A CUDA compute capability is a pair of major and minor versions, e.g. 8.0,
// 9.2, 10.1 plus potentially a feature extension, e.g. 9.0a, 10.0f.
//
// Feature extensions only make sense when talking about a compilation target
// compute capability since they restrict on which GPUs the generated code can
// be run.
//
// 1. The accelerated feature extension "a":
//   Kernels compiled with this extension can only be run on GPUs with the same
//   compute capability. For example, a sm_90a kernel can run on a sm_90 GPU,
//   but not on a sm_91 GPU.
//
// 2. The forward compatible feature extension "f":
//   Kernels compiled with this extension can only be run on GPUs with the same
//   major version and a later or same minor version. For example, a sm_100f
//   kernel can run on a sm_100 or sm_103 GPUs, but not on a sm_120 GPU.
struct CudaComputeCapability {
  int major = 0;
  int minor = 0;

  enum class FeatureExtension : uint8_t {
    kNone,  // No additional features - Generated PTX will run on all GPUs with
            // a higher compute capability. Example: sm_90
    kAcceleratedFeatures,  // Enables features that only work on GPUs with the
                           // same compute capability. Example: sm_90a
    kFamilyCompatibleFeatures  // Enables features that only work on GPUs
                               // within the same major version and a later
                               // minor version. Example: sm_100f
  };
  FeatureExtension feature_extension = FeatureExtension::kNone;

  // MSVC does not like "PASCAL" symbol.
  enum CudaComputeCapabilities {
    kPascal = 6,
    kVolta = 7,
    kAmpere = 8,
    kHopper = 9,
    kBlackwell = 10,
    kBlackwell_11 = 11,
    kBlackwell_12 = 12
  };

  constexpr CudaComputeCapability() = default;
  constexpr CudaComputeCapability(int major, int minor)
      : CudaComputeCapability(major, minor, FeatureExtension::kNone) {}

  constexpr CudaComputeCapability(int major, int minor,
                                  FeatureExtension feature_extension)
      : major{major}, minor{minor}, feature_extension{feature_extension} {}

  static absl::StatusOr<CudaComputeCapability> FromProto(
      const CudaComputeCapabilityProto& proto);

  // Parses the architecture name in the format
  // "major.minor<feature_extension>", example: "8.6" or "9.0a" or "10.0f".
  static absl::StatusOr<CudaComputeCapability> FromString(
      absl::string_view cuda_arch_name);

  constexpr static CudaComputeCapability Pascal() {
    return CudaComputeCapability{kPascal, 0};
  }

  constexpr static CudaComputeCapability Volta() {
    return CudaComputeCapability{kVolta, 0};
  }

  constexpr static CudaComputeCapability Ampere() {
    return CudaComputeCapability{kAmpere, 0};
  }

  // Includes all GPUs with compute capability 9.0, notably H100, H200, and
  // GH200. When comparing with `IsAtLeast` this will only be true for GPUs with
  // compute capability 9.0.
  constexpr static CudaComputeCapability H100Accelerated() {
    return CudaComputeCapability{kHopper, 0,
                                 FeatureExtension::kAcceleratedFeatures};
  }

  // Includes all GPUs with compute capability 9.x. When comparing with
  // `IsAtLeast` this will return true for all compute capabilities of at
  // least 9.0.
  constexpr static CudaComputeCapability Hopper() {
    return CudaComputeCapability{kHopper, 0, FeatureExtension::kNone};
  }

  // Includes all GPUs with compute capability 10.0, notably B200 and GB200.
  // When comparing with `IsAtLeast` this will only be true for GPUs with
  // compute capability 10.0.
  constexpr static CudaComputeCapability B200Accelerated() {
    return CudaComputeCapability{kBlackwell, 0,
                                 FeatureExtension::kAcceleratedFeatures};
  }

  // Includes all GPUs with compute capability 10.x. When comparing with
  // `IsAtLeast` this will true for all compute capabilities of 10.0 or higher.
  constexpr static CudaComputeCapability Blackwell() {
    return CudaComputeCapability{kBlackwell, 0, FeatureExtension::kNone};
  }

  // Includes all GPUs with compute capability 10.x. When comparing with
  // `IsAtLeast` this will true for all 10.x compute capabilities but not for
  // compute capabilities with a higher major version.
  constexpr static CudaComputeCapability BlackwellFamily() {
    return CudaComputeCapability{kBlackwell, 0,
                                 FeatureExtension::kFamilyCompatibleFeatures};
  }

  // Returns true if the compute capability is at least
  // `other_major.other_minor`. It is equivalent to
  // this->SupportsAllFeaturesOf(CudaComputeCapability{other_major,
  // other_minor}).
  bool IsAtLeast(int other_major, int other_minor = 0) const {
    return SupportsAllFeaturesOf(
        CudaComputeCapability{other_major, other_minor});
  }

  bool IsAtLeastPascal() const {
    return major >= CudaComputeCapabilities::kPascal;
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

  bool IsPascal() const { return major == CudaComputeCapabilities::kPascal; }

  bool IsVolta() const { return major == CudaComputeCapabilities::kVolta; }

  bool IsAmpere() const { return major == CudaComputeCapabilities::kAmpere; }

  bool IsAda() const {
    constexpr int kAdaMinor = 9;
    return major == CudaComputeCapabilities::kAmpere && minor == kAdaMinor;
  }

  bool IsHopper() const { return major == CudaComputeCapabilities::kHopper; }

  bool IsBlackwell() const {
    return major == CudaComputeCapabilities::kBlackwell;
  }

  // Returns true if a kernel compiled for compute capability `other` can be run
  // on a GPU with compute capability `this`.
  bool SupportsAllFeaturesOf(const CudaComputeCapability& other) const {
    switch (other.feature_extension) {
      case FeatureExtension::kNone:
        return std::tie(major, minor) >= std::tie(other.major, other.minor);
      case FeatureExtension::kAcceleratedFeatures:
        return std::tie(major, minor) == std::tie(other.major, other.minor);
      case FeatureExtension::kFamilyCompatibleFeatures:
        return major == other.major && minor >= other.minor;
    }
  }

  // Returns true if a kernel compiled for compute capability `this` can be run
  // on a GPU with compute capability `other`.
  bool CanRunOn(const CudaComputeCapability& other) const {
    return other.SupportsAllFeaturesOf(*this);
  }

  // Returns a copy of this compute capability without any feature extension
  // set.
  CudaComputeCapability WithoutAnyFeatureExtension() const {
    return CudaComputeCapability{major, minor, FeatureExtension::kNone};
  }

  // Returns a string representation of the compute capability. The format is
  // not guaranteed to follow any standard and should only be used for logging.
  std::string ToString() const;

  friend bool operator==(const CudaComputeCapability& lhs,
                         const CudaComputeCapability& rhs) {
    return std::tie(lhs.major, lhs.minor, lhs.feature_extension) ==
           std::tie(rhs.major, rhs.minor, rhs.feature_extension);
  }

  friend bool operator!=(const CudaComputeCapability& lhs,
                         const CudaComputeCapability& rhs) {
    return !(lhs == rhs);
  }

  CudaComputeCapabilityProto ToProto() const;

  template <typename H>
  friend H AbslHashValue(H state, const CudaComputeCapability& cc) {
    return H::combine(std::move(state), cc.major, cc.minor,
                      cc.feature_extension);
  }

  // Represents the compile mode as it can be passed to tools like NVCC,
  // ptxas, or nvprune.
  enum class CompileMode : uint8_t {
    kPtx,  // This means nvcc/ptxas will generate PTX for the given compute
           // architecture.
    kLto,  // This means nvcc/ptxas will generate NVVM-IR for Link Time
           // Optimization.
    kSass  // This means nvcc/ptxas will generate SASS for the given compute
           // architecture.
  };

  // Returns the target identifier that can be passed to ptxas's `--gpu-name`
  // option.
  std::string GetPtxAsTargetName(
      CompileMode compile_mode = CompileMode::kSass) const;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMPUTE_CAPABILITY_H_
