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

#include "xla/stream_executor/cuda/cuda_compute_capability.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"

namespace stream_executor {

absl::StatusOr<CudaComputeCapability> CudaComputeCapability::FromString(
    absl::string_view cuda_arch_name) {
  std::vector<absl::string_view> split = absl::StrSplit(cuda_arch_name, '.');
  if (split.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid CUDA architecture name: ", cuda_arch_name));
  }

  FeatureExtension feature_extension = FeatureExtension::kNone;
  if (!split[1].empty() && (split[1].back() == 'a' || split[1].back() == 'A')) {
    feature_extension = FeatureExtension::kAcceleratedFeatures;
    split[1].remove_suffix(1);
  }

  if (!split[1].empty() && (split[1].back() == 'f' || split[1].back() == 'F')) {
    feature_extension = FeatureExtension::kFamilyCompatibleFeatures;
    split[1].remove_suffix(1);
  }

  int major, minor;
  if (!absl::SimpleAtoi(split[0], &major) ||
      !absl::SimpleAtoi(split[1], &minor)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid CUDA architecture name: ", cuda_arch_name));
  }
  return CudaComputeCapability{major, minor, feature_extension};
}

static std::string FeatureExtensionToString(
    CudaComputeCapability::FeatureExtension feature_extension) {
  switch (feature_extension) {
    case CudaComputeCapability::FeatureExtension::kNone:
      return "";
    case CudaComputeCapability::FeatureExtension::kAcceleratedFeatures:
      return "a";
    case CudaComputeCapability::FeatureExtension::kFamilyCompatibleFeatures:
      return "f";
  }
}

std::string CudaComputeCapability::ToString() const {
  return absl::StrCat(major, ".", minor,
                      FeatureExtensionToString(feature_extension));
}

std::string CudaComputeCapability::GetPtxAsTargetName(
    CompileMode compile_mode) const {
  absl::string_view prefix = [&]() {
    switch (compile_mode) {
      case CompileMode::kPtx:
        return "compute";
      case CompileMode::kLto:
        return "lto";
      case CompileMode::kSass:
        return "sm";
    }
  }();
  return absl::StrFormat("%s_%d%d%s", prefix, major, minor,
                         FeatureExtensionToString(feature_extension));
}

absl::StatusOr<CudaComputeCapability> CudaComputeCapability::FromProto(
    const CudaComputeCapabilityProto& proto) {
  CudaComputeCapability cc;
  cc.major = proto.major();
  cc.minor = proto.minor();
  switch (proto.feature_extension()) {
    case CudaComputeCapabilityProto::UNSPECIFIED:
      // For backward compatibility we assume sm_90a and sm_100a for Hopper and
      // Blackwell generation GPUs.
      if (cc.major == 9 || cc.major == 10) {
        cc.feature_extension = FeatureExtension::kAcceleratedFeatures;
      } else {
        cc.feature_extension = FeatureExtension::kNone;
      }
      break;
    case CudaComputeCapabilityProto::NONE:
      cc.feature_extension = FeatureExtension::kNone;
      break;
    case CudaComputeCapabilityProto::ACCELERATED_FEATURES:
      cc.feature_extension = FeatureExtension::kAcceleratedFeatures;
      break;
    case CudaComputeCapabilityProto::FAMILY_COMPATIBLE_FEATURES:
      cc.feature_extension = FeatureExtension::kFamilyCompatibleFeatures;
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid feature extension: ", proto.feature_extension()));
  }
  return cc;
}

CudaComputeCapabilityProto CudaComputeCapability::ToProto() const {
  CudaComputeCapabilityProto proto;
  proto.set_major(major);
  proto.set_minor(minor);

  switch (feature_extension) {
    case FeatureExtension::kNone:
      proto.set_feature_extension(CudaComputeCapabilityProto::NONE);
      break;
    case FeatureExtension::kAcceleratedFeatures:
      proto.set_feature_extension(
          CudaComputeCapabilityProto::ACCELERATED_FEATURES);
      break;
    case FeatureExtension::kFamilyCompatibleFeatures:
      proto.set_feature_extension(
          CudaComputeCapabilityProto::FAMILY_COMPATIBLE_FEATURES);
      break;
  }
  return proto;
}

}  // namespace stream_executor
