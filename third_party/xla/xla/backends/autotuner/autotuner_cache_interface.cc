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

#include "xla/backends/autotuner/autotuner_cache_interface.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/util/sorted_range.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

namespace {

std::string ComputeDeviceName(
    const stream_executor::DeviceDescription& device_description) {
  // raw_device computation is taken from legacy_cache implementation except
  // dnn version.
  // Additionally, we fingerprint the string to avoid excessive length. We
  // would use data collected using the VLOG to see if we can use name
  // directly in place of the fingerprint, to improve readability.
  std::string compute_capability;
  if (auto* ccc = device_description.gpu_compute_capability()
                      .cuda_compute_capability()) {
    compute_capability = absl::StrCat("CUDA: ", ccc->major, ".", ccc->minor);
  } else if (auto* rcc = device_description.gpu_compute_capability()
                             .rocm_compute_capability()) {
    compute_capability = absl::StrCat("ROCM: ", rcc->gfx_version());
  } else if (auto* oneapi_cc = device_description.gpu_compute_capability()
                                   .oneapi_compute_capability()) {
    compute_capability = absl::StrCat("oneAPI: ", oneapi_cc->ToString());
  } else {
    LOG(FATAL) << "Unknown compute capability type";
  }

  // The string below should include only as much information as is needed to
  // make it a valid key. Information that should not be included is:
  // - specs that are directly derivable from the compute capability, e.g.
  //   shared memory size. For NVIDIA GPUs, you can see what is derivable from
  //   the SM version here:
  //   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
  // - specs that are irrelevant for autotuning. E.g. the total available
  //   memory on a device is not relevant, because by itself, it does not
  //   affect the performance of single kernels.
  //
  // See b/344573710 for some discussion.

  double memory_bandwidth = device_description.memory_bandwidth() / 1e9;
  memory_bandwidth = std::round(memory_bandwidth);

  constexpr double kBytesPerMegabyte = 1 << 20;
  double l2_cache_size = device_description.l2_cache_size() / kBytesPerMegabyte;

  std::string raw_device = absl::StrCat(
      compute_capability, ", Cores: ", device_description.core_count(),
      ", GPU clock: ", device_description.clock_rate_ghz(),
      " GHz, Memory bandwidth: ", memory_bandwidth,
      " GB/s, L2 cache: ", l2_cache_size, " MB");
  std::string fingerprint =
      absl::StrCat(absl::Hex(tsl::Fingerprint64(raw_device), absl::kZeroPad16));
  VLOG(1) << "Device fingerprint: " << fingerprint << " from " << raw_device
          << ", device name: " << device_description.name();
  return fingerprint;
}

std::string ComputeCodegenVersion(
    const absl::flat_hash_map<autotuner::Backend, std::string>&
        per_backend_versions) {
  if (per_backend_versions.empty()) {
    LOG(ERROR) << "No codegen backends found, cannot compute codegen version.";
    return "unknown";
  }
  std::vector<std::string> parts;
  for (const auto& [backend, version] :
       tsl::KeySortedRange(per_backend_versions)) {
    parts.push_back(
        absl::StrCat(autotuner::Backend_Name(backend), ":", version));
  }
  std::string raw_version = absl::StrJoin(parts, ",");
  std::string fingerprint = absl::StrCat(
      absl::Hex(tsl::Fingerprint64(raw_version), absl::kZeroPad16));
  VLOG(1) << "Codegen version: " << fingerprint << " from " << raw_version;
  return fingerprint;
}

}  // namespace

AutotuneCacheContext AutotuneCacheContext::Create(
    const stream_executor::DeviceDescription& device_description,
    absl::Span<const std::unique_ptr<CodegenBackend>> backends,
    std::string explicit_version) {
  absl::flat_hash_map<autotuner::Backend, std::string> per_backend_versions;
  for (const auto& backend : backends) {
    per_backend_versions[backend->backend()] = backend->version();
  }
  std::string codegen_version = ComputeCodegenVersion(per_backend_versions);
  return AutotuneCacheContext(
      ComputeDeviceName(device_description), std::move(explicit_version),
      std::move(codegen_version), std::move(per_backend_versions));
}

std::string AutotuneCacheContext::GetId() const {
  return absl::StrCat(device_, explicit_version_, codegen_version_);
}

bool AutotuneCacheContext::operator==(const AutotuneCacheContext& other) const {
  return GetId() == other.GetId();
}

bool AutotuneCacheContext::operator!=(const AutotuneCacheContext& other) const {
  return !(*this == other);
}

}  // namespace xla
