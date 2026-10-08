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

#ifndef XLA_STREAM_EXECUTOR_CUDA_LOCALITY_DOMAIN_H_
#define XLA_STREAM_EXECUTOR_CUDA_LOCALITY_DOMAIN_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/green_context.h"

namespace stream_executor::gpu {

// A LocalityDomain represents one locality domain (die) of a multi-die GPU,
// backed by a GreenContext over that die's SM partition. Locality domains are a
// particular use of green contexts: the SM partition is chosen by locality
// domain id rather than by a raw SM count. Locality domains are a CUDA 13.4+
// feature.
class LocalityDomain {
 public:
  LocalityDomain(int locality_domain_id,
                 std::unique_ptr<GreenContext> green_context);

  LocalityDomain(const LocalityDomain&) = delete;
  LocalityDomain& operator=(const LocalityDomain&) = delete;

  int locality_domain_id() const { return locality_domain_id_; }
  const GreenContext& green_context() const { return *green_context_; }

  // Number of SMs in this locality domain's partition.
  int sm_count() const { return green_context_->sm_count(); }

  // Creates a CUstream that launches work onto this locality domain's SM
  // partition. See GreenContext::CreateStream.
  absl::StatusOr<CUstream> CreateStream(int priority) const {
    return green_context_->CreateStream(priority);
  }

 private:
  int locality_domain_id_;
  std::unique_ptr<GreenContext> green_context_;
};

// Returns the number of locality domains on `device`, read from
// CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT. Returns an internal error if the
// attribute query fails or the reported count is not positive, and an
// unimplemented error when the CUDA toolkit or driver is older than 13.4.
absl::StatusOr<int> GetLocalityDomainCount(CUdevice device);

// Splits the device's SMs by locality domain and creates one LocalityDomain
// (each backed by a GreenContext) per domain, returned in domain-id order.
// The device's primary CUDA context must be current on the calling thread.
// Returns an unimplemented error when the CUDA toolkit or driver is older
// than 13.4.
absl::StatusOr<std::vector<std::unique_ptr<LocalityDomain>>>
CreateLocalityDomains(CUdevice device);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_LOCALITY_DOMAIN_H_
