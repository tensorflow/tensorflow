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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MULTIMEM_REGISTRY_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MULTIMEM_REGISTRY_H_

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_multimem.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// A request for a multimem for a given clique on a given address space.
struct MultimemRequest {
  static std::tuple<GpuCliqueKey, void*, uint64_t> CmpKey(
      const MultimemRequest& key) {
    return {key.key, key.map_to.opaque(), key.map_to.size()};
  }

  friend bool operator==(const MultimemRequest& a, const MultimemRequest& b) {
    return a.key == b.key && a.map_to == b.map_to;
  }

  template <typename H>
  friend H AbslHashValue(H h, const MultimemRequest& key) {
    return H::combine(std::move(h), key.key, key.map_to.opaque(),
                      key.map_to.size());
  }

  GpuCliqueKey key;
  se::DeviceAddressBase map_to;
};

class CollectiveMultimemRequests {
 public:
  virtual void Request(const MultimemRequest& request) = 0;
  virtual ~CollectiveMultimemRequests() = default;
};

class CollectiveMultimemProvider {
 public:
  virtual absl::StatusOr<std::shared_ptr<CollectiveMultimem>> Get(
      const MultimemRequest& request) const = 0;
  virtual ~CollectiveMultimemProvider() = default;
};

// Allocates and provides thunks requested multimem objects.
class CollectiveMultimemRegistry : public CollectiveMultimemRequests,
                                   public CollectiveMultimemProvider {
 public:
  // Does not take ownership of `executor`, which must outlive this object.
  CollectiveMultimemRegistry(se::StreamExecutor* absl_nonnull executor,
                             GlobalDeviceId global_device_id)
      : executor_(*executor), global_device_id_(global_device_id) {}

  void Request(const MultimemRequest& request) override;

  absl::Status Build();

  absl::StatusOr<std::shared_ptr<CollectiveMultimem>> Get(
      const MultimemRequest& request) const override;

 private:
  std::vector<MultimemRequest> requests_;
  absl::flat_hash_map<MultimemRequest, std::shared_ptr<CollectiveMultimem>>
      multimems_;
  se::StreamExecutor& executor_;
  GlobalDeviceId global_device_id_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MULTIMEM_REGISTRY_H_
