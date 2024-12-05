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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "absl/status/statusor.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// XLA:GPU extension of the Collectives interface with GPU-specific APIs.
class GpuCollectives : public Collectives {
 public:
  // A callback to get a unique clique id.
  using CliqueIdCallback =  // NOLINT
      std::function<absl::StatusOr<CliqueId>(const CliqueKey&)>;

  // GPU collectives device is just a wrapper around the StreamExecutor.
  class Device : public Collectives::Device {
   public:
    explicit Device(stream_executor::StreamExecutor* stream_executor);
    stream_executor::StreamExecutor* stream_executor() const;

   private:
    stream_executor::StreamExecutor* stream_executor_;
  };

  // Casts a Collectives::Device to a GPU device and returns an error if it's
  // not a GPU device.
  static absl::StatusOr<Device*> TryCast(Collectives::Device* device);

  // GPU communicator configuration.
  //
  // For NCCL backend see configuration options documentation at:
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig
  struct Config {
    bool split_share = false;
    int64_t max_nchannels = 0;
  };

  // Returns true if collectives backend uses global config.
  virtual bool IsGlobalConfig() const = 0;

  // Returns a clique id callback passed as an argument if it's not null or a
  // default callback to get create a clique id if we are running in local mode.
  virtual absl::StatusOr<const CliqueIdCallback*> GetCliqueIdCallback(
      const CliqueIdCallback* clique_id_callback, bool is_local) = 0;

  // Returns a slice of device memory `buff` containing `count` values of data
  // type `dtype` starting from `offset`.
  static stream_executor::DeviceMemoryBase Slice(
      stream_executor::DeviceMemoryBase buff, PrimitiveType dtype,
      size_t offset, size_t count);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_H_
