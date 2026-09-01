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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GXL_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GXL_COMMUNICATOR_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace se = ::stream_executor;

// GxlCommunicator is an interface for GXL-accelerated NCCL communicators.
class GxlCommunicator {
 public:
  virtual ~GxlCommunicator() = default;

  // Runs a GXL-accelerated AllGather operation.
  virtual absl::Status RunAllGatherGxl(se::Stream* stream,
                                       PrimitiveType element_type,
                                       se::DeviceAddressBase input_buffer,
                                       se::DeviceAddressBase output_buffer,
                                       int64_t element_count, int64_t rank) {
    return absl::UnimplementedError("GXL is not supported in this build.");
  }

  // Runs a GXL-accelerated RaggedAllToAll operation.
  virtual absl::Status RunRaggedAllToAllGxl(
      se::Stream* stream, PrimitiveType element_type,
      se::DeviceAddressBase input_buffer, se::DeviceAddressBase output_buffer,
      se::DeviceAddressBase input_offsets_buffer,
      se::DeviceAddressBase send_sizes_buffer,
      se::DeviceAddressBase output_offsets_buffer,
      se::DeviceAddressBase recv_sizes_buffer, int64_t num_row_elements,
      int64_t num_total_updates, int64_t rank) {
    return absl::UnimplementedError("GXL is not supported in this build.");
  }
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GXL_COMMUNICATOR_H_
