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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_device.h"

namespace xla {

TfrtGpuDevice::TfrtGpuDevice(Options&& options)
    : id_(options.id),
      local_device_id_(options.local_device_id),
      local_hardware_id_(options.local_hardware_id),
      executor_(options.executor),
      stream_(options.executor == nullptr
                  ? nullptr
                  : options.executor->CreateStream().value()),
      prng_seed_generator_(prng_seed_device_()),
      prng_seed_distribution_(std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()),
      last_collective_launch_event_(
          tsl::MakeAvailableAsyncValueRef<GpuEvent>()),
      description_(options.id, local_device_id_.value(), options.process_index,
                   options.process_index_in_partition, options.partition_index,
                   options.platform_version),
      max_inflight_computations_semaphore_(
          /*capacity=*/options.max_inflight_computations) {
  std::vector<int64_t> v_coords(description_.coords().begin(),
                                description_.coords().end());

  description_.SetAttributes({
      {"coords", xla::PjRtDeviceAttribute(v_coords)},
      {"device_vendor", options.device_vendor},
      // TODO - b/435521225: `slice_index` is deprecated. Use `partition_index`,
      // which better aligns with NVIDIA's terminology.
      {"slice_index", static_cast<int64_t>(options.partition_index)},
      {"partition_index", static_cast<int64_t>(options.partition_index)},
      {"compute_capability",
       xla::PjRtDeviceAttribute(options.compute_capability)},
      {"core_count", static_cast<int64_t>(options.core_count)},
  });

  description_.SetDebugString(absl::StrCat("TFRT_GPU_", id_));
  description_.SetToString(absl::StrCat("GpuDevice(id=", id_, ")"));
}

TfrtGpuDevice::~TfrtGpuDevice() {
  // Block the host until all pending work on the stream is done. This is to
  // avoid user-after-free errors in host callbacks.
  if (stream_ != nullptr) {
    absl::Status status = stream_->BlockHostUntilDone();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to wait for stream to finish: " << status;
    }
  }
}

}  // namespace xla
