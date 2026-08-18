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

#include "xla/tools/cost_model/gpu_bandwidth_benchmark.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"

namespace xla::gpu {

absl::StatusOr<double> GetPeakBandwidthBytesPerSec(int device_id) {
  absl::StatusOr<stream_executor::Platform*> platform =
      stream_executor::PlatformManager::PlatformWithName(
          stream_executor::GpuPlatformName());
  if (!platform.ok()) {
    return platform.status();
  }
  absl::StatusOr<stream_executor::StreamExecutor*> executor =
      (*platform)->ExecutorForDevice(device_id);
  if (!executor.ok()) {
    return executor.status();
  }
  const int64_t bandwidth =
      (*executor)->GetDeviceDescription().memory_bandwidth();
  if (bandwidth <= 0) {
    return absl::InternalError(absl::StrFormat(
        "Failed to determine peak memory bandwidth for device %d: "
        "memory_bandwidth is %v.",
        device_id, bandwidth));
  }
  return static_cast<double>(bandwidth);
}

namespace {

// Enqueues `runs` back-to-back memcpy operations within a single event timer
// window. Batching operations amortizes event timer quantization overhead and
// eliminates host-device synchronization variance, measuring true sustained
// memory bandwidth.
absl::StatusOr<absl::Duration> MeasureMemcpyDuration(
    stream_executor::Stream* stream, stream_executor::DeviceAddressBase* dst,
    const stream_executor::DeviceAddressBase& src, int64_t size_bytes,
    int runs) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<stream_executor::EventBasedTimer> timer,
                   stream->CreateEventBasedTimer(/*use_delay_kernel=*/false));
  for (int r = 0; r < runs; ++r) {
    ABSL_RETURN_IF_ERROR(stream->Memcpy(dst, src, size_bytes));
  }
  return timer->GetElapsedDuration();
}

absl::StatusOr<absl::Duration> TimeD2dMemcpy(
    stream_executor::Stream* stream, stream_executor::DeviceAddressBase* in,
    stream_executor::DeviceAddressBase* out, int64_t size_bytes,
    int warmup_runs, int measurement_runs) {
  ABSL_RETURN_IF_ERROR(stream->MemZero(in, size_bytes));
  ABSL_RETURN_IF_ERROR(stream->MemZero(out, size_bytes));

  for (int w = 0; w < warmup_runs; ++w) {
    ABSL_RETURN_IF_ERROR(stream->Memcpy(out, *in, size_bytes));
  }
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  return MeasureMemcpyDuration(stream, out, *in, size_bytes, measurement_runs);
}

}  // namespace

absl::StatusOr<double> MeasureD2dBandwidthBytesPerSec(int device_id,
                                                      int64_t size_bytes,
                                                      int warmup_runs,
                                                      int measurement_runs) {
  if (device_id < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid device_id: %d. Must be >= 0.", device_id));
  }
  if (size_bytes <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid size_bytes: %d. Must be > 0.", size_bytes));
  }
  if (warmup_runs < 0 || measurement_runs <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid warmup (%d) or measurement (%d) run count.",
                        warmup_runs, measurement_runs));
  }

  ABSL_ASSIGN_OR_RETURN(stream_executor::Platform * platform,
                   stream_executor::PlatformManager::PlatformWithName(
                       stream_executor::GpuPlatformName()));
  ABSL_ASSIGN_OR_RETURN(stream_executor::StreamExecutor * executor,
                   platform->ExecutorForDevice(device_id));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<stream_executor::Stream> stream,
                   executor->CreateStream());

  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  ABSL_ASSIGN_OR_RETURN(stream_executor::ScopedDeviceAddress<uint8_t> d_in,
                   allocator.Allocate(device_id, size_bytes));
  ABSL_ASSIGN_OR_RETURN(stream_executor::ScopedDeviceAddress<uint8_t> d_out,
                   allocator.Allocate(device_id, size_bytes));

  ABSL_ASSIGN_OR_RETURN(absl::Duration elapsed,
                   TimeD2dMemcpy(stream.get(), d_in.ptr(), d_out.ptr(),
                                 size_bytes, warmup_runs, measurement_runs));

  const double elapsed_sec = absl::ToDoubleSeconds(elapsed);
  if (elapsed_sec <= 0.0) {
    return absl::InternalError("EventBasedTimer reported non-positive time.");
  }
  const double avg_sec = elapsed_sec / measurement_runs;
  // A D2D memcpy reads `size_bytes` from the source buffer and writes
  // `size_bytes` to the destination buffer, totaling 2 * size_bytes traversed.
  return (2.0 * static_cast<double>(size_bytes)) / avg_sec;
}

std::string FormatBandwidthTable(absl::Span<const BandwidthEntry> entries) {
  std::string result =
      "DMA Size (Bytes)    Bandwidth Fraction\n"
      "----------------    ------------------\n";
  for (const BandwidthEntry& entry : entries) {
    absl::StrAppendFormat(&result, "%16d    %18.8f\n", entry.dma_size_bytes,
                          entry.bandwidth_fraction);
  }
  return result;
}

}  // namespace xla::gpu
