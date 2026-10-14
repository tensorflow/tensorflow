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
#include "xla/stream_executor/stream_executor_address_allocator.h"

namespace xla::gpu {
namespace {

using ::stream_executor::DeviceAddressBase;
using ::stream_executor::DeviceDescription;
using ::stream_executor::EventBasedTimer;
using ::stream_executor::GpuPlatformName;
using ::stream_executor::Platform;
using ::stream_executor::PlatformManager;
using ::stream_executor::ScopedDeviceAddress;
using ::stream_executor::Stream;
using ::stream_executor::StreamExecutor;
using ::stream_executor::StreamExecutorAddressAllocator;

absl::StatusOr<absl::Duration> TimeD2dMemcpy(
    int64_t size_bytes, int warmup_runs, int measurement_runs, Stream& stream,
    DeviceAddressBase& src, DeviceAddressBase& dst) {
  ABSL_RETURN_IF_ERROR(stream.MemZero(&src, size_bytes));
  ABSL_RETURN_IF_ERROR(stream.MemZero(&dst, size_bytes));

  for (int w = 0; w < warmup_runs; ++w) {
    ABSL_RETURN_IF_ERROR(stream.Memcpy(&dst, src, size_bytes));
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<EventBasedTimer> timer,
                   stream.CreateEventBasedTimer(/*use_delay_kernel=*/true));
  for (int r = 0; r < measurement_runs; ++r) {
    ABSL_RETURN_IF_ERROR(stream.Memcpy(&dst, src, size_bytes));
  }
  return timer->GetElapsedDuration();
}

}  // namespace

absl::StatusOr<double> GetPeakBandwidthBytesPerSec(int ordinal) {
  ABSL_ASSIGN_OR_RETURN(Platform * platform,
                   PlatformManager::PlatformWithName(GpuPlatformName()));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<DeviceDescription> description,
                   platform->DescriptionForDevice(ordinal));
  const int64_t bandwidth = description->memory_bandwidth();
  if (bandwidth <= 0) {
    return absl::InternalError(absl::StrFormat(
        "Failed to determine peak memory bandwidth for device %d: "
        "memory_bandwidth is %v.",
        ordinal, bandwidth));
  }
  return static_cast<double>(bandwidth);
}

absl::StatusOr<double> MeasureD2dBandwidthBytesPerSec(int ordinal,
                                                      int64_t size_bytes,
                                                      int warmup_runs,
                                                      int measurement_runs) {
  if (ordinal < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid ordinal: %d. Must be >= 0.", ordinal));
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

  ABSL_ASSIGN_OR_RETURN(Platform * platform,
                   PlatformManager::PlatformWithName(GpuPlatformName()));
  ABSL_ASSIGN_OR_RETURN(StreamExecutor * executor,
                   platform->ExecutorForDevice(ordinal));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Stream> stream, executor->CreateStream());

  StreamExecutorAddressAllocator allocator(executor);
  ABSL_ASSIGN_OR_RETURN(ScopedDeviceAddress<uint8_t> src,
                   allocator.Allocate(ordinal, size_bytes));
  ABSL_ASSIGN_OR_RETURN(ScopedDeviceAddress<uint8_t> dst,
                   allocator.Allocate(ordinal, size_bytes));

  ABSL_ASSIGN_OR_RETURN(absl::Duration elapsed,
                   TimeD2dMemcpy(size_bytes, warmup_runs, measurement_runs,
                                 *stream, *src.ptr(), *dst.ptr()));

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
