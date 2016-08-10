/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_init.h"

#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {

namespace {

std::unique_ptr<std::map<std::pair<int, int>, bool>> GetPeerAccessMap(
    gpu::Platform* platform, int device_count) {
  auto* map = new std::map<std::pair<int, int>, bool>;
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      gpu::StreamExecutor* from = platform->ExecutorForDevice(i).ValueOrDie();
      gpu::StreamExecutor* to = platform->ExecutorForDevice(j).ValueOrDie();
      (*map)[{i, j}] = from->CanEnablePeerAccessTo(to);
    }
  }

  return std::unique_ptr<std::map<std::pair<int, int>, bool>>{map};
}

Status EnablePeerAccess(gpu::Platform* platform, int device_count) {
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      gpu::StreamExecutor* from = platform->ExecutorForDevice(i).ValueOrDie();
      gpu::StreamExecutor* to = platform->ExecutorForDevice(j).ValueOrDie();

      if (from->CanEnablePeerAccessTo(to)) {
        auto status = from->EnablePeerAccessTo(to);
        if (!status.ok()) {
          return errors::Internal(status.ToString());
        }
      } else {
        LOG(INFO) << "cannot enable peer access from device ordinal " << i
                  << " to device ordinal " << j;
      }
    }
  }
  return Status::OK();
}

static void InitGPU() {
  auto result = gpu::MultiPlatformManager::PlatformWithName("CUDA");
  if (!result.ok()) {
    LOG(WARNING)
        << "Not initializing the GPU, could not create GPU MachineManager. "
        << "Error: " << result.status();
    return;
  }

  gpu::Platform* platform = result.ValueOrDie();

  int dev_count = platform->VisibleDeviceCount();

  if (dev_count <= 0) {
    LOG(INFO) << "No GPU devices available on machine.";
    return;
  }

  for (int i = 0; i < dev_count; ++i) {
    auto stream_exec = platform->ExecutorForDevice(i).ValueOrDie();
    int64 free_bytes;
    int64 total_bytes;
    if (!stream_exec->DeviceMemoryUsage(&free_bytes, &total_bytes)) {
      // Logs internally on failure.
      free_bytes = 0;
      total_bytes = 0;
    }
    const auto& description = stream_exec->GetDeviceDescription();
    int cc_major;
    int cc_minor;
    if (!description.cuda_compute_capability(&cc_major, &cc_minor)) {
      // Logs internally on failure.
      cc_major = 0;
      cc_minor = 0;
    }
    LOG(INFO) << "Found device " << i << " with properties: "
              << "\nname: " << description.name() << "\nmajor: " << cc_major
              << " minor: " << cc_minor << " memoryClockRate (GHz) "
              << description.clock_rate_ghz() << "\npciBusID "
              << description.pci_bus_id() << "\nTotal memory: "
              << strings::HumanReadableNumBytes(total_bytes)
              << "\nFree memory: "
              << strings::HumanReadableNumBytes(free_bytes);
  }

  // Enable peer access

  auto status = EnablePeerAccess(platform, dev_count);
  if (!status.ok()) {
    LOG(FATAL) << "could not enable peer access for GPU devices: " << status;
  }

  // Print out a matrix showing which devices can DMA to one
  // another.
  auto access_map = GetPeerAccessMap(platform, dev_count);
  string line_buf = "DMA: ";
  for (int i = 0; i < dev_count; ++i) {
    strings::StrAppend(&line_buf, i, " ");
  }
  LOG(INFO) << line_buf;
  for (int i = 0; i < dev_count; ++i) {
    line_buf = strings::StrCat(i, ":   ");
    for (int j = 0; j < dev_count; ++j) {
      if ((*access_map)[{i, j}]) {
        line_buf.append("Y ");
      } else {
        line_buf.append("N ");
      }
    }
    LOG(INFO) << line_buf;
  }
}

static bool InitModule() {
  InitGPU();
  return true;
}

}  // namespace

gpu::Platform* GPUMachineManager() {
  // Create the machine manager singleton and initialize the GPUs only
  // once.
  static bool init = InitModule();
  CHECK(init);  // Avoids compiler warning that init is unused.

  auto result = gpu::MultiPlatformManager::PlatformWithName("CUDA");
  if (!result.ok()) {
    return nullptr;
  }

  return result.ValueOrDie();
}

}  // namespace tensorflow
