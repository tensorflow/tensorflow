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

#include "xla/stream_executor/gpu/read_numa_node.h"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numa.h"

namespace stream_executor::gpu {

std::optional<int> ReadNumaNode(absl::string_view pci_bus_id,
                                int device_ordinal) {
  if (tsl::port::NUMANumNodes() < 2) {
    // NUMA support is not currently enabled, or there is only one node.
    return tsl::port::kNUMANoAffinity;
  }
  VLOG(2) << "trying to read NUMA node for device ordinal: " << device_ordinal;

  if (pci_bus_id.empty()) {
    LOG(INFO) << "no PCI bus ID for device ordinal: " << device_ordinal;
    return std::nullopt;
  }

  std::string filename =
      absl::StrFormat("/sys/bus/pci/devices/%s/numa_node", pci_bus_id);

  // We have to use fopen/fread here so that the device properties can be
  // populated before InitGoogle procedure has been completed (at which point we
  // could use the file::* utilities).
  FILE* file = fopen(filename.c_str(), "r");
  if (file == nullptr) {
    LOG(INFO) << "could not open file to read NUMA node: " << filename
              << "\nYour kernel may have been built without NUMA support.";
    return std::nullopt;
  }

  std::string content;
  char buf[32];
  size_t did_read = fread(buf, sizeof(buf[0]), sizeof(buf) - 1, file);
  fclose(file);
  buf[did_read] = '\0';
  content = buf;

  int32_t value;
  if (absl::SimpleAtoi(content, &value)) {
    if (value < 0) {  // See http://b/18228951 for details on this path.
      LOG(INFO) << "successful NUMA node read from SysFS had negative value ("
                << value
                << "), but there must be at least one NUMA node so this will "
                   " be massaged to NUMA node zero in some places."
                   " See more at "
                   "https://github.com/torvalds/linux/blob/v6.0/Documentation/"
                   "ABI/testing/sysfs-bus-pci#L344-L355";
      return tsl::port::kNUMANoAffinity;
    }
    return value;
  }

  LOG(WARNING)
      << "could not convert SysFS file contents to integral NUMA node value: "
      << content;

  return std::nullopt;
}

}  // namespace stream_executor::gpu
