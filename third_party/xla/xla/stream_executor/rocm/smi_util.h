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

#ifndef XLA_STREAM_EXECUTOR_ROCM_SMI_UTIL_H_
#define XLA_STREAM_EXECUTOR_ROCM_SMI_UTIL_H_

#include <cstdint>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

// Backend independent view of the SMI queries XLA needs. Exactly one backend
// implements them, smi_util_amd_smi.cc or smi_util_rocm_smi.cc, and
// the BUILD file picks the source and the library to link against. Nothing
// here exposes an SMI type, so callers never need to know which one they got.
// The rationale for the ROCm 7.13 boundary lives next to that choice, in
// xla/stream_executor/rocm/BUILD.

namespace stream_executor::gpu {

struct BdfComponents {
  uint64_t domain;
  uint64_t bus;
  uint64_t device;
  uint64_t function;
};

// Opaque identifier of an SMI visible GPU. Only the backend interprets the
// value: rocm_smi identifies a GPU by a dense monitor device index, amd_smi by
// an opaque processor handle, and both fit in a uintptr_t.
struct SmiDeviceHandle {
  uintptr_t value = 0;

  friend bool operator==(SmiDeviceHandle lhs, SmiDeviceHandle rhs) {
    return lhs.value == rhs.value;
  }
  friend bool operator!=(SmiDeviceHandle lhs, SmiDeviceHandle rhs) {
    return !(lhs == rhs);
  }
};

// Current PCIe link state of a device.
struct PcieLinkStatus {
  uint32_t speed_mt_per_sec;
  uint16_t width;
};

// Process-global lock serializing all SMI access from XLA. rocm_smi only
// guards state with a per-device mutex, but some of it is global (e.g. the
// shared gpu_metrics object), so concurrent queries on different devices race.
// amd_smi embeds rocm_smi and inherits the same problem. Hold this across each
// full SMI call sequence.
ABSL_CONST_INIT extern absl::Mutex smi_mutex;

// Initializes the SMI library at most once per process. Every later call
// reports the result of that one attempt.
absl::Status InitSmi() ABSL_EXCLUSIVE_LOCKS_REQUIRED(smi_mutex);

// Parses a PCI bus ID string (e.g., "0000:41:00.0") into its BDF components.
// Touches no SMI state, so it needs no lock.
absl::StatusOr<BdfComponents> ParseBdf(absl::string_view pci_bus_id);

// Returns handles for every SMI visible GPU, in SMI enumeration order.
absl::StatusOr<std::vector<SmiDeviceHandle>> EnumerateDevices()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(smi_mutex);

// Finds the SMI device that matches the given PCI bus ID.
absl::StatusOr<SmiDeviceHandle> FindDevice(const BdfComponents& target_bdf)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(smi_mutex);

// Returns the current PCIe link speed and width of the device.
absl::StatusOr<PcieLinkStatus> QueryPcieLinkStatus(SmiDeviceHandle device)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(smi_mutex);

// Returns the xGMI hive ID of the device. Fails if it is not in a hive, which
// SMI does not distinguish from a failed query.
absl::StatusOr<uint64_t> QueryHiveId(SmiDeviceHandle device)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(smi_mutex);

// Returns true if src reaches dst over an xGMI link.
absl::StatusOr<bool> IsXgmiPeer(SmiDeviceHandle src, SmiDeviceHandle dst)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(smi_mutex);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_SMI_UTIL_H_
