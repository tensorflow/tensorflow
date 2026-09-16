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

// Backend independent part of the SMI utilities. The SMI calls themselves live
// in smi_util_amd_smi.cc and smi_util_rocm_smi.cc.

#include "xla/stream_executor/rocm/smi_util.h"

#include <cstddef>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace stream_executor::gpu {

ABSL_CONST_INIT absl::Mutex smi_mutex(absl::kConstInit);

// Parses a PCI bus/device/function ID string into its numeric components.
// Accepts two formats:
//   DDDD:BB:DD.F - domain:bus:device.function (e.g. "0000:41:00.0")
//   BB:DD.F      - bus:device.function, domain defaults to 0
// All fields are hex
absl::StatusOr<BdfComponents> ParseBdf(absl::string_view pci_bus_id) {
  const auto invalid = [&] {
    return absl::InvalidArgumentError(
        absl::StrCat("cannot parse PCI bus ID: ", pci_bus_id));
  };

  BdfComponents bdf = {};

  // Determine which format we have by counting colons.
  // Two colons -> DDDD:BB:DD.F, one colon -> BB:DD.F.
  size_t first_colon = pci_bus_id.find(':');
  if (first_colon == absl::string_view::npos) return invalid();

  size_t second_colon = pci_bus_id.find(':', first_colon + 1);
  size_t dot;

  if (second_colon != absl::string_view::npos) {
    // DDDD:BB:DD.F format
    dot = pci_bus_id.find('.', second_colon + 1);
    if (dot == absl::string_view::npos) return invalid();

    if (!absl::SimpleHexAtoi(pci_bus_id.substr(0, first_colon), &bdf.domain))
      return invalid();
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(first_colon + 1, second_colon - first_colon - 1),
            &bdf.bus))
      return invalid();
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(second_colon + 1, dot - second_colon - 1),
            &bdf.device))
      return invalid();
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(dot + 1), &bdf.function))
      return invalid();
  } else {
    // BB:DD.F format (domain = 0)
    dot = pci_bus_id.find('.', first_colon + 1);
    if (dot == absl::string_view::npos) return invalid();

    bdf.domain = 0;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(0, first_colon), &bdf.bus))
      return invalid();
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(first_colon + 1, dot - first_colon - 1),
            &bdf.device))
      return invalid();
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(dot + 1), &bdf.function))
      return invalid();
  }

  return bdf;
}

}  // namespace stream_executor::gpu
