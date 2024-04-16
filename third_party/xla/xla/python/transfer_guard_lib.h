/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_TRANSFER_GUARD_LIB_H_
#define XLA_PYTHON_TRANSFER_GUARD_LIB_H_

#include <optional>
#include <string>

// placeholder for index annotation headers
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"

namespace jax {

// Transfer guard level chosen by the user code.
enum class TransferGuardLevel {
  // Explicit transfers: allow
  // Implicit transfers: allow
  kAllow,
  // Explicit transfers: allow
  // Implicit transfers: log
  kLog,
  // Explicit transfers: allow
  // Implicit transfers: disallow
  kDisallow,
  // Explicit transfers: log
  // Implicit transfers: log
  kLogExplicit,
  // Explicit transfers: disallow
  // Implicit transfers: disallow
  kDisallowExplicit,
};

// Flags for transfer guard levels are controlled by:
// - a global flag value,
//   e.g., associated to --jax_transfer_guard_device_to_host
//   which defaults to TransferGuardLevel::kAllow.
// - possibly a thread-local value, which initially is std::nullopt and
//   overrides the global value if set. The thread-local state is used to
//   implement context managers that locally override the global state.
//
// Explicit device_put/device_get contexts are tracked by context managers.
struct TransferGuardState {
  std::optional<TransferGuardLevel> host_to_device;
  std::optional<TransferGuardLevel> device_to_device;
  std::optional<TransferGuardLevel> device_to_host;
  bool explicit_device_put = false;
  bool explicit_device_get = false;
};

// Resulting action for a transfer given the transfer guard level and the
// transfer type.
enum class TransferGuardAction {
  // Silently allow the transfer.
  kAllow,
  // Log and allow the transfer.
  kLog,
  // Disallow the transfer.
  kDisallow,
};

// Guards a host-to-device transfer. formatter is called to describe the
// transfer in a log message or error status.
// REQUIRES: Python GIL.
absl::Status ApplyTransferGuardToHostToDevice(
    absl::FunctionRef<std::string()> formatter);

// Guards a device-to-device transfer. formatter is called to describe the
// transfer in a log message or error status.
// REQUIRES: Python GIL.
absl::Status ApplyTransferGuardToDeviceToDevice(
    absl::FunctionRef<std::string()> formatter);

// Guards a device-to-host transfer. formatter is called to describe the
// transfer in a log message or error status.
// REQUIRES: Python GIL.
absl::Status ApplyTransferGuardToDeviceToHost(
    absl::FunctionRef<std::string()> formatter);

// The function to call in `xla.cc` to add the bindings for this module.
void BuildTransferGuardSubmodule(nanobind::module_& m);

}  // namespace jax

#endif  // XLA_PYTHON_TRANSFER_GUARD_LIB_H_
