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

// This files implements the configuration management for different types of
// guards.
// C++ backends are responsible for enforcing transfer guard levels.

#include "xla/python/guard_lib.h"

#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "xla/util.h"

namespace jax {

namespace nb = ::nanobind;

namespace {

// Protected by the GIL.
GuardState& global_state = *new GuardState();

ABSL_CONST_INIT thread_local GuardState thread_local_state;

// The default transfer guard level.
constexpr TransferGuardLevel kDefaultGuardLevel = TransferGuardLevel::kAllow;

// The default garbage collection guard level.
constexpr GarbageCollectionGuardLevel kDefaultGarbageCollectionGuardLevel =
    GarbageCollectionGuardLevel::kAllow;

// Returns the transfer guard action for a transfer.
TransferGuardAction GetTransferGuardAction(TransferGuardLevel guard_level,
                                           bool explicit_transfer) {
  switch (guard_level) {
    case TransferGuardLevel::kAllow:
      return TransferGuardAction::kAllow;
    case TransferGuardLevel::kLog:
      if (explicit_transfer) {
        return TransferGuardAction::kAllow;
      } else {
        return TransferGuardAction::kLog;
      }
    case TransferGuardLevel::kDisallow:
      if (explicit_transfer) {
        return TransferGuardAction::kAllow;
      } else {
        return TransferGuardAction::kDisallow;
      }
    case TransferGuardLevel::kLogExplicit:
      return TransferGuardAction::kLog;
    case TransferGuardLevel::kDisallowExplicit:
      return TransferGuardAction::kDisallow;
    default:
      // Unreachable; gracefully handle the unexpected guard level and prevent a
      // compiler warning.
      return TransferGuardAction::kDisallow;
  }
}

// Returns the transfer guard action for a host-to-device transfer.
// REQUIRES: Python GIL.
TransferGuardAction GetTransferGuardActionForHostToDevice() {
  return GetTransferGuardAction(
      thread_local_state.host_to_device.value_or(
          global_state.host_to_device.value_or(kDefaultGuardLevel)),
      thread_local_state.explicit_device_put);
}

// Returns the transfer guard action for a device-to-device transfer.
// REQUIRES: Python GIL.
TransferGuardAction GetTransferGuardActionForDeviceToDevice() {
  return GetTransferGuardAction(
      thread_local_state.device_to_device.value_or(
          global_state.device_to_device.value_or(kDefaultGuardLevel)),
      thread_local_state.explicit_device_put);
}

// Returns the transfer guard action for a device-to-host transfer.
// REQUIRES: Python GIL.
TransferGuardAction GetTransferGuardActionForDeviceToHost() {
  return GetTransferGuardAction(
      thread_local_state.device_to_host.value_or(
          global_state.device_to_host.value_or(kDefaultGuardLevel)),
      thread_local_state.explicit_device_get);
}

}  // namespace

absl::Status ApplyTransferGuardToHostToDevice(
    absl::FunctionRef<std::string()> formatter) {
  switch (GetTransferGuardActionForHostToDevice()) {
    case TransferGuardAction::kAllow:
      break;
    case TransferGuardAction::kLog:
      LOG(WARNING) << "host-to-device transfer: " << formatter();
      break;
    case TransferGuardAction::kDisallow:
      return xla::InvalidArgument("Disallowed host-to-device transfer: %s",
                                  formatter());
  }
  return absl::OkStatus();
}

absl::Status ApplyTransferGuardToDeviceToDevice(
    absl::FunctionRef<std::string()> formatter) {
  switch (GetTransferGuardActionForDeviceToDevice()) {
    case TransferGuardAction::kAllow:
      break;
    case TransferGuardAction::kLog:
      LOG(WARNING) << "device-to-device transfer: " << formatter();
      break;
    case TransferGuardAction::kDisallow:
      return xla::InvalidArgument("Disallowed device-to-device transfer: %s",
                                  formatter());
  }
  return absl::OkStatus();
}

absl::Status ApplyTransferGuardToDeviceToHost(
    absl::FunctionRef<std::string()> formatter) {
  switch (GetTransferGuardActionForDeviceToHost()) {
    case TransferGuardAction::kAllow:
      break;
    case TransferGuardAction::kLog:
      LOG(WARNING) << "device-to-host transfer: " << formatter();
      break;
    case TransferGuardAction::kDisallow:
      return xla::InvalidArgument("Disallowed device-to-host transfer: %s",
                                  formatter());
  }
  return absl::OkStatus();
}

GarbageCollectionGuardLevel GetGarbageCollectArrayGuard() {
  return thread_local_state.garbage_collect_array.value_or(
      global_state.garbage_collect_array.value_or(
          kDefaultGarbageCollectionGuardLevel));
}

void BuildGuardSubmodule(nb::module_& m) {
  nb::module_ glib =
      m.def_submodule("guard_lib", "Jax support library for guards");

  nb::enum_<TransferGuardLevel> tglevel(glib, "TransferGuardLevel");
  tglevel.value("ALLOW", TransferGuardLevel::kAllow);
  tglevel.value("LOG", TransferGuardLevel::kLog);
  tglevel.value("DISALLOW", TransferGuardLevel::kDisallow);
  tglevel.value("LOG_EXPLICIT", TransferGuardLevel::kLogExplicit);
  tglevel.value("DISALLOW_EXPLICIT", TransferGuardLevel::kDisallowExplicit);

  nb::enum_<GarbageCollectionGuardLevel> gcglevel(
      glib, "GarbageCollectionGuardLevel");
  gcglevel.value("ALLOW", GarbageCollectionGuardLevel::kAllow);
  gcglevel.value("LOG", GarbageCollectionGuardLevel::kLog);
  gcglevel.value("FATAL", GarbageCollectionGuardLevel::kFatal);

  nb::class_<GuardState> tgstate(glib, "GuardState");
  tgstate.def_rw("host_to_device", &GuardState::host_to_device,
                 nb::arg().none());
  tgstate.def_rw("device_to_device", &GuardState::device_to_device,
                 nb::arg().none());
  tgstate.def_rw("device_to_host", &GuardState::device_to_host,
                 nb::arg().none());
  tgstate.def_rw("explicit_device_put", &GuardState::explicit_device_put);
  tgstate.def_rw("explicit_device_get", &GuardState::explicit_device_get);
  tgstate.def_rw("garbage_collect_array", &GuardState::garbage_collect_array,
                 nb::arg().none());

  glib.def(
      "global_state", [&]() { return &global_state; },
      nb::rv_policy::reference);
  glib.def(
      "thread_local_state", [&]() { return &thread_local_state; },
      nb::rv_policy::reference);
}

}  // namespace jax
