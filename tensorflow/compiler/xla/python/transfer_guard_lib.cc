/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This files implements the configuration management for transfer guards.
// C++ backends responsible for enforcing transfer guard levels.

#include "tensorflow/compiler/xla/python/transfer_guard_lib.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"

namespace jax {

namespace py = ::pybind11;

namespace {

// Protected by the GIL.
TransferGuardState& global_state = *new TransferGuardState();

ABSL_CONST_INIT thread_local TransferGuardState thread_local_state;

// The default transfer guard level.
constexpr TransferGuardLevel kDefaultGuardLevel = TransferGuardLevel::kAllow;

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

xla::Status ApplyTransferGuardToHostToDevice(
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
  return ::tsl::OkStatus();
}

xla::Status ApplyTransferGuardToDeviceToDevice(
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
  return ::tsl::OkStatus();
}

xla::Status ApplyTransferGuardToDeviceToHost(
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
  return ::tsl::OkStatus();
}

void BuildTransferGuardSubmodule(py::module& m) {
  py::module tglib = m.def_submodule("transfer_guard_lib",
                                     "Jax transfer guard support library");

  py::enum_<TransferGuardLevel> tglevel(tglib, "TransferGuardLevel");
  tglevel.value("ALLOW", TransferGuardLevel::kAllow);
  tglevel.value("LOG", TransferGuardLevel::kLog);
  tglevel.value("DISALLOW", TransferGuardLevel::kDisallow);
  tglevel.value("LOG_EXPLICIT", TransferGuardLevel::kLogExplicit);
  tglevel.value("DISALLOW_EXPLICIT", TransferGuardLevel::kDisallowExplicit);

  py::class_<TransferGuardState> tgstate(tglib, "TransferGuardState");
  tgstate.def_readwrite("host_to_device", &TransferGuardState::host_to_device);
  tgstate.def_readwrite("device_to_device",
                        &TransferGuardState::device_to_device);
  tgstate.def_readwrite("device_to_host", &TransferGuardState::device_to_host);
  tgstate.def_readwrite("explicit_device_put",
                        &TransferGuardState::explicit_device_put);
  tgstate.def_readwrite("explicit_device_get",
                        &TransferGuardState::explicit_device_get);

  tglib.def(
      "global_state", [&]() { return &global_state; },
      py::return_value_policy::reference);
  tglib.def(
      "thread_local_state", [&]() { return &thread_local_state; },
      py::return_value_policy::reference);
}

}  // namespace jax
