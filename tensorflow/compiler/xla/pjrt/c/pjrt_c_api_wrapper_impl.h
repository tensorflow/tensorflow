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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_WRAPPER_IMPL_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_WRAPPER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

struct PJRT_Error {
  xla::Status status;
};

struct PJRT_Client {
  std::unique_ptr<xla::PjRtClient> client;
  std::vector<PJRT_Device> owned_devices;
  // `devices` contains the addresses of the contents of `owned_devices`.
  std::vector<PJRT_Device*> devices;
  // `addressable_devices` contains pointers to the `owned_devices` that the
  // client can issue commands to.
  std::vector<PJRT_Device*> addressable_devices;
  // Map from wrapped C++ devices to C devices. The values are the same as
  // `owned_devices`.
  absl::flat_hash_map<xla::PjRtDevice*, PJRT_Device*> c_device_from_cpp_device;
};

// PJRT_Devices are owned by their corresponding PJRT_Client.
struct PJRT_Device {
  // The xla::PjRtDevice* is owned by the corresponding xla::PjRtClient.
  xla::PjRtDevice* device;
};

struct PJRT_Executable {
  std::unique_ptr<xla::PjRtExecutable> executable;
  PJRT_Client* client;
  // These pointers are a subset of `client`'s `addressable_devices`, i.e. those
  // addressed by the compiled executable program. `client` owns the objects
  // these point to.
  std::vector<PJRT_Device*> addressable_devices;
  // TODO(b/237545405): Remove `populated` once we implement creation methods
  // for PJRT_Executable that can populate addressable_devices on instantiation.
  bool populated = false;
};

struct PJRT_Buffer {
  std::unique_ptr<xla::PjRtBuffer> buffer;
};

namespace pjrt {

// C API definitions

void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args);
void PJRT_Error_Message(PJRT_Error_Message_Args* args);

PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args);
PJRT_Error* PJRT_Client_PlatformName(PJRT_Client_PlatformName_Args* args);
PJRT_Error* PJRT_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args);
PJRT_Error* PJRT_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args);
PJRT_Error* PJRT_Client_Devices(PJRT_Client_Devices_Args* args);
PJRT_Error* PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args);
PJRT_Error* PJRT_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args);

PJRT_Error* PJRT_Device_Id(PJRT_Device_Id_Args* args);
PJRT_Error* PJRT_Device_ProcessIndex(PJRT_Device_ProcessIndex_Args* args);
PJRT_Error* PJRT_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args);
PJRT_Error* PJRT_Device_Kind(PJRT_Device_Kind_Args* args);

PJRT_Error* PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args);
PJRT_Error* PJRT_Executable_Name(PJRT_Executable_Name_Args* args);
PJRT_Error* PJRT_Executable_AddressableDevices(
    PJRT_Executable_AddressableDevices_Args* args);
PJRT_Error* PJRT_Executable_Delete(PJRT_Executable_Delete_Args* args);
PJRT_Error* PJRT_Executable_IsDeleted(PJRT_Executable_IsDeleted_Args* args);

PJRT_Error* PJRT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args);
PJRT_Error* PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args* args);
PJRT_Error* PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args);
PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args);

// Helper macros and functions

#define PJRT_RETURN_IF_ERROR(expr)                                \
  do {                                                            \
    xla::Status _status = (expr);                                 \
    if (!_status.ok()) {                                          \
      PJRT_Error* _c_status = new PJRT_Error{std::move(_status)}; \
      return _c_status;                                           \
    }                                                             \
  } while (false)

#define PJRT_ASSIGN_OR_RETURN(lhs, rexpr)                                  \
  _PJRT_ASSIGN_OR_RETURN_IMPL(_PJRT_CONCAT(_status_or_value, __COUNTER__), \
                              lhs, rexpr,                                  \
                              _PJRT_CONCAT(_c_status, __COUNTER__));

#define _PJRT_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr, c_status) \
  auto statusor = (rexpr);                                          \
  if (!statusor.ok()) {                                             \
    PJRT_Error* c_status = new PJRT_Error();                        \
    c_status->status = statusor.status();                           \
    return c_status;                                                \
  }                                                                 \
  lhs = std::move(*statusor)

#define _PJRT_CONCAT(x, y) _PJRT_CONCAT_IMPL(x, y)
#define _PJRT_CONCAT_IMPL(x, y) x##y

// Helper function for checking C API argument struct sizes. Returns a non-OK
// status if the expected and actual sizes aren't equal (i.e. no ABI
// compatibility guarantees).
xla::Status CheckMatchingStructSizes(absl::string_view struct_name,
                                     size_t expected_size, size_t actual_size);

// Helper function
std::string StructSizeErrorMsg(absl::string_view struct_name,
                               size_t expected_size, size_t actual_size);

}  // namespace pjrt

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_WRAPPER_IMPL_H_
