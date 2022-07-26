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

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/compiler/xla/shape.h"

// TODO(b/238999986): Remove this.
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"

namespace pjrt {

xla::Status CheckMatchingStructSizes(absl::string_view struct_name,
                                     size_t expected_size, size_t actual_size) {
  if (expected_size != actual_size) {
    return tensorflow::errors::InvalidArgument(
        StructSizeErrorMsg(struct_name, expected_size, actual_size));
  }
  return tensorflow::OkStatus();
}

std::string StructSizeErrorMsg(absl::string_view struct_name,
                               size_t expected_size, size_t actual_size) {
  return absl::StrCat("Unexpected ", struct_name, " size: expected ",
                      expected_size, ", got ", actual_size,
                      ". Check installed software versions.");
}

// Returns C device from wrapped C++ device.
static PJRT_Device* GetCDevice(const PJRT_Client* client,
                               const xla::PjRtDevice* device) {
  auto c_device_map = client->c_device_from_cpp_device;
  auto iter = c_device_map.find(device);
  CHECK(iter != c_device_map.end());
  return iter->second;
}

// ---------------------------------- Errors -----------------------------------

void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args) {
  xla::Status struct_size_check = CheckMatchingStructSizes(
      "PJRT_Error_Destroy_Args", PJRT_Error_Destroy_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.error_message();
  }
  if (args->struct_size >= PJRT_STRUCT_SIZE(PJRT_Error_Destroy_Args, error)) {
    delete args->error;
  }
}

void PJRT_Error_Message(PJRT_Error_Message_Args* args) {
  xla::Status struct_size_check = CheckMatchingStructSizes(
      "PJRT_Error_Message_Args", PJRT_Error_Message_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.error_message();
  }
  if (args->struct_size >= PJRT_STRUCT_SIZE(PJRT_Error_Destroy_Args, error)) {
    xla::Status* status = &args->error->status;
    args->message = status->error_message().data();
    args->message_size = status->error_message().size();
  }
}

// ---------------------------------- Client -----------------------------------

PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Destroy_Args", PJRT_Client_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->client;
  return nullptr;
}

PJRT_Error* PJRT_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_CLient_ProcessIndex_Args",
      PJRT_Client_ProcessIndex_Args_STRUCT_SIZE, args->struct_size));
  args->process_index = args->client->client->process_index();
  return nullptr;
}

PJRT_Error* PJRT_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_PlatformName_Args",
      PJRT_Client_PlatformName_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view platform_name = args->client->client->platform_name();
  args->platform_name = platform_name.data();
  args->platform_name_size = platform_name.size();
  return nullptr;
}

PJRT_Error* PJRT_Client_PlatformVersion(
    PJRT_Client_PlatformVersion_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_CLient_PlatformVersion_Args",
      PJRT_Client_PlatformVersion_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view platform_version = args->client->client->platform_version();
  args->platform_version = platform_version.data();
  args->platform_version_size = platform_version.size();
  return nullptr;
}

PJRT_Error* PJRT_Client_Devices(PJRT_Client_Devices_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Devices_Args", PJRT_Client_Devices_Args_STRUCT_SIZE,
      args->struct_size));
  args->num_devices = args->client->devices.size();
  args->devices = args->client->devices.data();
  return nullptr;
}

PJRT_Error* PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_AddressableDevices_Args",
      PJRT_Client_AddressableDevices_Args_STRUCT_SIZE, args->struct_size));
  args->num_addressable_devices = args->client->addressable_devices.size();
  args->addressable_devices = args->client->addressable_devices.data();
  return nullptr;
}

PJRT_Error* PJRT_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_LookupDevice_Args",
      PJRT_Client_LookupDevice_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(xla::PjRtDevice * device,
                        args->client->client->LookupDevice(args->id));
  args->device = GetCDevice(args->client, device);
  return nullptr;
}

// --------------------------------- Devices -----------------------------------

PJRT_Error* PJRT_Device_Id(PJRT_Device_Id_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes("PJRT_Device_Id_Args",
                                                PJRT_Device_Id_Args_STRUCT_SIZE,
                                                args->struct_size));

  args->id = args->device->device->id();
  return nullptr;
}

PJRT_Error* PJRT_Device_ProcessIndex(PJRT_Device_ProcessIndex_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_ProcessIndex_Args",
      PJRT_Device_ProcessIndex_Args_STRUCT_SIZE, args->struct_size));
  args->process_index = args->device->device->process_index();
  return nullptr;
}

PJRT_Error* PJRT_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_IsAddressable_Args",
      PJRT_Device_IsAddressable_Args_STRUCT_SIZE, args->struct_size));
  args->is_addressable = args->device->device->IsAddressable();
  return nullptr;
}

PJRT_Error* PJRT_Device_Attributes(PJRT_Device_Attributes_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_Attributes_Args", PJRT_Device_Attributes_Args_STRUCT_SIZE,
      args->struct_size));

  // Returns the attributes that were initialized during PJRT_Device creation.
  args->num_attributes = args->device->attributes.size();
  args->attributes = args->device->attributes.data();

  return nullptr;
}

PJRT_Error* PJRT_Device_Kind(PJRT_Device_Kind_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_Kind_Args", PJRT_Device_Kind_Args_STRUCT_SIZE,
      args->struct_size));

  args->device_kind = args->device->device->device_kind().data();
  args->device_kind_size = args->device->device->device_kind().size();
  return nullptr;
}

PJRT_Error* PJRT_Device_LocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_LocalHardwareId_Args",
      PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE, args->struct_size));
  args->local_hardware_id = args->device->device->local_hardware_id();
  return nullptr;
}

// ------------------------------- Executables ---------------------------------

PJRT_Error* PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_Destroy_Args", PJRT_Executable_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->executable;
  return nullptr;
}

PJRT_Error* PJRT_Executable_Name(PJRT_Executable_Name_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_Name_Args", PJRT_Executable_Name_Args_STRUCT_SIZE,
      args->struct_size));
  absl::string_view executable_name = args->executable->executable->name();
  args->executable_name = executable_name.data();
  args->executable_name_size = executable_name.size();
  return nullptr;
}

// Searches `device_list` for a PJRT_Device* that wraps a provided
// `xla::PjRtDevice *` (`cpp_device`). If a match is found, that PJRT_Device* is
// returned. Otherwise, returns nullptr.
static PJRT_Device* FindDeviceWrapper(
    xla::PjRtDevice* cpp_device, absl::Span<PJRT_Device* const> device_list) {
  for (PJRT_Device* device : device_list) {
    if (device->device == cpp_device) {
      return device;
    }
  }
  return nullptr;
}

static void PopulatePjrtExecutableAddressableDevices(
    PJRT_Executable* executable) {
  CHECK(executable->client != nullptr) << ": client was null";
  absl::Span<xla::PjRtDevice* const> cpp_devices =
      executable->executable->addressable_devices();
  const size_t num_addressable_devices = cpp_devices.size();
  std::vector<PJRT_Device*>& exec_devices = executable->addressable_devices;
  exec_devices.reserve(num_addressable_devices);

  const std::vector<PJRT_Device*>& client_devices =
      executable->client->addressable_devices;

  CHECK(client_devices.size() >= num_addressable_devices)
      << ": client->addressable_devices is not bigger than "
         "executable->addressable_devices()";

  for (int i = 0; i < num_addressable_devices; ++i) {
    xla::PjRtDevice* cpp_device = cpp_devices[i];
    PJRT_Device* device = FindDeviceWrapper(cpp_device, client_devices);
    CHECK(device != nullptr)
        << ": No PJRT_Device* found in client->addressable_devices"
        << " that wraps executable->addressable_devices()[" << i << "] ("
        << cpp_devices[i] << ")";
    exec_devices.push_back(device);
  }
}

PJRT_Error* PJRT_Executable_AddressableDevices(
    PJRT_Executable_AddressableDevices_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_AddressableDevices_Args",
      PJRT_Executable_AddressableDevices_Args_STRUCT_SIZE, args->struct_size));

  // TODO(b/237545405): Implement creation methods for PJRT_Executable that can
  // populate addressable_devices on instantiation,  and use this logic there
  if (!args->executable->populated) {
    PopulatePjrtExecutableAddressableDevices(args->executable);
    args->executable->populated = true;
  }

  args->num_addressable_devices = args->executable->addressable_devices.size();
  args->addressable_devices = args->executable->addressable_devices.data();
  return nullptr;
}

PJRT_Error* PJRT_Executable_Delete(PJRT_Executable_Delete_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_Delete_Args", PJRT_Executable_Delete_Args_STRUCT_SIZE,
      args->struct_size));
  args->executable->executable->Delete();
  return nullptr;
}

PJRT_Error* PJRT_Executable_IsDeleted(PJRT_Executable_IsDeleted_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_IsDeleted_Args",
      PJRT_Executable_IsDeleted_Args_STRUCT_SIZE, args->struct_size));
  args->is_deleted = args->executable->executable->IsDeleted();
  return nullptr;
}

static std::vector<std::vector<xla::PjRtBuffer*>> Convert2DCBuffersToCppBuffers(
    PJRT_Buffer*** c_lists, size_t outer_size, size_t inner_size) {
  std::vector<std::vector<xla::PjRtBuffer*>> cpp_lists;
  cpp_lists.reserve(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    auto& cpp_list = cpp_lists.emplace_back();
    cpp_list.reserve(inner_size);
    for (int j = 0; j < inner_size; ++j) {
      cpp_list.push_back(c_lists[i][j]->buffer.get());
    }
  }
  return cpp_lists;
}

PJRT_Error* PJRT_Executable_Execute(PJRT_Executable_Execute_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_Execute_Args", PJRT_Executable_Execute_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes("PJRT_ExecuteOptions",
                                                PJRT_ExecuteOptions_STRUCT_SIZE,
                                                args->options->struct_size));
  xla::ExecuteOptions options;
  options.launch_id = args->options->launch_id;
  options.strict_shape_checking = true;
  options.arguments_are_tupled = false;
  options.untuple_result = true;
  options.context = nullptr;
  options.multi_slice_config = nullptr;
  std::vector<std::vector<xla::PjRtBuffer*>> cpp_argument_lists =
      Convert2DCBuffersToCppBuffers(args->argument_lists, args->num_devices,
                                    args->num_args);

  PJRT_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
          cpp_buffer_lists,
      args->executable->executable->Execute(cpp_argument_lists, options));

  for (int i = 0; i < cpp_buffer_lists.size(); ++i) {
    for (int j = 0; j < cpp_buffer_lists[i].size(); ++j) {
      args->output_lists[i][j] =
          new PJRT_Buffer{std::move(cpp_buffer_lists[i][j])};
    }
  }
  return nullptr;
}

// ---------------------------------- Buffers ----------------------------------
// TODO(b/238999986): Replace this with decomposed shape methods.
PJRT_Error* PJRT_Buffer_OnDeviceTrimmedShape(
    PJRT_Buffer_OnDeviceTrimmedShape_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_OnDeviceTrimmedShape_Args",
      PJRT_Buffer_OnDeviceTrimmedShape_Args_STRUCT_SIZE, args->struct_size));

  const xla::Shape& shape = args->buffer->buffer->on_device_shape();
  args->element_type = shape.element_type();
  ApiConverter::CreateVector(shape.dimensions(), &args->dimensions);
  ApiConverter::CreateVector(shape.dynamic_dimensions(),
                             &args->dynamic_dimensions);

  if (shape.has_layout()) {
    ApiConverter::ToC(shape.layout(), &args->layout);
  } else {
    args->layout.format = xla::INVALID_FORMAT;
  }

  return nullptr;
}

PJRT_Error* PJRT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_OnDeviceSizeInBytes_Args",
      PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(args->on_device_size_in_bytes,
                        args->buffer->buffer->GetOnDeviceSizeInBytes());
  return nullptr;
}

PJRT_Error* PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_Delete_Args", PJRT_Buffer_Delete_Args_STRUCT_SIZE,
      args->struct_size));
  args->buffer->buffer->Delete();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_IsDeleted_Args", PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE,
      args->struct_size));
  args->is_deleted = args->buffer->buffer->IsDeleted();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_IsOnCpu_Args", PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE,
      args->struct_size));
  args->is_on_cpu = args->buffer->buffer->IsOnCpu();
  return nullptr;
}

}  // namespace pjrt
