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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

// TODO(b/238999986): Remove this.
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_conversions.h"

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

std::string ProgramFormatErrorMsg(absl::string_view program_format) {
  return absl::StrCat("Unknown program format '", program_format, "'.");
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
    const xla::Status* status = &args->error->status;
    args->message = status->error_message().data();
    args->message_size = status->error_message().size();
  }
}

PJRT_Error* PJRT_Error_GetCode(PJRT_Error_GetCode_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Error_GetCode_Args", PJRT_Error_GetCode_Args_STRUCT_SIZE,
      args->struct_size));
  args->code = StatusCodeToPjrtErrorCode(args->error->status.code());
  return nullptr;
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

static xla::StatusOr<xla::CompileOptions>
ConvertCCompileOptionstoCppCompileOptions(PJRT_CompileOptions* c_option) {
  xla::CompileOptions ret;
  ret.parameter_is_tupled_arguments = c_option->parameter_is_tupled_arguments;
  if (c_option->device_ordinal != -1) {
    ret.executable_build_options.set_device_ordinal(c_option->device_ordinal);
  }
  ret.executable_build_options.set_num_replicas(c_option->num_replicas);
  ret.executable_build_options.set_num_partitions(c_option->num_partitions);
  ret.executable_build_options.set_use_spmd_partitioning(
      c_option->use_spmd_partitioning);
  ret.executable_build_options.set_allow_spmd_sharding_propagation_to_output(
      c_option->allow_spmd_sharding_propagation_to_output);
  if (c_option->device_assignment_size > 0) {
    absl::string_view device_assignment_sv(c_option->device_assignment,
                                           c_option->device_assignment_size);
    std::string device_assignment_str(device_assignment_sv);
    xla::DeviceAssignmentProto proto;
    proto.ParseFromString(device_assignment_str);
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::DeviceAssignment> device_assignment,
        xla::DeviceAssignment::Deserialize(proto));
    ret.executable_build_options.set_device_assignment(*device_assignment);
  }
  return ret;
}

PJRT_Error* PJRT_Client_Compile(PJRT_Client_Compile_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Compile_Args", PJRT_Client_Compile_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes("PJRT_CompileOptions",
                                                PJRT_CompileOptions_STRUCT_SIZE,
                                                args->options->struct_size));
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Program", PJRT_Program_STRUCT_SIZE, args->program->struct_size));

  PJRT_ASSIGN_OR_RETURN(
      xla::CompileOptions options,
      ConvertCCompileOptionstoCppCompileOptions(args->options));

  absl::string_view format_str(args->program->format,
                               args->program->format_size);
  absl::string_view module_str(args->program->code, args->program->code_size);

  std::unique_ptr<xla::PjRtLoadedExecutable> executable;
  if (format_str == pjrt::kMlirFormat) {
    mlir::MLIRContext context;
    PJRT_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          xla::ParseMlirModuleString(module_str, context));

    PJRT_ASSIGN_OR_RETURN(executable,
                          args->client->client->Compile(*module, options));
  } else if (format_str == pjrt::kHloFormat) {
    xla::HloModuleProto module_proto;
    module_proto.ParseFromString(module_str);
    xla::XlaComputation computation(module_proto);
    PJRT_ASSIGN_OR_RETURN(executable,
                          args->client->client->Compile(computation, options));
  } else {
    PJRT_RETURN_IF_ERROR(
        tensorflow::errors::InvalidArgument(ProgramFormatErrorMsg(format_str)));
  }
  // TODO(b/237545405): Implement creation methods for PJRT_Executable.
  args->executable = new PJRT_Executable{std::move(executable), args->client};
  PopulatePjrtExecutableAddressableDevices(args->executable);
  args->executable->populated = true;
  return nullptr;
}

static void PopulateDeviceAssignment(int* const device_assignment_buffer,
                                     int num_replicas, int num_partitions,
                                     xla::DeviceAssignment device_assignment) {
  int* iterator = device_assignment_buffer;
  for (int replica = 0; replica < num_replicas; ++replica) {
    for (int partition = 0; partition < num_partitions; ++partition) {
      *(iterator++) = device_assignment(replica, partition);
    }
  }
}

PJRT_Error* PJRT_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_DefaultAssignment_Args",
      PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE, args->struct_size));

  const int replicas = args->num_replicas;
  const int partitions = args->num_partitions;
  const size_t buffer_size = args->default_assignment_size;
  if (buffer_size < replicas * partitions) {
    xla::Status status = tensorflow::errors::FailedPrecondition(
        absl::StrCat(__func__, ": `default_assignment_size` ", buffer_size,
                     " < `num_replicas * num_partitions`, ", replicas, " * ",
                     partitions, " = ", replicas * partitions));
    return new PJRT_Error{status};
  }

  PJRT_ASSIGN_OR_RETURN(
      xla::DeviceAssignment device_assignment,
      args->client->client->GetDefaultDeviceAssignment(replicas, partitions));

  PopulateDeviceAssignment(args->default_assignment, replicas, partitions,
                           std::move(device_assignment));
  return nullptr;
}

PJRT_Error* PJRT_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_BufferFromHostBuffer_Args",
      PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE, args->struct_size));

  absl::Span<const int64_t> dims =
      absl::Span<const int64_t>(args->dims, args->num_dims);

  std::optional<absl::Span<int64_t const>> byte_strides = std::nullopt;
  if (args->byte_strides != nullptr) {
    byte_strides =
        absl::Span<const int64_t>(args->byte_strides, args->num_byte_strides);
  }

  xla::PjRtFuture<xla::Status>::Promise promise =
      xla::PjRtFuture<xla::Status>::CreatePromise();

  std::function<void()> on_done_with_host_buffer = [promise]() mutable {
    promise.Set(xla::Status::OK());
  };

  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      args->client->client->BufferFromHostBuffer(
          args->data, ::pjrt::ConvertFromPjRtBufferType(args->type), dims,
          byte_strides,
          ::pjrt::ConvertFromPjRtHostBufferSemantics(
              args->host_buffer_semantics),
          on_done_with_host_buffer, args->device->device));

  args->buffer = new PJRT_Buffer{std::move(buffer), args->client};
  args->done_with_host_buffer =
      new PJRT_Event{xla::PjRtFuture<xla::Status>(std::move(promise))};

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

PJRT_Error* PJRT_Device_DebugString(PJRT_Device_DebugString_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_DebugString_Args", PJRT_Device_DebugString_Args_STRUCT_SIZE,
      args->struct_size));

  args->debug_string = args->device->device->DebugString().data();
  args->debug_string_size = args->device->device->DebugString().size();
  return nullptr;
}

PJRT_Error* PJRT_Device_ToString(PJRT_Device_ToString_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Device_ToString_Args", PJRT_Device_ToString_Args_STRUCT_SIZE,
      args->struct_size));
  args->to_string = args->device->device->ToString().data();
  args->to_string_size = args->device->device->ToString().size();
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
      args->output_lists[i][j] = new PJRT_Buffer{
          std::move(cpp_buffer_lists[i][j]), args->executable->client};
    }
  }
  return nullptr;
}

PJRT_Error* PJRT_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Executable_NumOutputs_Args",
      PJRT_Executable_NumOutputs_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<xla::HloModule>> hlo_modules,
      args->executable->executable->GetHloModules());
  if (hlo_modules.empty()) {
    return new PJRT_Error{
        xla::InvalidArgument("Can't get number of executable outputs, Hlo "
                             "modules is empty for executable %s.",
                             args->executable->executable->name())};
  }
  if (hlo_modules.size() != 1) {
    return new PJRT_Error{
        xla::Unimplemented("MPMD execution not supported by PJRT C API (in "
                           "function PJRT_Executable_NumOutputs).")};
  }
  xla::Shape shape = hlo_modules[0].get()->result_shape();
  if (shape.IsTuple()) {
    args->num_outputs = shape.tuple_shapes_size();
  } else {
    // The output size is 1 is it is not a tuple.
    args->num_outputs = 1;
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
    args->has_layout = true;
    ApiConverter::ToC(shape.layout(), &args->layout);
  } else {
    args->has_layout = false;
  }

  return nullptr;
}

PJRT_Error* PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_Destroy_Args", PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->buffer;
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

PJRT_Error* PJRT_Buffer_Device(PJRT_Buffer_Device_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_Device_Args", PJRT_Buffer_Device_Args_STRUCT_SIZE,
      args->struct_size));
  args->device = FindDeviceWrapper(args->buffer->buffer->device(),
                                   args->buffer->client->addressable_devices);
  CHECK(args->device != nullptr)
      << "No PJRT_Device* found in the client's `addressable_devices` that "
         "wraps this "
      << args->buffer->buffer->device()->DebugString();
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

PJRT_Error* PJRT_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_CopyToDevice_Args",
      PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer> dst_buffer,
      args->buffer->buffer->CopyToDevice(args->dst_device->device));
  args->dst_buffer =
      new PJRT_Buffer{std::move(dst_buffer), args->buffer->client};
  return nullptr;
}

PJRT_Error* PJRT_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_ToHostBuffer_Args",
      PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE, args->struct_size));

  const xla::Shape& host_shape = xla::ShapeUtil::DeviceShapeToHostShape(
      args->src->buffer->on_device_shape());

  size_t host_buffer_size = xla::ShapeUtil::ByteSizeOfElements(host_shape);

  if (args->dst == nullptr) {
    args->dst_size = host_buffer_size;
    return nullptr;
  }

  if (args->dst_size < host_buffer_size) {
    return new PJRT_Error{
        xla::InvalidArgument("`dst_size` must be >= %zu, got %zu.",
                             host_buffer_size, args->dst_size)};
  }

  auto literal = std::make_unique<xla::MutableBorrowingLiteral>(
      static_cast<char*>(args->dst), host_shape);
  xla::PjRtFuture<xla::Status> future =
      args->src->buffer->ToLiteral(literal.get());

  args->event = new PJRT_Event{std::move(future)};
  args->event->future.OnReady(
      [literal{std::move(literal)}](xla::Status status) {
        /* To keep literal alive */
      });

  return nullptr;
}

PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_IsOnCpu_Args", PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE,
      args->struct_size));
  args->is_on_cpu = args->buffer->buffer->IsOnCpu();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Buffer_ReadyEvent_Args", PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE,
      args->struct_size));
  xla::PjRtFuture<xla::Status> wrapped_promise =
      args->buffer->buffer->GetReadyFuture();
  args->event = new PJRT_Event{std::move(wrapped_promise)};
  return nullptr;
}

// -------------------------------- Events -------------------------------------

PJRT_Error* PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_Destroy", PJRT_Event_Destroy_Args_STRUCT_SIZE,
      args->struct_size));

  delete args->event;
  return nullptr;
}

PJRT_Error* PJRT_Event_IsReady(PJRT_Event_IsReady_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_IsReady", PJRT_Event_IsReady_Args_STRUCT_SIZE,
      args->struct_size));

  args->is_ready = args->event->future.IsReady();
  return nullptr;
}

PJRT_Error* PJRT_Event_Await(PJRT_Event_Await_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_Await", PJRT_Event_Await_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Event* event = args->event;
  event->status.emplace(event->future.Await());
  PJRT_RETURN_IF_ERROR(event->status.value());
  return nullptr;
}

PJRT_Error* PJRT_Event_Error(PJRT_Event_Error_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_Error", PJRT_Event_Error_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Event* event = args->event;
  CHECK(event->future.IsReady());
  if (!event->status.has_value()) {
    PJRT_Event_Await_Args await_args;
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.priv = nullptr;
    await_args.event = event;
    return PJRT_Event_Await(&await_args);
  }
  PJRT_RETURN_IF_ERROR(event->status.value());
  return nullptr;
}

PJRT_Error* PJRT_Event_OnReady(PJRT_Event_OnReady_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Event_OnReady", PJRT_Event_OnReady_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Event_OnReadyCallback callback = args->callback;
  void* user_arg = args->user_arg;
  auto impl_callback = [callback, user_arg](xla::Status status) -> void {
    PJRT_Error* error = nullptr;
    if (!status.ok()) {
      error = new PJRT_Error{status};
    }
    callback(error, user_arg);
  };
  args->event->future.OnReady(impl_callback);
  return nullptr;
}

}  // namespace pjrt
