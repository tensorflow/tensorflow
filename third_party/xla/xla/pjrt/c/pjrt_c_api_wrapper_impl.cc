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

#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/compile_options.pb.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"

namespace pjrt {

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

// Returns C memory from wrapped C++ memory.
static PJRT_Memory* GetCMemory(const PJRT_Client* client,
                               const xla::PjRtMemorySpace* memory) {
  auto c_memory_map = client->c_memory_from_cpp_memory;
  auto iter = c_memory_map.find(memory);
  CHECK(iter != c_memory_map.end());
  return iter->second;
}

// Performs one-time cost-analysis on an executable, and populates its cost
// analysis properties. After this returns successfully, cost analysis
// properties of the executable can be accessed without mutex.
static absl::Status PopulateExecutableCostAnalysis(
    PJRT_Executable* executable) {
  // Call GetCostAnalysis in the underlying PjRtExecutable
  using PropertiesMapType =
      absl::flat_hash_map<std::string, xla::PjRtValueType>;
  TF_ASSIGN_OR_RETURN(const PropertiesMapType properties,
                      executable->get()->GetCostAnalysis());
  // If no output, return empty result
  if (properties.empty()) {
    return absl::OkStatus();
  }

  // Copy each returned property to cost analysis vectors in PJRT_Executable
  std::vector<PJRT_NamedValue>& cost_analysis_properties =
      executable->cost_analysis_properties;
  cost_analysis_properties.resize((properties.size()));
  std::vector<std::string>& cost_analysis_names =
      executable->cost_analysis_names;
  cost_analysis_names.resize(properties.size());
  size_t i = 0;
  for (const auto& property : properties) {
    PJRT_NamedValue& cost_analysis_property = cost_analysis_properties[i];
    std::string& property_name = cost_analysis_names[i];

    cost_analysis_property.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    cost_analysis_property.extension_start = nullptr;

    property_name = property.first;
    cost_analysis_property.name = property_name.c_str();
    cost_analysis_property.name_size = property_name.size();

    const xla::PjRtValueType& property_value = property.second;
    CHECK(std::holds_alternative<float>(property_value))
        << property_value.index();
    cost_analysis_property.type = PJRT_NamedValue_Type::PJRT_NamedValue_kFloat;
    cost_analysis_property.float_value = std::get<float>(property_value);
    cost_analysis_property.value_size = 1;

    ++i;
  }

  return absl::OkStatus();
}

static absl::Status PopulateExecutableOutputElementTypes(
    PJRT_Executable* executable) {
  TF_ASSIGN_OR_RETURN(auto output_types,
                      executable->get()->GetOutputElementTypes());
  if (output_types.empty()) {
    return xla::InvalidArgument(
        "Can't get output element types, the list is empty for executable "
        "%s.",
        executable->get()->name());
  }
  if (output_types.size() != 1) {
    return xla::Unimplemented(
        "MPMD execution not supported by PJRT C API (in function "
        "PJRT_Executable_OutputElementTypes).");
  }

  std::vector<xla::PrimitiveType>& inner_output_types = output_types[0];
  std::vector<PJRT_Buffer_Type>& out_types = executable->out_types;
  out_types.reserve(inner_output_types.size());
  for (const auto& element_type : inner_output_types) {
    out_types.push_back(ConvertToPjRtBufferType(element_type));
  }

  return absl::OkStatus();
}

static absl::Status PopulateExecutableOutputDimensions(
    PJRT_Executable* executable) {
  TF_ASSIGN_OR_RETURN(auto output_dims,
                      executable->get()->GetOutputDimensions());
  if (output_dims.empty()) {
    return xla::InvalidArgument(
        "Can't get output dimensions, the list is empty for executable %s.",
        executable->get()->name());
  }
  if (output_dims.size() != 1) {
    return xla::Unimplemented(
        "MPMD execution not supported by PJRT C API (in function "
        "PJRT_Executable_OutputDimensions).");
  }

  std::vector<xla::DimensionVector>& inner_output_dims = output_dims[0];
  std::vector<size_t>& out_dimension_sizes = executable->out_dimension_sizes;
  out_dimension_sizes.reserve(inner_output_dims.size());
  size_t total_size = 0;
  for (const auto& dimension_vector : inner_output_dims) {
    out_dimension_sizes.push_back(dimension_vector.size());
    total_size += dimension_vector.size();
  }
  std::vector<int64_t>& out_dimensions = executable->out_dimensions;
  out_dimensions.reserve(total_size);
  for (const auto& dimension_vector : inner_output_dims) {
    for (int i = 0; i < dimension_vector.size(); ++i) {
      out_dimensions.push_back(dimension_vector[i]);
    }
  }

  return absl::OkStatus();
}

static absl::Status PopulateExecutableOutputMemoryKinds(
    PJRT_Executable* executable) {
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<absl::string_view>> output_memories,
      executable->get()->GetOutputMemoryKinds());
  if (output_memories.empty()) {
    return xla::InvalidArgument(
        "Can't get output memory kinds, the list is empty for executable %s.",
        executable->get()->name());
  }
  if (output_memories.size() != 1) {
    return xla::Unimplemented(
        "MPMD execution not supported by PJRT C API (in "
        "function PJRT_Executable_GetOutputMemoryKinds).");
  }

  std::vector<absl::string_view>& inner_output_memories = output_memories[0];
  std::vector<const char*>& memory_kinds = executable->memory_kinds;
  std::vector<size_t>& memory_kind_sizes = executable->memory_kind_sizes;
  memory_kinds.reserve(inner_output_memories.size());
  memory_kind_sizes.reserve(inner_output_memories.size());
  for (absl::string_view memory : inner_output_memories) {
    memory_kinds.push_back(memory.data());
    memory_kind_sizes.push_back(memory.size());
  }

  return absl::OkStatus();
}

class CApiKeyValueStore : public xla::KeyValueStoreInterface {
 public:
  CApiKeyValueStore(PJRT_KeyValueGetCallback c_get_callback, void* get_user_arg,
                    PJRT_KeyValueTryGetCallback c_try_get_callback,
                    void* try_get_user_arg,
                    PJRT_KeyValuePutCallback c_put_callback, void* put_user_arg)
      : c_get_callback_(c_get_callback),
        get_user_arg_(get_user_arg),
        c_try_get_callback_(c_try_get_callback),
        try_get_user_arg_(try_get_user_arg),
        c_put_callback_(c_put_callback),
        put_user_arg_(put_user_arg) {}

  absl::StatusOr<std::string> Get(absl::string_view key,
                                  absl::Duration timeout) override {
    PJRT_CallbackError callback_error = [](PJRT_Error_Code code,
                                           const char* message,
                                           size_t message_size) {
      return new PJRT_Error{absl::Status(static_cast<absl::StatusCode>(code),
                                         std::string(message, message_size))};
    };
    PJRT_KeyValueGetCallback_Args args;
    args.key = key.data();
    args.key_size = key.size();
    args.timeout_in_ms = timeout / absl::Milliseconds(1);
    args.callback_error = &callback_error;
    args.user_arg = get_user_arg_;
    std::unique_ptr<PJRT_Error> error(c_get_callback_(&args));
    if (error != nullptr) {
      return error->status;
    }
    auto result = std::string(args.value, args.value_size);
    args.value_deleter_callback(args.value);
    return result;
  }

  absl::StatusOr<std::string> TryGet(absl::string_view key) override {
    PJRT_CallbackError callback_error = [](PJRT_Error_Code code,
                                           const char* message,
                                           size_t message_size) {
      return new PJRT_Error{absl::Status(static_cast<absl::StatusCode>(code),
                                         std::string(message, message_size))};
    };
    PJRT_KeyValueTryGetCallback_Args args;
    args.key = key.data();
    args.key_size = key.size();
    args.callback_error = &callback_error;
    args.user_arg = try_get_user_arg_;
    std::unique_ptr<PJRT_Error> error(c_try_get_callback_(&args));
    if (error != nullptr) {
      return error->status;
    }
    auto result = std::string(args.value, args.value_size);
    args.value_deleter_callback(args.value);
    return result;
  }

  absl::Status Set(absl::string_view key, absl::string_view value) override {
    PJRT_CallbackError callback_error = [](PJRT_Error_Code code,
                                           const char* message,
                                           size_t message_size) {
      return new PJRT_Error{absl::Status(static_cast<absl::StatusCode>(code),
                                         std::string(message, message_size))};
    };
    PJRT_KeyValuePutCallback_Args args;
    args.key = key.data();
    args.key_size = key.size();
    args.value = value.data();
    args.value_size = value.size();
    args.callback_error = &callback_error;
    args.user_arg = put_user_arg_;
    std::unique_ptr<PJRT_Error> error(c_put_callback_(&args));
    if (error != nullptr) {
      return error->status;
    }
    return absl::OkStatus();
  }

 private:
  PJRT_KeyValueGetCallback c_get_callback_;
  void* get_user_arg_;
  PJRT_KeyValueTryGetCallback c_try_get_callback_;
  void* try_get_user_arg_;
  PJRT_KeyValuePutCallback c_put_callback_;
  void* put_user_arg_;
};

std::shared_ptr<xla::KeyValueStoreInterface> ToCppKeyValueStore(
    PJRT_KeyValueGetCallback c_get_callback, void* get_user_arg,
    PJRT_KeyValueTryGetCallback c_try_get_callback, void* try_get_user_arg,
    PJRT_KeyValuePutCallback c_put_callback, void* put_user_arg) {
  if (c_get_callback == nullptr || c_try_get_callback == nullptr ||
      c_put_callback == nullptr) {
    return nullptr;
  }
  return std::make_shared<CApiKeyValueStore>(
      c_get_callback, get_user_arg, c_try_get_callback, try_get_user_arg,
      c_put_callback, put_user_arg);
}

// ---------------------------------- Errors -----------------------------------

void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args) {
  absl::Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
      "PJRT_Error_Destroy_Args", PJRT_Error_Destroy_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  if (args->struct_size >= PJRT_STRUCT_SIZE(PJRT_Error_Destroy_Args, error)) {
    delete args->error;
  }
}

void PJRT_Error_Message(PJRT_Error_Message_Args* args) {
  absl::Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
      "PJRT_Error_Message_Args", PJRT_Error_Message_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  if (args->struct_size >= PJRT_STRUCT_SIZE(PJRT_Error_Destroy_Args, error)) {
    const absl::Status* status = &args->error->status;
    args->message = status->message().data();
    args->message_size = status->message().size();
  }
}

PJRT_Error* PJRT_Error_GetCode(PJRT_Error_GetCode_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Error_GetCode_Args", PJRT_Error_GetCode_Args_STRUCT_SIZE,
      args->struct_size));
  args->code = StatusCodeToPjrtErrorCode(
      static_cast<absl::StatusCode>(args->error->status.code()));
  return nullptr;
}

// ---------------------------------- Plugin -----------------------------------

PJRT_Error* PJRT_Plugin_Attributes_Empty(PJRT_Plugin_Attributes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Plugin_Attributes_Args", PJRT_Plugin_Attributes_Args_STRUCT_SIZE,
      args->struct_size));
  args->num_attributes = 0;
  args->attributes = nullptr;
  return nullptr;
}

PJRT_Error* PJRT_Plugin_Attributes_Xla(PJRT_Plugin_Attributes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Plugin_Attributes_Args", PJRT_Plugin_Attributes_Args_STRUCT_SIZE,
      args->struct_size));
  const std::vector<PJRT_NamedValue>& attributes =
      pjrt::GetXlaPluginCAttributes();
  args->num_attributes = attributes.size();
  args->attributes = attributes.data();
  return nullptr;
}

PJRT_Error* PJRT_Plugin_Initialize_NoOp(PJRT_Plugin_Initialize_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Plugin_Initialize_Args", PJRT_Plugin_Initialize_Args_STRUCT_SIZE,
      args->struct_size));
  return nullptr;
}

// ---------------------------------- Client -----------------------------------

PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Destroy_Args", PJRT_Client_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->client;
  return nullptr;
}

PJRT_Error* PJRT_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_CLient_ProcessIndex_Args",
      PJRT_Client_ProcessIndex_Args_STRUCT_SIZE, args->struct_size));
  args->process_index = args->client->client->process_index();
  return nullptr;
}

PJRT_Error* PJRT_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_PlatformName_Args",
      PJRT_Client_PlatformName_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view platform_name = args->client->client->platform_name();
  args->platform_name = platform_name.data();
  args->platform_name_size = platform_name.size();
  return nullptr;
}

PJRT_Error* PJRT_Client_PlatformVersion(
    PJRT_Client_PlatformVersion_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_CLient_PlatformVersion_Args",
      PJRT_Client_PlatformVersion_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view platform_version = args->client->client->platform_version();
  args->platform_version = platform_version.data();
  args->platform_version_size = platform_version.size();
  return nullptr;
}

PJRT_Error* PJRT_Client_TopologyDescription(
    PJRT_Client_TopologyDescription_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_TopologyDescription_Args",
      PJRT_Client_TopologyDescription_Args_STRUCT_SIZE, args->struct_size));

  PJRT_RETURN_IF_ERROR(args->client->topology.status());
  args->topology = args->client->topology->get();
  return nullptr;
}

PJRT_Error* PJRT_Client_Devices(PJRT_Client_Devices_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Devices_Args", PJRT_Client_Devices_Args_STRUCT_SIZE,
      args->struct_size));
  args->num_devices = args->client->devices.size();
  args->devices = args->client->devices.data();
  return nullptr;
}

PJRT_Error* PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_AddressableDevices_Args",
      PJRT_Client_AddressableDevices_Args_STRUCT_SIZE, args->struct_size));
  args->num_addressable_devices = args->client->addressable_devices.size();
  args->addressable_devices = args->client->addressable_devices.data();
  return nullptr;
}

PJRT_Error* PJRT_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_LookupDevice_Args",
      PJRT_Client_LookupDevice_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      xla::PjRtDevice * device,
      args->client->client->LookupDevice(xla::PjRtGlobalDeviceId(args->id)));
  args->device = GetCDevice(args->client, device);
  return nullptr;
}

PJRT_Error* PJRT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_LookupAddressableDevice_Args",
      PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(xla::PjRtDevice * addressable_device,
                        args->client->client->LookupAddressableDevice(
                            xla::PjRtLocalDeviceId(args->local_hardware_id)));
  args->addressable_device = GetCDevice(args->client, addressable_device);
  return nullptr;
}

// TODO: b/306669267 - this method is deprecated. Return unimplemented error,
// until the next major version upgrade.
PJRT_Error* PJRT_LoadedExecutable_Fingerprint(
    PJRT_LoadedExecutable_Fingerprint_Args* args) {
  return new PJRT_Error{
      xla::Unimplemented("PJRT_LoadedExecutable_Fingerprint is deprecated, use "
                         "PJRT_Executable_Fingerprint instead.")};
}

PJRT_Error* PJRT_Client_AddressableMemories(
    PJRT_Client_AddressableMemories_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_AddressableMemories_Args",
      PJRT_Client_AddressableMemories_Args_STRUCT_SIZE, args->struct_size));
  args->num_addressable_memories = args->client->addressable_memories.size();
  args->addressable_memories = args->client->addressable_memories.data();
  return nullptr;
}

PJRT_Error* PJRT_Client_CreateBuffersForAsyncHostToDevice(
    PJRT_Client_CreateBuffersForAsyncHostToDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_CreateBuffersForAsyncHostToDevice_Args",
      PJRT_Client_CreateBuffersForAsyncHostToDevice_Args_STRUCT_SIZE,
      args->struct_size));
  std::vector<std::optional<xla::Layout>> device_layouts;
  absl::InlinedVector<xla::PjRtClient::ShapeSpec, 4> shape_specs;
  shape_specs.reserve(args->num_shape_specs);
  for (int i = 0; i < args->num_shape_specs; ++i) {
    shape_specs.push_back(pjrt::ConvertFromPjrtShapeSpec(args->shape_specs[i]));
  }
  std::optional<absl::Span<const std::optional<xla::Layout>>>
      arg_device_layouts;
  if (args->num_device_layouts == 0) {
    arg_device_layouts = std::nullopt;
  } else {
    device_layouts.reserve(args->num_device_layouts);
    for (int i = 0; i < args->num_device_layouts; ++i) {
      std::optional<xla::Layout> optional_layout;
      if (args->device_layouts[i] != nullptr) {
        xla::Layout cpp_layout;
        PJRT_Buffer_MemoryLayout* layout = args->device_layouts[i];
        switch (layout->type) {
          case PJRT_Buffer_MemoryLayout_Type::
              PJRT_Buffer_MemoryLayout_Type_Tiled: {
            PJRT_ASSIGN_OR_RETURN(cpp_layout, ConvertToLayout(layout->tiled));
            break;
          }
          case PJRT_Buffer_MemoryLayout_Type::
              PJRT_Buffer_MemoryLayout_Type_Strides: {
            PJRT_RETURN_IF_ERROR(absl::InvalidArgumentError(
                "PJRT_Buffer_MemoryLayout_Type_Strides is not supported to be "
                "converted to a xla::Layout."));
            break;
          }
          default: {
            PJRT_RETURN_IF_ERROR(absl::InvalidArgumentError(
                absl::StrCat("Unexpected PJRT_Buffer_MemoryLayout_Type type: ",
                             layout->type)));
          }
        }
        device_layouts.push_back(cpp_layout);
      } else {
        device_layouts.push_back(std::nullopt);
      }
    }
    arg_device_layouts = absl::MakeSpan(device_layouts);
  }

  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager>
          transfer_manager,
      args->client->client->CreateBuffersForAsyncHostToDevice(
          absl::MakeSpan(shape_specs), arg_device_layouts,
          args->memory->memory_space));
  args->transfer_manager = new PJRT_AsyncHostToDeviceTransferManager{
      std::move(transfer_manager), args->client};
  return nullptr;
}

PJRT_Error* PJRT_Client_DmaMap(PJRT_Client_DmaMap_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_DmaMap_Args", PJRT_Client_DmaMap_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_RETURN_IF_ERROR(args->client->client->DmaMap(args->data, args->size));
  return nullptr;
}

PJRT_Error* PJRT_Client_DmaUnmap(PJRT_Client_DmaUnmap_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_DmaUnmap_Args", PJRT_Client_DmaUnmap_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_RETURN_IF_ERROR(args->client->client->DmaUnmap(args->data));
  return nullptr;
}

// Searches `device_list` for a PJRT_Device* that wraps a provided
// `xla::PjRtDevice *` (`cpp_device`). If a match is found, that PJRT_Device*
// is returned. Otherwise, returns nullptr.
static PJRT_Device* FindDeviceWrapper(
    xla::PjRtDevice* cpp_device, absl::Span<PJRT_Device* const> device_list) {
  for (PJRT_Device* device : device_list) {
    if (device->device == cpp_device) {
      return device;
    }
  }
  return nullptr;
}

PJRT_Memory* PJRT_Client_FindMemoryWrapper(xla::PjRtMemorySpace* cpp_memory,
                                           PJRT_Client* client) {
  for (PJRT_Memory* memory : client->addressable_memories) {
    if (memory->memory_space == cpp_memory) {
      return memory;
    }
  }
  return nullptr;
}

static void PopulatePjrtExecutableAddressableDevices(
    PJRT_LoadedExecutable* executable) {
  CHECK(executable->client != nullptr) << ": client was null";
  absl::Span<xla::PjRtDevice* const> cpp_devices =
      executable->get()->addressable_devices();
  const size_t num_addressable_devices = cpp_devices.size();
  std::vector<PJRT_Device*>& exec_devices = executable->addressable_devices;
  exec_devices.reserve(num_addressable_devices);

  const std::vector<PJRT_Device*>& client_devices =
      executable->client->addressable_devices;

  CHECK_GE(client_devices.size(), num_addressable_devices);

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

//-------------------- AsyncHostToDeviceTransferManager ---------------------

PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_Destroy(
    PJRT_AsyncHostToDeviceTransferManager_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_AsyncHostToDeviceTransferManager_Destroy_Args",
      PJRT_AsyncHostToDeviceTransferManager_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->transfer_manager;
  return nullptr;
}

PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_TransferData(
    PJRT_AsyncHostToDeviceTransferManager_TransferData_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_AsyncHostToDeviceTransferManager_TransferData_Args",
      PJRT_AsyncHostToDeviceTransferManager_TransferData_Args_STRUCT_SIZE,
      args->struct_size));
  xla::PjRtFuture<>::Promise promise = xla::PjRtFuture<>::CreatePromise();
  absl::AnyInvocable<void() &&> on_done_with_d2h_transfer =
      [promise]() mutable { promise.Set(); };
  PJRT_RETURN_IF_ERROR(
      args->transfer_manager->transfer_manager->TransferRawDataToSubBuffer(
          args->buffer_index, args->data, args->offset, args->transfer_size,
          args->is_last_transfer, std::move(on_done_with_d2h_transfer)));
  args->done_with_h2d_transfer =
      new PJRT_Event{xla::PjRtFuture<>(std::move(promise))};
  return nullptr;
}

PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(
    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args",
      PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args_STRUCT_SIZE,
      args->struct_size));
  std::unique_ptr<xla::PjRtBuffer> buffer_out =
      args->transfer_manager->transfer_manager->RetrieveBuffer(
          args->buffer_index);
  args->buffer_out =
      new PJRT_Buffer{std::move(buffer_out), args->transfer_manager->client};
  return nullptr;
}

PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_Device(
    PJRT_AsyncHostToDeviceTransferManager_Device_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_AsyncHostToDeviceTransferManager_Device_Args",
      PJRT_AsyncHostToDeviceTransferManager_Device_Args_STRUCT_SIZE,
      args->struct_size));
  args->device_out =
      FindDeviceWrapper(args->transfer_manager->transfer_manager->device(),
                        args->transfer_manager->client->addressable_devices);
  CHECK(args->device_out != nullptr)
      << "No PJRT_Device* found in the client's `addressable_devices` that "
         "wraps this "
      << args->transfer_manager->transfer_manager->device()->DebugString();
  return nullptr;
}

PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_BufferCount(
    PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args",
      PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args_STRUCT_SIZE,
      args->struct_size));
  args->buffer_count = args->transfer_manager->transfer_manager->buffer_count();
  return nullptr;
}

PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_BufferSize(
    PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args",
      PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args_STRUCT_SIZE,
      args->struct_size));
  args->buffer_size =
      args->transfer_manager->transfer_manager->buffer_size(args->buffer_index);
  return nullptr;
}

PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_SetBufferError(
    PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args",
      PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args_STRUCT_SIZE,
      args->struct_size));
  auto error_message =
      absl::string_view(args->error_message, args->error_message_size);
  auto error = absl::Status(pjrt::PjrtErrorCodeToStatusCode(args->error_code),
                            error_message);
  args->transfer_manager->transfer_manager->SetBufferError(args->buffer_index,
                                                           error);
  return nullptr;
}

PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
    PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args",
      PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args_STRUCT_SIZE,
      args->struct_size));

  auto pjrt_metadata = ConvertFromPjRtNamedValueList(args->transfer_metadata,
                                                     args->num_metadata);
  absl::flat_hash_map<std::string, std::string> metadata;
  for (const auto& [key, value] : pjrt_metadata) {
    metadata[key] = std::get<std::string>(value);
  }
  args->transfer_manager->transfer_manager->AddTransferMetadata(metadata);
  return nullptr;
}

namespace {

absl::StatusOr<xla::CompileOptions> ParseCompileOptions(
    absl::string_view options_str) {
  xla::CompileOptionsProto options_proto;
  // Open source ParseFromString doesn't support string_view.
  if (!options_proto.ParseFromArray(options_str.data(), options_str.size())) {
    return tsl::errors::InvalidArgument(
        "PJRT_Client_Compile: failed to deserialize CompileOptionsProto");
  }
  return xla::CompileOptions::FromProto(options_proto);
}

using ProgramVariant =
    std::variant<mlir::OwningOpRef<mlir::ModuleOp>, xla::XlaComputation>;
absl::StatusOr<
    std::variant<mlir::OwningOpRef<mlir::ModuleOp>, xla::XlaComputation>>
ParsePjrtProgram(std::optional<mlir::MLIRContext>& context,
                 const PJRT_Program* program) {
  auto format_str = absl::string_view(program->format, program->format_size);
  auto module_str = absl::string_view(program->code, program->code_size);

  if (format_str == pjrt::kMlirFormat) {
    if (!context.has_value()) {
      context.emplace();
    }
    TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                        xla::ParseMlirModuleString(module_str, *context));

    return ProgramVariant(std::move(module));
  } else if (format_str == pjrt::kHloFormat) {
    xla::HloModuleProto module_proto;
    // Open source ParseFromString doesn't support string_view.
    if (!module_proto.ParseFromArray(module_str.data(), module_str.size())) {
      return tsl::errors::InvalidArgument(
          "PJRT_Client_Compile: failed to deserialize HloModuleProto");
    }
    return ProgramVariant(xla::XlaComputation(module_proto));
  } else {
    return tsl::errors::InvalidArgument(ProgramFormatErrorMsg(format_str));
  }
}

mlir::ModuleOp UnpackPjrtProgram(mlir::OwningOpRef<mlir::ModuleOp>& module) {
  return *module;
}
const xla::XlaComputation& UnpackPjrtProgram(
    const xla::XlaComputation& computation) {
  return computation;
}

}  // namespace

PJRT_Error* PJRT_Client_Compile(PJRT_Client_Compile_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Compile_Args", PJRT_Client_Compile_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Program", PJRT_Program_STRUCT_SIZE, args->program->struct_size));

  int64_t traceme_context_id = pjrt::GetTracemeContextId(args);
  tsl::profiler::TraceMeConsumer consumer(
      "PJRT_Client_Compile", tsl::profiler::ContextType::kPjrtLibraryCall,
      traceme_context_id);

  PJRT_ASSIGN_OR_RETURN(
      xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));

  std::optional<mlir::MLIRContext> context;
  PJRT_ASSIGN_OR_RETURN(auto module_or_hlo,
                        ParsePjrtProgram(context, args->program));
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                        std::visit(
                            [args, &options](auto& program) {
                              return args->client->client->CompileAndLoad(
                                  UnpackPjrtProgram(program), options);
                            },
                            module_or_hlo));
  args->executable =
      new PJRT_LoadedExecutable(std::move(executable), args->client);
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
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_DefaultAssignment_Args",
      PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE, args->struct_size));

  const int replicas = args->num_replicas;
  const int partitions = args->num_partitions;
  const size_t buffer_size = args->default_assignment_size;
  if (buffer_size < replicas * partitions) {
    absl::Status status = absl::FailedPreconditionError(
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
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_BufferFromHostBuffer_Args",
      PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE, args->struct_size));

  absl::Span<const int64_t> dims =
      absl::Span<const int64_t>(args->dims, args->num_dims);

  std::optional<absl::Span<int64_t const>> byte_strides = std::nullopt;
  if (args->byte_strides != nullptr) {
    byte_strides =
        absl::Span<const int64_t>(args->byte_strides, args->num_byte_strides);
  }
  std::optional<xla::Layout> layout = std::nullopt;
  if (args->device_layout != nullptr) {
    switch (args->device_layout->type) {
      case PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled: {
        PJRT_ASSIGN_OR_RETURN(layout,
                              ConvertToLayout(args->device_layout->tiled));
        break;
      }
      case PJRT_Buffer_MemoryLayout_Type::
          PJRT_Buffer_MemoryLayout_Type_Strides: {
        PJRT_RETURN_IF_ERROR(absl::InvalidArgumentError(absl::StrCat(
            "PJRT_Buffer_MemoryLayout_Type_Strides in device_layout is not "
            "supported in  PJRT_Client_BufferFromHostBuffer for platform ",
            args->client->client->platform_name())));
        break;
      }
      default: {
        PJRT_RETURN_IF_ERROR(absl::InvalidArgumentError(
            absl::StrCat("Unexpected PJRT_Buffer_MemoryLayout_Type type: ",
                         args->device_layout->type)));
      }
    }
  }

  xla::PjRtFuture<>::Promise promise = xla::PjRtFuture<>::CreatePromise();

  absl::AnyInvocable<void() &&> on_done_with_host_buffer = [promise]() mutable {
    promise.Set();
  };

  std::unique_ptr<xla::PjRtBuffer> buffer;
  bool has_layout_and_memory = layout.has_value() && args->memory != nullptr;
  bool has_layout_and_no_memory = layout.has_value() && args->memory == nullptr;
  bool has_memory_and_no_layout =
      !layout.has_value() && args->memory != nullptr;
  if (has_layout_and_memory) {
    PJRT_ASSIGN_OR_RETURN(
        buffer, args->client->client->BufferFromHostBuffer(
                    args->data, ::pjrt::ConvertFromPjRtBufferType(args->type),
                    dims, byte_strides,
                    ::pjrt::ConvertFromPjRtHostBufferSemantics(
                        args->host_buffer_semantics),
                    std::move(on_done_with_host_buffer),
                    args->memory->memory_space, &layout.value()));
  } else if (has_layout_and_no_memory) {
    PJRT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                          args->device->device->default_memory_space());
    PJRT_ASSIGN_OR_RETURN(
        buffer, args->client->client->BufferFromHostBuffer(
                    args->data, ::pjrt::ConvertFromPjRtBufferType(args->type),
                    dims, byte_strides,
                    ::pjrt::ConvertFromPjRtHostBufferSemantics(
                        args->host_buffer_semantics),
                    std::move(on_done_with_host_buffer), memory_space,
                    &layout.value()));
  } else if (has_memory_and_no_layout) {
    PJRT_ASSIGN_OR_RETURN(
        buffer,
        args->client->client->BufferFromHostBuffer(
            args->data, ::pjrt::ConvertFromPjRtBufferType(args->type), dims,
            byte_strides,
            ::pjrt::ConvertFromPjRtHostBufferSemantics(
                args->host_buffer_semantics),
            std::move(on_done_with_host_buffer), args->memory->memory_space,
            /*device_layout=*/nullptr));
  } else {
    PJRT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                          args->device->device->default_memory_space());
    PJRT_ASSIGN_OR_RETURN(
        buffer, args->client->client->BufferFromHostBuffer(
                    args->data, ::pjrt::ConvertFromPjRtBufferType(args->type),
                    dims, byte_strides,
                    ::pjrt::ConvertFromPjRtHostBufferSemantics(
                        args->host_buffer_semantics),
                    std::move(on_done_with_host_buffer), memory_space,
                    /*device_layout=*/nullptr));
  }

  args->buffer = new PJRT_Buffer{std::move(buffer), args->client};
  args->done_with_host_buffer =
      new PJRT_Event{xla::PjRtFuture<>(std::move(promise))};

  return nullptr;
}

PJRT_Error* PJRT_Client_CreateViewOfDeviceBuffer(
    PJRT_Client_CreateViewOfDeviceBuffer_Args* args) {
  PJRT_ASSIGN_OR_RETURN(xla::Shape shape,
                        pjrt::BuildXlaShapeFromC(args->element_type, args->dims,
                                                 args->num_dims, args->layout));
  std::function<void()> on_delete_callback;
  if (args->on_delete_callback != nullptr) {
    on_delete_callback = [on_delete_callback = args->on_delete_callback,
                          user_arg = args->on_delete_callback_arg,
                          device_buffer_ptr = args->device_buffer_ptr]() {
      on_delete_callback(device_buffer_ptr, user_arg);
    };
  }
  std::optional<std::intptr_t> stream = std::nullopt;
  if (reinterpret_cast<void*>(args->stream) != nullptr) {
    stream = args->stream;
  }
  std::unique_ptr<xla::PjRtBuffer> buffer;
  bool has_memory_space = args->struct_size >=
                          PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
  xla::PjRtMemorySpace* memory_space = nullptr;
  if (has_memory_space && args->memory != nullptr) {
    memory_space = args->memory->memory_space;
  } else if (args->device != nullptr) {
    PJRT_ASSIGN_OR_RETURN(memory_space,
                          args->device->device->default_memory_space());
  } else {
    return new PJRT_Error{
        absl::InvalidArgumentError("PJRT_Client_CreateViewOfDeviceBuffer "
                                   "requires either a device or a memory")};
  }
  PJRT_ASSIGN_OR_RETURN(buffer, args->client->client->CreateViewOfDeviceBuffer(
                                    args->device_buffer_ptr, shape,
                                    memory_space, on_delete_callback, stream));
  args->buffer = new PJRT_Buffer{std::move(buffer), args->client};
  return nullptr;
}

// --------------------------------- Devices -----------------------------------

PJRT_Error* PJRT_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_DeviceDescription_Id_Args",
      PJRT_DeviceDescription_Id_Args_STRUCT_SIZE, args->struct_size));

  args->id = args->device_description->device_description->id();
  return nullptr;
}

PJRT_Error* PJRT_DeviceDescription_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_DeviceDescription_ProcessIndex_Args",
      PJRT_DeviceDescription_ProcessIndex_Args_STRUCT_SIZE, args->struct_size));
  args->process_index =
      args->device_description->device_description->process_index();
  return nullptr;
}

PJRT_Error* PJRT_DeviceDescription_Attributes(
    PJRT_DeviceDescription_Attributes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_DeviceDescription_Attributes_Args",
      PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE, args->struct_size));

  // Returns the attributes that were initialized during PJRT_Device creation.
  args->num_attributes = args->device_description->attributes.size();
  args->attributes = args->device_description->attributes.data();

  return nullptr;
}

PJRT_Error* PJRT_DeviceDescription_Kind(
    PJRT_DeviceDescription_Kind_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_DeviceDescription_Kind_Args",
      PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE, args->struct_size));

  args->device_kind =
      args->device_description->device_description->device_kind().data();
  args->device_kind_size =
      args->device_description->device_description->device_kind().size();
  return nullptr;
}

PJRT_Error* PJRT_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_DeviceDescription_DebugString_Args",
      PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE, args->struct_size));

  args->debug_string =
      args->device_description->device_description->DebugString().data();
  args->debug_string_size =
      args->device_description->device_description->DebugString().size();
  return nullptr;
}

PJRT_Error* PJRT_DeviceDescription_MemoryDescriptions(
    PJRT_DeviceDescription_MemoryDescriptions_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_DeviceDescription_MemoryDescriptions_Args",
      PJRT_DeviceDescription_MemoryDescriptions_Args_STRUCT_SIZE,
      args->struct_size));

  absl::Span<const xla::PjRtMemorySpaceDescription* const> memory_spaces =
      args->device_description->device_description->memory_spaces();

  // We pass each xla::PjRtMemorySpaceDescriptions to the caller through an
  // opaque pointer.
  args->memory_descriptions =
      reinterpret_cast<const PJRT_MemoryDescription* const*>(
          memory_spaces.data());

  absl::StatusOr<const xla::PjRtMemorySpaceDescription*> default_memory =
      args->device_description->device_description->default_memory_space();
  args->default_memory_index = -1;
  for (int i = 0; i < memory_spaces.size(); i++) {
    if (default_memory.ok() && *default_memory == memory_spaces[i]) {
      args->default_memory_index = i;
    }
  }

  args->num_memory_descriptions = memory_spaces.size();
  return nullptr;
}

PJRT_Error* PJRT_DeviceDescription_ToString(
    PJRT_DeviceDescription_ToString_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_DeviceDescription_ToString_Args",
      PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE, args->struct_size));
  args->to_string =
      args->device_description->device_description->ToString().data();
  args->to_string_size =
      args->device_description->device_description->ToString().size();
  return nullptr;
}

PJRT_Error* PJRT_MemoryDescription_Kind(
    PJRT_MemoryDescription_Kind_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_MemoryDescription_Kind_Args",
      PJRT_MemoryDescription_Kind_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view kind =
      args->memory_description->memory_space_description.kind();
  args->kind = kind.data();
  args->kind_size = kind.size();
  args->kind_id = args->memory_description->memory_space_description.kind_id();
  return nullptr;
}

PJRT_Error* PJRT_Device_GetDescription(PJRT_Device_GetDescription_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Device_GetDescription_Args",
      PJRT_Device_GetDescription_Args_STRUCT_SIZE, args->struct_size));
  args->device_description = &args->device->description;
  return nullptr;
}

PJRT_Error* PJRT_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Device_IsAddressable_Args",
      PJRT_Device_IsAddressable_Args_STRUCT_SIZE, args->struct_size));
  args->is_addressable = args->device->device->IsAddressable();
  return nullptr;
}

PJRT_Error* PJRT_Device_LocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Device_LocalHardwareId_Args",
      PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE, args->struct_size));
  args->local_hardware_id = args->device->device->local_hardware_id().value();
  return nullptr;
}

PJRT_Error* PJRT_Device_AddressableMemories(
    PJRT_Device_AddressableMemories_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Device_AddressableMemories_Args",
      PJRT_Device_AddressableMemories_Args_STRUCT_SIZE, args->struct_size));
  args->memories = args->device->addressable_memories.data();
  args->num_memories = args->device->addressable_memories.size();
  return nullptr;
}

PJRT_Error* PJRT_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Device_DefaultMemory_Args",
      PJRT_Device_DefaultMemory_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                        args->device->device->default_memory_space());
  args->memory = GetCMemory(args->device->client, memory_space);
  return nullptr;
}

PJRT_Error* PJRT_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Device_MemoryStats_Args", PJRT_Device_MemoryStats_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_ASSIGN_OR_RETURN(tsl::AllocatorStats stats,
                        args->device->device->GetAllocatorStats());

  args->bytes_in_use = stats.bytes_in_use;

  args->peak_bytes_in_use_is_set = true;
  args->peak_bytes_in_use = stats.peak_bytes_in_use;
  args->num_allocs_is_set = true;
  args->num_allocs = stats.num_allocs;
  args->largest_alloc_size_is_set = true;
  args->largest_alloc_size = stats.largest_alloc_size;

  args->bytes_limit_is_set = stats.bytes_limit.has_value();
  if (stats.bytes_limit) {
    args->bytes_limit = *stats.bytes_limit;
  }

  args->bytes_reserved_is_set = true;
  args->bytes_reserved = stats.bytes_reserved;
  args->peak_bytes_reserved_is_set = true;
  args->peak_bytes_reserved = stats.peak_bytes_reserved;

  args->bytes_reservable_limit_is_set =
      stats.bytes_reservable_limit.has_value();
  if (stats.bytes_reservable_limit) {
    args->bytes_reservable_limit = *stats.bytes_reservable_limit;
  }

  args->largest_free_block_bytes_is_set = true;
  args->largest_free_block_bytes = stats.largest_free_block_bytes;

  args->pool_bytes_is_set = stats.pool_bytes.has_value();
  if (stats.pool_bytes) {
    args->pool_bytes = *stats.pool_bytes;
  }

  args->peak_pool_bytes_is_set = stats.peak_pool_bytes.has_value();
  if (stats.peak_pool_bytes) {
    args->peak_pool_bytes = *stats.peak_pool_bytes;
  }

  return nullptr;
}

// ------------------------------- Memory --------------------------------------

PJRT_Error* PJRT_Memory_Id(PJRT_Memory_Id_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Memory_Id_Args", PJRT_Memory_Id_Args_STRUCT_SIZE,
      args->struct_size));

  args->id = args->memory->memory_space->id();
  return nullptr;
}

PJRT_Error* PJRT_Memory_Kind(PJRT_Memory_Kind_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Memory_Kind_Args", PJRT_Memory_Kind_Args_STRUCT_SIZE,
      args->struct_size));
  args->kind = args->memory->memory_space->kind().data();
  args->kind_size = args->memory->memory_space->kind().size();
  return nullptr;
}

PJRT_Error* PJRT_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Memory_Kind_Id_Args", PJRT_Memory_Kind_Id_Args_STRUCT_SIZE,
      args->struct_size));
  args->kind_id = args->memory->memory_space->kind_id();
  return nullptr;
}

PJRT_Error* PJRT_Memory_DebugString(PJRT_Memory_DebugString_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Memory_DebugString_Args", PJRT_Memory_DebugString_Args_STRUCT_SIZE,
      args->struct_size));

  args->debug_string = args->memory->memory_space->DebugString().data();
  args->debug_string_size = args->memory->memory_space->DebugString().size();
  return nullptr;
}

PJRT_Error* PJRT_Memory_ToString(PJRT_Memory_ToString_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Memory_ToString_Args", PJRT_Memory_ToString_Args_STRUCT_SIZE,
      args->struct_size));

  args->to_string = args->memory->memory_space->ToString().data();
  args->to_string_size = args->memory->memory_space->ToString().size();
  return nullptr;
}

PJRT_Error* PJRT_Memory_AddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Memory_AddressableByDevices_Args",
      PJRT_Memory_AddressableByDevices_Args_STRUCT_SIZE, args->struct_size));
  args->devices = args->memory->devices.data();
  args->num_devices = args->memory->devices.size();
  return nullptr;
}

// ------------------------------- Execute Context -----------------------------

PJRT_Error* PJRT_ExecuteContext_Destroy(
    PJRT_ExecuteContext_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteContext_Destroy_Args",
      PJRT_ExecuteContext_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->context;
  return nullptr;
}

// ------------------------------- Executables ---------------------------------

PJRT_Error* PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_Destroy_Args", PJRT_Executable_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->executable;
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_LoadedExecutable_Destroy_Args",
      PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->executable;
  return nullptr;
}

PJRT_Error* PJRT_Executable_Name(PJRT_Executable_Name_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_Name_Args", PJRT_Executable_Name_Args_STRUCT_SIZE,
      args->struct_size));
  absl::string_view executable_name = args->executable->get()->name();
  args->executable_name = executable_name.data();
  args->executable_name_size = executable_name.size();
  return nullptr;
}

PJRT_Error* PJRT_Executable_NumReplicas(
    PJRT_Executable_NumReplicas_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_NumReplicas_Args",
      PJRT_Executable_NumReplicas_Args_STRUCT_SIZE, args->struct_size));
  args->num_replicas = args->executable->get()->num_replicas();
  return nullptr;
}

PJRT_Error* PJRT_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_NumPartitions_Args",
      PJRT_Executable_NumPartitions_Args_STRUCT_SIZE, args->struct_size));
  args->num_partitions = args->executable->get()->num_partitions();
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_LoadedExecutable_AddressableDevices_Args",
      PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE,
      args->struct_size));

  args->num_addressable_devices = args->executable->addressable_devices.size();
  args->addressable_devices = args->executable->addressable_devices.data();
  return nullptr;
}

PJRT_Error* PJRT_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_NumOutputs_Args",
      PJRT_Executable_NumOutputs_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(std::vector<xla::Shape> output_shapes,
                        args->executable->get()->GetOutputShapes());
  if (output_shapes.empty()) {
    return new PJRT_Error{
        xla::InvalidArgument("Can't get number of executable outputs, output "
                             "shapes is empty for executable %s.",
                             args->executable->get()->name())};
  }
  if (output_shapes.size() != 1) {
    return new PJRT_Error{
        xla::Unimplemented("MPMD execution not supported by PJRT C API (in "
                           "function PJRT_Executable_NumOutputs).")};
  }
  const xla::Shape& shape = output_shapes[0];
  if (shape.IsTuple()) {
    args->num_outputs = shape.tuple_shapes().size();
  } else {
    // The output size is 1, as it is not a tuple.
    args->num_outputs = 1;
  }
  return nullptr;
}

PJRT_Error* PJRT_Executable_SizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_SizeOfGeneratedCodeInBytes_Args",
      PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE,
      args->struct_size));

  args->size_in_bytes = args->executable->get()->SizeOfGeneratedCodeInBytes();
  return nullptr;
}

static absl::Status VerifyOptimizedProgramArgs(
    PJRT_Executable_OptimizedProgram_Args* args) {
  TF_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_OptimizedProgram_Args",
      PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE, args->struct_size));
  TF_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Program", PJRT_Program_STRUCT_SIZE, args->program->struct_size));
  return absl::OkStatus();
}

static absl::StatusOr<std::shared_ptr<xla::HloModule>>
GetOptimizedProgramModule(const PJRT_Executable_OptimizedProgram_Args* args) {
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<xla::HloModule>> hlo_modules,
                      args->executable->get()->GetHloModules());
  if (hlo_modules.empty()) {
    return xla::InvalidArgument(
        "Can't get the optimized program for executable "
        "`%s`: HLO modules is empty.",
        args->executable->get()->name());
  }
  if (hlo_modules.size() > 1) {
    return xla::Unimplemented(
        "Can't get the optimized program for executable "
        "`%s`: MPMD execution is not supported by PJRT C API",
        args->executable->get()->name());
  }
  return std::move(hlo_modules[0]);
}

PJRT_Error* PJRT_Executable_OptimizedProgram(
    PJRT_Executable_OptimizedProgram_Args* args) {
  PJRT_RETURN_IF_ERROR(VerifyOptimizedProgramArgs(args));
  PJRT_Program* program = args->program;
  program->format = kHloWithConfigFormat.data();
  program->format_size = kHloWithConfigFormat.size();
  PJRT_ASSIGN_OR_RETURN(std::shared_ptr<xla::HloModule> hlo_module,
                        GetOptimizedProgramModule(args));
  xla::HloModuleProtoWithConfig proto = hlo_module->ToProtoWithConfig();
  if (program->code == nullptr) {
    program->code_size = proto.ByteSizeLong();
    if (program->code_size >= 2ull * 1024 * 1024 * 1024) {
      return new PJRT_Error{xla::ResourceExhausted(
          "%s: HLO program serialization would require more than the max "
          "supported protobuff size of 2 GiB.",
          __func__)};
    }
    return nullptr;
  } else {
    if (program->code_size < proto.ByteSizeLong()) {
      return new PJRT_Error{
          xla::InvalidArgument("`program->code_size` %d < required bytes %d",
                               program->code_size, proto.ByteSizeLong()),
      };
    }
    bool succeeded = proto.SerializeToArray(program->code, program->code_size);
    if (!succeeded) {
      return new PJRT_Error{
          xla::ResourceExhausted("%s: HLO program serialization exceeds max "
                                 "supported protobuff size of 2 GiB.",
                                 __func__)};
    }
    return nullptr;
  }
}

PJRT_Error* PJRT_Executable_Fingerprint(
    PJRT_Executable_Fingerprint_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_Fingerprint_Args",
      PJRT_Executable_Fingerprint_Args_STRUCT_SIZE, args->struct_size));
  PJRT_RETURN_IF_ERROR(args->executable->fingerprint.status());
  args->executable_fingerprint = args->executable->fingerprint.value().c_str();
  args->executable_fingerprint_size =
      args->executable->fingerprint.value().size();
  return nullptr;
}

PJRT_Error* PJRT_Executable_GetCostAnalysis(
    PJRT_Executable_GetCostAnalysis_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_GetCostAnalysis_Args",
      PJRT_Executable_GetCostAnalysis_Args_STRUCT_SIZE, args->struct_size));

  {
    absl::MutexLock lock(&args->executable->mutex);
    if (!args->executable->cost_analysis_ran) {
      PJRT_RETURN_IF_ERROR(PopulateExecutableCostAnalysis(args->executable));
      args->executable->cost_analysis_ran = true;
    }
  }

  // Output cost analysis data in PJRT_Executable
  args->num_properties = args->executable->cost_analysis_properties.size();
  if (args->num_properties > 0) {
    args->properties = args->executable->cost_analysis_properties.data();
  } else {
    args->properties = nullptr;
  }
  return nullptr;
}

PJRT_Error* PJRT_Executable_OutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_OutputElementTypes_Args",
      PJRT_Executable_OutputElementTypes_Args_STRUCT_SIZE, args->struct_size));

  {
    absl::MutexLock lock(&args->executable->mutex);
    if (!args->executable->out_type_ran) {
      PJRT_RETURN_IF_ERROR(
          PopulateExecutableOutputElementTypes(args->executable));
      args->executable->out_type_ran = true;
    }
  }

  args->num_output_types = args->executable->out_types.size();
  args->output_types = args->executable->out_types.data();
  return nullptr;
}

PJRT_Error* PJRT_Executable_OutputDimensions(
    PJRT_Executable_OutputDimensions_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_OutputDimensions_Args",
      PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE, args->struct_size));

  {
    absl::MutexLock lock(&args->executable->mutex);
    if (!args->executable->out_dimension_ran) {
      PJRT_RETURN_IF_ERROR(
          PopulateExecutableOutputDimensions(args->executable));
      args->executable->out_dimension_ran = true;
    }
  }

  args->num_outputs = args->executable->out_dimension_sizes.size();
  args->dim_sizes = args->executable->out_dimension_sizes.data();
  args->dims = args->executable->out_dimensions.data();
  return nullptr;
}

PJRT_Error* PJRT_Executable_OutputMemoryKinds(
    PJRT_Executable_OutputMemoryKinds_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_OutputMemoryKinds_Args",
      PJRT_Executable_OutputMemoryKinds_Args_STRUCT_SIZE, args->struct_size));

  {
    absl::MutexLock lock(&args->executable->mutex);
    if (!args->executable->memory_kind_ran) {
      PJRT_RETURN_IF_ERROR(
          PopulateExecutableOutputMemoryKinds(args->executable));
      args->executable->memory_kind_ran = true;
    }
  }

  args->num_outputs = args->executable->memory_kinds.size();
  args->memory_kinds = args->executable->memory_kinds.data();
  args->memory_kind_sizes = args->executable->memory_kind_sizes.data();
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_LoadedExecutable_Delete_Args",
      PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE, args->struct_size));
  args->executable->get()->Delete();
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_LoadedExecutable_IsDeleted_Args",
      PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE, args->struct_size));
  args->is_deleted = args->executable->get()->IsDeleted();
  return nullptr;
}

static xla::SendCallback CSendCallbackToCpp(
    const PJRT_SendCallbackInfo& c_callback) {
  return xla::SendCallback{
      c_callback.channel_id,
      // Transfer metadata is unused because PJRT C API doesn't support
      // use_major_to_minor_data_layout_for_callbacks = false
      [user_arg = c_callback.user_arg, callback = c_callback.send_callback](
          const xla::PjRtTransferMetadata& unused_metadata,
          xla::PjRtChunk input, size_t total_size_in_bytes,
          bool done) -> absl::Status {
        PJRT_Chunk c_chunk = ConvertFromCppChunk(std::move(input));
        // PJRT_CallbackError creates PJRT_Error in the implementation, but
        // using the caller's callback status code & message. This way, the
        // caller avoids creating PJRT_Error itself, and the PJRT_Error is
        // fully managed in the implementation layer.
        PJRT_CallbackError c_callback_error =
            [](PJRT_Error_Code code, const char* message, size_t message_size) {
              return new PJRT_Error{
                  absl::Status(static_cast<absl::StatusCode>(code),
                               std::string(message, message_size))};
            };

        std::unique_ptr<PJRT_Error> error(callback(
            &c_chunk, &c_callback_error, total_size_in_bytes, done, user_arg));
        if (error == nullptr) {
          return absl::OkStatus();
        }
        return error->status;
      }};
}

// Create new libtpu C++ callbacks that calls C API callback with converted
// arguments.
static void CSendCallbackListsToCpp(
    PJRT_SendCallbackInfo** c_lists, size_t outer_size, size_t inner_size,
    std::vector<std::vector<xla::SendCallback>>& cpp_lists) {
  cpp_lists.reserve(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    std::vector<xla::SendCallback>& cpp_list = cpp_lists.emplace_back();
    cpp_list.reserve(inner_size);
    for (int j = 0; j < inner_size; ++j) {
      cpp_list.push_back(CSendCallbackToCpp(c_lists[i][j]));
    }
  }
}

static xla::RecvCallback CRecvCallbackToCpp(
    const PJRT_RecvCallbackInfo& c_callback) {
  return xla::RecvCallback{
      c_callback.channel_id,
      // Transfer metadata is unused because PJRT C API doesn't support
      // use_major_to_minor_data_layout_for_callbacks = false
      [user_arg = c_callback.user_arg, callback = c_callback.recv_callback](
          const xla::PjRtTransferMetadata& unused_metadata,
          std::unique_ptr<xla::CopyToDeviceStream> stream) {
        auto c_stream = std::make_unique<PJRT_CopyToDeviceStream>();
        c_stream->stream = std::move(stream);
        // The callback takes the ownership of the stream and will be
        // responsible for calling its deleter.
        callback(c_stream.release(), user_arg);
      }};
}

static void CRecvCallbackListsToCpp(
    PJRT_RecvCallbackInfo** c_lists, size_t outer_size, size_t inner_size,
    std::vector<std::vector<xla::RecvCallback>>& cpp_lists) {
  cpp_lists.reserve(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    auto& cpp_list = cpp_lists.emplace_back();
    cpp_list.reserve(inner_size);
    for (int j = 0; j < inner_size; ++j) {
      cpp_list.push_back(CRecvCallbackToCpp(c_lists[i][j]));
    }
  }
}

static std::vector<std::vector<xla::PjRtBuffer*>> Convert2DCBuffersToCppBuffers(
    PJRT_Buffer* const* const* c_lists, size_t outer_size, size_t inner_size) {
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

PJRT_Error* PJRT_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_LoadedExecutable_Execute_Args",
      PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE, args->struct_size));
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteOptions", PJRT_ExecuteOptions_STRUCT_SIZE,
      args->options->struct_size));

  int64_t traceme_context_id = pjrt::GetTracemeContextId(args);
  tsl::profiler::TraceMeConsumer consumer(
      "PJRT_LoadedExecutable_Execute",
      tsl::profiler::ContextType::kPjrtLibraryCall, traceme_context_id);

  xla::ExecuteOptions options;
  options.launch_id = args->options->launch_id;
  options.strict_shape_checking = true;
  options.arguments_are_tupled = false;
  options.untuple_result = true;
  options.context = args->options->context
                        ? args->options->context->execute_context.get()
                        : nullptr;
  options.multi_slice_config = nullptr;
  options.use_major_to_minor_data_layout_for_callbacks = true;
  if (args->options->num_non_donatable_input_indices > 0) {
    for (int i = 0; i < args->options->num_non_donatable_input_indices; ++i) {
      options.non_donatable_input_indices.insert(
          args->options->non_donatable_input_indices[i]);
    }
  }

  std::vector<std::vector<xla::PjRtBuffer*>> cpp_argument_lists =
      Convert2DCBuffersToCppBuffers(args->argument_lists, args->num_devices,
                                    args->num_args);

  // Set send/recv callbacks in ExecuteOptions. The callbacks
  // should call the C callbacks provided by the caller.
  auto cpp_send_callbacks =
      std::make_shared<std::vector<std::vector<xla::SendCallback>>>();
  if (args->options->num_send_ops > 0) {
    CSendCallbackListsToCpp(args->options->send_callbacks, args->num_devices,
                            args->options->num_send_ops, *cpp_send_callbacks);
    options.send_callbacks = *cpp_send_callbacks;
    CHECK_EQ(options.send_callbacks.size(), args->num_devices);
  }

  auto cpp_recv_callbacks =
      std::make_shared<std::vector<std::vector<xla::RecvCallback>>>();
  if (args->options->num_recv_ops > 0) {
    CRecvCallbackListsToCpp(args->options->recv_callbacks, args->num_devices,
                            args->options->num_recv_ops, *cpp_recv_callbacks);
    options.recv_callbacks = *cpp_recv_callbacks;
    CHECK_EQ(options.recv_callbacks.size(), args->num_devices);
  }

  if (args->execute_device == nullptr) {
    std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> cpp_buffer_lists;
    if (args->device_complete_events != nullptr ||
        !cpp_send_callbacks->empty() || !cpp_recv_callbacks->empty()) {
      std::optional<std::vector<xla::PjRtFuture<>>> returned_futures;
      returned_futures.emplace();
      PJRT_ASSIGN_OR_RETURN(cpp_buffer_lists,
                            args->executable->get()->Execute(
                                cpp_argument_lists, options, returned_futures));
      CHECK_EQ(returned_futures->size(), args->num_devices);

      // We assume that these OnReady callbacks will fire even if
      // returned_futures is destroyed first. This is true for the
      // AsyncValue-based implementation of PjRtFuture.
      if (!cpp_send_callbacks->empty() || !cpp_recv_callbacks->empty()) {
        for (int i = 0; i < returned_futures->size(); ++i) {
          (*returned_futures)[i].OnReady(
              [cpp_send_callbacks, cpp_recv_callbacks](absl::Status status) {
                // Keeps C++ callbacks alive until execution completes on all
                // devices.
              });
        }
      }

      if (args->device_complete_events != nullptr) {
        for (int i = 0; i < returned_futures->size(); ++i) {
          args->device_complete_events[i] =
              new PJRT_Event{std::move((*returned_futures)[i])};
        }
      }
    } else {
      PJRT_ASSIGN_OR_RETURN(cpp_buffer_lists, args->executable->get()->Execute(
                                                  cpp_argument_lists, options));
    }

    for (int i = 0; i < cpp_buffer_lists.size(); ++i) {
      for (int j = 0; j < cpp_buffer_lists[i].size(); ++j) {
        args->output_lists[i][j] = new PJRT_Buffer{
            std::move(cpp_buffer_lists[i][j]), args->executable->client};
      }
    }
  } else {
    if (args->num_devices != 1) {
      return new PJRT_Error{xla::InvalidArgument(
          "num_devices and corresponding output list sizes must be 1 when "
          "calling PJRT_LoadedExecutable_Execute with non-null "
          "execute_device. "
          "Got "
          "num_devices=%i",
          args->num_devices)};
    }
    if (!cpp_send_callbacks->empty() || !cpp_recv_callbacks->empty()) {
      return new PJRT_Error{xla::Unimplemented(
          "PJRT_Executable_Execute doesn't support using send/recv callbacks "
          "with `execute_device`.")};
    }

    std::vector<std::unique_ptr<xla::PjRtBuffer>> cpp_buffer_list;
    std::optional<xla::PjRtFuture<>> returned_future;
    bool fill_future = args->device_complete_events != nullptr;
    PJRT_ASSIGN_OR_RETURN(
        xla::CompileOptions compile_options,
        args->executable->get()->GetExecutable()->GetCompileOptions());
    if (compile_options.compile_portable_executable) {
      PJRT_ASSIGN_OR_RETURN(
          cpp_buffer_list,
          args->executable->get()->ExecutePortable(
              cpp_argument_lists[0], args->execute_device->device, options,
              returned_future, fill_future));
    } else {
      PJRT_ASSIGN_OR_RETURN(
          cpp_buffer_list,
          args->executable->get()->ExecuteSharded(
              cpp_argument_lists[0], args->execute_device->device, options,
              returned_future, fill_future));
    }
    for (int i = 0; i < cpp_buffer_list.size(); ++i) {
      args->output_lists[0][i] = new PJRT_Buffer{std::move(cpp_buffer_list[i]),
                                                 args->executable->client};
    }
    if (fill_future) {
      args->device_complete_events[0] =
          new PJRT_Event{std::move((*returned_future))};
    }
  }

  return nullptr;
}

PJRT_Error* PJRT_Executable_Serialize(PJRT_Executable_Serialize_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_Serialize_Args",
      PJRT_Executable_Serialize_Args_STRUCT_SIZE, args->struct_size));
  std::string serialization;
  PJRT_ASSIGN_OR_RETURN(serialization,
                        args->executable->executable->SerializeExecutable());

  PJRT_SerializedExecutable* serialized_exec = new PJRT_SerializedExecutable;
  if (serialized_exec == nullptr) {
    return new PJRT_Error{xla::ResourceExhausted(
        "Out of memory for `PJRT_Executable_Serialize()`")};
  }
  serialized_exec->serialized = std::move(serialization);
  args->serialized_executable = serialized_exec;
  args->serialized_bytes = serialized_exec->serialized.data();
  args->serialized_bytes_size = serialized_exec->serialized.size();
  args->serialized_executable_deleter =
      +[](PJRT_SerializedExecutable* serialized_executable) {
        delete serialized_executable;
      };
  return nullptr;
}

PJRT_Error* PJRT_Executable_GetCompiledMemoryStats(
    PJRT_Executable_GetCompiledMemoryStats_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_Serialize_Args",
      PJRT_Executable_Serialize_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(auto memory_stats,
                        args->executable->executable->GetCompiledMemoryStats());
  args->generated_code_size_in_bytes =
      memory_stats.generated_code_size_in_bytes;
  args->argument_size_in_bytes = memory_stats.argument_size_in_bytes;
  args->output_size_in_bytes = memory_stats.output_size_in_bytes;
  args->alias_size_in_bytes = memory_stats.alias_size_in_bytes;
  args->temp_size_in_bytes = memory_stats.temp_size_in_bytes;
  args->host_generated_code_size_in_bytes =
      memory_stats.host_generated_code_size_in_bytes;
  args->host_argument_size_in_bytes = memory_stats.host_argument_size_in_bytes;
  args->host_output_size_in_bytes = memory_stats.host_output_size_in_bytes;
  args->host_alias_size_in_bytes = memory_stats.host_alias_size_in_bytes;
  args->host_temp_size_in_bytes = memory_stats.host_temp_size_in_bytes;
  return nullptr;
}

PJRT_Error* PJRT_Executable_DeserializeAndLoad(
    PJRT_Executable_DeserializeAndLoad_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_DeserializeAndLoad_Args",
      PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE, args->struct_size));
  absl::string_view serialized(args->serialized_executable,
                               args->serialized_executable_size);

  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      args->client->client->LoadSerializedExecutable(
          serialized, /*options=*/std::nullopt, xla::LoadOptions()));

  args->loaded_executable =
      new PJRT_LoadedExecutable(std::move(executable), args->client);
  return nullptr;
}

PJRT_Error* PJRT_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_LoadedExecutable_GetExecutable_Args",
      PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE, args->struct_size));
  args->executable =
      new PJRT_Executable{args->loaded_executable->executable->GetExecutable()};
  return nullptr;
}

// ---------------------------------- Buffers ----------------------------------

PJRT_Error* PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_Destroy_Args", PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->buffer;
  return nullptr;
}

PJRT_Error* PJRT_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_ElementType_Args", PJRT_Buffer_ElementType_Args_STRUCT_SIZE,
      args->struct_size));
  args->type = ConvertToPjRtBufferType(args->buffer->buffer->element_type());
  return nullptr;
}

PJRT_Error* PJRT_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_Dimensions_Args", PJRT_Buffer_Dimensions_Args_STRUCT_SIZE,
      args->struct_size));
  args->dims = args->buffer->buffer->dimensions().data();
  args->num_dims = args->buffer->buffer->dimensions().size();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_UnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_UnpaddedDimensions_Args",
      PJRT_Buffer_UnpaddedDimensions_Args_STRUCT_SIZE, args->struct_size));

  std::optional<std::vector<int64_t>>& unpadded_dims =
      args->buffer->unpadded_dims;
  {
    absl::MutexLock lock(&args->buffer->mu);
    if (!unpadded_dims.has_value()) {
      PJRT_ASSIGN_OR_RETURN(std::vector<int64_t> dims,
                            args->buffer->buffer->logical_dimensions());
      unpadded_dims.emplace(std::move(dims));
    }
  }
  args->unpadded_dims = unpadded_dims->data();
  args->num_dims = unpadded_dims->size();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_DynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_DynamicDimensionIndices_Args",
      PJRT_Buffer_DynamicDimensionIndices_Args_STRUCT_SIZE, args->struct_size));
  absl::Span<const bool> is_dyn_dim =
      args->buffer->buffer->is_dynamic_dimension();
  std::optional<std::vector<size_t>>& dyn_dim_indices =
      args->buffer->dynamic_dim_indices;
  {
    absl::MutexLock lock(&args->buffer->mu);
    if (!dyn_dim_indices.has_value()) {
      std::vector<size_t>& dyn_dim_indices_value = dyn_dim_indices.emplace();
      for (int i = 0; i < is_dyn_dim.size(); ++i) {
        if (is_dyn_dim[i]) {
          dyn_dim_indices_value.push_back(i);
        }
      }
    }
  }
  args->dynamic_dim_indices = dyn_dim_indices->data();
  args->num_dynamic_dims = dyn_dim_indices->size();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_GetMemoryLayout(
    PJRT_Buffer_GetMemoryLayout_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_GetMemoryLayout_Args",
      PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE, args->struct_size));

  std::optional<BufferMemoryLayoutData>& layout_data =
      args->buffer->layout_data;
  {
    absl::MutexLock lock(&args->buffer->mu);
    if (!layout_data.has_value()) {
      // TODO(skyewm): change PJRT C API to also use opaque layout type
      std::shared_ptr<const xla::PjRtLayout> pjrt_layout =
          args->buffer->buffer->layout();
      const xla::Layout& xla_layout = pjrt_layout->xla_layout();

      PJRT_ASSIGN_OR_RETURN(BufferMemoryLayoutData data,
                            ConvertToBufferMemoryLayoutData(xla_layout));
      layout_data.emplace(std::move(data));
    }
  }
  args->layout = layout_data->c_layout;
  return nullptr;
}

PJRT_Error* PJRT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_OnDeviceSizeInBytes_Args",
      PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(args->on_device_size_in_bytes,
                        args->buffer->buffer->GetOnDeviceSizeInBytes());
  return nullptr;
}

PJRT_Error* PJRT_Buffer_Device(PJRT_Buffer_Device_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
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

PJRT_Error* PJRT_Buffer_Memory(PJRT_Buffer_Memory_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_Memory_Args", PJRT_Buffer_Memory_Args_STRUCT_SIZE,
      args->struct_size));
  args->memory = PJRT_Client_FindMemoryWrapper(
      args->buffer->buffer->memory_space(), args->buffer->client);
  if (args->memory == nullptr) {
    return new PJRT_Error{xla::Unimplemented(
        "PJRT_Buffer_Memory not implemented for platform '%s'",
        args->buffer->client->client->platform_name())};
  }
  return nullptr;
}

PJRT_Error* PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_Delete_Args", PJRT_Buffer_Delete_Args_STRUCT_SIZE,
      args->struct_size));
  args->buffer->buffer->Delete();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_IsDeleted_Args", PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE,
      args->struct_size));
  args->is_deleted = args->buffer->buffer->IsDeleted();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_CopyRawToHost(PJRT_Buffer_CopyRawToHost_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_CopyRawToHost_Args",
      PJRT_Buffer_CopyRawToHost_Args_STRUCT_SIZE, args->struct_size));
  xla::PjRtFuture<> wrapped_promise = args->buffer->buffer->CopyRawToHost(
      args->dst, args->offset, args->transfer_size);
  args->event = new PJRT_Event{std::move(wrapped_promise)};
  return nullptr;
}

PJRT_Error* PJRT_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_CopyToDevice_Args",
      PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                        args->dst_device->device->default_memory_space());
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> dst_buffer,
                        args->buffer->buffer->CopyToMemorySpace(memory_space));
  args->dst_buffer =
      new PJRT_Buffer{std::move(dst_buffer), args->dst_device->client};
  return nullptr;
}

PJRT_Error* PJRT_Buffer_CopyToMemory(PJRT_Buffer_CopyToMemory_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_CopyToMemory_Args",
      PJRT_Buffer_CopyToMemory_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer> dst_buffer,
      args->buffer->buffer->CopyToMemorySpace(args->dst_memory->memory_space));
  args->dst_buffer =
      new PJRT_Buffer{std::move(dst_buffer), args->dst_memory->client};
  return nullptr;
}

PJRT_Error* PJRT_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_ToHostBuffer_Args",
      PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE, args->struct_size));

  xla::Shape device_shape;
  if (args->src->buffer->on_device_shape().is_dynamic()) {
    PJRT_ASSIGN_OR_RETURN(device_shape,
                          args->src->buffer->logical_on_device_shape());
  } else {
    device_shape = args->src->buffer->on_device_shape();
  }
  xla::Shape host_shape = xla::ShapeUtil::DeviceShapeToHostShape(device_shape);
  if (args->host_layout != nullptr) {
    if (args->host_layout->type ==
        PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Strides) {
      PJRT_RETURN_IF_ERROR(absl::InvalidArgumentError(
          absl::StrCat("PJRT_Buffer_ToHostBuffer does not support host_layout "
                       "with strides for platform ",
                       args->src->buffer->client()->platform_name())));
    }
    if (args->host_layout->tiled.num_tiles > 0) {
      PJRT_RETURN_IF_ERROR(absl::InvalidArgumentError(
          absl::StrCat("PJRT_Buffer_ToHostBuffer does not support host_layout "
                       "with tiled dimension for platform ",
                       args->src->buffer->client()->platform_name())));
    }
    PJRT_ASSIGN_OR_RETURN(*host_shape.mutable_layout(),
                          ConvertToLayout(args->host_layout->tiled));
  }

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
  xla::PjRtFuture<> future = args->src->buffer->ToLiteral(literal.get());

  args->event = new PJRT_Event{std::move(future)};
  args->event->future.OnReady(
      [literal{std::move(literal)}](absl::Status status) {
        /* To keep literal alive */
      });

  return nullptr;
}

PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_IsOnCpu_Args", PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE,
      args->struct_size));
  args->is_on_cpu = args->buffer->buffer->IsOnCpu();
  return nullptr;
}

PJRT_Error* PJRT_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_ReadyEvent_Args", PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE,
      args->struct_size));
  xla::PjRtFuture<> wrapped_promise = args->buffer->buffer->GetReadyFuture();
  args->event = new PJRT_Event{std::move(wrapped_promise)};
  return nullptr;
}

PJRT_Error* PJRT_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_UnsafePointer_Args",
      PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE, args->struct_size));

  PJRT_ASSIGN_OR_RETURN(args->buffer_pointer,
                        args->buffer->client->client->UnsafeBufferPointer(
                            args->buffer->buffer.get()));
  return nullptr;
}

PJRT_Error* PJRT_Buffer_IncreaseExternalReferenceCount(
    PJRT_Buffer_IncreaseExternalReferenceCount_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_IncreaseExternalReferenceCount_Args",
      PJRT_Buffer_IncreaseExternalReferenceCount_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference,
      args->buffer->buffer->AcquireExternalReference());
  args->buffer->external_references.push_back(std::move(external_reference));
  return nullptr;
}

PJRT_Error* PJRT_Buffer_DecreaseExternalReferenceCount(
    PJRT_Buffer_DecreaseExternalReferenceCount_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_DecreaseExternalReferenceCount_Args",
      PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE,
      args->struct_size));

  if (!args->buffer->external_references.empty()) {
    args->buffer->external_references.pop_back();
    return nullptr;
  }
  absl::Status status = xla::InvalidArgument(
      "Attempting to decrease reference on a buffer with zero reference "
      "count.");
  PJRT_Error* error = new PJRT_Error{std::move(status)};
  return error;
}

PJRT_Error* PJRT_Buffer_OpaqueDeviceMemoryDataPointer(
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args",
      PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference,
      args->buffer->buffer->AcquireExternalReference());
  args->device_memory_ptr = external_reference->OpaqueDeviceMemoryDataPointer();
  return nullptr;
}

// ---------------------------- CopyToDeviceStream -----------------------------

PJRT_Error* PJRT_CopyToDeviceStream_Destroy(
    PJRT_CopyToDeviceStream_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_CopyToDeviceStream_Destroy",
      PJRT_CopyToDeviceStream_Destroy_Args_STRUCT_SIZE, args->struct_size));

  delete args->stream;
  return nullptr;
}

PJRT_Error* PJRT_CopyToDeviceStream_AddChunk(
    PJRT_CopyToDeviceStream_AddChunk_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_CopyToDeviceStream_AddChunk_Args",
      PJRT_CopyToDeviceStream_AddChunk_Args_STRUCT_SIZE, args->struct_size));

  xla::PjRtFuture<> future =
      args->stream->stream->AddChunk(ConvertToCppChunk(*args->chunk));
  args->transfer_complete = new PJRT_Event{std::move(future)};
  return nullptr;
}

PJRT_Error* PJRT_CopyToDeviceStream_TotalBytes(
    PJRT_CopyToDeviceStream_TotalBytes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_CopyToDeviceStream_TotalBytes_Args",
      PJRT_CopyToDeviceStream_TotalBytes_Args_STRUCT_SIZE, args->struct_size));

  args->total_bytes = args->stream->stream->total_bytes();
  return nullptr;
}

PJRT_Error* PJRT_CopyToDeviceStream_GranuleSize(
    PJRT_CopyToDeviceStream_GranuleSize_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_CopyToDeviceStream_GranuleSize_Args",
      PJRT_CopyToDeviceStream_GranuleSize_Args_STRUCT_SIZE, args->struct_size));

  args->granule_size_in_bytes = args->stream->stream->granule_size_in_bytes();
  return nullptr;
}

PJRT_Error* PJRT_CopyToDeviceStream_CurrentBytes(
    PJRT_CopyToDeviceStream_CurrentBytes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_CopyToDeviceStream_CurrentBytes_Args",
      PJRT_CopyToDeviceStream_CurrentBytes_Args_STRUCT_SIZE,
      args->struct_size));

  args->current_bytes = args->stream->stream->current_bytes();
  return nullptr;
}

// -------------------------------- Events -------------------------------------

PJRT_Error* PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Event_Destroy", PJRT_Event_Destroy_Args_STRUCT_SIZE,
      args->struct_size));

  delete args->event;
  return nullptr;
}

PJRT_Error* PJRT_Event_IsReady(PJRT_Event_IsReady_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Event_IsReady", PJRT_Event_IsReady_Args_STRUCT_SIZE,
      args->struct_size));

  args->is_ready = args->event->future.IsReady();
  return nullptr;
}

PJRT_Error* PJRT_Event_Await(PJRT_Event_Await_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Event_Await", PJRT_Event_Await_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Event* event = args->event;
  PJRT_RETURN_IF_ERROR(event->future.Await());
  return nullptr;
}

PJRT_Error* PJRT_Event_Error(PJRT_Event_Error_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Event_Error", PJRT_Event_Error_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Event* event = args->event;
  CHECK(event->future.IsReady());
  PJRT_RETURN_IF_ERROR(event->future.Await());
  return nullptr;
}

PJRT_Error* PJRT_Event_OnReady(PJRT_Event_OnReady_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Event_OnReady", PJRT_Event_OnReady_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Event_OnReadyCallback callback = args->callback;
  void* user_arg = args->user_arg;
  auto impl_callback = [callback, user_arg](absl::Status status) -> void {
    PJRT_Error* error = nullptr;
    if (!status.ok()) {
      error = new PJRT_Error{status};
    }
    callback(error, user_arg);
  };
  args->event->future.OnReady(impl_callback);
  return nullptr;
}

// ------------------------------ Device Topology ------------------------------

PJRT_Error* PJRT_TopologyDescription_Destroy(
    PJRT_TopologyDescription_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_Destroy_Args",
      PJRT_TopologyDescription_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->topology;
  return nullptr;
}

PJRT_Error* PJRT_TopologyDescription_PlatformName(
    PJRT_TopologyDescription_PlatformName_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_PlatformName_Args",
      PJRT_TopologyDescription_PlatformName_Args_STRUCT_SIZE,
      args->struct_size));
  absl::string_view platform_name = args->topology->topology->platform_name();
  args->platform_name = platform_name.data();
  args->platform_name_size = platform_name.size();
  return nullptr;
}

PJRT_Error* PJRT_TopologyDescription_PlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_PlatformVersion_Args",
      PJRT_TopologyDescription_PlatformVersion_Args_STRUCT_SIZE,
      args->struct_size));
  absl::string_view platform_version =
      args->topology->topology->platform_version();
  args->platform_version = platform_version.data();
  args->platform_version_size = platform_version.size();
  return nullptr;
}

PJRT_Error* PJRT_TopologyDescription_GetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_GetDeviceDescriptions_Args",
      PJRT_TopologyDescription_GetDeviceDescriptions_Args_STRUCT_SIZE,
      args->struct_size));
  args->descriptions = args->topology->description_pointers.data();
  args->num_descriptions = args->topology->description_pointers.size();
  return nullptr;
}

PJRT_Error* PJRT_TopologyDescription_Serialize(
    PJRT_TopologyDescription_Serialize_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_Serialize_Args",
      PJRT_TopologyDescription_Serialize_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(std::string out, args->topology->topology->Serialize());
  auto* storage = new PJRT_SerializedTopology{std::move(out)};
  args->serialized_topology = storage;
  args->serialized_topology_deleter =
      +[](PJRT_SerializedTopology* serialized_topology) {
        delete serialized_topology;
      };
  args->serialized_bytes = storage->serialized.data();
  args->serialized_bytes_size = storage->serialized.size();
  return nullptr;
}

PJRT_Error* PJRT_TopologyDescription_Attributes(
    PJRT_TopologyDescription_Attributes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_Attributes_Args",
      PJRT_TopologyDescription_Attributes_Args_STRUCT_SIZE, args->struct_size));
  args->attributes = args->topology->attributes.data();
  args->num_attributes = args->topology->attributes.size();
  return nullptr;
}

PJRT_Error* PJRT_Compile(PJRT_Compile_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Compile_Args", PJRT_Compile_Args_STRUCT_SIZE, args->struct_size));
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Program", PJRT_Program_STRUCT_SIZE, args->program->struct_size));

  xla::PjRtClient* client = nullptr;
  if (args->client != nullptr) {
    client = args->client->client.get();
  }
  PJRT_ASSIGN_OR_RETURN(
      xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));

  std::optional<mlir::MLIRContext> context;
  PJRT_ASSIGN_OR_RETURN(auto module_or_hlo,
                        ParsePjrtProgram(context, args->program));
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtExecutable> executable,
                        std::visit(
                            [&](auto& program) {
                              return PjRtCompile(
                                  options, UnpackPjrtProgram(program),
                                  *args->topology->topology, client);
                            },
                            module_or_hlo));
  args->executable = new PJRT_Executable(std::move(executable));
  return nullptr;
}

PJRT_Error* PJRT_Layouts_MemoryLayout_Destroy(
    PJRT_Layouts_MemoryLayout_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Layouts_MemoryLayout_Destroy_Args",
      PJRT_Layouts_MemoryLayout_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->layout;
  return nullptr;
}

PJRT_Error* PJRT_Layouts_MemoryLayout_Serialize(
    PJRT_Layouts_MemoryLayout_Serialize_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Layouts_MemoryLayout_Serialize_Args",
      PJRT_Layouts_MemoryLayout_Serialize_Args_STRUCT_SIZE, args->struct_size));

  PJRT_Layouts_SerializedLayout* s_layout = new PJRT_Layouts_SerializedLayout{
      /* .serialized = */ args->layout->layout->Serialize()};
  args->serialized_layout = s_layout;
  args->serialized_bytes = s_layout->serialized.data();
  args->serialized_bytes_size = s_layout->serialized.size();
  args->serialized_layout_deleter =
      +[](PJRT_Layouts_SerializedLayout* s_lay) { delete s_lay; };
  return nullptr;
}

PJRT_Error* PJRT_Layouts_PJRT_Client_GetDefaultLayout(
    PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args",
      PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_ASSIGN_OR_RETURN(xla::Layout xla_layout,
                        args->client->client->GetDefaultLayout(
                            pjrt::ConvertFromPjRtBufferType(args->type),
                            {args->dims, args->num_dims}));
  auto pjrt_xla_layout = std::make_shared<xla::PjRtLayout>(xla_layout);
  args->layout = new PJRT_Layouts_MemoryLayout{std::move(pjrt_xla_layout)};
  return nullptr;
}

PJRT_Error* PJRT_Layouts_PJRT_Buffer_MemoryLayout(
    PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args",
      PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args_STRUCT_SIZE,
      args->struct_size));

  args->layout = new PJRT_Layouts_MemoryLayout{args->buffer->buffer->layout()};
  return nullptr;
}

static std::vector<PJRT_NamedValue> PopulatePjrtAttributes(
    const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&
        attributes) {
  std::vector<PJRT_NamedValue> c_attributes;
  c_attributes.resize(attributes.size());
  int ind = 0;
  // Doing shallow copy of attribute names and values when it's string or an
  // array.
  for (auto const& [name, value] : attributes) {
    PJRT_NamedValue& cur_attribute = c_attributes[ind];
    cur_attribute.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    cur_attribute.extension_start = nullptr;
    cur_attribute.name = name.c_str();
    cur_attribute.name_size = name.size();
    if (const std::string* string_val = std::get_if<std::string>(&value)) {
      cur_attribute.type = PJRT_NamedValue_Type::PJRT_NamedValue_kString;
      cur_attribute.string_value = string_val->c_str();
      cur_attribute.value_size = string_val->size();
    } else if (const std::vector<int64_t>* vector_val =
                   std::get_if<std::vector<int64_t>>(&value)) {
      cur_attribute.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List;
      cur_attribute.int64_array_value = vector_val->data();
      cur_attribute.value_size = vector_val->size();
    } else if (const int64_t* int_value = std::get_if<int64_t>(&value)) {
      cur_attribute.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
      cur_attribute.int64_value = *int_value;
      cur_attribute.value_size = 1;
    } else {
      // Do not allow other types (such as
      // PJRT_NamedValue::PJRT_NamedValue_kFloat) since device attributes
      // currently should not return other types.
      CHECK(false) << "Unexpected attribute type " << value.index() << " for "
                   << name;
    }
    ++ind;
  }
  return c_attributes;
}

static void PopulatePjrtClientDevices(PJRT_Client* c_client) {
  absl::Span<xla::PjRtDevice* const> cpp_devices = c_client->client->devices();
  const size_t num_devices = cpp_devices.size();
  c_client->owned_devices.reserve(num_devices);
  c_client->devices.reserve(num_devices);
  c_client->addressable_devices.reserve(
      c_client->client->addressable_device_count());

  for (xla::PjRtDevice* device : cpp_devices) {
    c_client->owned_devices.push_back(
        PJRT_Device{device, {&device->description()}});
    PJRT_Device* c_device = &c_client->owned_devices.back();
    c_device->client = c_client;
    c_device->description.attributes =
        PopulatePjrtAttributes(device->description().Attributes());
    c_client->devices.push_back(c_device);
    if (device->IsAddressable()) {
      c_client->addressable_devices.push_back(c_device);
    }
    c_client->c_device_from_cpp_device[device] = c_device;
  }
  CHECK_EQ(c_client->addressable_devices.size(),
           c_client->client->addressable_device_count());
}

static void PopulatePjrtClientMemories(PJRT_Client* c_client) {
  absl::Span<xla::PjRtMemorySpace* const> memory_spaces =
      c_client->client->memory_spaces();
  // TODO(yueshengys): After global memories are supported, `owned_memories`
  // should eventually contain all memories not just addressable ones.
  c_client->owned_memories.reserve(memory_spaces.size());
  c_client->addressable_memories.reserve(memory_spaces.size());
  for (xla::PjRtMemorySpace* memory_space : memory_spaces) {
    c_client->owned_memories.push_back(PJRT_Memory{memory_space});
    PJRT_Memory* c_memory = &c_client->owned_memories.back();
    c_memory->client = c_client;
    c_client->addressable_memories.push_back(c_memory);
    c_client->c_memory_from_cpp_memory[memory_space] = c_memory;
  }
}

static void AttachDevicesAndMemories(PJRT_Client* c_client) {
  for (PJRT_Device* c_device : c_client->devices) {
    // TODO(yueshengys): Remove this when global memories are supported.
    if (!c_device->device->IsAddressable()) {
      continue;
    }
    absl::Span<xla::PjRtMemorySpace* const> cpp_memories =
        c_device->device->memory_spaces();
    c_device->addressable_memories.reserve(cpp_memories.size());
    for (xla::PjRtMemorySpace* memory_space : cpp_memories) {
      c_device->addressable_memories.push_back(
          GetCMemory(c_client, memory_space));
    }
  }

  // TODO(yueshengys): Expand this to all memories when supported, not just
  // addressable ones.
  for (PJRT_Memory* c_memory : c_client->addressable_memories) {
    absl::Span<xla::PjRtDevice* const> cpp_devices =
        c_memory->memory_space->devices();
    c_memory->devices.reserve(cpp_devices.size());
    for (xla::PjRtDevice* cpp_device : cpp_devices) {
      c_memory->devices.push_back(GetCDevice(c_client, cpp_device));
    }
  }
}

static absl::StatusOr<std::unique_ptr<PJRT_TopologyDescription>>
GetStatusOrTopologyDescription(const xla::PjRtClient& cpp_client) {
  absl::StatusOr<const xla::PjRtTopologyDescription*> status_or_cpp_topo =
      cpp_client.GetTopologyDescription();
  if (!status_or_cpp_topo.ok()) {
    return status_or_cpp_topo.status();
  }
  return std::unique_ptr<PJRT_TopologyDescription>(
      CreateWrapperDeviceTopology(*status_or_cpp_topo));
}

PJRT_Client* CreateWrapperClient(std::unique_ptr<xla::PjRtClient> cpp_client) {
  PJRT_Client* c_client = new PJRT_Client(std::move(cpp_client));
  PopulatePjrtClientDevices(c_client);
  PopulatePjrtClientMemories(c_client);
  AttachDevicesAndMemories(c_client);
  return c_client;
}

PJRT_ExecuteContext* CreateWrapperExecuteContext(
    std::unique_ptr<xla::ExecuteContext> cpp_execute_context) {
  PJRT_ExecuteContext* execute_context =
      new PJRT_ExecuteContext{std::move(cpp_execute_context)};
  return execute_context;
}

PJRT_TopologyDescription* CreateWrapperDeviceTopology(
    const xla::PjRtTopologyDescription* cpp_topology) {
  PJRT_TopologyDescription* c_topology =
      new PJRT_TopologyDescription{/*owned_topology=*/nullptr, cpp_topology};
  c_topology->cpp_descriptions = c_topology->topology->DeviceDescriptions();
  c_topology->descriptions.reserve(c_topology->cpp_descriptions.size());
  c_topology->description_pointers.reserve(c_topology->cpp_descriptions.size());
  for (auto& description : c_topology->cpp_descriptions) {
    c_topology->descriptions.emplace_back(
        PJRT_DeviceDescription{description.get()});
    c_topology->description_pointers.emplace_back(
        &c_topology->descriptions.back());
    c_topology->descriptions.back().attributes =
        PopulatePjrtAttributes(description->Attributes());
  }
  c_topology->attributes =
      PopulatePjrtAttributes(c_topology->topology->Attributes());
  return c_topology;
}

PJRT_TopologyDescription* CreateWrapperDeviceTopology(
    std::unique_ptr<xla::PjRtTopologyDescription> cpp_topology) {
  PJRT_TopologyDescription* topo_desc =
      CreateWrapperDeviceTopology(cpp_topology.get());
  topo_desc->owned_topology = std::move(cpp_topology);
  return topo_desc;
}

}  // namespace pjrt

PJRT_Client::PJRT_Client(std::unique_ptr<xla::PjRtClient> cpp_client)
    : client(std::move(cpp_client)),
      topology(pjrt::GetStatusOrTopologyDescription(*client)) {}

PJRT_Executable::PJRT_Executable(
    std::shared_ptr<xla::PjRtExecutable> shared_executable)
    : shared_executable(std::move(shared_executable)),
      fingerprint(this->shared_executable->FingerprintExecutable()) {
  executable = this->shared_executable.get();
}

PJRT_Executable::PJRT_Executable(xla::PjRtExecutable* unowned_executable)
    : executable(unowned_executable),
      fingerprint(executable->FingerprintExecutable()) {}

PJRT_LoadedExecutable::PJRT_LoadedExecutable(
    std::shared_ptr<xla::PjRtLoadedExecutable> executable, PJRT_Client* client)
    : executable(std::move(executable)), client(client) {
  pjrt::PopulatePjrtExecutableAddressableDevices(this);
}

namespace pjrt {

PJRT_Api CreatePjrtApi(PJRT_Client_Create* create_fn,
                       PJRT_ExecuteContext_Create* execute_context_create_fn,
                       PJRT_TopologyDescription_Create* topology_create_fn,
                       PJRT_Plugin_Initialize* plugin_initialize_fn,
                       PJRT_Extension_Base* extension_start,
                       PJRT_Plugin_Attributes* plugin_attributes_fn) {
  return PJRT_Api{
      /*struct_size=*/PJRT_Api_STRUCT_SIZE,
      /*extension_start=*/extension_start,

      /*pjrt_api_version=*/
      PJRT_Api_Version{/*struct_size=*/PJRT_Api_Version_STRUCT_SIZE,
                       /*priv=*/nullptr,
                       /*major_version=*/PJRT_API_MAJOR,
                       /*minor_version=*/PJRT_API_MINOR},

      /*PJRT_Error_Destroy=*/pjrt::PJRT_Error_Destroy,
      /*PJRT_Error_Message=*/pjrt::PJRT_Error_Message,
      /*PJRT_Error_GetCode=*/pjrt::PJRT_Error_GetCode,

      /*PJRT_Plugin_Initialize=*/plugin_initialize_fn,
      /*PJRT_Plugin_Attributes=*/plugin_attributes_fn,

      /*PJRT_Event_Destroy=*/pjrt::PJRT_Event_Destroy,
      /*PJRT_Event_IsReady=*/pjrt::PJRT_Event_IsReady,
      /*PJRT_Event_Error=*/pjrt::PJRT_Event_Error,
      /*PJRT_Event_Await=*/pjrt::PJRT_Event_Await,
      /*PJRT_Event_OnReady=*/pjrt::PJRT_Event_OnReady,

      /*PJRT_Client_Create=*/create_fn,
      /*PJRT_Client_Destroy=*/pjrt::PJRT_Client_Destroy,
      /*PJRT_Client_PlatformName=*/pjrt::PJRT_Client_PlatformName,
      /*PJRT_Client_ProcessIndex=*/pjrt::PJRT_Client_ProcessIndex,
      /*PJRT_Client_PlatformVersion= */ pjrt::PJRT_Client_PlatformVersion,
      /*PJRT_Client_Devices= */ pjrt::PJRT_Client_Devices,
      /*PJRT_Client_AddressableDevices=*/
      pjrt::PJRT_Client_AddressableDevices,
      /*PJRT_Client_LookupDevice=*/pjrt::PJRT_Client_LookupDevice,
      /*PJRT_Client_LookupAddressableDevice=*/
      pjrt::PJRT_Client_LookupAddressableDevice,
      /*PJRT_Client_AddressableMemories=*/pjrt::PJRT_Client_AddressableMemories,
      /*PJRT_Client_Compile=*/pjrt::PJRT_Client_Compile,
      /*PJRT_Client_DefaultDeviceAssignment=*/
      pjrt::PJRT_Client_DefaultDeviceAssignment,
      /*PJRT_Client_BufferFromHostBuffer=*/
      pjrt::PJRT_Client_BufferFromHostBuffer,

      /*PJRT_DeviceDescription_Id=*/pjrt::PJRT_DeviceDescription_Id,
      /*PJRT_DeviceDescription_ProcessIndex=*/
      pjrt::PJRT_DeviceDescription_ProcessIndex,
      /*PJRT_DeviceDescription_Attributes=*/
      pjrt::PJRT_DeviceDescription_Attributes,
      /*PJRT_DeviceDescription_Kind=*/pjrt::PJRT_DeviceDescription_Kind,
      /*PJRT_DeviceDescription_DebugString=*/
      pjrt::PJRT_DeviceDescription_DebugString,
      /*PJRT_DeviceDescription_ToString=*/
      pjrt::PJRT_DeviceDescription_ToString,

      /*PJRT_Device_GetDescription=*/pjrt::PJRT_Device_GetDescription,
      /*PJRT_Device_IsAddressable=*/pjrt::PJRT_Device_IsAddressable,
      /*PJRT_Device_LocalHardwareId=*/pjrt::PJRT_Device_LocalHardwareId,
      /*PJRT_Device_AddressableMemories=*/pjrt::PJRT_Device_AddressableMemories,
      /*PJRT_Device_DefaultMemory=*/pjrt::PJRT_Device_DefaultMemory,
      /*PJRT_Device_MemoryStats=*/pjrt::PJRT_Device_MemoryStats,

      /*PJRT_Memory_Id=*/pjrt::PJRT_Memory_Id,
      /*PJRT_Memory_Kind=*/pjrt::PJRT_Memory_Kind,
      /*PJRT_Memory_DebugString=*/pjrt::PJRT_Memory_DebugString,
      /*PJRT_Memory_ToString=*/pjrt::PJRT_Memory_ToString,
      /*PJRT_Memory_AddressableByDevices=*/
      pjrt::PJRT_Memory_AddressableByDevices,

      /*PJRT_Executable_Destroy=*/pjrt::PJRT_Executable_Destroy,
      /*PJRT_Executable_Name=*/pjrt::PJRT_Executable_Name,
      /*PJRT_Executable_NumReplicas=*/pjrt::PJRT_Executable_NumReplicas,
      /*PJRT_Executable_NumPartitions=*/
      pjrt::PJRT_Executable_NumPartitions,
      /*PJRT_Executable_NumOutputs=*/pjrt::PJRT_Executable_NumOutputs,
      /*PJRT_Executable_SizeOfGeneratedCodeInBytes=*/
      pjrt::PJRT_Executable_SizeOfGeneratedCodeInBytes,
      /*PJRT_Executable_GetCostAnalysis=*/pjrt::PJRT_Executable_GetCostAnalysis,
      /*PJRT_Executable_OutputMemoryKinds=*/
      pjrt::PJRT_Executable_OutputMemoryKinds,
      /*PJRT_Executable_OptimizedProgram=*/
      pjrt::PJRT_Executable_OptimizedProgram,
      /*PJRT_Executable_Serialize=*/pjrt::PJRT_Executable_Serialize,

      /*PJRT_LoadedExecutable_Destroy=*/pjrt::PJRT_LoadedExecutable_Destroy,
      /*PJRT_LoadedExecutable_GetExecutable=*/
      pjrt::PJRT_LoadedExecutable_GetExecutable,
      /*PJRT_LoadedExecutable_AddressableDevices=*/
      pjrt::PJRT_LoadedExecutable_AddressableDevices,
      /*PJRT_LoadedExecutable_Delete=*/pjrt::PJRT_LoadedExecutable_Delete,
      /*PJRT_LoadedExecutable_IsDeleted=*/
      pjrt::PJRT_LoadedExecutable_IsDeleted,
      /*PJRT_LoadedExecutable_Execute=*/pjrt::PJRT_LoadedExecutable_Execute,
      /*PJRT_Executable_DeserializeAndLoad=*/
      pjrt::PJRT_Executable_DeserializeAndLoad,
      /*PJRT_LoadedExecutable_Fingerprint=*/
      pjrt::PJRT_LoadedExecutable_Fingerprint,

      /*PJRT_Buffer_Destroy=*/pjrt::PJRT_Buffer_Destroy,
      /*PJRT_Buffer_ElementType=*/pjrt::PJRT_Buffer_ElementType,
      /*PJRT_Buffer_Dimensions=*/pjrt::PJRT_Buffer_Dimensions,
      /*PJRT_Buffer_UnpaddedDimensions=*/
      pjrt::PJRT_Buffer_UnpaddedDimensions,
      /*PJRT_Buffer_DynamicDimensionIndices=*/
      pjrt::PJRT_Buffer_DynamicDimensionIndices,
      /*PJRT_Buffer_GetMemoryLayout=*/
      pjrt::PJRT_Buffer_GetMemoryLayout,
      /*PJRT_Buffer_OnDeviceSizeInBytes=*/
      pjrt::PJRT_Buffer_OnDeviceSizeInBytes,
      /*PJRT_Buffer_Device=*/pjrt::PJRT_Buffer_Device,
      /*PJRT_Buffer_Memory=*/pjrt::PJRT_Buffer_Memory,
      /*PJRT_Buffer_Delete=*/pjrt::PJRT_Buffer_Delete,
      /*PJRT_Buffer_IsDeleted=*/pjrt::PJRT_Buffer_IsDeleted,
      /*PJRT_Buffer_CopyToDevice=*/pjrt::PJRT_Buffer_CopyToDevice,
      /*PJRT_Buffer_ToHostBuffer=*/pjrt::PJRT_Buffer_ToHostBuffer,
      /*PJRT_Buffer_IsOnCpu=*/pjrt::PJRT_Buffer_IsOnCpu,
      /*PJRT_Buffer_ReadyEvent=*/pjrt::PJRT_Buffer_ReadyEvent,
      /*PJRT_Buffer_UnsafePointer=*/pjrt::PJRT_Buffer_UnsafePointer,
      /*PJRT_Buffer_IncreaseExternalReferenceCount=*/
      pjrt::PJRT_Buffer_IncreaseExternalReferenceCount,
      /*PJRT_Buffer_DecreaseExternalReferenceCount=*/
      pjrt::PJRT_Buffer_DecreaseExternalReferenceCount,
      /*PJRT_Buffer_OpaqueDeviceMemoryDataPointer=*/
      pjrt::PJRT_Buffer_OpaqueDeviceMemoryDataPointer,

      /*PJRT_CopyToDeviceStream_Destroy=*/
      pjrt::PJRT_CopyToDeviceStream_Destroy,
      /*PJRT_CopyToDeviceStream_AddChunk=*/
      pjrt::PJRT_CopyToDeviceStream_AddChunk,
      /*PJRT_CopyToDeviceStream_TotalBytes=*/
      pjrt::PJRT_CopyToDeviceStream_TotalBytes,
      /*PJRT_CopyToDeviceStream_GranuleSize=*/
      pjrt::PJRT_CopyToDeviceStream_GranuleSize,
      /*PJRT_CopyToDeviceStream_CurrentBytes=*/
      pjrt::PJRT_CopyToDeviceStream_CurrentBytes,

      /*PJRT_TopologyDescription_Create=*/topology_create_fn,
      /*PJRT_TopologyDescription_Destroy=*/
      pjrt::PJRT_TopologyDescription_Destroy,
      /*PJRT_TopologyDescription_PlatformName=*/
      pjrt::PJRT_TopologyDescription_PlatformName,
      /*PJRT_TopologyDescription_PlatformVersion=*/
      pjrt::PJRT_TopologyDescription_PlatformVersion,
      /*PJRT_TopologyDescription_GetDeviceDescriptions=*/
      pjrt::PJRT_TopologyDescription_GetDeviceDescriptions,
      /*PJRT_TopologyDescription_Serialize=*/
      pjrt::PJRT_TopologyDescription_Serialize,
      /*PJRT_TopologyDescription_Attributes=*/
      pjrt::PJRT_TopologyDescription_Attributes,

      /*PJRT_Compile=*/pjrt::PJRT_Compile,

      // Always add new fields to the end of the struct. Move fields below to
      // their corresponding places after each major version bump.
      /*PJRT_Executable_OutputElementTypes=*/
      pjrt::PJRT_Executable_OutputElementTypes,
      /*PJRT_Executable_OutputDimensions=*/
      pjrt::PJRT_Executable_OutputDimensions,
      /*PJRT_Buffer_CopyToMemory=*/
      pjrt::PJRT_Buffer_CopyToMemory,
      /*PJRT_Client_CreateViewOfDeviceBuffer=*/
      pjrt::PJRT_Client_CreateViewOfDeviceBuffer,
      /*PJRT_Executable_Fingerprint=*/pjrt::PJRT_Executable_Fingerprint,
      /*PJRT_Client_TopologyDescription= */
      pjrt::PJRT_Client_TopologyDescription,
      /*PJRT_Executable_GetCompiledMemoryStats= */
      pjrt::PJRT_Executable_GetCompiledMemoryStats,
      /*PJRT_Memory_Kind_Id=*/pjrt::PJRT_Memory_Kind_Id,

      /*PJRT_ExecuteContext_Create=*/execute_context_create_fn,
      /*PJRT_ExecuteContext_Destroy=*/pjrt::PJRT_ExecuteContext_Destroy,
      /*PJRT_Buffer_CopyRawToHost=*/pjrt::PJRT_Buffer_CopyRawToHost,
      /*PJRT_AsyncHostToDeviceTransferManager_Destroy=*/
      pjrt::PJRT_AsyncHostToDeviceTransferManager_Destroy,
      /*PJRT_AsyncHostToDeviceTransferManager_TransferData=*/
      pjrt::PJRT_AsyncHostToDeviceTransferManager_TransferData,
      /*PJRT_Client_CreateBuffersForAsyncHostToDevice=*/
      pjrt::PJRT_Client_CreateBuffersForAsyncHostToDevice,
      /*PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer=*/
      pjrt::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer,
      /*PJRT_AsyncHostToDeviceTransferManager_Device=*/
      pjrt::PJRT_AsyncHostToDeviceTransferManager_Device,
      /*PJRT_AsyncHostToDeviceTransferManager_BufferCount=*/
      pjrt::PJRT_AsyncHostToDeviceTransferManager_BufferCount,
      /*PJRT_AsyncHostToDeviceTransferManager_BufferSize=*/
      pjrt::PJRT_AsyncHostToDeviceTransferManager_BufferSize,
      /*PJRT_AsyncHostToDeviceTransferManager_SetBufferError=*/
      pjrt::PJRT_AsyncHostToDeviceTransferManager_SetBufferError,
      /*PJRT_AsyncHostToDeviceTransferManager_AddMetadata=*/
      pjrt::PJRT_AsyncHostToDeviceTransferManager_AddMetadata,
      /*PJRT_Client_DmaMap=*/pjrt::PJRT_Client_DmaMap,
      /*PJRT_Client_DmaUnmap=*/pjrt::PJRT_Client_DmaUnmap,
  };
}

PJRT_Layouts_Extension CreateLayoutsExtension(PJRT_Extension_Base* next) {
  return PJRT_Layouts_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_Layouts_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_Layouts,
          /*next=*/next,
      },
      /*PJRT_Layouts_MemoryLayout_Destroy=*/
      pjrt::PJRT_Layouts_MemoryLayout_Destroy,
      /*PJRT_Layouts_MemoryLayout_Serialize=*/
      pjrt::PJRT_Layouts_MemoryLayout_Serialize,
      /*PJRT_Layouts_PJRT_Client_GetDefaultLayout=*/
      pjrt::PJRT_Layouts_PJRT_Client_GetDefaultLayout,
      /*PJRT_Layouts_PJRT_Buffer_MemoryLayout=*/
      pjrt::PJRT_Layouts_PJRT_Buffer_MemoryLayout,
  };
}

PJRT_MemoryDescriptions_Extension CreateMemoryDescriptionsExtension(
    PJRT_Extension_Base* next) {
  return PJRT_MemoryDescriptions_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_MemoryDescriptions_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_MemoryDescriptions,
          /*next=*/next,
      },
      /*PJRT_DeviceDescription_MemorySpaces=*/
      pjrt::PJRT_DeviceDescription_MemoryDescriptions,
      /*PJRT_MemoryDescription_Kind=*/
      pjrt::PJRT_MemoryDescription_Kind};
}

}  // namespace pjrt
