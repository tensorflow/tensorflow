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

#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/ffi/execution_context.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/c/pjrt_c_api_stream_extension.h"
#include "xla/pjrt/c/pjrt_c_api_tpu_topology_extension.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_phase_compiler.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/extensions/cross_host_transfers/pjrt_c_api_cross_host_transfers_extension.h"
#include "xla/pjrt/extensions/executable_metadata/executable_metadata_extension.h"
#include "xla/pjrt/extensions/host_allocator/host_allocator_extension.h"
#include "xla/pjrt/extensions/host_allocator/host_allocator_interface_impl.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/pjrt/scoped_async_tracking_event.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

constexpr int kMaxDims = 4;

// Helper macros

// Return error future if not success and frees the PJRT_Error returned by
// `expr`.
#define RETURN_FUTURE_IF_ERROR(expr, c_api)                              \
  do {                                                                   \
    PJRT_Error* error = (expr);                                          \
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> _error(         \
        error, pjrt::MakeErrorDeleter(c_api));                           \
    absl::Status _status = pjrt::PjrtErrorToStatus(_error.get(), c_api); \
    if (!_status.ok()) {                                                 \
      return Future<>(_status);                                          \
    }                                                                    \
  } while (false)

// ---------------------------------- Client -----------------------------------

static absl::StatusOr<const PjRtCApiTopologyDescription> InitClientTopoDesc(
    const PJRT_Api* c_api, PJRT_Client* c_client) {
  absl::StatusOr<PJRT_TopologyDescription*> c_topo =
      pjrt::GetTopologyDescription(c_client, c_api);
  TF_RETURN_IF_ERROR(c_topo.status());
  return PjRtCApiTopologyDescription(c_api, *c_topo, /*owned=*/false);
}

static absl::flat_hash_map<PJRT_Extension_Type, PJRT_Extension_Base*>
InitExtensions(const PJRT_Api* c_api) {
  absl::flat_hash_map<PJRT_Extension_Type, PJRT_Extension_Base*> extensions;
  for (PJRT_Extension_Base* ext = c_api->extension_start; ext != nullptr;
       ext = ext->next) {
    extensions.emplace(ext->type, ext);
  }
  return extensions;
}

static absl::StatusOr<std::unique_ptr<PjRtClient::HostAllocator>>
InitHostAllocator(const PJRT_Api* c_api, PJRT_Client* c_client) {
  PJRT_HostAllocator_Extension* extension =
      pjrt::FindExtension<PJRT_HostAllocator_Extension>(
          c_api, PJRT_Extension_Type::PJRT_Extension_Type_HostAllocator);
  if (extension == nullptr) {
    return absl::UnimplementedError("HostAllocator extension not found");
  }
  return std::make_unique<HostAllocatorInterfaceImpl>(c_client, extension);
}

PjRtCApiClient::PjRtCApiClient(
    const PJRT_Api* c_api, PJRT_Client* c_client,
    std::unique_ptr<pjrt::PJRT_KeyValueCallbackData> kv_callback_data)
    : c_api_(c_api),
      c_client_(std::unique_ptr<PJRT_Client, ::pjrt::PJRT_ClientDeleter>(
          c_client, ::pjrt::MakeClientDeleter(c_api))),
      kv_callback_data_(std::move(kv_callback_data)),
      topo_desc_(InitClientTopoDesc(c_api, c_client)),
      extensions_(InitExtensions(c_api)),
      host_allocator_(InitHostAllocator(c_api, c_client)),
      // Example platform version string:
      //   PJRT C API
      //   TFRT TPU v2
      //   Built on Mar 4 2021 15:25:57 (1614900357) cl/360760169
      platform_version_(absl::StrCat(
          "PJRT C API\n", ::pjrt::GetPlatformVersion(c_client, c_api))),
      platform_name_(::pjrt::GetPlatformName(c_client, c_api)),
      platform_id_(tsl::Fingerprint64(platform_name_)) {
  InitDevicesAndMemorySpaces();
  InitAttributes();
  LOG(INFO) << "PjRtCApiClient created.";
}

void PjRtCApiClient::InitDevicesAndMemorySpaces() {
  // Initialize devices.
  PJRT_Client_Devices_Args devices_args;
  devices_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
  devices_args.extension_start = nullptr;
  devices_args.client = c_client_.get();

  pjrt::LogFatalIfPjrtError(c_api_->PJRT_Client_Devices(&devices_args), c_api_);

  const size_t num_devices = devices_args.num_devices;
  c_to_cpp_device_map_.reserve(num_devices);
  owned_devices_.reserve(num_devices);
  devices_.reserve(num_devices);

  for (int i = 0; i < num_devices; ++i) {
    PJRT_Device* device = devices_args.devices[i];
    std::unique_ptr<PjRtCApiDevice>& cpp_device = owned_devices_.emplace_back(
        std::make_unique<PjRtCApiDevice>(device, this));
    devices_.push_back(cpp_device.get());
    c_to_cpp_device_map_[device] = cpp_device.get();
  }

  // Initialize addressable devices.
  PJRT_Client_AddressableDevices_Args address_args;
  address_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  address_args.extension_start = nullptr;
  address_args.client = c_client_.get();

  pjrt::LogFatalIfPjrtError(
      c_api_->PJRT_Client_AddressableDevices(&address_args), c_api_);

  const size_t num_addressable_devices = address_args.num_addressable_devices;
  addressable_devices_.reserve(num_addressable_devices);

  for (int i = 0; i < num_addressable_devices; ++i) {
    PJRT_Device* c_device = address_args.addressable_devices[i];
    addressable_devices_.push_back(GetCppDevice(c_device));
  }

  // Initialize addressable memory spaces.
  PJRT_Client_AddressableMemories_Args memory_args;
  memory_args.struct_size = PJRT_Client_AddressableMemories_Args_STRUCT_SIZE;
  memory_args.extension_start = nullptr;
  memory_args.client = c_client_.get();

  std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> client_error(
      c_api_->PJRT_Client_AddressableMemories(&memory_args),
      pjrt::MakeErrorDeleter(c_api_));
  if (client_error == nullptr) {
    const size_t num_memories = memory_args.num_addressable_memories;
    c_to_cpp_memory_map_.reserve(num_memories);
    owned_memory_spaces_.reserve(num_memories);
    addressable_memory_spaces_.reserve(num_memories);

    for (int i = 0; i < num_memories; ++i) {
      PJRT_Memory* memory = memory_args.addressable_memories[i];
      std::unique_ptr<PjRtCApiMemorySpace>& cpp_memory =
          owned_memory_spaces_.emplace_back(
              std::make_unique<PjRtCApiMemorySpace>(memory, this));
      addressable_memory_spaces_.push_back(cpp_memory.get());
      c_to_cpp_memory_map_[memory] = cpp_memory.get();
    }
  } else if (pjrt::GetErrorCode(client_error.get(), c_api_) !=
             PJRT_Error_Code_UNIMPLEMENTED) {
    pjrt::LogFatalIfPjrtError(client_error.get(), c_api_);
  }

  // Attach memory spaces to devices.
  // TODO(yueshengys): switch to global devices when supported.
  for (const auto& device : addressable_devices_) {
    PjRtCApiDevice* cpp_device = tensorflow::down_cast<PjRtCApiDevice*>(device);
    PJRT_Device* c_device = cpp_device->c_device();
    PJRT_Device_AddressableMemories_Args args;
    args.struct_size = PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.device = c_device;

    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> device_error(
        c_api_->PJRT_Device_AddressableMemories(&args),
        pjrt::MakeErrorDeleter(c_api_));
    if (device_error != nullptr) {
      if (pjrt::GetErrorCode(device_error.get(), c_api_) !=
          PJRT_Error_Code_UNIMPLEMENTED) {
        pjrt::LogFatalIfPjrtError(device_error.get(), c_api_);
      }
      break;
    }

    const size_t num_memories = args.num_memories;
    cpp_device->memory_spaces_.reserve(num_memories);
    for (int i = 0; i < num_memories; ++i) {
      cpp_device->memory_spaces_.push_back(GetCppMemory(args.memories[i]));
    }
  }

  // Attach devices to memory spaces.
  // TODO(yueshengys): switch to global memories when supported.
  for (const auto& memory : addressable_memory_spaces_) {
    PjRtCApiMemorySpace* cpp_memory =
        tensorflow::down_cast<PjRtCApiMemorySpace*>(memory);
    PJRT_Memory* c_memory = cpp_memory->c_memory();
    PJRT_Memory_AddressableByDevices_Args args;
    args.struct_size = PJRT_Memory_AddressableByDevices_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.memory = c_memory;
    pjrt::LogFatalIfPjrtError(c_api_->PJRT_Memory_AddressableByDevices(&args),
                              c_api_);

    const size_t num_attached_devices = args.num_devices;
    cpp_memory->devices_.reserve(num_attached_devices);

    for (int i = 0; i < num_attached_devices; ++i) {
      cpp_memory->devices_.push_back(GetCppDevice(args.devices[i]));
    }
  }
}

void PjRtCApiClient::InitAttributes() {
  PJRT_Plugin_Attributes_Args args;
  args.struct_size = PJRT_Plugin_Attributes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_Plugin_Attributes(&args), c_api_);
  attributes_ =
      pjrt::ConvertFromPjRtNamedValueList(args.attributes, args.num_attributes);
  attributes_["serialize_with_sdy"] = true;
  PJRT_CrossHostTransfers_Extension* extension =
      FindExtension<PJRT_CrossHostTransfers_Extension>(
          PJRT_Extension_Type::PJRT_Extension_Type_CrossHostTransfers);
  if (extension != nullptr) {
    attributes_["supports_cross_host_transfers"] = true;
  }
}

PJRT_Extension_Base* PjRtCApiClient::FindExtensionImpl(
    PJRT_Extension_Type type) const {
  auto it = extensions_.find(type);
  if (it == extensions_.end()) {
    return nullptr;
  }
  return it->second;
}

int PjRtCApiClient::device_count() const { return devices_.size(); }

int PjRtCApiClient::addressable_device_count() const {
  return addressable_devices_.size();
}

absl::Span<PjRtDevice* const> PjRtCApiClient::devices() const {
  return devices_;
}

absl::Span<PjRtDevice* const> PjRtCApiClient::addressable_devices() const {
  return addressable_devices_;
}

int PjRtCApiClient::process_index() const {
  PJRT_Client_ProcessIndex_Args process_index_args;
  process_index_args.struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE;
  process_index_args.extension_start = nullptr;
  process_index_args.client = c_client_.get();
  pjrt::LogFatalIfPjrtError(
      c_api_->PJRT_Client_ProcessIndex(&process_index_args), c_api_);

  return process_index_args.process_index;
}

absl::string_view PjRtCApiClient::platform_version() const {
  return platform_version_;
}

std::optional<PjRtPluginAttributes> PjRtCApiClient::plugin_attributes() const {
  return PjRtPluginAttributes{c_api_->pjrt_api_version.major_version,
                              c_api_->pjrt_api_version.minor_version,
                              attributes_};
}

static DeviceAssignment CalculateDefaultAssignment(
    int num_replicas, int num_partitions,
    absl::Span<const int> device_assignment) {
  DeviceAssignment cpp_device_assignment(num_replicas, num_partitions);
  const int* iterator = device_assignment.begin();
  for (int replica = 0; replica < num_replicas; ++replica) {
    for (int partition = 0; partition < num_partitions; ++partition) {
      cpp_device_assignment(replica, partition) = *(iterator++);
    }
  }
  return cpp_device_assignment;
}

absl::StatusOr<DeviceAssignment> PjRtCApiClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  PJRT_Client_DefaultDeviceAssignment_Args args;
  args.struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.num_replicas = num_replicas;
  args.num_partitions = num_partitions;
  std::vector<int> assignment_buffer(num_replicas * num_partitions);
  args.default_assignment_size = assignment_buffer.size();
  args.default_assignment = assignment_buffer.data();
  RETURN_STATUS_IF_PJRT_ERROR(
      c_api_->PJRT_Client_DefaultDeviceAssignment(&args), c_api_);
  absl::Span<const int> param{args.default_assignment,
                              args.default_assignment_size};
  return CalculateDefaultAssignment(args.num_replicas, args.num_partitions,
                                    param);
}

absl::StatusOr<PjRtDevice*> PjRtCApiClient::LookupDevice(
    PjRtGlobalDeviceId global_device_id) const {
  PJRT_Client_LookupDevice_Args args;
  args.struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.id = global_device_id.value();
  RETURN_STATUS_IF_PJRT_ERROR(c_api_->PJRT_Client_LookupDevice(&args), c_api_);
  return GetCppDevice(args.device);
}

absl::StatusOr<PjRtDevice*> PjRtCApiClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_device_id) const {
  PJRT_Client_LookupAddressableDevice_Args args;
  args.struct_size = PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.local_hardware_id = local_device_id.value();
  RETURN_STATUS_IF_PJRT_ERROR(
      c_api_->PJRT_Client_LookupAddressableDevice(&args), c_api_);
  return GetCppDevice(args.addressable_device);
}

void PjRtCApiClient::UpdateGlobalProcessInfo(
    absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) {
  auto translate_state = [](tensorflow::CoordinatedTaskState state) {
    switch (state) {
      case tensorflow::CoordinatedTaskState::TASKSTATE_UNSPECIFIED:
        return PJRT_ProcessState_kUnspecified;
      case tensorflow::CoordinatedTaskState::TASKSTATE_UNINITIALIZED:
        return PJRT_ProcessState_kUninitialized;
      case tensorflow::CoordinatedTaskState::TASKSTATE_DISCONNECTED:
        return PJRT_ProcessState_kDisconnected;
      case tensorflow::CoordinatedTaskState::TASKSTATE_CONNECTED:
        return PJRT_ProcessState_kConnected;
      case tensorflow::CoordinatedTaskState::TASKSTATE_ERROR:
        return PJRT_ProcessState_kError;
      default:
        LOG(FATAL) << "Unexpected CoordinatedTaskState " << state;
        return PJRT_ProcessState_kUnspecified;
    }
  };

  std::vector<PJRT_ProcessInfo> process_infos;
  for (const tensorflow::CoordinatedTaskStateInfo& info : infos) {
    PJRT_ProcessInfo process_info;
    process_info.struct_size = PJRT_ProcessInfo_STRUCT_SIZE;
    process_info.task_id = info.task().task_id();
    process_info.incarnation_id = info.incarnation();
    process_info.state = translate_state(info.state());
    process_info.error_code = info.error_code();
    process_info.error_message = info.error_message().data();
    process_info.error_message_size = info.error_message().size();
    process_infos.push_back(std::move(process_info));
  }

  PJRT_Client_UpdateGlobalProcessInfo_Args args;
  args.struct_size = PJRT_Client_UpdateGlobalProcessInfo_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.process_infos = process_infos.data();
  args.num_process_infos = process_infos.size();
  absl::Status status = pjrt::PjrtErrorToStatus(
      c_api_->PJRT_Client_UpdateGlobalProcessInfo(&args), c_api_);
  if (!status.ok()) {
    LOG(FATAL) << "PJRT_Client_UpdateGlobalProcessInfo failed: " << status;
  }
}

absl::Span<PjRtMemorySpace* const> PjRtCApiClient::memory_spaces() const {
  return addressable_memory_spaces_;
}

// Initializes `PJRT_Client_Compile_Args`, which will be used to call
// API PJRT_Client_Compile().
static absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
InitializeArgsAndCompile(PjRtCApiClient* api_client, const PJRT_Api* c_api,
                         PJRT_Client* client, const CompileOptions& options,
                         const std::string& code, const std::string& format) {
  PJRT_Client_Compile_Args args;
  args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  PJRT_Profiler_Extension profiler_extension =
      pjrt::CreatePjrtProfilerExtension("PJRT_Client_Compile linkage");
  args.extension_start = &profiler_extension.base;
  args.client = client;
  TF_ASSIGN_OR_RETURN(const CompileOptionsProto options_proto,
                      options.ToProto());
  std::string options_str = options_proto.SerializeAsString();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  PJRT_Program program;
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.extension_start = nullptr;
  program.code = const_cast<char*>(code.c_str());
  program.code_size = code.size();
  program.format = format.c_str();
  program.format_size = format.size();
  args.program = &program;

  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Client_Compile(&args), c_api);
  std::unique_ptr<PjRtLoadedExecutable> ret =
      std::make_unique<PjRtCApiLoadedExecutable>(api_client, args.executable);
  return ret;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCApiClient::CompileAndLoad(const XlaComputation& computation,
                               CompileOptions options) {
  std::string module_str = computation.proto().SerializeAsString();
  std::string format(pjrt::kHloFormat);
  return InitializeArgsAndCompile(this, c_api_, c_client_.get(), options,
                                  module_str, format);
}

namespace {

std::string GetPluginStablehloVersionOrDefault(PjRtClient* client) {
  // If the plugin is not set, use the default.
  if (!client) {
    return xla::GetDefaultStablehloVersion();
  }

  // If the plugin doesn't have attributes, use the default.
  auto attributes = client->plugin_attributes();
  if (!attributes.has_value()) {
    return xla::GetDefaultStablehloVersion();
  }

  // If plugin doesn't report it StableHLO version, use the default.
  auto attr_map = attributes->attributes;
  auto version = attr_map.find("stablehlo_current_version");
  if (version == attr_map.end()) {
    return xla::GetDefaultStablehloVersion();
  }

  std::vector<int64_t> v = std::get<std::vector<int64_t>>(version->second);
  return absl::StrFormat("%d.%d.%d", v[0], v[1], v[2]);
}

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCApiClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) {
  if (!pjrt_c_api()) llvm::report_fatal_error("pjrt_c_api is null");

  std::string version_string = GetPluginStablehloVersionOrDefault(this);

  TF_ASSIGN_OR_RETURN(
      std::string serialized,
      xla::Serialize(module, version_string,
                     /*inplace=*/options.allow_in_place_mlir_modification));
  if (options.allow_in_place_mlir_modification) {
    // If we're allowed to modify the computation, free the functions in the
    // MLIR. We don't use them anymore, and this reduces peak memory.
    module.getBody()->clear();
  }
  std::string format(pjrt::kMlirFormat);
  return InitializeArgsAndCompile(this, c_api_, c_client_.get(), options,
                                  serialized, format);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCApiClient::LoadSerializedExecutable(absl::string_view serialized,
                                         std::optional<CompileOptions> options,
                                         const LoadOptions& load_options) {
  PJRT_Executable_DeserializeAndLoad_Args des_args;

  des_args.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
  des_args.extension_start = nullptr;
  des_args.client = c_client_.get();
  des_args.serialized_executable = serialized.data();
  des_args.serialized_executable_size = serialized.length();
  des_args.overridden_serialized_compile_options = nullptr;
  des_args.overridden_serialized_compile_options_size = 0;

  std::string options_str;
  if (options) {
    TF_ASSIGN_OR_RETURN(const CompileOptionsProto options_proto,
                        options->ToProto());
    options_str = options_proto.SerializeAsString();
    des_args.overridden_serialized_compile_options = options_str.c_str();
    des_args.overridden_serialized_compile_options_size = options_str.size();
  }

  const PJRT_Api* api = pjrt_c_api();

  RETURN_STATUS_IF_PJRT_ERROR(
      api->PJRT_Executable_DeserializeAndLoad(&des_args), api);
  PJRT_LoadedExecutable* c_exec = des_args.loaded_executable;
  CHECK(c_exec != nullptr);
  return std::unique_ptr<PjRtLoadedExecutable>(
      std::make_unique<PjRtCApiLoadedExecutable>(this, c_exec));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtCApiClient::CreateUninitializedBuffer(const Shape& shape,
                                          PjRtMemorySpace* memory_space) {
  if (pjrt_c_api()->pjrt_api_version.major_version == 0 &&
      pjrt_c_api()->pjrt_api_version.minor_version < 69) {
    return absl::UnimplementedError(
        "PJRT_Client_CreateUninitializedBuffer available in this version of "
        "the PjRT plugin");
  }

  PJRT_Client_CreateUninitializedBuffer_Args args;
  args.struct_size = PJRT_Client_CreateUninitializedBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.device = nullptr;

  args.shape_dims = shape.dimensions().data();
  args.shape_num_dims = shape.dimensions().size();
  args.shape_element_type = pjrt::ConvertToPjRtBufferType(shape.element_type());

  pjrt::BufferMemoryLayoutData c_layout_data;
  if (shape.has_layout()) {
    TF_ASSIGN_OR_RETURN(c_layout_data,
                        pjrt::ConvertToBufferMemoryLayoutData(shape.layout()));
    args.shape_layout = &c_layout_data.c_layout;
  } else {
    args.shape_layout = nullptr;
  }

  args.memory =
      tensorflow::down_cast<PjRtCApiMemorySpace*>(memory_space)->c_memory();

  RETURN_STATUS_IF_PJRT_ERROR(
      c_api_->PJRT_Client_CreateUninitializedBuffer(&args), c_api_);

  auto buffer = std::unique_ptr<PjRtBuffer>(
      std::make_unique<PjRtCApiBuffer>(this, args.buffer));

  return buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> PjRtCApiClient::CreateErrorBuffer(
    absl::Status error, const Shape& shape, PjRtMemorySpace* memory) {
  if (c_api_->pjrt_api_version.major_version == 0 &&
      c_api_->pjrt_api_version.minor_version < 82) {
    return absl::UnimplementedError(
        "PJRT_Client_CreateErrorBuffer requires PJRT C API version 0.82 or "
        "higher.");
  }

  PJRT_Client_CreateErrorBuffer_Args args;
  args.struct_size = PJRT_Client_CreateErrorBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();

  args.error_code = pjrt::StatusCodeToPjrtErrorCode(error.code());
  args.error_message = error.message().data();
  args.error_message_size = error.message().size();

  args.shape_dims = shape.dimensions().data();
  args.shape_num_dims = shape.dimensions().size();
  args.shape_element_type = pjrt::ConvertToPjRtBufferType(shape.element_type());

  pjrt::BufferMemoryLayoutData c_layout_data;
  if (shape.has_layout()) {
    TF_ASSIGN_OR_RETURN(c_layout_data,
                        pjrt::ConvertToBufferMemoryLayoutData(shape.layout()));
    args.shape_layout = &c_layout_data.c_layout;
  } else {
    args.shape_layout = nullptr;
  }

  args.memory = tensorflow::down_cast<PjRtCApiMemorySpace*>(memory)->c_memory();

  RETURN_STATUS_IF_PJRT_ERROR(c_api_->PJRT_Client_CreateErrorBuffer(&args),
                              c_api_);

  auto buffer = std::unique_ptr<PjRtBuffer>(
      std::make_unique<PjRtCApiBuffer>(this, args.buffer));

  return buffer;
}

absl::Status FulfillAliasBuffer(
    const PJRT_Api* pjrt_c_api, absl::StatusOr<PjRtBuffer*> real_buffer_or,
    PJRT_FulfillAliasBufferCallback* fulfill_alias_buffer_cb) {
  if (pjrt_c_api->pjrt_api_version.major_version == 0 &&
      pjrt_c_api->pjrt_api_version.minor_version < 76) {
    return absl::UnimplementedError(
        "PJRT_Client_FulfillAliasBuffer requires PJRT C API version 0.76 or "
        "higher.");
  }
  PJRT_Client_FulfillAliasBuffer_Args args;
  args.struct_size = PJRT_Client_FulfillAliasBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.fulfill_alias_buffer_cb = fulfill_alias_buffer_cb;

  if (real_buffer_or.ok()) {
    // We have a real buffer, make sure it's a PjRtCApiBuffer and pass it to the
    // C API.
    PjRtCApiBuffer* c_buffer =
        tensorflow::down_cast<PjRtCApiBuffer*>(real_buffer_or.value());
    args.buffer = c_buffer->c_buffer();
    args.status_code = PJRT_Error_Code_OK;
    args.error_message = nullptr;
    args.error_message_size = 0;
  } else {
    // If the real buffer is an error, then we need to fulfill that alias
    // buffer with a nullptr.
    args.buffer = nullptr;
    args.status_code =
        pjrt::StatusCodeToPjrtErrorCode(real_buffer_or.status().code());
    args.error_message = real_buffer_or.status().message().data();
    args.error_message_size = real_buffer_or.status().message().size();
  }

  PJRT_Error* error = pjrt_c_api->PJRT_Client_FulfillAliasBuffer(&args);
  if (error != nullptr) {
    return pjrt::PjrtErrorToStatus(error, pjrt_c_api);
  }
  return absl::OkStatus();
}

absl::StatusOr<
    std::pair<std::unique_ptr<PjRtBuffer>, PjRtFulfillAliasBufferCallback>>
PjRtCApiClient::CreateAliasBuffer(const Shape& shape,
                                  PjRtMemorySpace* memory_space) {
  if (pjrt_c_api()->pjrt_api_version.major_version == 0 &&
      pjrt_c_api()->pjrt_api_version.minor_version < 76) {
    return absl::UnimplementedError(
        "PJRT_Client_CreateBufferAlias requires PJRT C API version 0.76 or "
        "higher.");
  }

  PJRT_Client_CreateAliasBuffer_Args args;
  args.struct_size = PJRT_Client_CreateAliasBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();

  args.shape_dims = shape.dimensions().data();
  args.shape_num_dims = shape.dimensions().size();
  args.shape_element_type = pjrt::ConvertToPjRtBufferType(shape.element_type());

  pjrt::BufferMemoryLayoutData c_layout_data;
  if (shape.has_layout()) {
    TF_ASSIGN_OR_RETURN(c_layout_data,
                        pjrt::ConvertToBufferMemoryLayoutData(shape.layout()));
    args.shape_layout = &c_layout_data.c_layout;
  } else {
    args.shape_layout = nullptr;
  }

  args.memory =
      tensorflow::down_cast<PjRtCApiMemorySpace*>(memory_space)->c_memory();
  args.alias_buffer = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(c_api_->PJRT_Client_CreateAliasBuffer(&args),
                              c_api_);

  std::unique_ptr<PjRtBuffer> alias_buffer(
      std::make_unique<PjRtCApiBuffer>(this, args.alias_buffer));

  PjRtFulfillAliasBufferCallback fulfill_alias_buffer_cb =
      [pjrt_c_api = pjrt_c_api(),
       fulfill_alias_buffer_cb = args.fulfill_alias_buffer_cb](
          absl::StatusOr<PjRtBuffer*> real_buffer) -> absl::Status {
    return FulfillAliasBuffer(pjrt_c_api, real_buffer, fulfill_alias_buffer_cb);
  };

  return std::make_pair(std::move(alias_buffer),
                        std::move(fulfill_alias_buffer_cb));
}

absl::StatusOr<const PjRtTopologyDescription*>
PjRtCApiClient::GetTopologyDescription() const {
  if (!topo_desc_.ok()) {
    return topo_desc_.status();
  }
  return &(*topo_desc_);
}

absl::StatusOr<PjRtClient::HostAllocator*> PjRtCApiClient::GetHostAllocator()
    const {
  if (!host_allocator_.ok()) {
    return host_allocator_.status();
  }
  return host_allocator_->get();
}

absl::StatusOr<std::uintptr_t> PjRtCApiClient::UnsafeBufferPointer(
    PjRtBuffer* buffer) {
  // Validate that the buffer's client matches the function call's client, since
  // that could be a common error.
  // Not doing input nullptr validation since such cases should be rare, and
  // crashes should bubble up the call stack to higher layers. See b/248334153
  // for the considerations that went into this.
  if (buffer->client() != this) {
    return InvalidArgument(
        "buffer passed to PjRtCApiClient::UnsafeBufferPointer() is from a "
        "different client than that of the function call. Buffer's client "
        "platform: '%s', function call's client platform: '%s'.",
        buffer->client()->platform_name(), this->platform_name());
  }

  PJRT_Buffer_UnsafePointer_Args args;
  args.struct_size = PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer =
      tensorflow::down_cast<const PjRtCApiBuffer*>(buffer)->c_buffer();

  RETURN_STATUS_IF_PJRT_ERROR(c_api_->PJRT_Buffer_UnsafePointer(&args), c_api_);

  return args.buffer_pointer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtCApiClient::BufferFromHostBufferInternalImpl(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    std::variant<PjRtDevice*, PjRtMemorySpace*> device_or_memory,
    const Layout* device_layout) {
  if (host_buffer_semantics != HostBufferSemantics::kImmutableOnlyDuringCall &&
      host_buffer_semantics != HostBufferSemantics::kImmutableZeroCopy &&
      host_buffer_semantics !=
          HostBufferSemantics::kImmutableUntilTransferCompletes) {
    return Unimplemented(
        "PJRT C API does not support HostBufferSemantics other than "
        "HostBufferSemantics::kImmutableOnlyDuringCall, "
        "HostBufferSemantics::kImmutableZeroCopy and "
        "HostBufferSemantics::kImmutableUntilTransferCompletes.");
  }

  PJRT_Client_BufferFromHostBuffer_Args args;
  args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.data = data;
  args.type = ::pjrt::ConvertToPjRtBufferType(type);

  args.dims = dims.data();
  args.num_dims = dims.size();
  if (byte_strides.has_value()) {
    args.byte_strides = byte_strides.value().data();
    args.num_byte_strides = byte_strides.value().size();
  } else {
    args.byte_strides = nullptr;
    args.num_byte_strides = 0;
  }
  pjrt::BufferMemoryLayoutData c_layout_data;
  if (device_layout != nullptr) {
    TF_ASSIGN_OR_RETURN(c_layout_data,
                        pjrt::ConvertToBufferMemoryLayoutData(*device_layout));
    args.device_layout = &c_layout_data.c_layout;
  } else {
    args.device_layout = nullptr;
  }

  args.host_buffer_semantics =
      ::pjrt::ConvertToPjRtHostBufferSemantics(host_buffer_semantics);
  if (std::holds_alternative<PjRtDevice*>(device_or_memory)) {
    args.device = tensorflow::down_cast<PjRtCApiDevice*>(
                      std::get<PjRtDevice*>(device_or_memory))
                      ->c_device();
    args.memory = nullptr;
  } else {
    CHECK(std::holds_alternative<PjRtMemorySpace*>(device_or_memory));
    args.device = nullptr;
    args.memory = tensorflow::down_cast<PjRtCApiMemorySpace*>(
                      std::get<PjRtMemorySpace*>(device_or_memory))
                      ->c_memory();
  }

  RETURN_STATUS_IF_PJRT_ERROR(c_api_->PJRT_Client_BufferFromHostBuffer(&args),
                              c_api_);

  auto buffer = std::unique_ptr<PjRtBuffer>(
      std::make_unique<PjRtCApiBuffer>(this, args.buffer));

  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event(
      args.done_with_host_buffer, ::pjrt::MakeEventDeleter(c_api_));

  RETURN_STATUS_IF_PJRT_ERROR(
      pjrt::InvokePjRtEventWhenReady(c_api_, event.get(),
                                     std::move(on_done_with_host_buffer)),
      c_api_);

  return buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtCApiClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  return BufferFromHostBufferInternalImpl(
      data, type, dims, byte_strides, host_buffer_semantics,
      std::move(on_done_with_host_buffer), memory_space, device_layout);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtCApiClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                      PjRtMemorySpace* memory_space,
                                      const Layout* device_layout) {
  if (literal.shape().is_dynamic()) {
    return Unimplemented(
        "PJRT C API does not support dynamic shapes for "
        "BufferFromHostLiteral.");
  }
  absl::InlinedVector<int64_t, 4> strides(literal.shape().dimensions().size());
  TF_RETURN_IF_ERROR(
      ShapeUtil::UnpackedByteStrides(literal.shape(), absl::MakeSpan(strides)));
  return BufferFromHostBufferInternalImpl(
      literal.untyped_data(), literal.shape().element_type(),
      literal.shape().dimensions(), strides,
      HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/nullptr, memory_space, device_layout);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtCApiClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream) {
  PJRT_Client_CreateViewOfDeviceBuffer_Args args;
  args.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.device_buffer_ptr = device_ptr;
  args.dims = shape.dimensions().data();
  args.num_dims = shape.dimensions().size();
  args.element_type = pjrt::ConvertToPjRtBufferType(shape.element_type());
  pjrt::BufferMemoryLayoutData c_layout_data;
  if (shape.has_layout()) {
    TF_ASSIGN_OR_RETURN(c_layout_data,
                        pjrt::ConvertToBufferMemoryLayoutData(shape.layout()));
    args.layout = &(c_layout_data.c_layout);
  } else {
    args.layout = nullptr;
  }
  if (on_delete_callback != nullptr) {
    args.on_delete_callback_arg =
        new std::function(std::move(on_delete_callback));
    args.on_delete_callback = [](void* device_buffer_ptr, void* user_arg) {
      auto* c_func = reinterpret_cast<std::function<void()>*>(user_arg);
      (*c_func)();
      delete c_func;
    };
  } else {
    args.on_delete_callback = nullptr;
    args.on_delete_callback_arg = nullptr;
  }
  args.device = nullptr;
  args.memory =
      tensorflow::down_cast<PjRtCApiMemorySpace*>(memory_space)->c_memory();
  if (stream.has_value()) {
    args.stream = *stream;
  } else {
    args.stream = reinterpret_cast<intptr_t>(nullptr);
  }
  const PJRT_Api* c_api = pjrt_c_api();

  RETURN_STATUS_IF_PJRT_ERROR(
      c_api->PJRT_Client_CreateViewOfDeviceBuffer(&args), c_api);

  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<PjRtCApiBuffer>(this, args.buffer));
}

absl::StatusOr<Layout> PjRtCApiClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  const PJRT_Api* c_api = pjrt_c_api();
  PJRT_Layouts_Extension* extension = FindExtension<PJRT_Layouts_Extension>(
      PJRT_Extension_Type::PJRT_Extension_Type_Layouts);
  if (extension == nullptr) {
    return LayoutUtil::MakeDescendingLayout(dims.size());
  }
  PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args args;
  args.struct_size = PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.type = pjrt::ConvertToPjRtBufferType(element_type);
  args.dims = dims.data();
  args.num_dims = dims.size();
  RETURN_STATUS_IF_PJRT_ERROR(
      extension->PJRT_Layouts_PJRT_Client_GetDefaultLayout(&args), c_api);

  // Clean up `PJRT_Layouts_MemoryLayout`.
  std::unique_ptr<PJRT_Layouts_MemoryLayout,
                  pjrt::PJRT_Layouts_MemoryLayoutDeleter>
      layout_destroyer(args.layout, pjrt::MakeMemoryLayoutDeleter(c_api));

  // TODO(yueshengys): once b/338478940 is fixed, we can get rid of the
  // serialization here and wrap the `args.layout` into a subclass of
  // `PjRtLayout`.
  PJRT_Layouts_MemoryLayout_Serialize_Args serialize_args;
  serialize_args.struct_size =
      PJRT_Layouts_MemoryLayout_Serialize_Args_STRUCT_SIZE;
  serialize_args.extension_start = nullptr;
  serialize_args.layout = args.layout;
  RETURN_STATUS_IF_PJRT_ERROR(
      extension->PJRT_Layouts_MemoryLayout_Serialize(&serialize_args), c_api);

  // Clean up `PJRT_Layouts_SerializedLayout`.
  absl::Cleanup cleanup = [&serialize_args] {
    serialize_args.serialized_layout_deleter(serialize_args.serialized_layout);
  };

  std::string serialized_layout(serialize_args.serialized_bytes,
                                serialize_args.serialized_bytes_size);
  TF_ASSIGN_OR_RETURN(std::shared_ptr<const PjRtLayout> pjrt_layout,
                      PjRtLayout::Deserialize(serialized_layout));

  return pjrt_layout->xla_layout();
}

absl::Status PjRtCApiClient::DmaMap(void* data, size_t size) {
  const PJRT_Api* c_api = pjrt_c_api();
  PJRT_Client_DmaMap_Args args;
  args.struct_size = PJRT_Client_DmaMap_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.data = data;
  args.size = size;
  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Client_DmaMap(&args), c_api);
  return absl::OkStatus();
}

absl::Status PjRtCApiClient::DmaUnmap(void* data) {
  const PJRT_Api* c_api = pjrt_c_api();
  PJRT_Client_DmaUnmap_Args args;
  args.struct_size = PJRT_Client_DmaUnmap_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.data = data;
  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Client_DmaUnmap(&args), c_api);
  return absl::OkStatus();
}

// Helper struct and method used to serialize shapes past the C API boundary.
struct ShapesInfo {
  std::vector<size_t> shape_num_dims;
  std::vector<std::optional<pjrt::BufferMemoryLayoutData>> layout_list;
  std::vector<const int64_t*> num_dims;
  std::vector<PJRT_Buffer_Type> element_type_list;
};

ShapesInfo MakeShapesInfo(absl::Span<const Shape> shapes) {
  std::vector<size_t> shape_num_dims;
  shape_num_dims.reserve(shapes.size());
  std::vector<std::optional<pjrt::BufferMemoryLayoutData>> layout_list;
  layout_list.reserve(shapes.size());
  std::vector<const int64_t*> num_dims;
  num_dims.reserve(shapes.size());
  std::vector<PJRT_Buffer_Type> element_type_list;
  element_type_list.reserve(shapes.size());

  for (int i = 0; i < shapes.size(); ++i) {
    shape_num_dims.push_back(shapes[i].dimensions().size());

    num_dims.push_back(shapes[i].dimensions().data());
    element_type_list.push_back(
        pjrt::ConvertToPjRtBufferType(shapes[i].element_type()));

    if (shapes[i].has_layout()) {
      auto& layout = shapes[i].layout();
      absl::StatusOr<pjrt::BufferMemoryLayoutData> c_layout_data =
          pjrt::ConvertToBufferMemoryLayoutData(layout);
      if (c_layout_data.ok()) {
        layout_list.push_back(std::optional<pjrt::BufferMemoryLayoutData>(
            std::move(*c_layout_data)));
      } else {
        layout_list.push_back({});
      }
    } else {
      layout_list.push_back({});
    }
  }

  return ShapesInfo{
      /*shape_num_dims=*/std::move(shape_num_dims),
      /*layout_list=*/std::move(layout_list),
      /*num_dims=*/std::move(num_dims),
      /*element_type_list=*/std::move(element_type_list),
  };
}

// Helper method to convert a list of PJRT_Buffer* to a list of PjRtBuffer*.
std::vector<std::unique_ptr<PjRtBuffer>> MakePjRtBuffersFromPJRT_Buffers(
    PjRtCApiClient* client, PJRT_Buffer** c_buffers, size_t num_buffers) {
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.reserve(num_buffers);
  for (int i = 0; i < num_buffers; ++i) {
    buffers.emplace_back(
        std::make_unique<PjRtCApiBuffer>(client, c_buffers[i]));
  }
  return buffers;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiClient::MakeCrossHostReceiveBuffers(
    absl::Span<const Shape> shapes, PjRtDevice* device,
    PjRtCrossHostRecvNotifier notifier) {
  const PJRT_Api* c_api = pjrt_c_api();
  PJRT_CrossHostTransfers_Extension* extension =
      FindExtension<PJRT_CrossHostTransfers_Extension>(
          PJRT_Extension_Type::PJRT_Extension_Type_CrossHostTransfers);
  if (extension == nullptr) {
    return absl::UnimplementedError(
        "MakeCrossHostReceiveBuffers is not implemented in this PJRT plugin.");
  }
  PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args args;
  args.struct_size =
      PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();

  ShapesInfo shapes_info = MakeShapesInfo(shapes);
  args.num_shapes = shapes.size();
  args.shape_num_dims = shapes_info.shape_num_dims.data();
  args.num_dims = shapes_info.num_dims.data();
  args.element_types = shapes_info.element_type_list.data();

  std::vector<PJRT_Buffer_MemoryLayout*> layout_list;
  for (int i = 0; i < shapes_info.layout_list.size(); i++) {
    if (shapes_info.layout_list[i].has_value()) {
      layout_list.push_back(&shapes_info.layout_list[i]->c_layout);
    } else {
      layout_list.push_back(nullptr);
    }
  }
  args.layouts = layout_list.data();

  args.notifier = pjrt::CppCrossHostRecvNotifierToC(c_api, std::move(notifier));
  args.device = tensorflow::down_cast<PjRtCApiDevice*>(device)->c_device();

  std::vector<PJRT_Buffer*> temp_buffers(shapes.size());
  args.buffers = temp_buffers.data();
  RETURN_STATUS_IF_PJRT_ERROR(
      extension->PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers(&args),
      c_api);

  return MakePjRtBuffersFromPJRT_Buffers(this, args.buffers,
                                         temp_buffers.size());
}

absl::StatusOr<std::vector<Future<>>> PjRtCApiClient::CrossHostSendBuffers(
    absl::Span<PjRtBuffer* const> buffers,
    absl::Span<const PjRtGlobalDeviceId> dst_global_device_ids,
    std::vector<CrossHostTransferKey> transfer_keys) {
  // Get C API extension.
  const PJRT_Api* c_api = pjrt_c_api();
  PJRT_CrossHostTransfers_Extension* extension =
      FindExtension<PJRT_CrossHostTransfers_Extension>(
          PJRT_Extension_Type::PJRT_Extension_Type_CrossHostTransfers);
  if (extension == nullptr) {
    return absl::UnimplementedError(
        "CrossHostSendBuffers is not implemented in this PJRT plugin.");
  }

  // Form inputs.
  PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args args;
  args.struct_size =
      PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();
  args.num_buffers = buffers.size();

  std::vector<PJRT_Buffer*> c_buffers;
  c_buffers.reserve(buffers.size());
  for (PjRtBuffer* buffer : buffers) {
    c_buffers.push_back(
        tensorflow::down_cast<const PjRtCApiBuffer*>(buffer)->c_buffer());
  }

  args.buffers = c_buffers.data();
  args.dst_global_device_ids = dst_global_device_ids.data();
  args.transfer_keys = transfer_keys.data();

  auto send_events = std::vector<PJRT_Event*>(args.num_buffers);
  args.send_events = send_events.data();

  RETURN_STATUS_IF_PJRT_ERROR(
      extension->PJRT_Transfers_PJRT_Client_CrossHostSendBuffers(&args), c_api);

  std::vector<Future<>> send_futures;
  send_futures.reserve(args.num_buffers);
  for (int i = 0; i < args.num_buffers; ++i) {
    send_futures.push_back(
        pjrt::ConvertCEventToCppFuture(args.send_events[i], c_api));
  }

  return send_futures;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiClient::CrossHostReceiveBuffers(
    xla::PjRtDevice* device, absl::Span<const xla::Shape> shapes,
    absl::Span<const PjRtGlobalDeviceId> src_global_device_ids,
    std::vector<CrossHostTransferKey> transfer_keys) {
  // Get C API extension.
  const PJRT_Api* c_api = pjrt_c_api();
  PJRT_CrossHostTransfers_Extension* extension =
      FindExtension<PJRT_CrossHostTransfers_Extension>(
          PJRT_Extension_Type::PJRT_Extension_Type_CrossHostTransfers);
  if (extension == nullptr) {
    return absl::UnimplementedError(
        "CrossHostReceiveBuffers is not implemented in this PJRT plugin.");
  }

  // Form inputs.
  PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args args;
  args.struct_size =
      PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();

  ShapesInfo shapes_info = MakeShapesInfo(shapes);
  args.num_shapes = shapes.size();
  args.shape_num_dims = shapes_info.shape_num_dims.data();
  args.num_dims = shapes_info.num_dims.data();
  args.element_types = shapes_info.element_type_list.data();

  std::vector<PJRT_Buffer_MemoryLayout*> layout_list;
  for (int i = 0; i < shapes_info.layout_list.size(); i++) {
    if (shapes_info.layout_list[i].has_value()) {
      layout_list.push_back(&shapes_info.layout_list[i]->c_layout);
    } else {
      layout_list.push_back(nullptr);
    }
  }
  args.layouts = layout_list.data();

  args.device = tensorflow::down_cast<PjRtCApiDevice*>(device)->c_device();
  args.src_global_device_ids = src_global_device_ids.data();
  args.transfer_keys = transfer_keys.data();

  std::vector<PJRT_Buffer*> temp_buffers(shapes.size());
  args.buffers = temp_buffers.data();

  RETURN_STATUS_IF_PJRT_ERROR(
      extension->PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers(&args),
      c_api);

  return MakePjRtBuffersFromPJRT_Buffers(this, args.buffers,
                                         temp_buffers.size());
}

class PjRtCApiAsyncHostToDeviceTransferManager
    : public PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  PjRtCApiAsyncHostToDeviceTransferManager(
      PjRtCApiClient* client,
      PJRT_AsyncHostToDeviceTransferManager* c_transfer_manager)
      : c_client_(client),
        c_transfer_manager_(c_transfer_manager,
                            ::pjrt::MakeAsyncHostToDeviceTransferManagerDeleter(
                                client->pjrt_c_api())) {}

  size_t buffer_count() const override {
    PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args args;
    args.struct_size =
        PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.transfer_manager = c_transfer_manager_.get();
    const PJRT_Api* api = c_client_->pjrt_c_api();
    pjrt::LogFatalIfPjrtError(
        api->PJRT_AsyncHostToDeviceTransferManager_BufferCount(&args), api);
    return args.buffer_count;
  }

  PjRtDevice* device() const override {
    PJRT_AsyncHostToDeviceTransferManager_Device_Args args;
    args.struct_size =
        PJRT_AsyncHostToDeviceTransferManager_Device_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.transfer_manager = c_transfer_manager_.get();
    const PJRT_Api* api = c_client_->pjrt_c_api();
    pjrt::LogFatalIfPjrtError(
        api->PJRT_AsyncHostToDeviceTransferManager_Device(&args), api);
    return c_client_->GetCppDevice(args.device_out);
  }

  std::unique_ptr<PjRtBuffer> RetrieveBuffer(int buffer_index) override {
    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args args;
    args.struct_size =
        PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.transfer_manager = c_transfer_manager_.get();
    args.buffer_index = buffer_index;
    const PJRT_Api* api = c_client_->pjrt_c_api();
    pjrt::LogFatalIfPjrtError(
        api->PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(&args), api);
    return std::make_unique<PjRtCApiBuffer>(c_client_, args.buffer_out);
  }

  absl::Status TransferLiteralToBuffer(
      int buffer_index, const LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override {
    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args args;
    args.struct_size =
        PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.transfer_manager = c_transfer_manager_.get();
    args.buffer_index = buffer_index;

    const xla::Shape& shape = literal.shape();
    args.shape_dims = shape.dimensions().data();
    args.shape_num_dims = shape.dimensions().size();
    args.shape_element_type =
        pjrt::ConvertToPjRtBufferType(shape.element_type());

    pjrt::BufferMemoryLayoutData c_layout_data;
    if (shape.has_layout()) {
      TF_ASSIGN_OR_RETURN(
          c_layout_data, pjrt::ConvertToBufferMemoryLayoutData(shape.layout()));
      args.shape_layout = &c_layout_data.c_layout;
    } else {
      args.shape_layout = nullptr;
    }

    args.data = literal.untyped_data();
    const PJRT_Api* api = c_client_->pjrt_c_api();
    RETURN_STATUS_IF_PJRT_ERROR(
        api->PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(&args), api);
    std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event(
        args.done_with_h2d_transfer, ::pjrt::MakeEventDeleter(api));
    RETURN_STATUS_IF_PJRT_ERROR(
        pjrt::InvokePjRtEventWhenReady(api, event.get(), std::move(on_done)),
        api);
    return absl::OkStatus();
  }

  size_t buffer_size(int buffer_index) const override {
    PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args args;
    args.struct_size =
        PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.transfer_manager = c_transfer_manager_.get();
    args.buffer_index = buffer_index;
    const PJRT_Api* api = c_client_->pjrt_c_api();
    pjrt::LogFatalIfPjrtError(
        api->PJRT_AsyncHostToDeviceTransferManager_BufferSize(&args), api);
    return args.buffer_size;
  }

  absl::Status TransferRawDataToBuffer(
      int buffer_index, absl::string_view data,
      absl::AnyInvocable<void() &&> on_done) override {
    return TransferRawDataToSubBuffer(buffer_index, data.data(), 0, data.size(),
                                      /*is_last_transfer=*/true,
                                      std::move(on_done));
  }

  absl::Status TransferRawDataToSubBuffer(
      int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) override {
    PJRT_AsyncHostToDeviceTransferManager_TransferData_Args args;
    args.struct_size =
        PJRT_AsyncHostToDeviceTransferManager_TransferData_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.transfer_manager = c_transfer_manager_.get();
    args.buffer_index = buffer_index;
    args.data = data;
    args.offset = offset;
    args.transfer_size = transfer_size;
    args.is_last_transfer = is_last_transfer;
    const PJRT_Api* api = c_client_->pjrt_c_api();
    RETURN_STATUS_IF_PJRT_ERROR(
        api->PJRT_AsyncHostToDeviceTransferManager_TransferData(&args), api);
    std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event(
        args.done_with_h2d_transfer, ::pjrt::MakeEventDeleter(api));
    RETURN_STATUS_IF_PJRT_ERROR(
        pjrt::InvokePjRtEventWhenReady(api, event.get(), std::move(on_done)),
        api);
    return absl::OkStatus();
  }

  void SetBufferError(int buffer_index, absl::Status error) override {
    PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args args;
    args.struct_size =
        PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.transfer_manager = c_transfer_manager_.get();
    args.buffer_index = buffer_index;
    args.error_code = pjrt::StatusCodeToPjrtErrorCode(error.code());
    args.error_message = error.message().data();
    args.error_message_size = error.message().size();
    const PJRT_Api* api = c_client_->pjrt_c_api();
    pjrt::LogFatalIfPjrtError(
        api->PJRT_AsyncHostToDeviceTransferManager_SetBufferError(&args), api);
  }

  using TransferMetadata = absl::flat_hash_map<std::string, std::string>;
  void AddTransferMetadata(const TransferMetadata& metadata) override {
    PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args args;
    args.struct_size =
        PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.transfer_manager = c_transfer_manager_.get();
    absl::flat_hash_map<std::string, xla::PjRtValueType> pjrt_metadata;
    for (const auto& [key, value] : metadata) {
      pjrt_metadata[key] = PjRtValueType(value);
    };
    absl::StatusOr<std::vector<PJRT_NamedValue>> result =
        pjrt::ConvertToPjRtNamedValueList(pjrt_metadata);
    CHECK_OK(result.status());
    std::vector<PJRT_NamedValue> c_metadata = result.value();
    args.transfer_metadata = c_metadata.data();
    args.num_metadata = c_metadata.size();
    const PJRT_Api* api = c_client_->pjrt_c_api();
    pjrt::LogFatalIfPjrtError(
        api->PJRT_AsyncHostToDeviceTransferManager_AddMetadata(&args), api);
  }

 private:
  PjRtCApiClient* c_client_;
  std::unique_ptr<PJRT_AsyncHostToDeviceTransferManager,
                  ::pjrt::PJRT_AsyncHostToDeviceTransferManagerDeleter>
      c_transfer_manager_;
};

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
PjRtCApiClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  const PJRT_Api* c_api = pjrt_c_api();
  PJRT_Client_CreateBuffersForAsyncHostToDevice_Args args;
  args.struct_size =
      PJRT_Client_CreateBuffersForAsyncHostToDevice_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client_.get();

  args.num_shape_specs = shape_specs.size();
  absl::InlinedVector<PJRT_ShapeSpec, 4> c_shape_specs;
  c_shape_specs.reserve(shape_specs.size());
  for (const ShapeSpec& shape_spec : shape_specs) {
    c_shape_specs.push_back(pjrt::ConvertToPjRtShapeSpec(shape_spec));
  }
  args.shape_specs = c_shape_specs.data();

  absl::InlinedVector<pjrt::BufferMemoryLayoutData, 4> layout_data_list;
  absl::InlinedVector<PJRT_Buffer_MemoryLayout*, 4> device_layout_list;
  if (device_layouts.has_value()) {
    args.num_device_layouts = device_layouts->size();
    device_layout_list.reserve(device_layouts->size());
    layout_data_list.reserve(device_layouts->size());
    for (int i = 0; i < device_layouts->size(); ++i) {
      if (device_layouts.has_value() && (*device_layouts)[i].has_value()) {
        const Layout& layout = (*device_layouts)[i].value();
        TF_ASSIGN_OR_RETURN(pjrt::BufferMemoryLayoutData c_layout_data,
                            pjrt::ConvertToBufferMemoryLayoutData(layout));
        layout_data_list.push_back(std::move(c_layout_data));
        device_layout_list.emplace_back(&(layout_data_list.back().c_layout));
      } else {
        device_layout_list.emplace_back(nullptr);
      }
    }
    args.device_layouts = device_layout_list.data();
  } else {
    args.num_device_layouts = 0;
    args.device_layouts = nullptr;
  }
  args.memory =
      tensorflow::down_cast<PjRtCApiMemorySpace*>(memory_space)->c_memory();

  RETURN_STATUS_IF_PJRT_ERROR(
      c_api->PJRT_Client_CreateBuffersForAsyncHostToDevice(&args), c_api);
  return std::make_unique<PjRtCApiAsyncHostToDeviceTransferManager>(
      this, args.transfer_manager);
}

const PJRT_Api* PjRtCApiClient::pjrt_c_api() const { return c_api_; }

// --------------------------------- Device Descriptions -----------------------

PjRtCApiDeviceDescription::PjRtCApiDeviceDescription(
    const PJRT_Api* c_api, PJRT_DeviceDescription* device_description)
    : c_api_(c_api), device_description_(device_description) {
  InitAttributes();
}

int PjRtCApiDeviceDescription::id() const {
  PJRT_DeviceDescription_Id_Args args;
  args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device_description = device_description_;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_DeviceDescription_Id(&args), c_api_);
  return args.id;
}

int PjRtCApiDeviceDescription::process_index() const {
  PJRT_DeviceDescription_ProcessIndex_Args args;
  args.struct_size = PJRT_DeviceDescription_ProcessIndex_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device_description = device_description_;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_DeviceDescription_ProcessIndex(&args),
                            c_api_);
  return args.process_index;
}

void PjRtCApiDeviceDescription::InitAttributes() {
  attributes_ = {};
  PJRT_DeviceDescription_Attributes_Args args;
  args.struct_size = PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device_description = device_description_;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_DeviceDescription_Attributes(&args),
                            c_api_);

  for (int i = 0; i < args.num_attributes; ++i) {
    const auto& attribute = args.attributes[i];
    std::string attribute_name(attribute.name, attribute.name_size);
    switch (attribute.type) {
      case PJRT_NamedValue_Type::PJRT_NamedValue_kString: {
        std::string string_value(attribute.string_value, attribute.value_size);
        attributes_[attribute_name] = PjRtDeviceAttribute(string_value);
        break;
      }
      case PJRT_NamedValue_Type::PJRT_NamedValue_kInt64: {
        attributes_[attribute_name] =
            PjRtDeviceAttribute(attribute.int64_value);
        break;
      }
      case PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List: {
        const int64_t* array_ptr(attribute.int64_array_value);
        std::vector<int64_t> int64_array(array_ptr,
                                         array_ptr + attribute.value_size);
        attributes_[attribute_name] = PjRtDeviceAttribute(int64_array);
        break;
      }
      // Do not allow other types (such as
      // PJRT_NamedValue::PJRT_NamedValue_kFloat) since device attributes
      // currently should not return other types. Also C API client currently
      // does not support forward compatibility (such as if the underlying
      // PJRT library is a newer version that returns types not supported by
      // this client). Failing here to prevent undefined behavior.
      default: {
        LOG(FATAL) << "PJRT_DeviceDescription_Attributes() returned attribute '"
                   << attribute_name << "' with unsupported type "
                   << attribute.type
                   << " to PjRtCApiDeviceDescription::InitAttributes()";
        break;
      }
    }
  }
}

const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
PjRtCApiDeviceDescription::Attributes() const {
  return attributes_;
}

absl::string_view PjRtCApiDeviceDescription::device_kind() const {
  PJRT_DeviceDescription_Kind_Args args;
  args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device_description = device_description_;

  pjrt::LogFatalIfPjrtError(c_api_->PJRT_DeviceDescription_Kind(&args), c_api_);

  absl::string_view device_kind(args.device_kind, args.device_kind_size);
  return device_kind;
}

absl::string_view PjRtCApiDeviceDescription::DebugString() const {
  PJRT_DeviceDescription_DebugString_Args args;
  args.struct_size = PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device_description = device_description_;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_DeviceDescription_DebugString(&args),
                            c_api_);
  absl::string_view debug_string(args.debug_string, args.debug_string_size);
  return debug_string;
}

absl::string_view PjRtCApiDeviceDescription::ToString() const {
  PJRT_DeviceDescription_ToString_Args args;
  args.struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device_description = device_description_;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_DeviceDescription_ToString(&args),
                            c_api_);
  absl::string_view to_string(args.to_string, args.to_string_size);
  return to_string;
}

void PjRtCApiDeviceDescription::InitMemoryDescriptions() const {
  const PJRT_MemoryDescriptions_Extension* extension =
      pjrt::FindExtension<PJRT_MemoryDescriptions_Extension>(
          c_api_, PJRT_Extension_Type::PJRT_Extension_Type_MemoryDescriptions);
  if (!extension) return;

  if (memory_space_description_pointers_.empty()) {
    memory_space_descriptions_ = pjrt::GetMemorySpaceDescriptions(
        device_description_, c_api_, &default_memory_space_description_);
    for (int i = 0; i < memory_space_descriptions_.size(); i++) {
      memory_space_description_pointers_.push_back(
          &memory_space_descriptions_[i]);
    }
  }
}

absl::Span<const PjRtMemorySpaceDescription* const>
PjRtCApiDeviceDescription::memory_spaces() const {
  if (memory_space_description_pointers_.empty()) {
    InitMemoryDescriptions();
  }
  return memory_space_description_pointers_;
}

absl::StatusOr<const PjRtMemorySpaceDescription*>
PjRtCApiDeviceDescription::default_memory_space() const {
  if (memory_space_description_pointers_.empty()) {
    InitMemoryDescriptions();
  }
  return default_memory_space_description_;
}

// ------------------------------- Devices -------------------------------------

PjRtCApiDevice::PjRtCApiDevice(PJRT_Device* device, PjRtCApiClient* client)
    : client_(client),
      device_(device),
      description_(client->pjrt_c_api(),
                   pjrt::GetDeviceDescription(client->pjrt_c_api(), device)) {}

PjRtClient* PjRtCApiDevice::client() const { return client_; }

bool PjRtCApiDevice::IsAddressable() const {
  PJRT_Device_IsAddressable_Args args;
  args.struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = device_;
  const PJRT_Api* api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_IsAddressable(&args), api);
  return args.is_addressable;
}

PjRtLocalHardwareId PjRtCApiDevice::local_hardware_id() const {
  PJRT_Device_LocalHardwareId_Args args;
  args.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = device_;
  const PJRT_Api* api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_LocalHardwareId(&args), api);
  return PjRtLocalHardwareId(args.local_hardware_id);
}

absl::StatusOr<PjRtMemorySpace*> PjRtCApiDevice::default_memory_space() const {
  PJRT_Device_DefaultMemory_Args args;
  args.struct_size = PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = device_;
  const PJRT_Api* api = client_->pjrt_c_api();
  RETURN_STATUS_IF_PJRT_ERROR(api->PJRT_Device_DefaultMemory(&args), api);
  return client_->GetCppMemory(args.memory);
}

absl::StatusOr<PjRtMemorySpace*> PjRtCApiDevice::memory_space_by_kind(
    absl::string_view kind) const {
  auto it = absl::c_find_if(memory_spaces_, [kind](PjRtMemorySpace* ms) {
    return ms->kind() == kind;
  });
  if (it != memory_spaces_.end()) {
    return *it;
  }
  return absl::InternalError(
      absl::StrCat("No memory space found (kind: ", kind, ")"));
}

absl::StatusOr<tsl::AllocatorStats> PjRtCApiDevice::GetAllocatorStats() const {
  PJRT_Device_MemoryStats_Args args;
  args.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = device_;
  const PJRT_Api* api = client_->pjrt_c_api();
  RETURN_STATUS_IF_PJRT_ERROR(api->PJRT_Device_MemoryStats(&args), api);

  tsl::AllocatorStats result;
  result.bytes_in_use = args.bytes_in_use;

  // The PJRT C API supports optionally returning most fields, but only some
  // fields in tsl::AllocatorStats are optional. Return -1 for unset,
  // non-optional fields. We could change tsl::AllocatorStats to have all
  // optional fields, but that requires changing a lot of callers.
  if (args.peak_bytes_in_use_is_set) {
    result.peak_bytes_in_use = args.peak_bytes_in_use;
  } else {
    result.peak_bytes_in_use = -1;
  }
  if (args.num_allocs_is_set) {
    result.num_allocs = args.num_allocs;
  } else {
    result.num_allocs = -1;
  }
  if (args.largest_alloc_size_is_set) {
    result.largest_alloc_size = args.largest_alloc_size;
  } else {
    result.largest_alloc_size = -1;
  }
  if (args.bytes_limit_is_set) {
    result.bytes_limit = args.bytes_limit;
  }
  if (args.bytes_reserved_is_set) {
    result.bytes_reserved = args.bytes_reserved;
  } else {
    result.bytes_reserved = -1;
  }
  if (args.peak_bytes_reserved_is_set) {
    result.peak_bytes_reserved = args.peak_bytes_reserved;
  } else {
    result.peak_bytes_reserved = -1;
  }
  if (args.bytes_reservable_limit_is_set) {
    result.bytes_reservable_limit = args.bytes_reservable_limit;
  }
  if (args.largest_free_block_bytes_is_set) {
    result.largest_free_block_bytes = args.largest_free_block_bytes;
  } else {
    result.largest_free_block_bytes = -1;
  }
  if (args.pool_bytes_is_set) {
    result.pool_bytes = args.pool_bytes;
  }
  if (args.peak_pool_bytes_is_set) {
    result.peak_pool_bytes = args.peak_pool_bytes;
  }
  return result;
}

absl::StatusOr<std::intptr_t> PjRtCApiDevice::GetStreamForExternalReadyEvents()
    const {
  const PJRT_Api* c_api = client_->pjrt_c_api();
  PJRT_Stream_Extension* extension =
      client_->FindExtension<PJRT_Stream_Extension>(
          PJRT_Extension_Type::PJRT_Extension_Type_Stream);
  if (extension == nullptr) {
    return absl::UnimplementedError(
        "Stream extension not implemented in this PJRT plugin.");
  }
  PJRT_Get_Stream_For_External_Ready_Events_Args args;
  args.struct_size = PJRT_Get_Stream_For_External_Ready_Events_Args_STRUCT_SIZE;
  args.device = device_;
  RETURN_STATUS_IF_PJRT_ERROR(extension->get_stream(&args), c_api);
  return args.stream;
}

absl::StatusOr<bool> PjRtCApiDevice::PoisonExecution(int32_t launch_id,
                                                     absl::Status error) {
  if (client_->pjrt_c_api()->pjrt_api_version.major_version == 0 &&
      client_->pjrt_c_api()->pjrt_api_version.minor_version < 85) {
    return absl::UnimplementedError(
        "PJRT_Device_PoisonExecution requires PJRT C API version 0.85 or "
        "higher.");
  }
  const PJRT_Api* c_api = client_->pjrt_c_api();
  PJRT_Device_PoisonExecution_Args args;
  args.struct_size = PJRT_Device_PoisonExecution_Args_STRUCT_SIZE;
  args.device = device_;
  args.launch_id = launch_id;

  args.error_code = pjrt::StatusCodeToPjrtErrorCode(error.code());
  args.error_message = error.message().data();
  args.error_message_size = error.message().size();

  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Device_PoisonExecution(&args), c_api);
  return args.poisoned;
}

std::unique_ptr<ScopedAsyncTrackingEvent>
PjRtCApiDevice::CreateAsyncTrackingEvent(absl::string_view description) const {
  if (client_->pjrt_c_api()->pjrt_api_version.major_version == 0 &&
      client_->pjrt_c_api()->pjrt_api_version.minor_version < 86) {
    return nullptr;
  }
  PJRT_Device_CreateAsyncTrackingEvent_Args args;
  args.struct_size = PJRT_Device_CreateAsyncTrackingEvent_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = c_device();
  args.description = description.data();
  args.description_size = description.size();
  args.event = nullptr;

  const PJRT_Api* api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_CreateAsyncTrackingEvent(&args),
                            api);

  if (args.event == nullptr) {
    return nullptr;
  }
  return std::make_unique<PjRtCApiAsyncTrackingEvent>(api, args.event);
}

PjRtCApiAsyncTrackingEvent::PjRtCApiAsyncTrackingEvent(
    const PJRT_Api* c_api, PJRT_AsyncTrackingEvent* event)
    : c_api_(c_api), event_(event) {}

PjRtCApiAsyncTrackingEvent::~PjRtCApiAsyncTrackingEvent() {
  PJRT_AsyncTrackingEvent_Destroy_Args args;
  args.struct_size = PJRT_AsyncTrackingEvent_Destroy_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.event = event_;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_AsyncTrackingEvent_Destroy(&args),
                            c_api_);
}

void PjRtCApiAsyncTrackingEvent::AddDependency(
    tsl::RCReference<tsl::AsyncValue> dependency) {
  LOG(FATAL) << "AddDependency is not supported in C API yet.";
}

// ------------------------------- Memory --------------------------------------

const PJRT_Api* PjRtCApiMemorySpace::pjrt_c_api() const {
  return client_->pjrt_c_api();
}

PjRtClient* PjRtCApiMemorySpace::client() const { return client_; }

int PjRtCApiMemorySpace::id() const {
  PJRT_Memory_Id_Args args;
  args.struct_size = PJRT_Memory_Id_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.memory = c_memory_;
  pjrt::LogFatalIfPjrtError(pjrt_c_api()->PJRT_Memory_Id(&args), pjrt_c_api());
  return args.id;
}

absl::string_view PjRtCApiMemorySpace::kind() const {
  PJRT_Memory_Kind_Args args;
  args.struct_size = PJRT_Memory_Kind_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.memory = c_memory_;

  pjrt::LogFatalIfPjrtError(pjrt_c_api()->PJRT_Memory_Kind(&args),
                            pjrt_c_api());

  return absl::string_view(args.kind, args.kind_size);
}

int PjRtCApiMemorySpace::kind_id() const {
  PJRT_Memory_Kind_Id_Args args;
  args.struct_size = PJRT_Memory_Kind_Id_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.memory = c_memory_;
  if (pjrt_c_api()->pjrt_api_version.major_version > 0 ||
      pjrt_c_api()->pjrt_api_version.minor_version >= 48) {
    // The `kind_id` API is added in version 0.48.
    pjrt::LogFatalIfPjrtError(pjrt_c_api()->PJRT_Memory_Kind_Id(&args),
                              pjrt_c_api());
    return args.kind_id;
  }
  return tsl::Fingerprint32(kind());
}

absl::string_view PjRtCApiMemorySpace::DebugString() const {
  PJRT_Memory_DebugString_Args args;
  args.struct_size = PJRT_Memory_DebugString_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.memory = c_memory_;
  pjrt::LogFatalIfPjrtError(pjrt_c_api()->PJRT_Memory_DebugString(&args),
                            pjrt_c_api());
  return absl::string_view(args.debug_string, args.debug_string_size);
}

absl::string_view PjRtCApiMemorySpace::ToString() const {
  PJRT_Memory_ToString_Args args;
  args.struct_size = PJRT_Memory_ToString_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.memory = c_memory_;
  pjrt::LogFatalIfPjrtError(pjrt_c_api()->PJRT_Memory_ToString(&args),
                            pjrt_c_api());
  return absl::string_view(args.to_string, args.to_string_size);
}

// ------------------------------- Executables ---------------------------------

PjRtCApiExecutable::PjRtCApiExecutable(const PJRT_Api* c_api,
                                       PJRT_Executable* executable)
    : c_api_(c_api),
      executable_(executable, ::pjrt::MakeExecutableDeleter(c_api)) {}

absl::string_view PjRtCApiExecutable::name() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_Name_Args args;
  args.executable = executable;
  args.struct_size = PJRT_Executable_Name_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  pjrt::LogFatalIfPjrtError(c_api->PJRT_Executable_Name(&args), c_api);

  return absl::string_view(args.executable_name, args.executable_name_size);
}

int PjRtCApiExecutable::num_replicas() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_NumReplicas_Args args;
  args.executable = executable;
  args.struct_size = PJRT_Executable_NumReplicas_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  pjrt::LogFatalIfPjrtError(c_api->PJRT_Executable_NumReplicas(&args), c_api);

  return args.num_replicas;
}

int PjRtCApiExecutable::num_partitions() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_NumPartitions_Args args;
  args.executable = executable;
  args.struct_size = PJRT_Executable_NumPartitions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  pjrt::LogFatalIfPjrtError(c_api->PJRT_Executable_NumPartitions(&args), c_api);

  return args.num_partitions;
}

int64_t PjRtCApiExecutable::SizeOfGeneratedCodeInBytes() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_SizeOfGeneratedCodeInBytes_Args args;
  args.struct_size =
      PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = executable;

  pjrt::LogFatalIfPjrtError(
      c_api->PJRT_Executable_SizeOfGeneratedCodeInBytes(&args), c_api);
  return args.size_in_bytes;
}

absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
PjRtCApiExecutable::GetCostAnalysis() const {
  // Initialize function call args
  PJRT_Executable_GetCostAnalysis_Args args;
  args.struct_size = PJRT_Executable_GetCostAnalysis_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_executable();

  // Make PJRT C API call
  const PJRT_Api* c_api = pjrt_c_api();
  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Executable_GetCostAnalysis(&args),
                              c_api);

  // Copy returned properties to output map
  return pjrt::ConvertFromPjRtNamedValueList(args.properties,
                                             args.num_properties);
}

absl::StatusOr<std::vector<std::vector<PrimitiveType>>>
PjRtCApiExecutable::GetOutputElementTypes() const {
  PJRT_Executable_OutputElementTypes_Args args;
  args.struct_size = PJRT_Executable_OutputElementTypes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_executable();

  const PJRT_Api* c_api = pjrt_c_api();

  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Executable_OutputElementTypes(&args),
                              c_api);

  std::vector<PrimitiveType> out;
  out.reserve(args.num_output_types);
  for (int i = 0; i < args.num_output_types; ++i) {
    out.push_back(pjrt::ConvertFromPjRtBufferType(args.output_types[i]));
  }
  return std::vector<std::vector<PrimitiveType>>{std::move(out)};
}

static absl::StatusOr<Shape> GetOutputShapeHelper(
    const std::vector<PrimitiveType>& element_types,
    const std::vector<DimensionVector>& dimensions,
    const std::vector<std::shared_ptr<const PjRtLayout>>& layouts) {
  CHECK_EQ(element_types.size(), dimensions.size());
  CHECK_EQ(element_types.size(), layouts.size());

  std::vector<xla::Shape> shapes;
  shapes.reserve(element_types.size());
  for (int i = 0; i < element_types.size(); ++i) {
    TF_ASSIGN_OR_RETURN(xla::Shape shape, ShapeUtil::MakeValidatedShape(
                                              element_types[i], dimensions[i]));
    *shape.mutable_layout() = layouts[i]->xla_layout();
    shapes.push_back(std::move(shape));
  }
  if (shapes.size() == 1) {
    return shapes[0];
  }
  return ShapeUtil::MakeTupleShape(shapes);
}

absl::StatusOr<std::vector<Shape>> PjRtCApiExecutable::GetOutputShapes() const {
  TF_ASSIGN_OR_RETURN(std::vector<std::vector<PrimitiveType>> element_types,
                      GetOutputElementTypes());
  TF_ASSIGN_OR_RETURN(std::vector<std::vector<DimensionVector>> dimensions,
                      GetOutputDimensions());
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<const PjRtLayout>> layouts,
                      GetOutputLayouts());

  // `PjRtExecutable::GetOutputLayouts` doesn't support MPMD executables.
  // Only one output is expected.
  CHECK_EQ(element_types.size(), 1);
  CHECK_EQ(dimensions.size(), 1);

  TF_ASSIGN_OR_RETURN(
      Shape shape,
      GetOutputShapeHelper(element_types[0], dimensions[0], layouts));
  return std::vector<Shape>{shape};
}

absl::StatusOr<std::string>
PjRtCApiExecutable::GetSerializedExecutableMetadata() const {
  auto executable_metadata_extension =
      pjrt::FindExtension<PJRT_ExecutableMetadata_Extension>(
          c_api_, PJRT_Extension_Type::PJRT_Extension_Type_ExecutableMetadata);
  if (executable_metadata_extension == nullptr) {
    return absl::UnimplementedError(
        "PJRT_ExecutableMetadata_Extension not implemented by this PJRT "
        "plugin.");
  }
  PJRT_ExecutableMetadata_GetExecutableMetadata_Args args;
  args.executable = c_executable();
  args.metadata = nullptr;
  executable_metadata_extension->get_executable_metadata(&args);
  absl::Cleanup cleanup = [&args, &executable_metadata_extension] {
    if (args.metadata != nullptr) {
      PJRT_ExecutableMetadata_DestroySerializedMetadata_Args free_args;
      free_args.metadata = args.metadata;
      executable_metadata_extension->destroy_serialized_metadata(&free_args);
    }
  };
  if (args.metadata == nullptr) {
    return absl::InternalError(
        "PJRT_ExecutableMetadata_Extension did not return metadata.");
  }
  return std::string(args.metadata->serialized_metadata,
                     args.metadata->serialized_metadata_size);
}

absl::StatusOr<std::vector<std::vector<DimensionVector>>>
PjRtCApiExecutable::GetOutputDimensions() const {
  PJRT_Executable_OutputDimensions_Args args;
  args.struct_size = PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_executable();

  const PJRT_Api* c_api = pjrt_c_api();

  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Executable_OutputDimensions(&args),
                              c_api);

  std::vector<DimensionVector> out;
  out.reserve(args.num_outputs);
  int index = 0;
  for (int i = 0; i < args.num_outputs; ++i) {
    DimensionVector dimensions;
    dimensions.reserve(args.dim_sizes[i]);
    for (int j = 0; j < args.dim_sizes[i]; ++j) {
      dimensions.push_back(args.dims[index++]);
    }
    out.push_back(std::move(dimensions));
  }
  return std::vector<std::vector<DimensionVector>>{std::move(out)};
}

absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
PjRtCApiExecutable::GetOutputLayouts() const {
  const PJRT_Api* c_api = pjrt_c_api();
  if (c_api->pjrt_api_version.major_version == 0 &&
      c_api->pjrt_api_version.minor_version < 81) {
    // If the PJRT C API version is too old, fall back to the default
    // implementation.
    return this->PjRtExecutable::GetOutputLayouts();
  }
  PJRT_Layouts_Extension* extension =
      pjrt::FindExtension<PJRT_Layouts_Extension>(
          c_api, PJRT_Extension_Type::PJRT_Extension_Type_Layouts);
  if (extension == nullptr ||
      extension->PJRT_Layouts_MemoryLayout_Serialize == nullptr ||
      extension->PJRT_Layouts_PJRT_Executable_GetOutputLayouts == nullptr) {
    // If we can't find PJRT_Layouts_PJRT_Executable_GetOutputLayouts support,
    // fall back to the default implementation.
    return this->PjRtExecutable::GetOutputLayouts();
  }

  PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args args;
  args.struct_size =
      PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_executable();
  RETURN_STATUS_IF_PJRT_ERROR(
      extension->PJRT_Layouts_PJRT_Executable_GetOutputLayouts(&args), c_api);

  std::vector<std::shared_ptr<const PjRtLayout>> layouts;
  layouts.reserve(args.num_outputs);
  for (int i = 0; i < args.num_outputs; ++i) {
    // TODO(b/343274093): returns a PjRtLayout that wraps a C API layout
    // directly instead of de/serializing into an xla::Layout.
    PJRT_Layouts_MemoryLayout_Serialize_Args serialize_args;
    serialize_args.struct_size =
        PJRT_Layouts_MemoryLayout_Serialize_Args_STRUCT_SIZE;
    serialize_args.extension_start = nullptr;
    serialize_args.layout = args.layouts[i];
    pjrt::LogFatalIfPjrtError(
        extension->PJRT_Layouts_MemoryLayout_Serialize(&serialize_args), c_api);

    // Clean up `PJRT_Layouts_SerializedLayout`.
    absl::Cleanup cleanup = [&serialize_args] {
      serialize_args.serialized_layout_deleter(
          serialize_args.serialized_layout);
    };

    std::string serialized_layout(serialize_args.serialized_bytes,
                                  serialize_args.serialized_bytes_size);
    absl::StatusOr<std::shared_ptr<const PjRtLayout>> pjrt_layout =
        PjRtLayout::Deserialize(serialized_layout);
    CHECK_OK(pjrt_layout.status());
    layouts.push_back(*std::move(pjrt_layout));
  }
  return layouts;
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
PjRtCApiExecutable::GetOutputMemoryKinds() const {
  PJRT_Executable_OutputMemoryKinds_Args args;
  args.struct_size = PJRT_Executable_OutputMemoryKinds_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_executable();

  const PJRT_Api* c_api = pjrt_c_api();
  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Executable_OutputMemoryKinds(&args),
                              c_api);

  std::vector<absl::string_view> out;
  out.reserve(args.num_outputs);
  for (int i = 0; i < args.num_outputs; ++i) {
    out.push_back(
        absl::string_view(args.memory_kinds[i], args.memory_kind_sizes[i]));
  }
  return std::vector<std::vector<absl::string_view>>{std::move(out)};
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
PjRtCApiExecutable::GetHloModules() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_OptimizedProgram_Args args;
  args.struct_size = PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = executable;
  PJRT_Program program;
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.extension_start = nullptr;
  program.code = nullptr;
  args.program = &program;

  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Executable_OptimizedProgram(&args),
                              c_api);

  constexpr size_t TWO_GIBIBYTES = 2ull * 1024 * 1024 * 1024;
  const size_t code_size = args.program->code_size;
  CHECK(code_size < TWO_GIBIBYTES);
  std::string code(code_size, ' ');
  args.program->code = code.data();
  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Executable_OptimizedProgram(&args),
                              c_api);

  absl::string_view program_format(program.format, program.format_size);
  if (program_format != ::pjrt::kHloWithConfigFormat &&
      program_format != ::pjrt::kMlirFormat) {
    return xla::Internal(
        "expected program format `hlo_with_config` or `mlir` but got %s",
        program_format);
  }

  if (program_format == ::pjrt::kMlirFormat) {
    mlir::MLIRContext ctx;
    TF_ASSIGN_OR_RETURN(  // NOLINT(clang-diagnostic-pre-c++20-compat)
        mlir::OwningOpRef<mlir::ModuleOp> module,
        ParseMlirModuleString(code, ctx));
    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    if (mlir::failed(pm.run(module.get())))
      return xla::Internal("failed to convert to MHLO");
    // TODO(jieying): Tuple args should really come from GetCompileOptions (or
    // equivalent) once implemented.
    mlir::MlirToHloConversionOptions options;
    options.return_tuple = false;
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloModule> hlo_module,
                        mlir::ConvertMlirHloToHloModule(module.get(), options));

    std::vector<std::shared_ptr<HloModule>> out;
    out.push_back(std::move(hlo_module));
    return out;
  }

  HloModuleProtoWithConfig proto;
  proto.ParseFromString(code);
  std::vector<std::shared_ptr<HloModule>> out;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProtoWithConfig(proto));
  out.push_back(std::move(module));
  return out;
}

absl::StatusOr<std::string> PjRtCApiExecutable::SerializeExecutable() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_Serialize_Args ser_args;
  ser_args.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE;
  ser_args.extension_start = nullptr;
  ser_args.executable = executable;
  ser_args.serialized_executable = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Executable_Serialize(&ser_args),
                              c_api);
  absl::Cleanup cleanup = [&ser_args] {
    ser_args.serialized_executable_deleter(ser_args.serialized_executable);
  };
  return std::string(ser_args.serialized_bytes, ser_args.serialized_bytes_size);
}

absl::StatusOr<std::string> PjRtCApiExecutable::FingerprintExecutable() const {
  const PJRT_Api* c_api_ = pjrt_c_api();
  PJRT_Executable_Fingerprint_Args args;
  args.struct_size = PJRT_Executable_Fingerprint_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_executable();
  RETURN_STATUS_IF_PJRT_ERROR(c_api_->PJRT_Executable_Fingerprint(&args),
                              c_api_);
  return std::string(args.executable_fingerprint,
                     args.executable_fingerprint_size);
}

absl::StatusOr<CompileOptions> PjRtCApiExecutable::GetCompileOptions() const {
  if (c_api_->pjrt_api_version.major_version == 0 &&
      c_api_->pjrt_api_version.minor_version < 87) {
    return absl::UnimplementedError(
        "PJRT_Executable_GetCompileOptions not implemented in this PJRT "
        "plugin.");
  }
  PJRT_Executable_GetCompileOptions_Args args;
  args.struct_size = PJRT_Executable_GetCompileOptions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_executable();
  RETURN_STATUS_IF_PJRT_ERROR(c_api_->PJRT_Executable_GetCompileOptions(&args),
                              c_api_);
  absl::Cleanup cleanup = [&args] {
    args.serialized_compile_options_deleter(args.serialized_compile_options);
  };
  CompileOptionsProto proto;
  if (!proto.ParseFromString(
          std::string(args.serialized_bytes, args.serialized_bytes_size))) {
    return absl::InternalError(
        "PjRtCApiExecutable::GetCompileOptions: Failed to parse "
        "CompileOptionsProto");
  }
  return CompileOptions::FromProto(proto);
}

// ------------------------ Loaded Executables ---------------------------------

PjRtCApiLoadedExecutable::PjRtCApiLoadedExecutable(
    PjRtCApiClient* client, PJRT_LoadedExecutable* executable)
    : client_(client),
      loaded_executable_(executable, ::pjrt::MakeLoadedExecutableDeleter(
                                         client->pjrt_c_api())) {
  PJRT_LoadedExecutable_GetExecutable_Args args;
  args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.loaded_executable = c_loaded_executable();
  args.executable = nullptr;
  pjrt::LogFatalIfPjrtError(
      pjrt_c_api()->PJRT_LoadedExecutable_GetExecutable(&args), pjrt_c_api());
  executable_ =
      std::make_unique<PjRtCApiExecutable>(pjrt_c_api(), args.executable);
  InitDevices();
  InitDeviceAssignment();
}

void PjRtCApiLoadedExecutable::InitDevices() {
  PJRT_LoadedExecutable_AddressableDevices_Args args;
  args.struct_size = PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_loaded_executable();
  args.addressable_devices = nullptr;
  args.num_addressable_devices = 0;

  const PJRT_Api* api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(
      api->PJRT_LoadedExecutable_AddressableDevices(&args), api);

  const size_t num_addressable_devices = args.num_addressable_devices;
  addressable_devices_.reserve(num_addressable_devices);

  for (size_t i = 0; i < num_addressable_devices; ++i) {
    PJRT_Device* device = args.addressable_devices[i];
    PjRtCApiDevice* c_api_device = client_->GetCppDevice(device);
    addressable_devices_.push_back(c_api_device);
  }
}

void PjRtCApiLoadedExecutable::InitDeviceAssignment() {
  if (pjrt_c_api()->pjrt_api_version.major_version == 0 &&
      pjrt_c_api()->pjrt_api_version.minor_version < 79) {
    device_assignment_ = nullptr;
    return;
  }
  PJRT_LoadedExecutable_GetDeviceAssignment_Args args;
  args.struct_size = PJRT_LoadedExecutable_GetDeviceAssignment_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_loaded_executable();

  const PJRT_Api* api = pjrt_c_api();

  pjrt::LogFatalIfPjrtError(
      api->PJRT_LoadedExecutable_GetDeviceAssignment(&args), api);

  absl::Cleanup cleanup = [&args] {
    args.serialized_device_assignment_deleter(
        args.serialized_device_assignment);
  };

  // If `serialized_bytes_size` is 0, this executable is portable and has no
  // device assignment.
  if (args.serialized_bytes_size == 0) {
    device_assignment_ = nullptr;
    return;
  }

  std::string serialized_proto(args.serialized_bytes,
                               args.serialized_bytes_size);
  DeviceAssignmentProto proto;
  CHECK(proto.ParseFromString(serialized_proto));

  absl::StatusOr<std::unique_ptr<DeviceAssignment>> device_assignment =
      DeviceAssignment::Deserialize(proto);
  CHECK_OK(device_assignment.status());
  device_assignment_ = std::move(*device_assignment);
}

static std::vector<std::vector<PJRT_Buffer*>> Convert2DCppBuffersToCBuffers(
    absl::Span<const std::vector<PjRtBuffer*>> cpp_lists) {
  std::vector<std::vector<PJRT_Buffer*>> c_lists;
  c_lists.reserve(cpp_lists.size());
  for (const auto& cpp_list : cpp_lists) {
    auto& c_list = c_lists.emplace_back();
    c_list.reserve(cpp_list.size());
    for (PjRtBuffer* buffer : cpp_list) {
      auto* c_api_argument = tensorflow::down_cast<PjRtCApiBuffer*>(buffer);
      c_list.push_back(c_api_argument->c_buffer());
    }
  }
  return c_lists;
}

static std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
Convert2DCBuffersToCppBuffers(PJRT_Buffer** const* c_lists, size_t outer_size,
                              int inner_size, xla::PjRtCApiClient* client) {
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> ret;
  for (size_t i = 0; i < outer_size; ++i) {
    auto& output_list = ret.emplace_back();
    output_list.reserve(inner_size);
    for (size_t j = 0; j < inner_size; ++j) {
      output_list.push_back(
          std::make_unique<PjRtCApiBuffer>(client, c_lists[i][j]));
    }
  }
  return ret;
}

// Wraps original `xla::SendCallback` inside `PJRT_SendCallbackInfo` using
// 1) void* `user_arg` to capture `cpp_send_callback.callback` (std::function)
// 2) `PJRT_SendCallback` function pointer, which reinterprets and calls
// `user_arg` to call `cpp_send_callback.callback` function. This appends to
// `send_callback_functions`, which must be kept alive for as lnog as the
// returned PJRT_SendCallbackInfo is needed.
//
// TODO(yeounoh) move this to pjrt_c_api_helpers after implementing C API for
// the opaque types `PJRT_Chunk` and `PJRT_CopyToDeviceStream`.
PJRT_SendCallbackInfo CppSendCallbackToC(
    const xla::SendCallback& cpp_send_callback,
    PjRtCApiLoadedExecutable::SendCallbackFunction* send_callback_function) {
  *send_callback_function =
      [&send_callback = cpp_send_callback.callback](
          PJRT_Chunk* chunk, PJRT_CallbackError* callback_error,
          size_t total_size_in_bytes, bool done) -> PJRT_Error* {
    // PJRT C API doesn't support
    // use_major_to_minor_data_layout_for_callbacks = false
    xla::Shape dummy_shape;
    absl::Status status = send_callback(xla::PjRtTransferMetadata{dummy_shape},
                                        ::pjrt::ConvertToCppChunk(*chunk),
                                        total_size_in_bytes, done);
    if (!status.ok()) {
      absl::string_view message = status.message();
      return (*callback_error)(pjrt::StatusCodeToPjrtErrorCode(status.code()),
                               message.data(), message.size());
    }
    return nullptr;
  };
  return PJRT_SendCallbackInfo{
      /*channel_id=*/cpp_send_callback.channel_id,
      /*user_arg=*/send_callback_function,
      /*send_callback=*/
      [](PJRT_Chunk* chunk, PJRT_CallbackError* callback_error,
         size_t total_size_in_bytes, bool done, void* user_arg) -> PJRT_Error* {
        // PJRT_SendCallback, `send_callback` is internal C interface callback
        // representation that cpatures the client C++ callback in void*
        // `user_arg` and reinterprets in the lower-level runtime for execution.
        // `user_arg` captures `send_callback_function` which is
        // SendCallbackFunction*.
        PjRtCApiLoadedExecutable::SendCallbackFunction* send_callback =
            reinterpret_cast<PjRtCApiLoadedExecutable::SendCallbackFunction*>(
                user_arg);
        return (*send_callback)(chunk, callback_error, total_size_in_bytes,
                                done);
      }};
}

CApiCopyToDeviceStream::CApiCopyToDeviceStream(
    PJRT_CopyToDeviceStream* c_stream, const PJRT_Api* c_api)
    : CopyToDeviceStream(/*total_bytes=*/0, /*granule_bytes=*/0),
      c_stream_(c_stream),
      c_api_(c_api) {
  PJRT_CopyToDeviceStream_TotalBytes_Args total_bytes_args;
  total_bytes_args.struct_size =
      PJRT_CopyToDeviceStream_TotalBytes_Args_STRUCT_SIZE;
  total_bytes_args.extension_start = nullptr;
  total_bytes_args.stream = c_stream_;
  pjrt::LogFatalIfPjrtError(
      c_api_->PJRT_CopyToDeviceStream_TotalBytes(&total_bytes_args), c_api_);
  total_bytes_ = total_bytes_args.total_bytes;

  PJRT_CopyToDeviceStream_GranuleSize_Args granule_size_args;
  granule_size_args.struct_size =
      PJRT_CopyToDeviceStream_GranuleSize_Args_STRUCT_SIZE;
  granule_size_args.extension_start = nullptr;
  granule_size_args.stream = c_stream_;
  pjrt::LogFatalIfPjrtError(
      c_api_->PJRT_CopyToDeviceStream_GranuleSize(&granule_size_args), c_api_);
  granule_bytes_ = granule_size_args.granule_size_in_bytes;
}

CApiCopyToDeviceStream::~CApiCopyToDeviceStream() {
  PJRT_CopyToDeviceStream_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_CopyToDeviceStream_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.stream = c_stream_;
  pjrt::LogFatalIfPjrtError(
      c_api_->PJRT_CopyToDeviceStream_Destroy(&destroy_args), c_api_);
}

Future<> CApiCopyToDeviceStream::AddChunk(PjRtChunk chunk) {
  PJRT_Chunk c_chunk = ::pjrt::ConvertFromCppChunk(std::move(chunk));

  PJRT_CopyToDeviceStream_AddChunk_Args add_chunk_args;
  add_chunk_args.struct_size =
      PJRT_CopyToDeviceStream_AddChunk_Args_STRUCT_SIZE;
  add_chunk_args.extension_start = nullptr;
  add_chunk_args.stream = c_stream_;
  add_chunk_args.chunk = &c_chunk;

  PJRT_CopyToDeviceStream_CurrentBytes_Args current_bytes_args;
  current_bytes_args.struct_size =
      PJRT_CopyToDeviceStream_CurrentBytes_Args_STRUCT_SIZE;
  current_bytes_args.extension_start = nullptr;
  current_bytes_args.stream = c_stream_;

  {
    absl::MutexLock lock(mu_);
    RETURN_FUTURE_IF_ERROR(
        c_api_->PJRT_CopyToDeviceStream_AddChunk(&add_chunk_args), c_api_);
    RETURN_FUTURE_IF_ERROR(
        c_api_->PJRT_CopyToDeviceStream_CurrentBytes(&current_bytes_args),
        c_api_);
    current_bytes_ = current_bytes_args.current_bytes;
  }

  CHECK(add_chunk_args.transfer_complete != nullptr);
  return ::pjrt::ConvertCEventToCppFuture(add_chunk_args.transfer_complete,
                                          c_api_);
}

// Wraps original `xla::RecvCallback` inside `PJRT_RecvCallbackInfo` using
// 1) void* `user_arg` to capture `cpp_recv_callback.callback` (std::function)
// 2) `PJRT_RecvCallback` function pointer, which reinterprets and calls
// `user_arg` to call `cpp_send_callback.callback` function. This appends to
// `recv_callback_functions`, which must be kept alive for as lnog as the
// returned PJRT_RecvCallbackInfo is needed.
//
// TODO(yeounoh) move this to pjrt_c_api_helpers after implementing C API for
// the opaque types `PJRT_Chunk` and `PJRT_CopyToDeviceStream`.
PJRT_RecvCallbackInfo CppRecvCallbackToC(
    const xla::RecvCallback& cpp_recv_callback, const PJRT_Api* c_api,
    PjRtCApiLoadedExecutable::RecvCallbackFunction* recv_callback_function) {
  *recv_callback_function = [&recv_callback = cpp_recv_callback.callback,
                             c_api](PJRT_CopyToDeviceStream* stream) {
    // PJRT C API doesn't support
    // use_major_to_minor_data_layout_for_callbacks = false
    xla::Shape dummy_shape;
    recv_callback(xla::PjRtTransferMetadata{dummy_shape},
                  std::make_unique<CApiCopyToDeviceStream>(stream, c_api));
  };
  return PJRT_RecvCallbackInfo{
      /*channel_id=*/cpp_recv_callback.channel_id,
      /*user_arg=*/recv_callback_function,
      /*recv_callback=*/
      [](PJRT_CopyToDeviceStream* stream, void* user_arg) {
        // PJRT_RecvCallback, `recv_callback` is internal C interface callback
        // representation that cpatures the client C++ callback in void*
        // `user_arg` and reinterprets in the lower-level runtime for execution.
        // `user_arg` captures `recv_callback_function` which is
        // RecvCallbackFunction*.
        PjRtCApiLoadedExecutable::RecvCallbackFunction* recv_callback =
            reinterpret_cast<PjRtCApiLoadedExecutable::RecvCallbackFunction*>(
                user_arg);
        (*recv_callback)(stream);
      }};
}

static void CppSendCallbackListsToC(
    absl::Span<const std::vector<xla::SendCallback>> cpp_lists,
    std::vector<PjRtCApiLoadedExecutable::SendCallbackFunction>&
        send_callback_functions,
    std::vector<std::vector<PJRT_SendCallbackInfo>>& c_lists) {
  if (cpp_lists.empty()) return;

  send_callback_functions.resize(cpp_lists.size() * cpp_lists[0].size());
  c_lists.reserve(cpp_lists.size());

  int func_count = 0;
  for (const std::vector<xla::SendCallback>& cpp_list : cpp_lists) {
    std::vector<PJRT_SendCallbackInfo>& c_list = c_lists.emplace_back();
    c_list.reserve(cpp_list.size());
    for (const xla::SendCallback& cpp_callback : cpp_list) {
      c_list.emplace_back(CppSendCallbackToC(
          cpp_callback, &send_callback_functions[func_count++]));
    }
  }
}

static void CppRecvCallbackListsToC(
    absl::Span<const std::vector<xla::RecvCallback>> cpp_lists,
    const PJRT_Api* c_api,
    std::vector<PjRtCApiLoadedExecutable::RecvCallbackFunction>&
        recv_callback_functions,
    std::vector<std::vector<PJRT_RecvCallbackInfo>>& c_lists) {
  if (cpp_lists.empty()) return;

  recv_callback_functions.resize(cpp_lists.size() * cpp_lists[0].size());
  c_lists.reserve(cpp_lists.size());

  int func_count = 0;
  for (const auto& cpp_list : cpp_lists) {
    std::vector<PJRT_RecvCallbackInfo>& c_list = c_lists.emplace_back();
    c_list.reserve(cpp_list.size());
    for (const auto& cpp_callback : cpp_list) {
      c_list.emplace_back(CppRecvCallbackToC(
          cpp_callback, c_api, &recv_callback_functions[func_count++]));
    }
  }
}

absl::StatusOr<size_t> PjRtCApiLoadedExecutable::GetNumOutputs() const {
  PJRT_Executable_NumOutputs_Args args;
  args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_executable();
  RETURN_STATUS_IF_PJRT_ERROR(pjrt_c_api()->PJRT_Executable_NumOutputs(&args),
                              pjrt_c_api());
  return args.num_outputs;
}

absl::StatusOr<std::vector<std::vector<PJRT_Buffer*>>>
PjRtCApiLoadedExecutable::InitializeOutputListsStorage(
    size_t outer_size) const {
  TF_ASSIGN_OR_RETURN(size_t inner_size, GetNumOutputs());
  std::vector<std::vector<PJRT_Buffer*>> c_output_lists_storage(
      outer_size, std::vector<PJRT_Buffer*>(inner_size));
  return c_output_lists_storage;
}

absl::StatusOr<std::vector<PJRT_Buffer**>>
PjRtCApiLoadedExecutable::InitializeOutputLists(
    std::vector<std::vector<PJRT_Buffer*>>& c_output_lists_storage) const {
  size_t outer_size = c_output_lists_storage.size();
  std::vector<PJRT_Buffer**> c_output_lists(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    c_output_lists[i] = c_output_lists_storage[i].data();
  }
  return c_output_lists;
}

absl::StatusOr<PJRT_LoadedExecutable_Execute_Args>
PjRtCApiLoadedExecutable::GetCommonExecuteArgs(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options, PJRT_ExecuteOptions& c_options,
    std::vector<std::vector<PJRT_Buffer*>>& c_argument_lists_storage,
    std::vector<PJRT_Buffer**>& c_arguments,
    std::optional<std::vector<PJRT_Event*>>& device_complete_events,
    SendRecvCallbackData& callback_data,
    std::vector<int64_t>& non_donatable_input_indices_storage,
    std::vector<int>& task_ids_storage,
    std::vector<int64_t>& incarnation_ids_storage) const {
  bool using_host_callbacks =
      !options.send_callbacks.empty() || !options.recv_callbacks.empty();
  if (using_host_callbacks &&
      !options.use_major_to_minor_data_layout_for_callbacks) {
    return Unimplemented(
        "PJRT C API doesn't support "
        "ExecuteOptions::use_major_to_minor_data_layout_for_callbacks = false");
  }

  PJRT_LoadedExecutable_Execute_Args args;
  args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  args.executable = c_loaded_executable();
  args.options = &c_options;
  args.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  args.options->launch_id = options.launch_id;
  for (auto i : options.non_donatable_input_indices) {
    non_donatable_input_indices_storage.push_back(i);
  }
  args.options->num_non_donatable_input_indices =
      options.non_donatable_input_indices.size();
  args.options->non_donatable_input_indices =
      non_donatable_input_indices_storage.data();
  args.num_devices = argument_handles.size();
  if (pjrt_c_api()->pjrt_api_version.minor_version >= 76) {
    args.options->call_location = options.call_location.c_str();
  }

  for (const auto& [task_id, incarnation_id] : options.incarnations) {
    task_ids_storage.push_back(task_id);
    incarnation_ids_storage.push_back(incarnation_id.value());
  }
  args.options->num_tasks = options.incarnations.size();
  args.options->task_ids = task_ids_storage.data();
  args.options->incarnation_ids = incarnation_ids_storage.data();

  // If the executable has no addressable devices, `num_args` cannot be
  // determined but it is unused. 0 serves as a placeholder.
  args.num_args = (args.num_devices > 0) ? argument_handles[0].size() : 0;
  if (device_complete_events.has_value() || using_host_callbacks) {
    device_complete_events->resize(args.num_devices);
    args.device_complete_events = device_complete_events->data();
  } else {
    args.device_complete_events = nullptr;
  }

  // Populates `args.argument_lists` from `argument_handles`.
  c_argument_lists_storage = Convert2DCppBuffersToCBuffers(argument_handles);
  c_arguments.reserve(c_argument_lists_storage.size());
  for (auto& argument_list : c_argument_lists_storage) {
    c_arguments.push_back(argument_list.data());
  }
  args.argument_lists = c_arguments.data();

  // Allocates memory for callbacks. `callback_data` needs to stay alive during
  // the execution.
  if (!options.send_callbacks.empty()) {
    CppSendCallbackListsToC(options.send_callbacks,
                            callback_data.send_callback_functions,
                            callback_data.c_send_callbacks);
    for (auto& c_send_callback_list : callback_data.c_send_callbacks) {
      callback_data.c_send_callback_lists.push_back(
          c_send_callback_list.data());
    }
    args.options->send_callbacks = callback_data.c_send_callback_lists.data();
    args.options->num_send_ops = options.send_callbacks[0].size();
  }
  if (!options.recv_callbacks.empty()) {
    CppRecvCallbackListsToC(options.recv_callbacks, pjrt_c_api(),
                            callback_data.recv_callback_functions,
                            callback_data.c_recv_callbacks);
    for (auto& c_recv_callback_list : callback_data.c_recv_callbacks) {
      callback_data.c_recv_callback_lists.push_back(
          c_recv_callback_list.data());
    }
    args.options->recv_callbacks = callback_data.c_recv_callback_lists.data();
    args.options->num_recv_ops = options.recv_callbacks[0].size();
  }

  return args;
}

static absl::StatusOr<PJRT_ExecuteContext*> ForwardExecuteContext(
    const PjRtCApiClient* client, const ExecuteContext* context) {
  const PJRT_Api* c_api = client->pjrt_c_api();
  // If the execute context is null, we don't have anything to forward.
  if (context == nullptr) return nullptr;

  // If we can't find the FFI extension, we can't forward anything from the
  // execute context to the C API.
  PJRT_FFI_Extension* ffi_extension = client->FindExtension<PJRT_FFI_Extension>(
      PJRT_Extension_Type::PJRT_Extension_Type_FFI);
  if (ffi_extension == nullptr) return nullptr;

  // Create a new instance of the PJRT_ExecuteContext.
  PJRT_ExecuteContext_Create_Args create_args = {
      PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, nullptr, nullptr};
  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_ExecuteContext_Create(&create_args),
                              c_api);

  // Forward FFI user data to the C API execute context.
  using TypeId = ffi::ExecutionContext::TypeId;
  auto forward_user_data = [&](TypeId type_id, void* data) -> absl::Status {
    PJRT_FFI_UserData_Add_Args add_args{
        PJRT_FFI_UserData_Add_Args_STRUCT_SIZE,
        nullptr,
        create_args.context,
        PJRT_FFI_UserData{type_id.value(), data},
    };
    RETURN_STATUS_IF_PJRT_ERROR(ffi_extension->user_data_add(&add_args), c_api);
    return absl::OkStatus();
  };

  TF_RETURN_IF_ERROR(
      context->ffi_context().ForEachWithStatus(forward_user_data));

  return create_args.context;
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
PjRtCApiLoadedExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<Future<>>>& returned_futures) const {
  std::vector<std::vector<PJRT_Buffer*>> c_argument_lists_storage;
  std::vector<int64_t> non_donatable_input_indices_storage;
  std::vector<int> task_ids_storage;
  std::vector<int64_t> incarnation_ids_storage;
  std::vector<PJRT_Buffer**> c_arguments;
  std::optional<std::vector<PJRT_Event*>> device_complete_events;
  if (returned_futures.has_value()) {
    device_complete_events.emplace();
  }

  PJRT_ExecuteOptions c_options = {PJRT_ExecuteOptions_STRUCT_SIZE, nullptr};
  TF_ASSIGN_OR_RETURN(c_options.context,
                      ForwardExecuteContext(client_, options.context));

  // Don't forget to destroy execute context if we created it.
  auto destroy_context = absl::MakeCleanup([&]() {
    if (c_options.context != nullptr) {
      PJRT_ExecuteContext_Destroy_Args destroy_args = {
          PJRT_ExecuteContext_Destroy_Args_STRUCT_SIZE, nullptr,
          c_options.context};
      pjrt::LogFatalIfPjrtError(
          pjrt_c_api()->PJRT_ExecuteContext_Destroy(&destroy_args),
          pjrt_c_api());
    }
  });

  auto callback_data = std::make_shared<SendRecvCallbackData>();
  TF_ASSIGN_OR_RETURN(
      PJRT_LoadedExecutable_Execute_Args args,
      GetCommonExecuteArgs(argument_handles, options, c_options,
                           c_argument_lists_storage, c_arguments,
                           device_complete_events, *callback_data,
                           non_donatable_input_indices_storage,
                           task_ids_storage, incarnation_ids_storage));

  // Allocates memory for output. `c_output_lists_storage` and `c_output_lists`
  // need to stay alive during the call of `PJRT_LoadedExecutable_Execute`.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<PJRT_Buffer*>> c_output_lists_storage,
      InitializeOutputListsStorage(args.num_devices));
  TF_ASSIGN_OR_RETURN(std::vector<PJRT_Buffer**> c_output_lists,
                      InitializeOutputLists(c_output_lists_storage));
  args.output_lists = c_output_lists.data();

  args.execute_device = nullptr;
  PJRT_Profiler_Extension profiler_extension =
      pjrt::CreatePjrtProfilerExtension(
          "PJRT_LoadedExecutable_Execute linkage");
  args.extension_start = &profiler_extension.base;

  RETURN_STATUS_IF_PJRT_ERROR(
      pjrt_c_api()->PJRT_LoadedExecutable_Execute(&args), pjrt_c_api());

  if (device_complete_events.has_value()) {
    std::vector<Future<>> device_complete_futures;
    device_complete_futures.reserve(args.num_devices);
    for (int i = 0; i < args.num_devices; ++i) {
      device_complete_futures.push_back(pjrt::ConvertCEventToCppFuture(
          args.device_complete_events[i], pjrt_c_api()));
      if (!callback_data->c_send_callbacks.empty() ||
          !callback_data->c_recv_callbacks.empty()) {
        device_complete_futures.back().OnReady(
            [callback_data](absl::Status status) {
              // Keeps C callbacks alive until execution completes on all
              // devices.
            });
      }
    }

    if (returned_futures.has_value()) {
      *returned_futures = std::move(device_complete_futures);
    }
  }

  int inner_size =
      c_output_lists_storage.empty() ? 0 : c_output_lists_storage[0].size();
  return Convert2DCBuffersToCppBuffers(args.output_lists, args.num_devices,
                                       inner_size, client_);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  if (!options.send_callbacks.empty() || !options.recv_callbacks.empty()) {
    return absl::Status(absl::StatusCode::kUnimplemented,
                        "Send/recv callbacks not implemented for "
                        "PjRtCApiLoadedExecutable::ExecuteWithSingleDevice.");
  }

  std::vector<std::vector<PjRtBuffer*>> argument_handles_vec = {
      {argument_handles.begin(), argument_handles.end()}};

  std::vector<std::vector<PJRT_Buffer*>> c_argument_lists_storage;
  std::vector<int64_t> non_donatable_input_indices_storage;
  std::vector<int> task_ids_storage;
  std::vector<int64_t> incarnation_ids_storage;
  std::vector<PJRT_Buffer**> c_arguments;
  std::optional<std::vector<PJRT_Event*>> device_complete_events;
  if (fill_future) {
    device_complete_events.emplace();
  }

  auto callback_data = std::make_shared<SendRecvCallbackData>();

  PJRT_ExecuteOptions c_options = {PJRT_ExecuteOptions_STRUCT_SIZE, nullptr};
  TF_ASSIGN_OR_RETURN(
      PJRT_LoadedExecutable_Execute_Args args,
      GetCommonExecuteArgs(argument_handles_vec, options, c_options,
                           c_argument_lists_storage, c_arguments,
                           device_complete_events, *callback_data,
                           non_donatable_input_indices_storage,
                           task_ids_storage, incarnation_ids_storage));

  // Allocates memory for output. `c_output_lists_storage` and `c_output_lists`
  // need to stay alive during the call of `PJRT_LoadedExecutable_Execute`.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<PJRT_Buffer*>> c_output_lists_storage,
      InitializeOutputListsStorage(args.num_devices));
  TF_ASSIGN_OR_RETURN(std::vector<PJRT_Buffer**> c_output_lists,
                      InitializeOutputLists(c_output_lists_storage));
  args.output_lists = c_output_lists.data();

  args.execute_device =
      tensorflow::down_cast<PjRtCApiDevice*>(device)->c_device();
  PJRT_Profiler_Extension profiler_extension =
      pjrt::CreatePjrtProfilerExtension(
          "PJRT_LoadedExecutable_Execute linkage");
  args.extension_start = &profiler_extension.base;

  RETURN_STATUS_IF_PJRT_ERROR(
      pjrt_c_api()->PJRT_LoadedExecutable_Execute(&args), pjrt_c_api());

  if (fill_future) {
    returned_future = pjrt::ConvertCEventToCppFuture(
        args.device_complete_events[0], pjrt_c_api());
  }
  return std::move(Convert2DCBuffersToCppBuffers(
      args.output_lists, args.num_devices, c_output_lists_storage[0].size(),
      client_)[0]);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiLoadedExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  return ExecuteWithSingleDevice(argument_handles, device, options,
                                 returned_future, fill_future);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiLoadedExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  return ExecuteWithSingleDevice(argument_handles, device, options,
                                 returned_future, fill_future);
}

void PjRtCApiLoadedExecutable::Delete() {
  PJRT_LoadedExecutable_Delete_Args args;
  args.struct_size = PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_loaded_executable();
  const PJRT_Api* c_api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(c_api->PJRT_LoadedExecutable_Delete(&args), c_api);
}

bool PjRtCApiLoadedExecutable::IsDeleted() const {
  PJRT_LoadedExecutable_IsDeleted_Args args;
  args.struct_size = PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = c_loaded_executable();

  const PJRT_Api* c_api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(c_api->PJRT_LoadedExecutable_IsDeleted(&args),
                            c_api);
  return args.is_deleted;
}

// ---------------------------------- Buffers ----------------------------------

PjRtCApiBuffer::PjRtCApiBuffer(PjRtCApiClient* client, PJRT_Buffer* buffer)
    : client_(client),
      buffer_(buffer, ::pjrt::MakeBufferDeleter(client->pjrt_c_api())),
      readiness_event_(nullptr,
                       ::pjrt::MakeEventDeleter(client->pjrt_c_api())) {}

PrimitiveType PjRtCApiBuffer::element_type() const {
  PJRT_Buffer_ElementType_Args args;
  args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  pjrt::LogFatalIfPjrtError(pjrt_c_api()->PJRT_Buffer_ElementType(&args),
                            pjrt_c_api());
  return pjrt::ConvertFromPjRtBufferType(args.type);
}

absl::Span<const int64_t> PjRtCApiBuffer::dimensions() const {
  PJRT_Buffer_Dimensions_Args args;
  args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  pjrt::LogFatalIfPjrtError(pjrt_c_api()->PJRT_Buffer_Dimensions(&args),
                            pjrt_c_api());
  return absl::Span<const int64_t>(args.dims, args.num_dims);
}

std::shared_ptr<const PjRtLayout> PjRtCApiBuffer::layout() const {
  {
    absl::MutexLock lock(mu_);
    if (layout_ == nullptr) {
      const PJRT_Api* c_api = pjrt_c_api();
      PJRT_Layouts_Extension* extension =
          client_->FindExtension<PJRT_Layouts_Extension>(
              PJRT_Extension_Type::PJRT_Extension_Type_Layouts);
      if (extension == nullptr) {
        layout_ = std::make_shared<PjRtLayout>(
            LayoutUtil::MakeDescendingLayout(dimensions().size()));
      } else {
        std::unique_ptr<PJRT_Layouts_MemoryLayout,
                        pjrt::PJRT_Layouts_MemoryLayoutDeleter>
            layout = pjrt::GetMemoryLayout(c_api, buffer_.get());

        // TODO(b/343274093): returns a PjRtLayout that wraps a C API layout
        // directly instead of de/serializing into an xla::Layout.
        PJRT_Layouts_MemoryLayout_Serialize_Args serialize_args;
        serialize_args.struct_size =
            PJRT_Layouts_MemoryLayout_Serialize_Args_STRUCT_SIZE;
        serialize_args.extension_start = nullptr;
        serialize_args.layout = layout.get();
        pjrt::LogFatalIfPjrtError(
            extension->PJRT_Layouts_MemoryLayout_Serialize(&serialize_args),
            c_api);

        // Clean up `PJRT_Layouts_SerializedLayout`.
        absl::Cleanup cleanup = [&serialize_args] {
          serialize_args.serialized_layout_deleter(
              serialize_args.serialized_layout);
        };

        std::string serialized_layout(serialize_args.serialized_bytes,
                                      serialize_args.serialized_bytes_size);
        absl::StatusOr<std::shared_ptr<const PjRtLayout>> pjrt_layout =
            PjRtLayout::Deserialize(serialized_layout);
        CHECK_OK(pjrt_layout.status());
        layout_ = *std::move(pjrt_layout);
      }
    }
  }
  return layout_;
}

const Shape& PjRtCApiBuffer::on_device_shape() const {
  if (!on_device_shape_.has_value()) {
    Shape shape(element_type(), dimensions(), is_dynamic_dimension());
    *shape.mutable_layout() = layout()->xla_layout();
    absl::MutexLock lock(mu_);
    on_device_shape_ = shape;
  }
  return *on_device_shape_;
}

absl::StatusOr<Shape> PjRtCApiBuffer::logical_on_device_shape() {
  absl::StatusOr<std::vector<int64_t>> dims = logical_dimensions();
  if (!dims.ok()) {
    return dims.status();
  }
  Shape result(element_type(), *dims, is_dynamic_dimension());
  *result.mutable_layout() = layout()->xla_layout();
  return result;
}

bool PjRtCApiBuffer::has_dynamic_dimensions() const {
  PJRT_Buffer_DynamicDimensionIndices_Args args;
  args.struct_size = PJRT_Buffer_DynamicDimensionIndices_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();

  const PJRT_Api* api = pjrt_c_api();
  std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error(
      api->PJRT_Buffer_DynamicDimensionIndices(&args),
      pjrt::MakeErrorDeleter(api));

  if (error &&
      pjrt::GetErrorCode(error.get(), api) == PJRT_Error_Code_UNIMPLEMENTED) {
    return false;
  }
  return args.num_dynamic_dims > 0;
}

absl::Span<const bool> PjRtCApiBuffer::is_dynamic_dimension() const {
  {
    absl::MutexLock lock(mu_);
    if (!is_dynamic_dimension_.has_value()) {
      absl::InlinedVector<bool, InlineRank()>& is_dynamic_dimension_value =
          is_dynamic_dimension_.emplace();
      is_dynamic_dimension_value.assign(dimensions().size(), false);

      PJRT_Buffer_DynamicDimensionIndices_Args args;
      args.struct_size = PJRT_Buffer_DynamicDimensionIndices_Args_STRUCT_SIZE;
      args.extension_start = nullptr;
      args.buffer = buffer_.get();
      const PJRT_Api* api = pjrt_c_api();
      std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error(
          api->PJRT_Buffer_DynamicDimensionIndices(&args),
          pjrt::MakeErrorDeleter(api));
      if (error && pjrt::GetErrorCode(error.get(), api) ==
                       PJRT_Error_Code_UNIMPLEMENTED) {
        return *is_dynamic_dimension_;
      }
      for (int i = 0; i < args.num_dynamic_dims; ++i) {
        is_dynamic_dimension_value[args.dynamic_dim_indices[i]] = true;
      }
    }
  }
  return *is_dynamic_dimension_;
}

absl::StatusOr<std::vector<int64_t>> PjRtCApiBuffer::logical_dimensions() {
  PJRT_Buffer_UnpaddedDimensions_Args args;
  args.struct_size = PJRT_Buffer_UnpaddedDimensions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  RETURN_STATUS_IF_PJRT_ERROR(
      pjrt_c_api()->PJRT_Buffer_UnpaddedDimensions(&args), pjrt_c_api());
  return std::vector<int64_t>(args.unpadded_dims,
                              args.unpadded_dims + args.num_dims);
}

Future<> PjRtCApiBuffer::LazyToLiteral(
    absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) {
  Future<MutableLiteralBase*> future = std::move(generator)();
  const absl::StatusOr<MutableLiteralBase*>& literal = future.Await();
  if (!literal.ok()) {
    return Future<>(literal.status());
  }
  return ToLiteral(literal.value());
}

Future<> PjRtCApiBuffer::ToLiteral(MutableLiteralBase* literal) {
  PJRT_Buffer_ToHostBuffer_Args args;
  args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.src = buffer_.get();
  args.event = nullptr;

  const xla::Shape& shape = literal->shape();

  if (!shape.IsArray()) {
    return Future<>(
        Unimplemented("PjRtCApiBuffer::ToLiteral: Shapes other than array are"
                      "not supported."));
  }

  args.dst_size = ShapeUtil::ByteSizeOfElements(shape);
  args.dst = literal->untyped_data();
  if (args.dst == nullptr) {
    // For zero-sized buffers, args.dst will be nullptr. In that case, the C API
    // will return early and not allocate an event.
    return Future<>(absl::OkStatus());
  }
  absl::StatusOr<pjrt::BufferMemoryLayoutData> c_layout_data;
  if (literal->shape().has_layout()) {
    c_layout_data =
        pjrt::ConvertToBufferMemoryLayoutData(literal->shape().layout());
    if (!c_layout_data.ok()) {
      return Future<>(c_layout_data.status());
    }
    args.host_layout = &(c_layout_data->c_layout);
  } else {
    args.host_layout = nullptr;
  }

  const PJRT_Api* api = pjrt_c_api();

  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> error{
      pjrt_c_api()->PJRT_Buffer_ToHostBuffer(&args),
      ::pjrt::MakeErrorDeleter(api)};

  if (error != nullptr) {
    return Future<>(::pjrt::PjrtErrorToStatus(error.get(), api));
  }
  CHECK(args.event != nullptr);
  return pjrt::ConvertCEventToCppFuture(args.event, api);
}

absl::StatusOr<size_t> PjRtCApiBuffer::GetOnDeviceSizeInBytes() const {
  PJRT_Buffer_OnDeviceSizeInBytes_Args args;
  args.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  RETURN_STATUS_IF_PJRT_ERROR(
      client_->pjrt_c_api()->PJRT_Buffer_OnDeviceSizeInBytes(&args),
      client_->pjrt_c_api());

  return args.on_device_size_in_bytes;
}

PjRtMemorySpace* PjRtCApiBuffer::memory_space() const {
  PJRT_Buffer_Memory_Args args;
  args.struct_size = PJRT_Buffer_Memory_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  const PJRT_Api* api = pjrt_c_api();
  std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error(
      api->PJRT_Buffer_Memory(&args), pjrt::MakeErrorDeleter(api));
  if (error == nullptr && args.memory != nullptr) {
    return client_->GetCppMemory(args.memory);
  } else if (error != nullptr && pjrt::GetErrorCode(error.get(), api) !=
                                     PJRT_Error_Code_UNIMPLEMENTED) {
    pjrt::LogFatalIfPjrtError(error.get(), api);
  }
  return nullptr;
}

PjRtDevice* PjRtCApiBuffer::device() const {
  PJRT_Buffer_Device_Args args;
  args.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  const PJRT_Api* api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_Device(&args), api);
  return client_->GetCppDevice(args.device);
}

void PjRtCApiBuffer::Delete() {
  PJRT_Buffer_Delete_Args args;
  args.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  const PJRT_Api* api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_Delete(&args), api);
}

bool PjRtCApiBuffer::IsDeleted() const {
  PJRT_Buffer_IsDeleted_Args args;
  args.struct_size = PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  const PJRT_Api* api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_IsDeleted(&args), api);
  return args.is_deleted;
}

Future<> PjRtCApiBuffer::CopyRawToHost(void* dst, int64_t offset,
                                       int64_t transfer_size) {
  PJRT_Buffer_CopyRawToHost_Args args;
  args.struct_size = PJRT_Buffer_CopyRawToHost_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  args.dst = dst;
  args.offset = offset;
  args.transfer_size = transfer_size;
  const PJRT_Api* api = pjrt_c_api();
  RETURN_FUTURE_IF_ERROR(api->PJRT_Buffer_CopyRawToHost(&args), api);
  CHECK(args.event != nullptr);
  return pjrt::ConvertCEventToCppFuture(args.event, api);
}

Future<> PjRtCApiBuffer::CopyRawToHostFuture(Future<void*> dst, int64_t offset,
                                             int64_t transfer_size) {
  if (pjrt_c_api()->pjrt_api_version.major_version == 0 &&
      pjrt_c_api()->pjrt_api_version.minor_version < 84) {
    return Future<>(absl::UnimplementedError(
        "PJRT_Buffer_CopyRawToHostFuture requires PJRT C API version 0.84 or "
        "higher."));
  }

  PJRT_Buffer_CopyRawToHostFuture_Args args;
  args.struct_size = PJRT_Buffer_CopyRawToHostFuture_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  args.offset = offset;
  args.transfer_size = transfer_size;
  const PJRT_Api* api = pjrt_c_api();
  RETURN_FUTURE_IF_ERROR(api->PJRT_Buffer_CopyRawToHostFuture(&args), api);
  dst.OnReady(
      [callback_data = args.callback_data,
       callback = args.future_ready_callback](absl::StatusOr<void*> dst) {
        PJRT_Buffer_CopyRawToHostFuture_Callback_Args callback_args;
        callback_args.struct_size =
            PJRT_Buffer_CopyRawToHostFuture_Callback_Args_STRUCT_SIZE;
        if (dst.ok()) {
          callback_args.dst = *dst;
          callback_args.error_code = PJRT_Error_Code_OK;
          callback_args.error_message = nullptr;
          callback_args.error_message_size = 0;
        } else {
          callback_args.dst = nullptr;
          callback_args.error_code =
              pjrt::StatusCodeToPjrtErrorCode(dst.status().code());
          callback_args.error_message = dst.status().message().data();
          callback_args.error_message_size = dst.status().message().size();
        }
        callback_args.callback_data = callback_data;
        callback(&callback_args);
      });
  CHECK(args.event != nullptr);
  return pjrt::ConvertCEventToCppFuture(args.event, api);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> PjRtCApiBuffer::CopyToMemorySpace(
    PjRtMemorySpace* dst_memory) {
  const PJRT_Api* api = pjrt_c_api();

  if (dst_memory->client() == client_) {
    PJRT_Buffer_CopyToMemory_Args args;
    args.struct_size = PJRT_Buffer_CopyToMemory_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = buffer_.get();
    args.dst_memory =
        tensorflow::down_cast<PjRtCApiMemorySpace*>(dst_memory)->c_memory();
    RETURN_STATUS_IF_PJRT_ERROR(api->PJRT_Buffer_CopyToMemory(&args), api);
    return std::unique_ptr<PjRtBuffer>(
        std::make_unique<PjRtCApiBuffer>(client_, args.dst_buffer));
  } else {
    // Copy across PjRtClients by copying through host
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal,
                        PjRtBuffer::ToLiteral().Await());
    absl::InlinedVector<int64_t, 4> byte_strides(
        literal->shape().dimensions().size());
    TF_RETURN_IF_ERROR(ShapeUtil::UnpackedByteStrides(
        literal->shape(), absl::MakeSpan(byte_strides)));
    // Avoid use-after-free on `literal` due to unsequenced move and use.
    Literal* literal_pointer = literal.get();
    return dst_memory->client()->BufferFromHostBuffer(
        literal_pointer->untyped_data(),
        literal_pointer->shape().element_type(),
        literal_pointer->shape().dimensions(), byte_strides,
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
        [literal{std::move(literal)}]() { /* frees literal */ }, dst_memory,
        /*device_layout=*/nullptr);
  }
}

bool PjRtCApiBuffer::IsOnCpu() const {
  PJRT_Buffer_IsOnCpu_Args args;
  args.struct_size = PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  const PJRT_Api* api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_IsOnCpu(&args), api);
  return args.is_on_cpu;
}

PJRT_Event* PjRtCApiBuffer::GetReadyEvent() {
  if (readiness_event_ == nullptr) {
    const PJRT_Api* api = pjrt_c_api();
    PJRT_Buffer_ReadyEvent_Args args;
    args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = buffer_.get();
    pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_ReadyEvent(&args), api);
    readiness_event_.reset(args.event);
  }
  return readiness_event_.get();
}

void PjRtCApiBuffer::MakePromiseTrackEvent() {
  CHECK(readiness_promise_ != nullptr);
  const PJRT_Api* api = pjrt_c_api();

  // Check if device execution has finished via `PJRT_Event_IsReady`. If true,
  // fetch the status with `PJRT_Event_Error()` and fulfill the promise
  // immediately. This avoids unnecessary overhead for already completed events.
  // Otherwise, register an asynchronous callback with
  // `PJRT_Event_OnReady` to be notified when the event is ready.
  PJRT_Event_IsReady_Args is_ready_args;
  is_ready_args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  is_ready_args.extension_start = nullptr;
  is_ready_args.event = GetReadyEvent();
  std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> is_ready_error{
      api->PJRT_Event_IsReady(&is_ready_args), pjrt::MakeErrorDeleter(api)};
  if (is_ready_error != nullptr) {
    readiness_promise_->Set(pjrt::PjrtErrorToStatus(is_ready_error.get(), api));
    return;
  }
  if (is_ready_args.is_ready) {
    PJRT_Event_Error_Args error_args;
    error_args.struct_size = PJRT_Event_Error_Args_STRUCT_SIZE;
    error_args.extension_start = nullptr;
    error_args.event = is_ready_args.event;
    PJRT_Error* error = api->PJRT_Event_Error(&error_args);
    readiness_promise_->Set(pjrt::PjrtErrorToStatus(error, api));
    pjrt::MakeErrorDeleter(api)(error);
    return;
  }

  PJRT_Event_OnReady_Args args;
  args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.event = GetReadyEvent();
  args.user_arg = new std::function<void(PJRT_Error*)>(
      [promise = readiness_promise_, api](PJRT_Error* error) -> void {
        promise->Set(::pjrt::PjrtErrorToStatus(error, api));
        ::pjrt::MakeErrorDeleter(api)(error);
      });
  args.callback = [](PJRT_Error* error, void* callback_ptr) {
    auto callback =
        static_cast<std::function<void(PJRT_Error*)>*>(callback_ptr);
    CHECK(callback != nullptr);
    (*callback)(error);
    delete callback;
  };

  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> error{
      api->PJRT_Event_OnReady(&args), ::pjrt::MakeErrorDeleter(api)};
  if (error != nullptr) {
    readiness_promise_->Set(::pjrt::PjrtErrorToStatus(error.get(), api));
  }
}

Future<> PjRtCApiBuffer::GetReadyFuture() {
  absl::MutexLock l(mu_);
  if (readiness_promise_ == nullptr) {
    auto [promise, future] = MakePromise<>();
    readiness_promise_ = std::move(promise).ToShared();
    readiness_future_ = std::move(future);
    MakePromiseTrackEvent();
  }
  return readiness_future_;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
PjRtCApiBuffer::AcquireExternalReference() {
  PJRT_Buffer_IncreaseExternalReferenceCount_Args increase_reference_count_args;
  increase_reference_count_args.buffer = c_buffer();
  increase_reference_count_args.struct_size =
      PJRT_Buffer_IncreaseExternalReferenceCount_Args_STRUCT_SIZE;
  increase_reference_count_args.extension_start = nullptr;
  RETURN_STATUS_IF_PJRT_ERROR(
      pjrt_c_api()->PJRT_Buffer_IncreaseExternalReferenceCount(
          &increase_reference_count_args),
      pjrt_c_api());

  PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args
      opaque_device_memory_data_pointer_args;
  opaque_device_memory_data_pointer_args.struct_size =
      PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
  opaque_device_memory_data_pointer_args.extension_start = nullptr;
  opaque_device_memory_data_pointer_args.buffer = c_buffer();
  RETURN_STATUS_IF_PJRT_ERROR(
      pjrt_c_api()->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(
          &opaque_device_memory_data_pointer_args),
      pjrt_c_api());

  void* device_memory_ptr =
      opaque_device_memory_data_pointer_args.device_memory_ptr;
  return std::make_unique<PjRtCApiExternalReference>(client_, this,
                                                     device_memory_ptr);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtCApiBuffer::DonateWithControlDependency(Future<> dependency) {
  if (client_->pjrt_c_api()->pjrt_api_version.major_version == 0 &&
      client_->pjrt_c_api()->pjrt_api_version.minor_version < 88) {
    return Unimplemented(
        "PJRT_Buffer_DonateWithControlDependency requires PJRT C API version "
        "0.88 or higher.");
  }
  PJRT_Buffer_DonateWithControlDependency_Args args;
  args.struct_size = PJRT_Buffer_DonateWithControlDependency_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = c_buffer();
  const PJRT_Api* api = pjrt_c_api();
  RETURN_STATUS_IF_PJRT_ERROR(
      api->PJRT_Buffer_DonateWithControlDependency(&args), api);

  dependency.OnReady([callback = args.dependency_ready_callback,
                      data = args.callback_data](absl::Status s) {
    PJRT_Buffer_DonateWithControlDependency_Callback_Args cb_args;
    cb_args.struct_size =
        PJRT_Buffer_DonateWithControlDependency_Callback_Args_STRUCT_SIZE;
    cb_args.callback_data = data;
    if (s.ok()) {
      cb_args.error_code = PJRT_Error_Code_OK;
      cb_args.error_message = nullptr;
      cb_args.error_message_size = 0;
    } else {
      cb_args.error_code = pjrt::StatusCodeToPjrtErrorCode(s.code());
      cb_args.error_message = s.message().data();
      cb_args.error_message_size = s.message().size();
    }
    callback(&cb_args);
  });
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<PjRtCApiBuffer>(client_, args.out_buffer));
}

void PjRtCApiBuffer::CopyToRemoteDevice(
    Future<std::string> serialized_descriptor, RemoteSendCallback on_done) {
  PJRT_CrossHostTransfers_Extension* extension =
      client_->FindExtension<PJRT_CrossHostTransfers_Extension>(
          PJRT_Extension_Type::PJRT_Extension_Type_CrossHostTransfers);
  if (extension == nullptr) {
    LOG(FATAL) << "PjRtBuffer::CopyToRemoteDevice: Cross host transfers "
                  "extension not found in PJRT plugin.";
  }
  PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args args;
  args.struct_size =
      PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = c_buffer();
  PJRT_Transfers_CrossHostRemoteSendCallbackInfo on_done_info =
      pjrt::CppCrossHostRemoteSendCallbackToC(pjrt_c_api(), std::move(on_done));
  args.on_done = on_done_info;

#if PJRT_API_CROSS_HOST_TRANSFERS_EXTENSION_VERSION < 5
  absl::StatusOr<std::string> descriptor = serialized_descriptor.Await();
  CHECK_OK(descriptor) << "Failed to copy buffer to remote device: "
                       << descriptor.status();
  args.serialized_descriptor = descriptor->c_str();
  args.serialized_descriptor_size = descriptor->size();
#else

  // When `serialized_descriptor` is ready, `descriptor_data` and
  // `descriptor_size` will be populated with the string data.
  size_t* descriptor_size = new size_t;
  char** descriptor_data = new char*;

  const PJRT_Api* c_api = pjrt_c_api();
  absl::StatusOr<std::string> descriptor;
  if (c_api->PJRT_Event_Create == nullptr || c_api->PJRT_Event_Set == nullptr) {
    // If `PJRT_Event_Create` or `PJRT_Event_Set` is not supported, block until
    // `serialized_descriptor` is ready and populate the descriptor data
    // synchronously.
    descriptor = serialized_descriptor.Await();
    CHECK_OK(descriptor) << "Failed to copy buffer to remote device: "
                         << descriptor.status();
    *descriptor_data = descriptor->data();
    *descriptor_size = descriptor->size();
    args.event = nullptr;
    args.serialized_descriptor = descriptor_data;
    args.serialized_descriptor_size = descriptor_size;
    extension->PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice(&args);
  } else {
    // Get a PJRT_Event to track `serialized_descriptor`.
    PJRT_Event_Create_Args event_args;
    event_args.struct_size = PJRT_Event_Create_Args_STRUCT_SIZE;
    event_args.extension_start = nullptr;
    pjrt::LogFatalIfPjrtError(c_api->PJRT_Event_Create(&event_args), c_api);

    // `PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice` registers an on-ready
    // callback for the event that reads the descriptor data. This callback
    // must be registered before the call to `serialized_descriptor.OnReady`
    // below, to ensure that callback is called before the descriptor data is
    // freed.
    args.event = event_args.event;
    args.serialized_descriptor = descriptor_data;
    args.serialized_descriptor_size = descriptor_size;
    extension->PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice(&args);

    // When `serialized_descriptor` is ready, populate the descriptor data and
    // then set the event.
    serialized_descriptor.OnReady([c_api, event = args.event, descriptor_data,
                                   descriptor_size](
                                      absl::StatusOr<std::string> descriptor) {
      if (descriptor.ok()) {
        *descriptor_data = descriptor->data();
        *descriptor_size = descriptor->size();
      }

      PJRT_Event_Set_Args event_set_args;
      event_set_args.struct_size = PJRT_Event_Set_Args_STRUCT_SIZE;
      event_set_args.extension_start = nullptr;
      event_set_args.event = event;
      event_set_args.error_code =
          pjrt::StatusCodeToPjrtErrorCode(descriptor.status().code());
      event_set_args.error_message = descriptor.status().message().data();
      event_set_args.error_message_size = descriptor.status().message().size();
      c_api->PJRT_Event_Set(&event_set_args);
    });
  }
#endif
}

PjRtCApiExternalReference::~PjRtCApiExternalReference() {
  PJRT_Buffer_DecreaseExternalReferenceCount_Args args;
  args.struct_size =
      PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_->c_buffer();
  pjrt::LogFatalIfPjrtError(
      client_->pjrt_c_api()->PJRT_Buffer_DecreaseExternalReferenceCount(&args),
      client_->pjrt_c_api());
}

absl::Status PjRtCApiExternalReference::WaitUntilBufferReadyOnStream(
    std::intptr_t stream) {
  const PJRT_Api* c_api = buffer_->pjrt_c_api();
  PJRT_Stream_Extension* extension =
      client_->FindExtension<PJRT_Stream_Extension>(
          PJRT_Extension_Type::PJRT_Extension_Type_Stream);
  if (extension == nullptr) {
    return absl::UnimplementedError(
        "Stream extension not implemented in this PJRT plugin.");
  }
  PJRT_Wait_Until_Buffer_Ready_On_Stream_Args args;
  args.struct_size = PJRT_Wait_Until_Buffer_Ready_On_Stream_Args_STRUCT_SIZE;
  args.stream = stream;
  args.buffer = buffer_->c_buffer();
  RETURN_STATUS_IF_PJRT_ERROR(extension->wait_stream(&args), c_api);
  return absl::OkStatus();
}

// ------------------------------ Device Topology ------------------------------

PjRtCApiTopologyDescription::PjRtCApiTopologyDescription(
    const PJRT_Api* c_api, PJRT_TopologyDescription* c_topology, bool owned)
    : compiler_(std::make_unique<PjRtCApiCompiler>(c_api)),
      c_api_(c_api),
      tpu_topology_extension_(pjrt::FindExtension<PJRT_TpuTopology_Extension>(
          c_api, PJRT_Extension_Type::PJRT_Extension_Type_TpuTopology)),
      c_topology_(c_topology),
      platform_version_(absl::StrCat(
          "PJRT C API\n", ::pjrt::GetPlatformVersion(c_topology, c_api))),
      platform_name_(::pjrt::PlatformName(c_api, c_topology)),
      platform_id_(tsl::Fingerprint64(platform_name_)) {
  if (owned) {
    owned_c_topology_ = std::unique_ptr<PJRT_TopologyDescription,
                                        pjrt::PJRT_TopologyDescriptionDeleter>(
        c_topology, pjrt::MakeTopologyDescriptionDeleter(c_api));
  }
  InitAttributes();
}

absl::string_view PjRtCApiTopologyDescription::platform_version() const {
  return platform_version_;
}

std::vector<std::unique_ptr<const PjRtDeviceDescription>>
PjRtCApiTopologyDescription::DeviceDescriptions() const {
  PJRT_TopologyDescription_GetDeviceDescriptions_Args args;
  args.struct_size =
      PJRT_TopologyDescription_GetDeviceDescriptions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = c_topology_;
  pjrt::LogFatalIfPjrtError(
      c_api_->PJRT_TopologyDescription_GetDeviceDescriptions(&args), c_api_);
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> out;
  out.reserve(args.num_descriptions);
  for (PJRT_DeviceDescription* device_desc :
       absl::Span<PJRT_DeviceDescription* const>(args.descriptions,
                                                 args.num_descriptions)) {
    out.push_back(
        std::make_unique<PjRtCApiDeviceDescription>(c_api_, device_desc));
  }
  return out;
}

absl::StatusOr<std::string> PjRtCApiTopologyDescription::Serialize() const {
  PJRT_TopologyDescription_Serialize_Args args;
  args.struct_size = PJRT_TopologyDescription_Serialize_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(c_api_->PJRT_TopologyDescription_Serialize(&args),
                              c_api_);
  auto out = std::string(args.serialized_bytes, args.serialized_bytes_size);
  args.serialized_topology_deleter(args.serialized_topology);
  return out;
}

absl::StatusOr<Layout> PjRtCApiTopologyDescription::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) const {
  const PJRT_Api* c_api = c_api_;
  PJRT_Layouts_Extension* extension =
      pjrt::FindExtension<PJRT_Layouts_Extension>(
          c_api, PJRT_Extension_Type::PJRT_Extension_Type_Layouts);
  if (extension == nullptr ||
      extension->PJRT_Layouts_PJRT_Topology_GetDefaultLayout == nullptr) {
    return Unimplemented(
        "PJRT C API does not implement "
        "PJRT_Layouts_PJRT_Topology_GetDefaultLayout.");
  }
  PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args args;
  args.struct_size =
      PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology_description = c_topology_;
  args.type = pjrt::ConvertToPjRtBufferType(element_type);
  args.dims = dims.data();
  args.num_dims = dims.size();
  RETURN_STATUS_IF_PJRT_ERROR(
      extension->PJRT_Layouts_PJRT_Topology_GetDefaultLayout(&args), c_api);

  // Clean up `PJRT_Layouts_MemoryLayout`.
  std::unique_ptr<PJRT_Layouts_MemoryLayout,
                  pjrt::PJRT_Layouts_MemoryLayoutDeleter>
      layout_destroyer(args.layout, pjrt::MakeMemoryLayoutDeleter(c_api));

  if (extension->PJRT_Layouts_MemoryLayout_Serialize == nullptr) {
    return Unimplemented(
        "PJRT_Layouts_MemoryLayout_Serialize is not implemented.");
  }

  // TODO(b/338478940): Wrap `args.layout` into a subclass of `PjRtLayout`.
  PJRT_Layouts_MemoryLayout_Serialize_Args serialize_args;
  serialize_args.struct_size =
      PJRT_Layouts_MemoryLayout_Serialize_Args_STRUCT_SIZE;
  serialize_args.extension_start = nullptr;
  serialize_args.layout = args.layout;
  RETURN_STATUS_IF_PJRT_ERROR(
      extension->PJRT_Layouts_MemoryLayout_Serialize(&serialize_args), c_api);

  // Clean up `PJRT_Layouts_SerializedLayout`.
  absl::Cleanup cleanup = [&serialize_args] {
    if (serialize_args.serialized_layout_deleter) {
      serialize_args.serialized_layout_deleter(
          serialize_args.serialized_layout);
    }
  };

  std::string serialized_layout(serialize_args.serialized_bytes,
                                serialize_args.serialized_bytes_size);
  TF_ASSIGN_OR_RETURN(std::shared_ptr<const PjRtLayout> pjrt_layout,
                      PjRtLayout::Deserialize(serialized_layout));

  return pjrt_layout->xla_layout();
}

void PjRtCApiTopologyDescription::InitAttributes() {
  PJRT_TopologyDescription_Attributes_Args args;
  args.struct_size = PJRT_TopologyDescription_Attributes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = c_topology_;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_TopologyDescription_Attributes(&args),
                            c_api_);
  attributes_ =
      pjrt::ConvertFromPjRtNamedValueList(args.attributes, args.num_attributes);
}

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
PjRtCApiTopologyDescription::Subslice(
    const PjRtDeviceDimensions& chips_per_host_bounds,
    const PjRtDeviceDimensions& host_bounds) const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented("Subslice is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_Subslice_Args args;
  args.struct_size = PJRT_TpuTopology_Subslice_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  args.chips_per_host_bounds = chips_per_host_bounds.data();
  args.chips_per_host_bounds_num_dims = chips_per_host_bounds.size();
  args.host_bounds = host_bounds.data();
  args.host_bounds_num_dims = host_bounds.size();
  args.subslice_topology = nullptr;
  RETURN_STATUS_IF_PJRT_ERROR(tpu_topology_extension_->subslice(&args), c_api_);
  return std::make_unique<PjRtCApiTopologyDescription>(c_api_,
                                                       args.subslice_topology,
                                                       /*owned=*/true);
}

bool PjRtCApiTopologyDescription::is_subslice_topology() const {
  CHECK(tpu_topology_extension_ != nullptr)
      << "Subslice is not supported by the PJRT C API.";
  PJRT_TpuTopology_IsSubsliceTopology_Args args;
  args.struct_size = PJRT_TpuTopology_IsSubsliceTopology_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  pjrt::LogFatalIfPjrtError(
      tpu_topology_extension_->is_subslice_topology(&args), c_api_);
  return args.is_subslice_topology;
}

absl::StatusOr<PjRtTopologyDescriptionProto>
PjRtCApiTopologyDescription::ToProto() const {
  TF_ASSIGN_OR_RETURN(std::string serialized, Serialize());
  PjRtTopologyDescriptionProto proto;
  if (!proto.ParseFromString(serialized)) {
    return Internal("Failed to parse serialized PjRtTopologyDescriptionProto.");
  }
  return proto;
}

absl::StatusOr<int> PjRtCApiTopologyDescription::ProcessCount() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented("ProcessCount is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_ProcessCount_Args args;
  args.struct_size = PJRT_TpuTopology_ProcessCount_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(tpu_topology_extension_->process_count(&args),
                              c_api_);
  return args.process_count;
}

absl::StatusOr<int> PjRtCApiTopologyDescription::ChipsPerProcess() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented("ChipsPerProcess is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_ChipsPerProcess_Args args;
  args.struct_size = PJRT_TpuTopology_ChipsPerProcess_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(tpu_topology_extension_->chips_per_process(&args),
                              c_api_);
  return args.chips_per_process;
}

absl::StatusOr<int> PjRtCApiTopologyDescription::CoreCountOfDefaultTypePerChip()
    const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "CoreCountOfDefaultTypePerChip is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_CoreCountPerChip_Args args;
  args.struct_size = PJRT_TpuTopology_CoreCountPerChip_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->core_count_per_chip(&args), c_api_);
  return args.core_count_of_default_type_per_chip;
}

absl::StatusOr<int> PjRtCApiTopologyDescription::ChipCount() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented("ChipCount is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_ChipCount_Args args;
  args.struct_size = PJRT_TpuTopology_ChipCount_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(tpu_topology_extension_->chip_count(&args),
                              c_api_);
  return args.chip_count;
}

absl::StatusOr<int> PjRtCApiTopologyDescription::CoreCountOfDefaultType()
    const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "CoreCountOfDefaultType is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_CoreCount_Args args;
  args.struct_size = PJRT_TpuTopology_CoreCount_Args_STRUCT_SIZE;

  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(tpu_topology_extension_->core_count(&args),
                              c_api_);
  return args.core_count_of_default_type;
}

absl::StatusOr<int>
PjRtCApiTopologyDescription::LogicalDeviceCountOfDefaultTypePerProcess() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "LogicalDeviceCountOfDefaultTypePerProcess is not supported by the "
        "PJRT C API.");
  }
  PJRT_TpuTopology_LogiDeviceCountPerProcess_Args args;
  args.struct_size =
      PJRT_TpuTopology_LogiDeviceCountPerProcess_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->logical_device_count_per_process(&args), c_api_);
  return args.logical_device_count_of_default_type_per_process;
}

absl::StatusOr<int>
PjRtCApiTopologyDescription::LogicalDeviceCountOfDefaultType() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "LogicalDeviceCountOfDefaultType is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_LogiDeviceCount_Args args;
  args.struct_size = PJRT_TpuTopology_LogiDeviceCount_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->logical_device_count(&args), c_api_);
  return args.logical_device_count_of_default_type;
}

absl::StatusOr<int>
PjRtCApiTopologyDescription::LogicalDeviceCountOfDefaultTypePerChip() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "LogicalDeviceCountOfDefaultTypePerChip is not supported by the PJRT C "
        "API.");
  }
  PJRT_TpuTopology_LogiDeviceCountPerChip_Args args;
  args.struct_size = PJRT_TpuTopology_LogiDeviceCountPerChip_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->logical_device_count_per_chip(&args), c_api_);
  return args.logical_device_count_of_default_type_per_chip;
}

absl::StatusOr<int>
PjRtCApiTopologyDescription::CoreCountOfDefaultTypePerProcess() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "CoreCountOfDefaultTypePerProcess is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_CoreCountPerProcess_Args args;
  args.struct_size = PJRT_TpuTopology_CoreCountPerProcess_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->core_count_per_process(&args), c_api_);
  return args.core_count_of_default_type_per_process;
}

absl::StatusOr<PjRtIdContainer<PjRtProcessId>>
PjRtCApiTopologyDescription::ProcessIds() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented("ProcessIds is not supported by the PJRT C API.");
  }
  TF_ASSIGN_OR_RETURN(int process_count, ProcessCount());
  std::vector<int> process_ids_storage(process_count);
  PJRT_TpuTopology_ProcessIds_Args args;
  args.struct_size = PJRT_TpuTopology_ProcessIds_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  args.max_process_ids = process_count;
  args.process_ids = process_ids_storage.data();
  RETURN_STATUS_IF_PJRT_ERROR(tpu_topology_extension_->process_ids(&args),
                              c_api_);
  PjRtIdContainer<PjRtProcessId> ids;
  ids.reserve(args.num_process_ids);
  for (size_t i = 0; i < args.num_process_ids; ++i) {
    ids.push_back(PjRtProcessId(args.process_ids[i]));
  }
  return ids;
}

absl::StatusOr<PjRtIdContainer<PjRtGlobalDeviceId>>
PjRtCApiTopologyDescription::LogicalDeviceOfDefaultTypeIdsOnProcess(
    PjRtProcessId process_id) const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "LogicalDeviceOfDefaultTypeIdsOnProcess is not supported by the PJRT "
        "C API.");
  }
  TF_ASSIGN_OR_RETURN(int logical_device_count,
                      LogicalDeviceCountOfDefaultTypePerProcess());
  std::vector<int> logical_device_ids_storage(logical_device_count);
  PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args args;
  args.struct_size = PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  args.process_id = process_id.value();
  args.max_logical_device_ids = logical_device_count;
  args.logical_device_of_default_type_ids = logical_device_ids_storage.data();
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->logical_device_ids_on_process(&args), c_api_);
  PjRtIdContainer<PjRtGlobalDeviceId> ids;
  ids.reserve(args.num_logical_device_ids);
  for (size_t i = 0; i < args.num_logical_device_ids; ++i) {
    ids.push_back(
        PjRtGlobalDeviceId(args.logical_device_of_default_type_ids[i]));
  }
  return ids;
}

absl::StatusOr<std::pair<PjRtProcessId, int>>
PjRtCApiTopologyDescription::ProcessIdAndIndexOnProcessForChip(
    PjRtGlobalChipId chip_id) const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "ProcessIdAndIndexOnProcessForChip is not supported by the PJRT C "
        "API.");
  }
  PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args args;
  args.struct_size =
      PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  args.chip_id = chip_id.value();
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->proc_id_and_idx_on_proc_for_chip(&args), c_api_);
  return std::make_pair(PjRtProcessId(args.process_id), args.index_on_process);
}

absl::StatusOr<std::pair<PjRtProcessId, int>> PjRtCApiTopologyDescription::
    ProcessIdAndIndexOnProcessForLogicalDeviceOfDefaultType(
        xla::PjRtGlobalDeviceId device_id) const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "ProcessIdAndIndexOnProcessForLogicalDeviceOfDefaultType is not "
        "supported by the PJRT C API.");
  }
  PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args args;
  args.struct_size =
      PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  args.device_id = device_id.value();
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->proc_id_and_idx_on_proc_for_logi_device(&args),
      c_api_);
  return std::make_pair(PjRtProcessId(args.process_id), args.index_on_process);
}

absl::StatusOr<PjRtDeviceDimensions>
PjRtCApiTopologyDescription::ProcessCoordFromId(
    PjRtProcessId process_id) const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "ProcessCoordFromId is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_ProcessCoordFromId_Args args;
  args.struct_size = PJRT_TpuTopology_ProcessCoordFromId_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  args.process_id = process_id.value();
  std::vector<int32_t> coords(kMaxDims);
  args.coords = coords.data();
  args.coords_max_dims = kMaxDims;
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->process_coord_from_id(&args), c_api_);
  return PjRtDeviceDimensions(
      absl::MakeSpan(coords.data(), args.coords_num_dims));
}

absl::StatusOr<PjRtGlobalChipId> PjRtCApiTopologyDescription::ChipIdFromCoord(
    const PjRtDeviceDimensions& chip) const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented("ChipIdFromCoord is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_ChipIdFromCoord_Args args;
  args.struct_size = PJRT_TpuTopology_ChipIdFromCoord_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  args.coords = chip.data();
  args.coords_num_dims = chip.size();
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->chip_id_from_coord(&args), c_api_);
  return PjRtGlobalChipId(args.chip_id);
}

absl::StatusOr<xla::PjRtGlobalDeviceId> PjRtCApiTopologyDescription::
    LogicalDeviceOfDefaultTypeIdFromChipCoordAndCoreIndex(
        const PjRtDeviceDimensions& chip, int core_index) const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "LogicalDeviceOfDefaultTypeIdFromChipCoordAndCoreIndex is not "
        "supported by the PJRT C API.");
  }
  std::vector<int32_t> chip_coords_storage(chip.size());
  for (size_t i = 0; i < chip.size(); ++i) {
    chip_coords_storage[i] = chip.data()[i];
  }
  PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args args;
  args.struct_size =
      PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  args.chip_coords = chip_coords_storage.data();
  args.chip_coords_num_dims = chip.size();
  args.logical_device_index_on_chip = core_index;
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->logical_device_id_from_chip_coord_and_idx(&args),
      c_api_);
  return PjRtGlobalDeviceId(args.logical_device_of_default_type_id);
}

absl::StatusOr<std::pair<PjRtDeviceDimensions, int32_t>>
PjRtCApiTopologyDescription::ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
    xla::PjRtGlobalDeviceId device_id) const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType is not supported "
        "by the PJRT C API.");
  }
  std::vector<int32_t> chip_coords_storage(kMaxDims);
  PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args args;
  args.struct_size =
      PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  args.device_id = device_id.value();
  args.chip_coords_max_dims = kMaxDims;
  args.chip_coords = chip_coords_storage.data();
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->chip_coord_and_idx_for_logi_device(&args),
      c_api_);
  return std::make_pair(
      PjRtDeviceDimensions(absl::MakeSpan(chip_coords_storage.data(),
                                          args.chip_coords_num_dims)),
      args.device_index_on_chip);
}

absl::StatusOr<PjRtDeviceDimensions>
PjRtCApiTopologyDescription::ChipsPerProcessBounds() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented(
        "ChipsPerProcessBounds is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_ChipsPerProcessBounds_Args args;
  args.struct_size = PJRT_TpuTopology_ChipsPerProcessBounds_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  std::vector<int32_t> bounds(kMaxDims);
  args.chip_per_process_bounds = bounds.data();
  args.chip_per_process_bounds_max_dims = kMaxDims;
  RETURN_STATUS_IF_PJRT_ERROR(
      tpu_topology_extension_->chips_per_process_bounds(&args), c_api_);
  return PjRtDeviceDimensions(
      absl::MakeSpan(bounds.data(), args.chip_per_process_bounds_num_dims));
}

absl::StatusOr<PjRtDeviceDimensions> PjRtCApiTopologyDescription::ChipBounds()
    const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented("ChipBounds is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_ChipBounds_Args args;
  args.struct_size = PJRT_TpuTopology_ChipBounds_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  std::vector<int32_t> bounds(kMaxDims);
  args.chip_bounds = bounds.data();
  args.chip_bounds_max_dims = kMaxDims;
  RETURN_STATUS_IF_PJRT_ERROR(tpu_topology_extension_->chip_bounds(&args),
                              c_api_);
  return PjRtDeviceDimensions(
      absl::MakeSpan(bounds.data(), args.chip_bounds_num_dims));
}

absl::StatusOr<PjRtDeviceDimensions>
PjRtCApiTopologyDescription::ProcessBounds() const {
  if (tpu_topology_extension_ == nullptr) {
    return Unimplemented("ProcessBounds is not supported by the PJRT C API.");
  }
  PJRT_TpuTopology_ProcessBounds_Args args;
  args.struct_size = PJRT_TpuTopology_ProcessBounds_Args_STRUCT_SIZE;
  args.topology = c_topology_;
  std::vector<int32_t> bounds(kMaxDims);
  args.process_bounds = bounds.data();
  args.process_bounds_max_dims = kMaxDims;
  RETURN_STATUS_IF_PJRT_ERROR(tpu_topology_extension_->process_bounds(&args),
                              c_api_);
  return PjRtDeviceDimensions(
      absl::MakeSpan(bounds.data(), args.process_bounds_num_dims));
}

// Initializes `PJRT_Compile_Args`, which will be used to call
// API PJRT_Compile().
static absl::StatusOr<std::unique_ptr<PjRtExecutable>>
InitializeArgsAndCompileAot(const PJRT_Api* c_api, PjRtClient* client,
                            const CompileOptions& options,
                            const PjRtTopologyDescription& topology,
                            const std::string& code,
                            const std::string& format) {
  PJRT_Compile_Args args;
  args.struct_size = PJRT_Compile_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  if (client == nullptr) {
    args.client = nullptr;
  } else {
    args.client =
        tensorflow::down_cast<PjRtCApiClient*>(client)->pjrt_c_client();
  }
  args.topology =
      tensorflow::down_cast<const PjRtCApiTopologyDescription*>(&topology)
          ->c_topology();
  TF_ASSIGN_OR_RETURN(const CompileOptionsProto options_proto,
                      options.ToProto());
  std::string options_str = options_proto.SerializeAsString();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  PJRT_Program program;
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.extension_start = nullptr;
  program.code = const_cast<char*>(code.c_str());
  program.code_size = code.size();
  program.format = format.c_str();
  program.format_size = format.size();
  args.program = &program;

  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Compile(&args), c_api);
  std::unique_ptr<PjRtExecutable> ret =
      std::make_unique<PjRtCApiExecutable>(c_api, args.executable);
  return ret;
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCApiCompiler::Compile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  std::string module_str = computation.proto().SerializeAsString();
  std::string format(pjrt::kHloFormat);
  return InitializeArgsAndCompileAot(c_api_, client, options, topology,
                                     module_str, format);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCApiCompiler::Compile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  std::string target_version = GetPluginStablehloVersionOrDefault(client);
  TF_ASSIGN_OR_RETURN(std::string serialized,
                      xla::Serialize(module, target_version));
  std::string format(pjrt::kMlirFormat);
  return InitializeArgsAndCompileAot(c_api_, client, options, topology,
                                     serialized, format);
}

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
PjRtCApiCompiler::DeserializePjRtTopologyDescription(
    const std::string& serialized_topology) {
  PJRT_TopologyDescription_Deserialize_Args args;
  args.struct_size = PJRT_TopologyDescription_Deserialize_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.serialized_topology = serialized_topology.data();
  args.serialized_topology_size = serialized_topology.size();
  args.topology = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(
      (*c_api_->PJRT_TopologyDescription_Deserialize)(&args), c_api_);

  return std::make_unique<PjRtCApiTopologyDescription>(c_api_, args.topology,
                                                       /*owned=*/true);
}

// -------------------------------- API access ---------------------------------

absl::StatusOr<std::unique_ptr<PjRtClient>> GetCApiClient(
    absl::string_view device_type,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(device_type));
  if (c_api == nullptr) {
    return Internal("PJRT C API is nullptr for %s", device_type);
  }
  return WrapClientAroundCApi(c_api, create_options, kv_store);
}

absl::StatusOr<std::unique_ptr<PjRtClient>> WrapClientAroundCApi(
    const PJRT_Api* c_api,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store) {
  PJRT_Client_Create_Args init_args;
  init_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  init_args.extension_start = nullptr;
  TF_ASSIGN_OR_RETURN(std::vector<PJRT_NamedValue> c_options,
                      pjrt::ConvertToPjRtNamedValueList(create_options));
  init_args.create_options = c_options.data();
  init_args.num_options = c_options.size();

  std::unique_ptr<pjrt::PJRT_KeyValueCallbackData> kv_callback_data;
  if (kv_store) {
    kv_callback_data = pjrt::ConvertToCKeyValueCallbacks(kv_store);
    init_args.kv_get_callback = kv_callback_data->c_kv_get;
    init_args.kv_get_user_arg = &kv_callback_data->kv_get_c_func;
    init_args.kv_try_get_callback = kv_callback_data->c_kv_try_get;
    init_args.kv_try_get_user_arg = &kv_callback_data->kv_try_get_c_func;
    init_args.kv_put_callback = kv_callback_data->c_kv_put;
    init_args.kv_put_user_arg = &kv_callback_data->kv_put_c_func;
  }

  RETURN_STATUS_IF_PJRT_ERROR(c_api->PJRT_Client_Create(&init_args), c_api);
  PJRT_Client* c_client = init_args.client;

  return std::unique_ptr<PjRtClient>(std::make_unique<PjRtCApiClient>(
      c_api, c_client, std::move(kv_callback_data)));
}

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetCApiTopology(
    absl::string_view device_type, absl::string_view topology_name,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(device_type));
  if (c_api == nullptr) {
    return Internal("PJRT C API is nullptr for %s", device_type);
  }
  return GetCApiTopology(c_api, topology_name, create_options);
}

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetCApiTopology(
    const PJRT_Api* c_api, absl::string_view topology_name,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options) {
  PJRT_TopologyDescription_Create_Args init_args;
  init_args.struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE;
  init_args.extension_start = nullptr;
  TF_ASSIGN_OR_RETURN(std::vector<PJRT_NamedValue> c_options,
                      pjrt::ConvertToPjRtNamedValueList(create_options));
  init_args.create_options = c_options.data();
  init_args.num_options = c_options.size();
  init_args.topology_name = topology_name.data();
  init_args.topology_name_size = topology_name.size();
  RETURN_STATUS_IF_PJRT_ERROR(
      c_api->PJRT_TopologyDescription_Create(&init_args), c_api);
  PJRT_TopologyDescription* c_topology = init_args.topology;
  return std::unique_ptr<PjRtTopologyDescription>(
      std::make_unique<PjRtCApiTopologyDescription>(c_api, c_topology,
                                                    /*owned=*/true));
}

absl::StatusOr<std::unique_ptr<PjRtCompiler>> GetCApiCompiler(
    absl::string_view device_type) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(device_type));
  if (c_api == nullptr) {
    return Internal("PJRT C API is nullptr for %s", device_type);
  }
  return std::make_unique<PjRtCApiCompiler>(c_api);
}

absl::StatusOr<std::unique_ptr<PjRtCompiler>> GetCApiCompiler() {
  TF_ASSIGN_OR_RETURN(std::vector<std::string> device_types,
                      pjrt::GetRegisteredPjrtApis());
  if (device_types.empty()) {
    return absl::FailedPreconditionError("PJRT_Api is not initialized.");
  }
  if (device_types.size() > 1) {
    return absl::FailedPreconditionError(
        "More than one device type registered. Please use "
        "GetCApiCompiler(absl::string_view device_type) "
        "instead.");
  }
  return GetCApiCompiler(device_types[0]);
}

absl::StatusOr<std::unique_ptr<PjRtPhaseCompiler>> GetCApiPhaseCompiler(
    absl::string_view device_type) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(device_type));
  if (c_api == nullptr) {
    return absl::InternalError(
        absl::StrCat("PJRT C API is nullptr for ", device_type));
  }
  // Ensure the Phase Compile extension is available and that the
  // 'phase_compile_get_compiler' and 'phase_compile_destroy_compiler'
  // callbacks are defined, as they are mandatory for the PjRtCApiPhaseCompiler
  // to function.
  auto phase_compile_extension =
      pjrt::FindExtension<PJRT_PhaseCompile_Extension>(
          c_api, PJRT_Extension_Type::PJRT_Extension_Type_PhaseCompile);
  if (phase_compile_extension == nullptr) {
    return absl::InternalError("Phase compile extension not found");
  }

  if (phase_compile_extension->phase_compile_get_compiler == nullptr) {
    return absl::InternalError(
        "phase_compile_get_compiler callback of the phase compile extension "
        "must not be null");
  }
  if (phase_compile_extension->phase_compile_destroy_compiler == nullptr) {
    return absl::InternalError(
        "phase_compile_destroy_compiler callback of the phase compile "
        "extension must not be null");
  }

  PJRT_PhaseCompile_Get_Compiler_Args get_compiler_args;
  get_compiler_args.struct_size =
      PJRT_PhaseCompile_Get_Compiler_Args_STRUCT_SIZE;
  get_compiler_args.extension_start = nullptr;
  RETURN_STATUS_IF_PJRT_ERROR(
      phase_compile_extension->phase_compile_get_compiler(&get_compiler_args),
      c_api);

  return std::make_unique<PjRtCApiPhaseCompiler>(
      c_api, phase_compile_extension, get_compiler_args.phase_compiler);
}

absl::StatusOr<std::unique_ptr<PjRtPhaseCompiler>> GetCApiPhaseCompiler() {
  TF_ASSIGN_OR_RETURN(std::vector<std::string> device_types,
                      pjrt::GetRegisteredPjrtApis());
  if (device_types.empty()) {
    return absl::FailedPreconditionError("PJRT_Api is not initialized.");
  }
  if (device_types.size() > 1) {
    return absl::FailedPreconditionError(
        "More than one device type registered. Please use "
        "GetCApiPhaseCompiler(absl::string_view device_type) "
        "instead.");
  }
  return GetCApiPhaseCompiler(device_types[0]);
}

}  // namespace xla
