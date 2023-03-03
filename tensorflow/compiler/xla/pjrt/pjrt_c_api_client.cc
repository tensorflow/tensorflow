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

#include "tensorflow/compiler/xla/pjrt/pjrt_c_api_client.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/string_view.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"
// TODO(skyewm): remove when everything goes through C API
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_api.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_initializer_helper.h"  // NOLINT(unused-includes): required for tensorflow::tpu::FindAndLoadTpuLibrary
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/status.h"

// TODO(b/238999986): Remove this when we have decomposed shape.
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_conversions.h"

namespace xla {

bool kPjRtCApiBypass = false;

// Helper macros

// Return error status if not success and frees the PJRT_Error returned by
// `expr`.
#define RETURN_STATUS_IF_ERROR(expr, c_api)                             \
  do {                                                                  \
    PJRT_Error* error = (expr);                                         \
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> _error(        \
        error, pjrt::MakeErrorDeleter(c_api));                          \
    xla::Status _status = pjrt::PjrtErrorToStatus(_error.get(), c_api); \
    if (!_status.ok()) {                                                \
      return _status;                                                   \
    }                                                                   \
  } while (false)

// ---------------------------------- Client -----------------------------------

PjRtCApiClient::PjRtCApiClient(const PJRT_Api* c_api, PJRT_Client* c_client)
    : c_api_(c_api),
      c_client_(std::unique_ptr<PJRT_Client, ::pjrt::PJRT_ClientDeleter>(
          c_client, ::pjrt::MakeClientDeleter(c_api))),
      // Example platform version string:
      //   PJRT C API
      //   TFRT TPU v2
      //   Built on Mar 4 2021 15:25:57 (1614900357) cl/360760169
      platform_version_(absl::StrCat(
          "PJRT C API\n", ::pjrt::GetPlatformVersion(c_client, c_api))) {
  if (kPjRtCApiBypass) {
    wrapped_ = c_client_->client.get();
  }

  InitDevices();
  LOG(INFO) << "PjRtCApiClient created.";
}

void PjRtCApiClient::InitDevices() {
  PJRT_Client_Devices_Args devices_args;
  devices_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
  devices_args.priv = nullptr;
  devices_args.client = c_client_.get();

  pjrt::LogFatalIfPjrtError(c_api_->PJRT_Client_Devices(&devices_args), c_api_);

  const size_t n = devices_args.num_devices;
  c_to_cpp_device_map_.reserve(n);
  owned_devices_.reserve(n);
  devices_.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    PJRT_Device* device = devices_args.devices[i];
    std::unique_ptr<PjRtCApiDevice>& cpp_device = owned_devices_.emplace_back(
        std::make_unique<PjRtCApiDevice>(device, this));
    devices_.push_back(cpp_device.get());
    c_to_cpp_device_map_[device] = cpp_device.get();
  }

  PJRT_Client_AddressableDevices_Args address_args;
  address_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  address_args.priv = nullptr;
  address_args.client = c_client_.get();

  pjrt::LogFatalIfPjrtError(
      c_api_->PJRT_Client_AddressableDevices(&address_args), c_api_);

  const size_t m = address_args.num_addressable_devices;
  addressable_devices_.reserve(m);

  for (size_t i = 0; i < m; ++i) {
    PJRT_Device* c_device = address_args.addressable_devices[i];
    addressable_devices_.push_back(GetCppDevice(c_device));
  }
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

absl::string_view PjRtCApiClient::platform_name() const {
  PJRT_Client_PlatformName_Args args;
  args.client = c_client_.get();
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.priv = nullptr;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_Client_PlatformName(&args), c_api_);

  absl::string_view platform_name(args.platform_name, args.platform_name_size);
  return platform_name;
}

int PjRtCApiClient::process_index() const {
  PJRT_Client_ProcessIndex_Args process_index_args;
  process_index_args.struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE;
  process_index_args.priv = nullptr;
  process_index_args.client = c_client_.get();
  pjrt::LogFatalIfPjrtError(
      c_api_->PJRT_Client_ProcessIndex(&process_index_args), c_api_);

  return process_index_args.process_index;
}

absl::string_view PjRtCApiClient::platform_version() const {
  return platform_version_;
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

StatusOr<DeviceAssignment> PjRtCApiClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  PJRT_Client_DefaultDeviceAssignment_Args args;
  args.struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.client = c_client_.get();
  args.num_replicas = num_replicas;
  args.num_partitions = num_partitions;
  std::vector<int> assignment_buffer(num_replicas * num_partitions);
  args.default_assignment_size = assignment_buffer.size();
  args.default_assignment = assignment_buffer.data();
  RETURN_STATUS_IF_ERROR(c_api_->PJRT_Client_DefaultDeviceAssignment(&args),
                         c_api_);
  absl::Span<const int> param{args.default_assignment,
                              args.default_assignment_size};
  return CalculateDefaultAssignment(args.num_replicas, args.num_partitions,
                                    param);
}

StatusOr<std::optional<std::string>> PjRtCApiClient::ExecutableFingerprint(
    const PjRtLoadedExecutable& executable) const {
  return {std::nullopt};
}

StatusOr<PjRtDevice*> PjRtCApiClient::LookupDevice(int device_id) const {
  PJRT_Client_LookupDevice_Args args;
  args.struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.client = c_client_.get();
  args.id = device_id;
  RETURN_STATUS_IF_ERROR(c_api_->PJRT_Client_LookupDevice(&args), c_api_);
  return GetCppDevice(args.device);
}

StatusOr<PjRtDevice*> PjRtCApiClient::LookupAddressableDevice(
    int local_hardware_id) const {
  PJRT_Client_LookupAddressableDevice_Args args;
  args.struct_size = PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.client = c_client_.get();
  args.local_hardware_id = local_hardware_id;
  RETURN_STATUS_IF_ERROR(c_api_->PJRT_Client_LookupAddressableDevice(&args),
                         c_api_);
  return GetCppDevice(args.addressable_device);
}

// Initializes `PJRT_Client_Compile_Args`, which will be used to call
// API PJRT_Client_Compile().
static StatusOr<std::unique_ptr<PjRtLoadedExecutable>> InitializeArgsAndCompile(
    PjRtCApiClient* api_client, const PJRT_Api* c_api, PJRT_Client* client,
    const CompileOptions& options, const std::string& code,
    const std::string& format) {
  PJRT_Client_Compile_Args args;
  args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.client = client;
  TF_ASSIGN_OR_RETURN(const CompileOptionsProto options_proto,
                      options.ToProto());
  std::string options_str = options_proto.SerializeAsString();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  PJRT_Program program;
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.priv = nullptr;
  program.code = const_cast<char*>(code.c_str());
  program.code_size = code.size();
  program.format = format.c_str();
  program.format_size = format.size();
  args.program = &program;

  RETURN_STATUS_IF_ERROR(c_api->PJRT_Client_Compile(&args), c_api);
  std::unique_ptr<PjRtLoadedExecutable> ret =
      std::make_unique<PjRtCApiLoadedExecutable>(api_client, args.executable);
  return ret;
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> PjRtCApiClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  std::string module_str = computation.proto().SerializeAsString();
  std::string format(pjrt::kHloFormat);
  return InitializeArgsAndCompile(this, c_api_, c_client_.get(), options,
                                  module_str, format);
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> PjRtCApiClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  std::string module_bytecode;
  {
    llvm::raw_string_ostream os(module_bytecode);
    mlir::writeBytecodeToFile(module, os);
  }
  std::string format(pjrt::kMlirFormat);
  return InitializeArgsAndCompile(this, c_api_, c_client_.get(), options,
                                  module_bytecode, format);
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCApiClient::DeserializeExecutable(absl::string_view serialized,
                                      std::optional<CompileOptions> options) {
  PJRT_Executable_DeserializeAndLoad_Args des_args;

  des_args.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
  des_args.priv = nullptr;
  des_args.client = c_client_.get();
  des_args.serialized_executable = serialized.data();
  des_args.serialized_executable_size = serialized.length();

  const PJRT_Api* api = pjrt_c_api();

  RETURN_STATUS_IF_ERROR(api->PJRT_Executable_DeserializeAndLoad(&des_args),
                         api);
  PJRT_LoadedExecutable* c_exec = des_args.loaded_executable;
  CHECK(c_exec != nullptr);
  return std::unique_ptr<PjRtLoadedExecutable>(
      std::make_unique<PjRtCApiLoadedExecutable>(this, c_exec));
}

StatusOr<std::uintptr_t> PjRtCApiClient::UnsafeBufferPointer(
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
  args.priv = nullptr;
  args.buffer =
      tensorflow::down_cast<const PjRtCApiBuffer*>(buffer)->c_buffer();

  RETURN_STATUS_IF_ERROR(c_api_->PJRT_Buffer_UnsafePointer(&args), c_api_);

  return args.buffer_pointer;
}

StatusOr<std::unique_ptr<PjRtBuffer>> PjRtCApiClient::WrapBuffer(
    StatusOr<std::unique_ptr<PjRtBuffer>> to_wrap) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> buffer, std::move(to_wrap));
  return std::unique_ptr<PjRtBuffer>(std::make_unique<PjRtCApiBuffer>(
      this, new PJRT_Buffer{std::move(buffer), pjrt_c_client()}));
}

StatusOr<std::unique_ptr<PjRtBuffer>> PjRtCApiClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    std::function<void()> on_done_with_host_buffer, PjRtDevice* device) {
  if (host_buffer_semantics != HostBufferSemantics::kImmutableOnlyDuringCall &&
      host_buffer_semantics != HostBufferSemantics::kZeroCopy &&
      host_buffer_semantics !=
          HostBufferSemantics::kImmutableUntilTransferCompletes) {
    return Unimplemented(
        "PJRT C API does not support HostBufferSemantics other than "
        "HostBufferSemantics::kImmutableOnlyDuringCall, "
        "HostBufferSemantics::kZeroCopy and "
        "HostBufferSemantics::kImmutableUntilTransferCompletes.");
  }

  PJRT_Client_BufferFromHostBuffer_Args args;
  args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  args.priv = nullptr;
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
  args.host_buffer_semantics =
      ::pjrt::ConvertToPjRtHostBufferSemantics(host_buffer_semantics);
  args.device = tensorflow::down_cast<PjRtCApiDevice*>(device)->c_device();

  RETURN_STATUS_IF_ERROR(c_api_->PJRT_Client_BufferFromHostBuffer(&args),
                         c_api_);

  auto buffer = std::unique_ptr<PjRtBuffer>(
      std::make_unique<PjRtCApiBuffer>(this, args.buffer));

  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event(
      args.done_with_host_buffer, ::pjrt::MakeEventDeleter(c_api_));

  if (on_done_with_host_buffer) {
    PJRT_Event_OnReady_Args event_args;
    event_args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
    event_args.priv = nullptr;
    event_args.event = event.get();
    event_args.user_arg = new std::function<void(PJRT_Error*)>(
        [on_done_with_host_buffer = std::move(on_done_with_host_buffer),
         c_api = c_api_](PJRT_Error* error) {
          if (error) {
            ::pjrt::MakeErrorDeleter(c_api)(error);
          }
          on_done_with_host_buffer();
        });
    event_args.callback = [](PJRT_Error* error, void* args) {
      std::function<void(PJRT_Error*)>* on_done_with_host_buffer =
          reinterpret_cast<std::function<void(PJRT_Error*)>*>(args);
      (*on_done_with_host_buffer)(error);
      delete on_done_with_host_buffer;
    };

    RETURN_STATUS_IF_ERROR(c_api_->PJRT_Event_OnReady(&event_args), c_api_);
  }

  return buffer;
}

const PJRT_Api* PjRtCApiClient::pjrt_c_api() const { return c_api_; }

// --------------------------------- Devices -----------------------------------

PjRtCApiDevice::PjRtCApiDevice(PJRT_Device* device, PjRtCApiClient* client)
    : client_(client), device_(device) {
  if (kPjRtCApiBypass) {
    wrapped_ = device_->device;
  }
  InitAttributes();
}

PjRtClient* PjRtCApiDevice::client() const { return client_; }

int PjRtCApiDevice::id() const {
  PJRT_Device_Id_Args args;
  args.struct_size = PJRT_Device_Id_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.device = device_;
  const PJRT_Api* api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_Id(&args), api);
  return args.id;
}

int PjRtCApiDevice::process_index() const {
  PJRT_Device_ProcessIndex_Args args;
  args.struct_size = PJRT_Device_ProcessIndex_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.device = device_;
  const PJRT_Api* api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_ProcessIndex(&args), api);
  return args.process_index;
}

bool PjRtCApiDevice::IsAddressable() const {
  PJRT_Device_IsAddressable_Args args;
  args.struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.device = device_;
  const PJRT_Api* api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_IsAddressable(&args), api);
  return args.is_addressable;
}

void PjRtCApiDevice::InitAttributes() {
  attributes_ = {};
  PJRT_Device_Attributes_Args args;
  args.struct_size = PJRT_Device_Attributes_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.device = device_;
  const PJRT_Api* api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_Attributes(&args), api);

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
        LOG(FATAL) << "PJRT_Device_Attributes() returned attribute '"
                   << attribute_name << "' with unsupported type "
                   << attribute.type << " to PjRtCApiDevice::InitAttributes()";
        break;
      }
    }
  }
}

const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
PjRtCApiDevice::Attributes() const {
  return attributes_;
}

absl::string_view PjRtCApiDevice::device_kind() const {
  PJRT_Device_Kind_Args args;
  args.struct_size = PJRT_Device_Kind_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.device = device_;

  const PJRT_Api* c_api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(c_api->PJRT_Device_Kind(&args), c_api);

  absl::string_view device_kind(args.device_kind, args.device_kind_size);
  return device_kind;
}

int PjRtCApiDevice::local_hardware_id() const {
  PJRT_Device_LocalHardwareId_Args args;
  args.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.device = device_;
  const PJRT_Api* api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_LocalHardwareId(&args), api);
  return args.local_hardware_id;
}

absl::string_view PjRtCApiDevice::DebugString() const {
  PJRT_Device_DebugString_Args args;
  args.struct_size = PJRT_Device_DebugString_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.device = device_;
  const PJRT_Api* c_api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(c_api->PJRT_Device_DebugString(&args), c_api);
  absl::string_view debug_string(args.debug_string, args.debug_string_size);
  return debug_string;
}

absl::string_view PjRtCApiDevice::ToString() const {
  PJRT_Device_ToString_Args args;
  args.struct_size = PJRT_Device_ToString_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.device = device_;
  const PJRT_Api* c_api = client_->pjrt_c_api();
  pjrt::LogFatalIfPjrtError(c_api->PJRT_Device_ToString(&args), c_api);
  absl::string_view to_string(args.to_string, args.to_string_size);
  return to_string;
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
  args.priv = nullptr;
  pjrt::LogFatalIfPjrtError(c_api->PJRT_Executable_Name(&args), c_api);

  return absl::string_view(args.executable_name, args.executable_name_size);
}

int PjRtCApiExecutable::num_replicas() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_NumReplicas_Args args;
  args.executable = executable;
  args.struct_size = PJRT_Executable_NumReplicas_Args_STRUCT_SIZE;
  args.priv = nullptr;
  pjrt::LogFatalIfPjrtError(c_api->PJRT_Executable_NumReplicas(&args), c_api);

  return args.num_replicas;
}

int PjRtCApiExecutable::num_partitions() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_NumPartitions_Args args;
  args.executable = executable;
  args.struct_size = PJRT_Executable_NumPartitions_Args_STRUCT_SIZE;
  args.priv = nullptr;
  pjrt::LogFatalIfPjrtError(c_api->PJRT_Executable_NumPartitions(&args), c_api);

  return args.num_partitions;
}

int64_t PjRtCApiExecutable::SizeOfGeneratedCodeInBytes() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_SizeOfGeneratedCodeInBytes_Args args;
  args.struct_size =
      PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.executable = executable;

  pjrt::LogFatalIfPjrtError(
      c_api->PJRT_Executable_SizeOfGeneratedCodeInBytes(&args), c_api);
  return args.size_in_bytes;
}

StatusOr<std::vector<std::shared_ptr<HloModule>>>
PjRtCApiExecutable::GetHloModules() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_OptimizedProgram_Args args;
  args.struct_size = PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.executable = executable;
  PJRT_Program program;
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.priv = nullptr;
  program.code = nullptr;
  args.program = &program;

  RETURN_STATUS_IF_ERROR(c_api->PJRT_Executable_OptimizedProgram(&args), c_api);

  constexpr size_t TWO_GIBIBYTES = 2ull * 1024 * 1024 * 1024;
  const size_t code_size = args.program->code_size;
  CHECK(code_size < TWO_GIBIBYTES);
  std::string code(code_size, ' ');
  args.program->code = code.data();
  RETURN_STATUS_IF_ERROR(c_api->PJRT_Executable_OptimizedProgram(&args), c_api);

  absl::string_view program_format(program.format, program.format_size);
  if (program_format != ::pjrt::kHloWithConfigFormat) {
    return xla::InternalError(
        "expected program format `hlo_with_config` but got %s", program_format);
  }

  HloModuleProtoWithConfig proto;
  proto.ParseFromString(code);
  std::vector<std::shared_ptr<HloModule>> out;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProtoWithConfig(proto));
  out.push_back(std::move(module));
  return out;
}

StatusOr<std::string> PjRtCApiExecutable::SerializeExecutable() const {
  auto* c_api = pjrt_c_api();
  auto* executable = c_executable();
  PJRT_Executable_Serialize_Args ser_args;
  ser_args.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE;
  ser_args.priv = nullptr;
  ser_args.executable = executable;
  ser_args.serialized_executable = nullptr;

  RETURN_STATUS_IF_ERROR(c_api->PJRT_Executable_Serialize(&ser_args), c_api);
  PJRT_SerializedExecutable* c_serialized_exec = ser_args.serialized_executable;
  std::unique_ptr<PJRT_SerializedExecutable,
                  ::pjrt::PJRT_SerializedExecutableDeleter>
      serialized_executable(c_serialized_exec,
                            ::pjrt::MakeSerializedExecutableDeleter(c_api));

  PJRT_SerializedExecutable_Data_Args data_args;
  data_args.struct_size = PJRT_SerializedExecutable_Data_Args_STRUCT_SIZE;
  data_args.priv = nullptr;
  data_args.serialized_executable = c_serialized_exec;
  data_args.data = nullptr;
  data_args.data_size = 0;

  RETURN_STATUS_IF_ERROR(c_api->PJRT_SerializedExecutable_Data(&data_args),
                         c_api);

  return std::string(data_args.data, data_args.data_size);
}

// ------------------------ Loaded Executables ---------------------------------

PjRtCApiLoadedExecutable::PjRtCApiLoadedExecutable(
    PjRtCApiClient* client, PJRT_LoadedExecutable* executable)
    : client_(client),
      loaded_executable_(executable, ::pjrt::MakeLoadedExecutableDeleter(
                                         client->pjrt_c_api())) {
  PJRT_LoadedExecutable_GetExecutable_Args args;
  args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.loaded_executable = c_loaded_executable();
  args.executable = nullptr;
  pjrt::LogFatalIfPjrtError(
      pjrt_c_api()->PJRT_LoadedExecutable_GetExecutable(&args), pjrt_c_api());
  executable_ =
      std::make_unique<PjRtCApiExecutable>(pjrt_c_api(), args.executable);
  InitDevices();
}

void PjRtCApiLoadedExecutable::InitDevices() {
  PJRT_LoadedExecutable_AddressableDevices_Args args;
  args.struct_size = PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
  args.priv = nullptr;
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
Convert2DCBuffersToCppBuffers(PJRT_Buffer*** c_lists, size_t outer_size,
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
  *send_callback_function = [&send_callback = cpp_send_callback.callback](
                                PJRT_TransferMetadata* metadata,
                                PJRT_Chunk* chunk, size_t total_size_in_bytes,
                                bool done) -> bool {
    // TODO(b/238999986) use `shape` with full information when available.
    xla::Shape shape = metadata->device_shape;
    xla::Status status = send_callback(
        xla::PjRtTransferMetadata{shape},
        // TODO(b/263390038) use PJRT_Chunk C API
        // instead of accessing the opaque type's field directly.
        xla::PjRtChunk(std::move(chunk->chunk)), total_size_in_bytes, done);
    if (!status.ok()) {
      return false;
    }
    return true;
  };
  return PJRT_SendCallbackInfo{
      /*channel_id=*/cpp_send_callback.channel_id,
      /*user_arg=*/send_callback_function,
      /*send_callback=*/
      [](PJRT_TransferMetadata* metadata, PJRT_Chunk* chunk,
         size_t total_size_in_bytes, bool done, void* user_arg) -> bool {
        // PJRT_SendCallback, `send_callback` is internal C interface callback
        // representation that cpatures the client C++ callback in void*
        // `user_arg` and reinterprets in the lower-level runtime for execution.
        // `user_arg` captures `send_callback_function` which is
        // SendCallbackFunction*.
        PjRtCApiLoadedExecutable::SendCallbackFunction* send_callback =
            reinterpret_cast<PjRtCApiLoadedExecutable::SendCallbackFunction*>(
                user_arg);
        return (*send_callback)(metadata, chunk, total_size_in_bytes, done);
      }};
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
    const xla::RecvCallback& cpp_recv_callback,
    PjRtCApiLoadedExecutable::RecvCallbackFunction* recv_callback_function) {
  *recv_callback_function = [&recv_callback = cpp_recv_callback.callback](
                                PJRT_TransferMetadata* metadata,
                                PJRT_CopyToDeviceStream* stream) {
    // TODO(b/238999986) use `shape` with full information when available.
    xla::Shape shape = metadata->device_shape;
    recv_callback(xla::PjRtTransferMetadata{shape},
                  // TODO(b/263390934) use PJRT_CopyToDeviceStream C API
                  // instead of accessing the opaque type's field directly.
                  std::move(stream->stream));
  };
  return PJRT_RecvCallbackInfo{
      /*channel_id=*/cpp_recv_callback.channel_id,
      /*user_arg=*/recv_callback_function,
      /*recv_callback=*/
      [](PJRT_TransferMetadata* metadata, PJRT_CopyToDeviceStream* stream,
         void* user_arg) {
        // PJRT_RecvCallback, `recv_callback` is internal C interface callback
        // representation that cpatures the client C++ callback in void*
        // `user_arg` and reinterprets in the lower-level runtime for execution.
        // `user_arg` captures `recv_callback_function` which is
        // RecvCallbackFunction*.
        PjRtCApiLoadedExecutable::RecvCallbackFunction* recv_callback =
            reinterpret_cast<PjRtCApiLoadedExecutable::RecvCallbackFunction*>(
                user_arg);
        (*recv_callback)(metadata, stream);
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
          cpp_callback, &recv_callback_functions[func_count++]));
    }
  }
}

xla::StatusOr<PJRT_LoadedExecutable_Execute_Args>
PjRtCApiLoadedExecutable::GetCommonExecuteArgs(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options, PJRT_ExecuteOptions& c_options,
    std::vector<std::vector<PJRT_Buffer*>>& c_argument_lists_storage,
    std::vector<PJRT_Buffer**>& c_arguments,
    std::vector<std::vector<PJRT_Buffer*>>& c_output_lists_storage,
    std::vector<PJRT_Buffer**>& c_output_lists,
    std::optional<std::vector<PJRT_Event*>>& device_complete_events,
    SendRecvCallbackData& callback_data) {
  PJRT_LoadedExecutable_Execute_Args args;
  args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.executable = c_loaded_executable();
  args.options = &c_options;
  args.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  args.options->launch_id = options.launch_id;
  args.num_devices = argument_handles.size();
  CHECK_GT(args.num_devices, 0);
  args.num_args = argument_handles[0].size();
  if (device_complete_events.has_value() || !options.send_callbacks.empty() ||
      !options.recv_callbacks.empty()) {
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

  // Allocates memory for output. `c_buffer_lists_storage` and `c_buffer_lists`
  // needs to stay alive during the call of `PJRT_LoadedExecutable_Execute`.

  PJRT_Executable_NumOutputs_Args numoutputs_args;
  numoutputs_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  numoutputs_args.priv = nullptr;
  numoutputs_args.executable = c_executable();
  RETURN_STATUS_IF_ERROR(
      pjrt_c_api()->PJRT_Executable_NumOutputs(&numoutputs_args), pjrt_c_api());
  size_t outer_size = args.num_devices;
  size_t inner_size = numoutputs_args.num_outputs;
  c_output_lists_storage.resize(outer_size);
  c_output_lists.resize(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    c_output_lists_storage[i].resize(inner_size);
    c_output_lists[i] = c_output_lists_storage[i].data();
  }
  args.output_lists = c_output_lists.data();

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
    CppRecvCallbackListsToC(options.recv_callbacks,
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

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
PjRtCApiLoadedExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  std::vector<std::vector<PJRT_Buffer*>> c_argument_lists_storage;
  std::vector<std::vector<PJRT_Buffer*>> c_output_lists_storage;
  std::vector<PJRT_Buffer**> c_output_lists;
  PJRT_ExecuteOptions c_options;
  std::vector<PJRT_Buffer**> c_arguments;
  std::optional<std::vector<PJRT_Event*>> device_complete_events;
  if (returned_futures.has_value()) {
    device_complete_events.emplace();
  }

  auto callback_data = std::make_shared<SendRecvCallbackData>();
  TF_ASSIGN_OR_RETURN(
      PJRT_LoadedExecutable_Execute_Args args,
      GetCommonExecuteArgs(argument_handles, options, c_options,
                           c_argument_lists_storage, c_arguments,
                           c_output_lists_storage, c_output_lists,
                           device_complete_events, *callback_data));

  args.execute_device = nullptr;

  RETURN_STATUS_IF_ERROR(pjrt_c_api()->PJRT_LoadedExecutable_Execute(&args),
                         pjrt_c_api());

  if (device_complete_events.has_value()) {
    std::vector<PjRtFuture<Status>> device_complete_futures;
    device_complete_futures.resize(args.num_devices);
    for (int i = 0; i < device_complete_futures.size(); ++i) {
      device_complete_futures[i] = pjrt::ConvertCEventToCppFuture(
          args.device_complete_events[i], pjrt_c_api());
      if (!callback_data->c_send_callbacks.empty() ||
          !callback_data->c_recv_callbacks.empty()) {
        device_complete_futures[i].OnReady([callback_data](xla::Status status) {
          // Keeps C callbacks alive until execution completes on all
          // devices.
        });
      }
    }

    if (returned_futures.has_value()) {
      returned_futures->resize(device_complete_futures.size());
      *returned_futures = std::move(device_complete_futures);
    }
  }

  return Convert2DCBuffersToCppBuffers(args.output_lists, args.num_devices,
                                       c_output_lists_storage[0].size(),
                                       client_);
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  if (!options.send_callbacks.empty() || !options.recv_callbacks.empty()) {
    return Status(tensorflow::error::UNIMPLEMENTED,
                  "Send/recv callbacks not implemented for "
                  "PjRtCApiLoadedExecutable::ExecuteWithSingleDevice.");
  }

  std::vector<std::vector<PjRtBuffer*>> argument_handles_vec = {
      {argument_handles.begin(), argument_handles.end()}};

  std::vector<std::vector<PJRT_Buffer*>> c_argument_lists_storage;
  std::vector<std::vector<PJRT_Buffer*>> c_output_lists_storage;
  std::vector<PJRT_Buffer**> c_output_lists;
  PJRT_ExecuteOptions c_options;
  std::vector<PJRT_Buffer**> c_arguments;
  std::optional<std::vector<PJRT_Event*>> device_complete_events;
  if (fill_future) {
    device_complete_events.emplace();
  }

  auto callback_data = std::make_shared<SendRecvCallbackData>();
  TF_ASSIGN_OR_RETURN(
      PJRT_LoadedExecutable_Execute_Args args,
      GetCommonExecuteArgs(argument_handles_vec, options, c_options,
                           c_argument_lists_storage, c_arguments,
                           c_output_lists_storage, c_output_lists,
                           device_complete_events, *callback_data));

  args.execute_device =
      tensorflow::down_cast<PjRtCApiDevice*>(device)->c_device();

  RETURN_STATUS_IF_ERROR(pjrt_c_api()->PJRT_LoadedExecutable_Execute(&args),
                         pjrt_c_api());

  if (fill_future) {
    *returned_future = pjrt::ConvertCEventToCppFuture(
        args.device_complete_events[0], pjrt_c_api());
  }
  return std::move(Convert2DCBuffersToCppBuffers(
      args.output_lists, args.num_devices, c_output_lists_storage[0].size(),
      client_)[0]);
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiLoadedExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  return ExecuteWithSingleDevice(argument_handles, device, options,
                                 returned_future, fill_future);
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiLoadedExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  return ExecuteWithSingleDevice(argument_handles, device, options,
                                 returned_future, fill_future);
}

void PjRtCApiLoadedExecutable::Delete() {
  PJRT_LoadedExecutable_Delete_Args args;
  args.struct_size = PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.executable = c_loaded_executable();
  const PJRT_Api* c_api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(c_api->PJRT_LoadedExecutable_Delete(&args), c_api);
}

bool PjRtCApiLoadedExecutable::IsDeleted() {
  PJRT_LoadedExecutable_IsDeleted_Args args;
  args.struct_size = PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.executable = c_loaded_executable();

  const PJRT_Api* c_api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(c_api->PJRT_LoadedExecutable_IsDeleted(&args),
                            c_api);
  return args.is_deleted;
}

StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
PjRtCApiLoadedExecutable::GetCostAnalysis() const {
  // Initialize function call args
  PJRT_LoadedExecutable_GetCostAnalysis_Args args;
  args.struct_size = PJRT_LoadedExecutable_GetCostAnalysis_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.executable = c_loaded_executable();

  // Make PJRT C API call
  const PJRT_Api* c_api = pjrt_c_api();
  RETURN_STATUS_IF_ERROR(c_api->PJRT_LoadedExecutable_GetCostAnalysis(&args),
                         c_api);

  // Copy returned properties to output map
  return pjrt::ConvertFromPjRtNamedValueList(args.properties,
                                             args.num_properties);
}

// ---------------------------------- Buffers ----------------------------------

PjRtCApiBuffer::PjRtCApiBuffer(PjRtCApiClient* client, PJRT_Buffer* buffer)
    : client_(client),
      buffer_(buffer, ::pjrt::MakeBufferDeleter(client->pjrt_c_api())),
      readiness_event_(nullptr,
                       ::pjrt::MakeEventDeleter(client->pjrt_c_api())) {
  set_shape();
}

const Shape& PjRtCApiBuffer::on_device_shape() const {
  CHECK(shape_.has_value())
      << "Shape should be initialized in PjRtCApiBuffer constructor.";
  return shape_.value();
}

void PjRtCApiBuffer::set_shape() {
  PJRT_Buffer_OnDeviceTrimmedShape_Args args;
  args.struct_size = PJRT_Buffer_OnDeviceTrimmedShape_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.buffer = buffer_.get();

  pjrt::LogFatalIfPjrtError(
      client_->pjrt_c_api()->PJRT_Buffer_OnDeviceTrimmedShape(&args),
      client_->pjrt_c_api());

  xla::PrimitiveType element_type =
      static_cast<xla::PrimitiveType>(args.element_type);

  CHECK_NE(element_type, xla::PrimitiveType::TUPLE);

  absl::Span<const int64_t> dims = ApiConverter::MakeSpan(args.dimensions);
  absl::Span<const bool> dynamic_dims =
      ApiConverter::MakeSpan(args.dynamic_dimensions);

  Shape trimmed_shape = Shape(element_type, dims, dynamic_dims, {});

  if (args.has_layout) {
    *(trimmed_shape.mutable_layout()) = ApiConverter::FromC(&args.layout);
  }

  shape_ = trimmed_shape;

  // TODO(amangu): Refactor the deletion.
  if (args.dimensions.size > TPU_C_API_MAX_INLINED) {
    delete[] args.dimensions.heap;
  }

  if (args.dynamic_dimensions.size > TPU_C_API_MAX_INLINED) {
    delete[] args.dynamic_dimensions.heap;
  }

  if (args.has_layout) {
    if (args.layout.minor_to_major.size > TPU_C_API_MAX_INLINED) {
      delete[] args.layout.minor_to_major.heap;
    }

    if (args.layout.tiles.size > TPU_C_API_MAX_INLINED) {
      delete[] args.layout.tiles.heap;
    }
  }
}

PjRtFuture<Status> PjRtCApiBuffer::ToLiteral(MutableLiteralBase* literal) {
  PJRT_Buffer_ToHostBuffer_Args args;
  args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.src = buffer_.get();

  const xla::Shape& shape = literal->shape();

  if (!shape.IsArray()) {
    return PjRtFuture<Status>(
        Unimplemented("PjRtCApiBuffer::ToLiteral: Shapes other than array are"
                      "not supported."));
  }

  args.dst_size = ShapeUtil::ByteSizeOfElements(shape);
  args.dst = literal->untyped_data();
  const PJRT_Api* api = pjrt_c_api();

  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> error{
      pjrt_c_api()->PJRT_Buffer_ToHostBuffer(&args),
      ::pjrt::MakeErrorDeleter(api)};

  if (error != nullptr) {
    xla::Status s = ::pjrt::PjrtErrorToStatus(error.get(), api);
    return PjRtFuture<Status>(s);
  }

  return pjrt::ConvertCEventToCppFuture(args.event, api);
}

StatusOr<size_t> PjRtCApiBuffer::GetOnDeviceSizeInBytes() const {
  PJRT_Buffer_OnDeviceSizeInBytes_Args args;
  args.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.buffer = buffer_.get();
  RETURN_STATUS_IF_ERROR(
      client_->pjrt_c_api()->PJRT_Buffer_OnDeviceSizeInBytes(&args),
      client_->pjrt_c_api());

  return args.on_device_size_in_bytes;
}

PjRtDevice* PjRtCApiBuffer::device() const {
  PJRT_Buffer_Device_Args args;
  args.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.buffer = buffer_.get();
  const PJRT_Api* api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_Device(&args), api);
  return client_->GetCppDevice(args.device);
}

void PjRtCApiBuffer::Delete() {
  PJRT_Buffer_Delete_Args args;
  args.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.buffer = buffer_.get();
  const PJRT_Api* api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_Delete(&args), api);
}

bool PjRtCApiBuffer::IsDeleted() {
  PJRT_Buffer_IsDeleted_Args args;
  args.struct_size = PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.buffer = buffer_.get();
  const PJRT_Api* api = pjrt_c_api();
  pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_IsDeleted(&args), api);
  return args.is_deleted;
}

StatusOr<std::unique_ptr<PjRtBuffer>> PjRtCApiBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  if (dst_device->client() == client_) {
    PJRT_Buffer_CopyToDevice_Args args;
    args.struct_size = PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.buffer = buffer_.get();
    args.dst_device =
        tensorflow::down_cast<PjRtCApiDevice*>(dst_device)->c_device();
    const PJRT_Api* api = pjrt_c_api();
    RETURN_STATUS_IF_ERROR(api->PJRT_Buffer_CopyToDevice(&args), api);
    return std::unique_ptr<PjRtBuffer>(
        std::make_unique<PjRtCApiBuffer>(client_, args.dst_buffer));
  } else {
    // Copy across PjRtClients by copying through host
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
    absl::InlinedVector<int64_t, 4> byte_strides(
        literal->shape().dimensions_size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
    // Avoid use-after-free on `literal` due to unsequenced move and use.
    Literal* literal_pointer = literal.get();
    return dst_device->client()->BufferFromHostBuffer(
        literal_pointer->untyped_data(),
        literal_pointer->shape().element_type(),
        literal_pointer->shape().dimensions(), byte_strides,
        PjRtClient::HostBufferSemantics::kZeroCopy,
        [literal{std::move(literal)}]() { /* frees literal */ }, dst_device);
  }
}

bool PjRtCApiBuffer::IsOnCpu() const {
  PJRT_Buffer_IsOnCpu_Args args;
  args.struct_size = PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE;
  args.priv = nullptr;
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
    args.priv = nullptr;
    args.buffer = buffer_.get();
    pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_ReadyEvent(&args), api);
    readiness_event_.reset(args.event);
  }
  return readiness_event_.get();
}

void PjRtCApiBuffer::MakePromiseTrackEvent() {
  CHECK(readiness_promise_ != nullptr);
  const PJRT_Api* api = pjrt_c_api();
  PJRT_Event_OnReady_Args args;
  args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.event = GetReadyEvent();
  args.user_arg = new std::function<void(PJRT_Error*)>(
      [promise = readiness_promise_, api](PJRT_Error* error) -> void {
        Status status = ::pjrt::PjrtErrorToStatus(error, api);
        promise->Set(status);
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

PjRtFuture<Status> PjRtCApiBuffer::GetReadyFuture() {
  if (readiness_promise_ == nullptr) {
    readiness_promise_ = std::make_shared<PjRtFuture<Status>::Promise>(
        PjRtFuture<Status>::CreatePromise());
    MakePromiseTrackEvent();
  }
  return PjRtFuture<Status>{*readiness_promise_};
}

// ------------------------------ Device Topology ------------------------------

PjRtCApiDeviceTopology::PjRtCApiDeviceTopology(const PJRT_Api* c_api,
                                               PJRT_DeviceTopology* c_topology)
    : compiler_(std::make_unique<PjRtCApiCompiler>(c_api)),
      c_api_(c_api),
      c_topology_(c_topology, ::pjrt::MakeDeviceTopologyDeleter(c_api)) {}

absl::string_view PjRtCApiDeviceTopology::platform_name() const {
  PJRT_DeviceTopology_PlatformName_Args args;
  args.topology = c_topology_.get();
  args.struct_size = PJRT_DeviceTopology_PlatformName_Args_STRUCT_SIZE;
  args.priv = nullptr;
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_DeviceTopology_PlatformName(&args),
                            c_api_);
  return absl::string_view(args.platform_name, args.platform_name_size);
}

absl::string_view PjRtCApiDeviceTopology::platform_version() const {
  PJRT_DeviceTopology_PlatformVersion_Args args;
  args.struct_size = PJRT_DeviceTopology_PlatformVersion_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.topology = c_topology_.get();
  pjrt::LogFatalIfPjrtError(c_api_->PJRT_DeviceTopology_PlatformVersion(&args),
                            c_api_);
  return absl::string_view(args.platform_version, args.platform_version_size);
}

// Initializes `PJRT_Compile_Args`, which will be used to call
// API PJRT_Compile().
static StatusOr<std::unique_ptr<PjRtExecutable>> InitializeArgsAndCompileAot(
    const PJRT_Api* c_api, PjRtClient* client, const CompileOptions& options,
    const PjRtDeviceTopology& topology, const std::string& code,
    const std::string& format) {
  PJRT_Compile_Args args;
  args.struct_size = PJRT_Compile_Args_STRUCT_SIZE;
  args.priv = nullptr;
  if (client == nullptr) {
    args.client = nullptr;
  } else {
    args.client =
        tensorflow::down_cast<PjRtCApiClient*>(client)->pjrt_c_client();
  }
  args.topology =
      tensorflow::down_cast<const PjRtCApiDeviceTopology*>(&topology)
          ->c_topology();
  TF_ASSIGN_OR_RETURN(const CompileOptionsProto options_proto,
                      options.ToProto());
  std::string options_str = options_proto.SerializeAsString();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  PJRT_Program program;
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.priv = nullptr;
  program.code = const_cast<char*>(code.c_str());
  program.code_size = code.size();
  program.format = format.c_str();
  program.format_size = format.size();
  args.program = &program;

  RETURN_STATUS_IF_ERROR(c_api->PJRT_Compile(&args), c_api);
  std::unique_ptr<PjRtExecutable> ret =
      std::make_unique<PjRtCApiExecutable>(c_api, args.executable);
  return ret;
}

StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCApiCompiler::Compile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtDeviceTopology& topology, PjRtClient* client) {
  std::string module_str = computation.proto().SerializeAsString();
  std::string format(pjrt::kHloFormat);
  return InitializeArgsAndCompileAot(c_api_, client, options, topology,
                                     module_str, format);
}

StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCApiCompiler::Compile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtDeviceTopology& topology, PjRtClient* client) {
  std::string module_bytecode;
  {
    llvm::raw_string_ostream os(module_bytecode);
    mlir::writeBytecodeToFile(module, os);
  }
  std::string format(pjrt::kMlirFormat);
  return InitializeArgsAndCompileAot(c_api_, client, options, topology,
                                     module_bytecode, format);
}

// -------------------------------- API access ---------------------------------

StatusOr<std::unique_ptr<PjRtClient>> GetCApiClient(
    absl::string_view device_type,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options) {
#if !defined(PLATFORM_GOOGLE) || defined(LIBTPU_STATIC)
  if (absl::AsciiStrToLower(device_type) == "tpu") {
    // TODO(b/261484192): handle device specific initialization.
    TF_RETURN_IF_ERROR(tensorflow::tpu::FindAndLoadTpuLibrary());
  }
#endif
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(device_type));
  if (c_api == nullptr) {
    return InternalError("PJRT C API is nullptr for %s", device_type);
  }

  PJRT_Client_Create_Args init_args;
  init_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  init_args.priv = nullptr;
  TF_ASSIGN_OR_RETURN(std::vector<PJRT_NamedValue> c_options,
                      pjrt::ConvertToPjRtNamedValueList(create_options));
  init_args.create_options = c_options.data();
  init_args.num_options = c_options.size();
  RETURN_STATUS_IF_ERROR(c_api->PJRT_Client_Create(&init_args), c_api);
  PJRT_Client* c_client = init_args.client;

  return std::unique_ptr<PjRtClient>(
      std::make_unique<PjRtCApiClient>(c_api, c_client));
}

StatusOr<std::unique_ptr<PjRtDeviceTopology>> GetCApiTopology(
    absl::string_view device_type) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(device_type));
  if (c_api == nullptr) {
    return InternalError("PJRT C API is nullptr for %s", device_type);
  }

  PJRT_DeviceTopology_Create_Args init_args;
  init_args.struct_size = PJRT_DeviceTopology_Create_Args_STRUCT_SIZE;
  init_args.priv = nullptr;
  RETURN_STATUS_IF_ERROR(c_api->PJRT_DeviceTopology_Create(&init_args), c_api);
  PJRT_DeviceTopology* c_topology = init_args.topology;
  return std::unique_ptr<PjRtDeviceTopology>(
      std::make_unique<PjRtCApiDeviceTopology>(c_api, c_topology));
}

}  // namespace xla
