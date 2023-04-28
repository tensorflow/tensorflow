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

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace pjrt {

const absl::string_view kHloFormat = "hlo";
const absl::string_view kMlirFormat = "mlir";
const absl::string_view kHloWithConfigFormat = "hlo_with_config";

PJRT_ClientDeleter MakeClientDeleter(const PJRT_Api* api) {
  return [api](PJRT_Client* client) -> void {
    PJRT_Client_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.client = client;

    PJRT_Error* error = api->PJRT_Client_Destroy(&destroy_args);
    // TODO(b/236710439): handle the error and remove this CHECK() call
    CHECK(error == nullptr);
  };
}

PJRT_ErrorDeleter MakeErrorDeleter(const PJRT_Api* api) {
  return [api](PJRT_Error* error) -> void {
    PJRT_Error_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.error = error;

    api->PJRT_Error_Destroy(&destroy_args);
  };
}

PJRT_BufferDeleter MakeBufferDeleter(const PJRT_Api* api) {
  return [api](PJRT_Buffer* buffer) -> void {
    PJRT_Buffer_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.buffer = buffer;

    pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_Destroy(&destroy_args), api);
  };
}

PJRT_ExecutableDeleter MakeExecutableDeleter(const PJRT_Api* api) {
  return [api](PJRT_Executable* executable) -> void {
    PJRT_Executable_Destroy_Args args;
    args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.executable = executable;
    pjrt::LogFatalIfPjrtError(api->PJRT_Executable_Destroy(&args), api);
  };
}

PJRT_LoadedExecutableDeleter MakeLoadedExecutableDeleter(const PJRT_Api* api) {
  return [api](PJRT_LoadedExecutable* executable) -> void {
    PJRT_LoadedExecutable_Destroy_Args args;
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.executable = executable;
    pjrt::LogFatalIfPjrtError(api->PJRT_LoadedExecutable_Destroy(&args), api);
  };
}

xla::Status PjrtErrorToStatus(const PJRT_Error* error, const PJRT_Api* api) {
  xla::Status status;
  if (error != nullptr) {
    status = xla::Status(PjrtErrorToStatusCode(error, api),
                         GetPjrtErrorMessage(error, api));
  }
  return status;
}

PJRT_TopologyDescriptionDeleter MakeTopologyDescriptionDeleter(
    const PJRT_Api* api) {
  return [api](PJRT_TopologyDescription* topology) -> void {
    PJRT_TopologyDescription_Destroy_Args destroy_args;
    destroy_args.struct_size =
        PJRT_TopologyDescription_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.topology = topology;

    pjrt::LogFatalIfPjrtError(
        api->PJRT_TopologyDescription_Destroy(&destroy_args), api);
  };
}

absl::StatusCode PjrtErrorToStatusCode(const PJRT_Error* error,
                                       const PJRT_Api* api) {
  PJRT_Error_GetCode_Args args;
  args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.error = error;
  api->PJRT_Error_GetCode(&args);
  PJRT_Error_Code code = args.code;
  switch (code) {
    case PJRT_Error_Code_CANCELLED:
    case PJRT_Error_Code_UNKNOWN:
    case PJRT_Error_Code_INVALID_ARGUMENT:
    case PJRT_Error_Code_DEADLINE_EXCEEDED:
    case PJRT_Error_Code_NOT_FOUND:
    case PJRT_Error_Code_ALREADY_EXISTS:
    case PJRT_Error_Code_PERMISSION_DENIED:
    case PJRT_Error_Code_RESOURCE_EXHAUSTED:
    case PJRT_Error_Code_FAILED_PRECONDITION:
    case PJRT_Error_Code_ABORTED:
    case PJRT_Error_Code_OUT_OF_RANGE:
    case PJRT_Error_Code_UNIMPLEMENTED:
    case PJRT_Error_Code_INTERNAL:
    case PJRT_Error_Code_UNAVAILABLE:
    case PJRT_Error_Code_DATA_LOSS:
    case PJRT_Error_Code_UNAUTHENTICATED:
      return static_cast<absl::StatusCode>(code);
  }
}

PJRT_Error_Code StatusCodeToPjrtErrorCode(absl::StatusCode code) {
  switch (static_cast<tsl::error::Code>(code)) {
    case tsl::error::CANCELLED:
    case tsl::error::UNKNOWN:
    case tsl::error::INVALID_ARGUMENT:
    case tsl::error::DEADLINE_EXCEEDED:
    case tsl::error::NOT_FOUND:
    case tsl::error::ALREADY_EXISTS:
    case tsl::error::PERMISSION_DENIED:
    case tsl::error::UNAUTHENTICATED:
    case tsl::error::RESOURCE_EXHAUSTED:
    case tsl::error::FAILED_PRECONDITION:
    case tsl::error::ABORTED:
    case tsl::error::OUT_OF_RANGE:
    case tsl::error::UNIMPLEMENTED:
    case tsl::error::INTERNAL:
    case tsl::error::UNAVAILABLE:
    case tsl::error::DATA_LOSS:
      return static_cast<PJRT_Error_Code>(code);
    case tsl::error::OK:
      CHECK(false) << "Status::OK() cannot be converted to PJRT_Error code, "
                      "use nullptr instead";
    case tensorflow::error::
        DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_:
      CHECK(false) << "got DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_"
                      "USE_DEFAULT_IN_SWITCH_INSTEAD_";
    case tensorflow::error::Code_INT_MIN_SENTINEL_DO_NOT_USE_:
      CHECK(false) << "got Code_INT_MIN_SENTINEL_DO_NOT_USE_";
    case tensorflow::error::Code_INT_MAX_SENTINEL_DO_NOT_USE_:
      CHECK(false) << "got Code_INT_MAX_SENTINEL_DO_NOT_USE_";
  }
}

absl::string_view GetPjrtErrorMessage(const PJRT_Error* error,
                                      const PJRT_Api* api) {
  PJRT_Error_Message_Args message_args;
  message_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  message_args.priv = nullptr;
  message_args.error = error;
  api->PJRT_Error_Message(&message_args);
  return absl::string_view(message_args.message, message_args.message_size);
}

void LogFatalIfPjrtError(PJRT_Error* error, const PJRT_Api* api) {
  std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> _error(
      error, MakeErrorDeleter(api));
  xla::Status _status = PjrtErrorToStatus(_error.get(), api);
  if (!_status.ok()) {
    LOG(FATAL) << "Unexpected error status " << _status.message();
  }
}

PJRT_EventDeleter MakeEventDeleter(const PJRT_Api* api) {
  CHECK(api != nullptr);
  return [api](PJRT_Event* managed) {
    PJRT_Event_Destroy_Args args;
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.event = managed;

    LogFatalIfPjrtError(api->PJRT_Event_Destroy(&args), api);
  };
}

PJRT_Buffer_Type ConvertToPjRtBufferType(xla::PrimitiveType type) {
  switch (type) {
    case xla::PrimitiveType::PRIMITIVE_TYPE_INVALID:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_INVALID;
    case xla::PrimitiveType::PRED:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_PRED;
    case xla::PrimitiveType::S8:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S8;
    case xla::PrimitiveType::S16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S16;
    case xla::PrimitiveType::S32:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S32;
    case xla::PrimitiveType::S64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S64;
    case xla::PrimitiveType::U8:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U8;
    case xla::PrimitiveType::U16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U16;
    case xla::PrimitiveType::U32:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U32;
    case xla::PrimitiveType::U64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U64;
    case xla::PrimitiveType::F16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F16;
    case xla::PrimitiveType::F32:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F32;
    case xla::PrimitiveType::BF16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_BF16;
    case xla::PrimitiveType::F64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F64;
    case xla::PrimitiveType::F8E5M2:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F8E5M2;
    case xla::PrimitiveType::F8E4M3FN:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F8E4M3FN;
    case xla::PrimitiveType::C64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_C64;
    case xla::PrimitiveType::C128:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_C128;
    default:
      CHECK(false)
          << "Element type of the shape is not supported in C API layer: "
          << xla::primitive_util::LowercasePrimitiveTypeName(type);
  }
}

xla::PrimitiveType ConvertFromPjRtBufferType(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type::PJRT_Buffer_Type_PRED:
      return xla::PrimitiveType::PRED;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S8:
      return xla::PrimitiveType::S8;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S16:
      return xla::PrimitiveType::S16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S32:
      return xla::PrimitiveType::S32;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S64:
      return xla::PrimitiveType::S64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U8:
      return xla::PrimitiveType::U8;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U16:
      return xla::PrimitiveType::U16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U32:
      return xla::PrimitiveType::U32;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U64:
      return xla::PrimitiveType::U64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F16:
      return xla::PrimitiveType::F16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F32:
      return xla::PrimitiveType::F32;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_BF16:
      return xla::PrimitiveType::BF16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F64:
      return xla::PrimitiveType::F64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_C64:
      return xla::PrimitiveType::C64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_C128:
      return xla::PrimitiveType::C128;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F8E5M2:
      return xla::PrimitiveType::F8E5M2;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F8E4M3FN:
      return xla::PrimitiveType::F8E4M3FN;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_INVALID:
      CHECK(false) << "Buffer type is not supported in C API layer.";
  }
}

const char* HostBufferSemanticsToString(
    xla::PjRtClient::HostBufferSemantics h) {
  switch (h) {
    case xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall:
      return "xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall";
    case xla::PjRtClient::HostBufferSemantics::kZeroCopy:
      return "xla::PjRtClient::HostBufferSemantics::kZeroCopy";
    case xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes:
      return "xla::PjRtClient::HostBufferSemantics::"
             "kImmutableUntilTransferCompletes";
  }
}

PJRT_HostBufferSemantics ConvertToPjRtHostBufferSemantics(
    xla::PjRtClient::HostBufferSemantics buffer_semantics) {
  switch (buffer_semantics) {
    case xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall:
      return PJRT_HostBufferSemantics::
          PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    case xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes:
      return PJRT_HostBufferSemantics::
          PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    case xla::PjRtClient::HostBufferSemantics::kZeroCopy:
      return PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kZeroCopy;
    default:
      CHECK(false)
          << "Input host buffer semantics is not supported in C API layer: "
          << HostBufferSemanticsToString(buffer_semantics);
  }
}

xla::PjRtClient::HostBufferSemantics ConvertFromPjRtHostBufferSemantics(
    PJRT_HostBufferSemantics buffer_semantics) {
  switch (buffer_semantics) {
    case PJRT_HostBufferSemantics::
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall:
      return xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
    case PJRT_HostBufferSemantics::
        PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes:
      return xla::PjRtClient::HostBufferSemantics::
          kImmutableUntilTransferCompletes;
    case PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kZeroCopy:
      return xla::PjRtClient::HostBufferSemantics::kZeroCopy;
  }
}

xla::PjRtFuture<xla::Status> ConvertCEventToCppFuture(PJRT_Event* c_event,
                                                      const PJRT_Api* c_api) {
  using xla::Status, xla::PjRtFuture;
  PJRT_Event_OnReady_Args event_onready_args;
  event_onready_args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  event_onready_args.priv = nullptr;
  event_onready_args.event = c_event;

  PjRtFuture<Status>::Promise promise = PjRtFuture<Status>::CreatePromise();
  event_onready_args.user_arg = new std::function<void(PJRT_Error*)>(
      [promise, c_event, c_api](PJRT_Error* error) mutable {
        if (error != nullptr) {
          xla::Status s = ::pjrt::PjrtErrorToStatus(error, c_api);
          promise.Set(s);
          ::pjrt::MakeErrorDeleter(c_api)(error);
        } else {
          promise.Set(tsl::OkStatus());
        }
        ::pjrt::MakeEventDeleter(c_api)(c_event);
      });
  event_onready_args.callback = [](PJRT_Error* error, void* arg) {
    std::function<void(PJRT_Error*)>* set_future =
        reinterpret_cast<std::function<void(PJRT_Error*)>*>(arg);
    (*set_future)(error);
    delete set_future;
  };

  PJRT_Error* error = c_api->PJRT_Event_OnReady(&event_onready_args);
  if (error != nullptr) {
    xla::Status s = ::pjrt::PjrtErrorToStatus(error, c_api);
    return PjRtFuture<Status>(s);
  }
  return PjRtFuture<Status>(std::move(promise));
}

static xla::StatusOr<PJRT_NamedValue> ConvertToPjRtNamedValue(
    const std::string& name, const xla::PjRtValueType& value) {
  PJRT_NamedValue c_value;
  c_value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  c_value.priv = nullptr;
  c_value.name = name.c_str();
  c_value.name_size = name.size();

  if (std::holds_alternative<std::string>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kString;
    const std::string& option_string_value = std::get<std::string>(value);
    c_value.string_value = option_string_value.c_str();
    c_value.value_size = option_string_value.size();
  } else if (std::holds_alternative<int64_t>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
    c_value.int64_value = std::get<int64_t>(value);
    c_value.value_size = 1;
  } else if (std::holds_alternative<std::vector<int64_t>>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List;
    const std::vector<int64_t>& option_int_list_value =
        std::get<std::vector<int64_t>>(value);
    c_value.int64_array_value = option_int_list_value.data();
    c_value.value_size = option_int_list_value.size();
  } else if (std::holds_alternative<float>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kFloat;
    c_value.float_value = std::get<float>(value);
    c_value.value_size = 1;
  } else {
    return tsl::errors::InvalidArgument("Unexpected PjRtValueType: '",
                                        value.index(), " with name: ", name);
  }

  return c_value;
}

xla::StatusOr<std::vector<PJRT_NamedValue>> ConvertToPjRtNamedValueList(
    const absl::flat_hash_map<std::string, xla::PjRtValueType>& cpp_value_map) {
  std::vector<PJRT_NamedValue> c_value_list;
  c_value_list.reserve(cpp_value_map.size());
  for (const auto& [name, value] : cpp_value_map) {
    TF_ASSIGN_OR_RETURN(PJRT_NamedValue c_value,
                        ConvertToPjRtNamedValue(name, value));
    c_value_list.push_back(c_value);
  }
  return c_value_list;
}

absl::flat_hash_map<std::string, xla::PjRtValueType>
ConvertFromPjRtNamedValueList(PJRT_NamedValue* c_value_list, size_t list_size) {
  absl::flat_hash_map<std::string, xla::PjRtValueType> cpp_value_map;
  for (int i = 0; i < list_size; ++i) {
    const PJRT_NamedValue& c_value = c_value_list[i];
    absl::string_view name = absl::string_view(c_value.name, c_value.name_size);
    switch (c_value.type) {
      case PJRT_NamedValue_Type::PJRT_NamedValue_kString: {
        std::string string_value(c_value.string_value, c_value.value_size);
        cpp_value_map[name] = xla::PjRtValueType(string_value);
        break;
      }
      case PJRT_NamedValue_Type::PJRT_NamedValue_kInt64: {
        cpp_value_map[name] = xla::PjRtValueType(c_value.int64_value);
        break;
      }
      case PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List: {
        const int64_t* array_ptr(c_value.int64_array_value);
        std::vector<int64_t> int64_array(array_ptr,
                                         array_ptr + c_value.value_size);
        cpp_value_map[name] = xla::PjRtValueType(int64_array);
        break;
      }
      case PJRT_NamedValue_Type::PJRT_NamedValue_kFloat: {
        cpp_value_map[name] = xla::PjRtValueType(c_value.float_value);
        break;
      }
      default: {
        LOG(FATAL) << "Unexpected PJRT_NamedValue type: " << c_value.type
                   << " with name: " << name;
        break;
      }
    }
  }
  return cpp_value_map;
}

static xla::StatusOr<PJRT_NamedValue_Type> GetPjrtNamedValueType(
    xla::PjRtValueType cpp_value) {
  if (std::holds_alternative<std::string>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kString;
  }
  if (std::holds_alternative<int64_t>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
  }
  if (std::holds_alternative<std::vector<int64_t>>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List;
  }
  if (std::holds_alternative<float>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kFloat;
  }
  return tsl::errors::InvalidArgument("Unexpected PjRtValueType with index",
                                      cpp_value.index());
}

xla::Status ValidateCreateOptions(
    const absl::flat_hash_map<std::string, xla::PjRtValueType>& value_map,
    const absl::flat_hash_map<std::string, PJRT_NamedValue_Type>&
        expected_name_and_types) {
  for (const auto& [name, value] : value_map) {
    auto it = expected_name_and_types.find(name);
    if (it == expected_name_and_types.end()) {
      return tsl::errors::InvalidArgument(
          "Unexpected option name passed to PJRT_Client_Create: ", name);
    }
    TF_ASSIGN_OR_RETURN(PJRT_NamedValue_Type type,
                        GetPjrtNamedValueType(value));
    if (type != it->second) {
      return tsl::errors::InvalidArgument(
          "Option passed to PJRT_Client_Create with name ", name,
          " has type index ", value.index(), " but expected type index is ",
          it->second);
    }
  }
  return tsl::OkStatus();
}

PJRT_SerializedExecutableDeleter MakeSerializedExecutableDeleter(
    const PJRT_Api* api) {
  return [api](PJRT_SerializedExecutable* serialized_executable) -> void {
    PJRT_SerializedExecutable_Destroy_Args destroy_args;
    destroy_args.struct_size =
        PJRT_SerializedExecutable_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.serialized_executable = serialized_executable;
    pjrt::LogFatalIfPjrtError(
        api->PJRT_SerializedExecutable_Destroy(&destroy_args), api);
  };
}

static std::string StructSizeErrorMsg(absl::string_view struct_name,
                                      size_t expected_size,
                                      size_t actual_size) {
  return absl::StrCat("Unexpected ", struct_name, " size: expected ",
                      expected_size, ", got ", actual_size,
                      ". Check installed software versions.");
}

xla::Status CheckMatchingStructSizes(absl::string_view struct_name,
                                     size_t expected_size, size_t actual_size) {
  if (expected_size != actual_size) {
    return tsl::errors::InvalidArgument(
        StructSizeErrorMsg(struct_name, expected_size, actual_size));
  }
  return tsl::OkStatus();
}

absl::string_view GetPlatformVersion(PJRT_Client* client, const PJRT_Api* api) {
  PJRT_Client_PlatformVersion_Args args;
  args.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.client = client;
  LogFatalIfPjrtError(api->PJRT_Client_PlatformVersion(&args), api);

  absl::string_view platform_version(args.platform_version,
                                     args.platform_version_size);
  return platform_version;
}

PJRT_Chunk ConvertFromCppChunk(xla::PjRtChunk chunk) {
  // `deleter_arg` holds a copy of the original xla::PjRtChunk
  // deleter. The original xla::PjRtChunk `input` releases its ownership
  // of data, which will subsequently be managed by `deleter` along with
  // `deleter_arg`.
  PJRT_Chunk c_chunk;
  c_chunk.data = chunk.data();
  c_chunk.size = static_cast<size_t>(chunk.size());
  c_chunk.deleter_arg = new std::function(chunk.deleter());
  c_chunk.deleter = [](void* data, void* deleter_arg) {
    auto* deleter = reinterpret_cast<std::function<void(void*)>*>(deleter_arg);
    (*deleter)(data);
    delete deleter;
  };

  // Release the ownership of `chunk.data()`, so it can be managed by `c_chunk`.
  chunk.release();

  return c_chunk;
}

xla::PjRtChunk ConvertToCppChunk(const PJRT_Chunk& chunk) {
  return xla::PjRtChunk(
      chunk.data, chunk.size,
      [deleter_arg = chunk.deleter_arg, deleter = chunk.deleter](void* data) {
        deleter(data, deleter_arg);
      });
}

}  // namespace pjrt
