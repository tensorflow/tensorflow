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

#include "xla/pjrt/c/pjrt_c_api_helpers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"

namespace pjrt {

const absl::string_view kHloFormat = "hlo";
const absl::string_view kMlirFormat = "mlir";
const absl::string_view kHloWithConfigFormat = "hlo_with_config";

PJRT_ClientDeleter MakeClientDeleter(const PJRT_Api* api) {
  return [api](PJRT_Client* client) -> void {
    PJRT_Client_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
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
    destroy_args.extension_start = nullptr;
    destroy_args.error = error;

    api->PJRT_Error_Destroy(&destroy_args);
  };
}

PJRT_BufferDeleter MakeBufferDeleter(const PJRT_Api* api) {
  return [api](PJRT_Buffer* buffer) -> void {
    PJRT_Buffer_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.buffer = buffer;

    pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_Destroy(&destroy_args), api);
  };
}

PJRT_ExecutableDeleter MakeExecutableDeleter(const PJRT_Api* api) {
  return [api](PJRT_Executable* executable) -> void {
    PJRT_Executable_Destroy_Args args;
    args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.executable = executable;
    pjrt::LogFatalIfPjrtError(api->PJRT_Executable_Destroy(&args), api);
  };
}

PJRT_LoadedExecutableDeleter MakeLoadedExecutableDeleter(const PJRT_Api* api) {
  return [api](PJRT_LoadedExecutable* executable) -> void {
    PJRT_LoadedExecutable_Destroy_Args args;
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.executable = executable;
    pjrt::LogFatalIfPjrtError(api->PJRT_LoadedExecutable_Destroy(&args), api);
  };
}

absl::Status PjrtErrorToStatus(const PJRT_Error* error, const PJRT_Api* api) {
  absl::Status status;
  if (error != nullptr) {
    status = absl::Status(PjrtErrorToStatusCode(error, api),
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
    destroy_args.extension_start = nullptr;
    destroy_args.topology = topology;

    pjrt::LogFatalIfPjrtError(
        api->PJRT_TopologyDescription_Destroy(&destroy_args), api);
  };
}

PJRT_Error_Code GetErrorCode(const PJRT_Error* error, const PJRT_Api* api) {
  PJRT_Error_GetCode_Args args;
  args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.error = error;
  pjrt::LogFatalIfPjrtError(api->PJRT_Error_GetCode(&args), api);
  return args.code;
}

absl::StatusCode PjrtErrorToStatusCode(const PJRT_Error* error,
                                       const PJRT_Api* api) {
  return PjrtErrorCodeToStatusCode(GetErrorCode(error, api));
}

absl::StatusCode PjrtErrorCodeToStatusCode(PJRT_Error_Code code) {
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
  message_args.extension_start = nullptr;
  message_args.error = error;
  api->PJRT_Error_Message(&message_args);
  return absl::string_view(message_args.message, message_args.message_size);
}

void LogFatalIfPjrtError(PJRT_Error* error, const PJRT_Api* api) {
  std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> _error(
      error, MakeErrorDeleter(api));
  absl::Status _status = PjrtErrorToStatus(_error.get(), api);
  if (!_status.ok()) {
    LOG(FATAL) << "Unexpected error status " << _status.message();
  }
}

PJRT_EventDeleter MakeEventDeleter(const PJRT_Api* api) {
  CHECK(api != nullptr);
  return [api](PJRT_Event* managed) {
    PJRT_Event_Destroy_Args args;
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
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
    case xla::PrimitiveType::S4:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S4;
    case xla::PrimitiveType::S8:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S8;
    case xla::PrimitiveType::S16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S16;
    case xla::PrimitiveType::S32:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S32;
    case xla::PrimitiveType::S64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S64;
    case xla::PrimitiveType::U4:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U4;
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
    case xla::PrimitiveType::F8E4M3B11FNUZ:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F8E4M3B11FNUZ;
    case xla::PrimitiveType::F8E5M2FNUZ:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F8E5M2FNUZ;
    case xla::PrimitiveType::F8E4M3FNUZ:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_F8E4M3FNUZ;
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
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S4:
      return xla::PrimitiveType::S4;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S8:
      return xla::PrimitiveType::S8;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S16:
      return xla::PrimitiveType::S16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S32:
      return xla::PrimitiveType::S32;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S64:
      return xla::PrimitiveType::S64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U4:
      return xla::PrimitiveType::U4;
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
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F8E4M3B11FNUZ:
      return xla::PrimitiveType::F8E4M3B11FNUZ;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F8E5M2FNUZ:
      return xla::PrimitiveType::F8E5M2FNUZ;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_F8E4M3FNUZ:
      return xla::PrimitiveType::F8E4M3FNUZ;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_INVALID:
      CHECK(false) << "Buffer type is not supported in C API layer.";
  }
}

const char* HostBufferSemanticsToString(
    xla::PjRtClient::HostBufferSemantics h) {
  switch (h) {
    case xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall:
      return "xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall";
    case xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy:
      return "xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy";
    case xla::PjRtClient::HostBufferSemantics::kMutableZeroCopy:
      return "xla::PjRtClient::HostBufferSemantics::kMutableZeroCopy";
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
    case xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy:
      return PJRT_HostBufferSemantics::
          PJRT_HostBufferSemantics_kImmutableZeroCopy;
    case xla::PjRtClient::HostBufferSemantics::kMutableZeroCopy:
      return PJRT_HostBufferSemantics::
          PJRT_HostBufferSemantics_kMutableZeroCopy;
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
    case PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kImmutableZeroCopy:
      return xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy;
    case PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kMutableZeroCopy:
      return xla::PjRtClient::HostBufferSemantics::kMutableZeroCopy;
  }
}

xla::PjRtFuture<> ConvertCEventToCppFuture(PJRT_Event* c_event,
                                           const PJRT_Api* c_api) {
  using absl::Status, xla::PjRtFuture;
  PJRT_Event_OnReady_Args event_onready_args;
  event_onready_args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  event_onready_args.extension_start = nullptr;
  event_onready_args.event = c_event;

  PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
  event_onready_args.user_arg = new std::function<void(PJRT_Error*)>(
      [promise, c_event, c_api](PJRT_Error* error) mutable {
        if (error != nullptr) {
          promise.Set(::pjrt::PjrtErrorToStatus(error, c_api));
          ::pjrt::MakeErrorDeleter(c_api)(error);
        } else {
          promise.Set();
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
    return PjRtFuture<>(::pjrt::PjrtErrorToStatus(error, c_api));
  }
  return PjRtFuture<>(std::move(promise));
}

static absl::StatusOr<PJRT_NamedValue> ConvertToPjRtNamedValue(
    const std::string& name, const xla::PjRtValueType& value) {
  PJRT_NamedValue c_value;
  c_value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  c_value.extension_start = nullptr;
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
  } else if (std::holds_alternative<bool>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kBool;
    c_value.bool_value = std::get<bool>(value);
    c_value.value_size = 1;
  } else {
    return tsl::errors::InvalidArgument("Unexpected PjRtValueType: '",
                                        value.index(), " with name: ", name);
  }

  return c_value;
}

absl::StatusOr<std::vector<PJRT_NamedValue>> ConvertToPjRtNamedValueList(
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
ConvertFromPjRtNamedValueList(const PJRT_NamedValue* c_value_list,
                              size_t list_size) {
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
      case PJRT_NamedValue_Type::PJRT_NamedValue_kBool: {
        cpp_value_map[name] = xla::PjRtValueType(c_value.bool_value);
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

static absl::StatusOr<PJRT_NamedValue_Type> GetPjrtNamedValueType(
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
  if (std::holds_alternative<bool>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kBool;
  }
  return tsl::errors::InvalidArgument("Unexpected PjRtValueType with index",
                                      cpp_value.index());
}

absl::Status ValidateCreateOptions(
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
  return absl::OkStatus();
}

const std::vector<PJRT_NamedValue>& GetXlaPluginCAttributes() {
  constexpr absl::string_view kXlaVersion = "xla_version";
  PJRT_NamedValue c_value;
  c_value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  c_value.extension_start = nullptr;
  c_value.name = kXlaVersion.data();
  c_value.name_size = kXlaVersion.size();
  c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
  // TODO(b/327203806): figure out where to keep the xla_version.
  c_value.int64_value = 1;
  c_value.value_size = 1;
  static const std::vector<PJRT_NamedValue>* c_values =
      new std::vector<PJRT_NamedValue>({c_value});
  return *c_values;
}

static std::string StructSizeErrorMsg(absl::string_view struct_name,
                                      size_t expected_size,
                                      size_t actual_size) {
  std::string error_msg = absl::StrCat(
      "Unexpected ", struct_name, " size: expected ", expected_size, ", got ",
      actual_size, ". Check installed software versions.");
#if defined(PJRT_API_MAJOR)
  absl::StrAppend(&error_msg, " The framework PJRT API version is ",
                  PJRT_API_MAJOR, ".", PJRT_API_MINOR, ".");
#endif  // PJRT_API_MAJOR
  return error_msg;
}

absl::Status ActualStructSizeIsGreaterOrEqual(absl::string_view struct_name,
                                              size_t expected_size,
                                              size_t actual_size) {
  if (actual_size < expected_size) {
    return tsl::errors::InvalidArgument(
        StructSizeErrorMsg(struct_name, expected_size, actual_size));
  }
  if (actual_size > expected_size) {
    VLOG(2) << StructSizeErrorMsg(struct_name, expected_size, actual_size);
  }
  return absl::OkStatus();
}

absl::string_view GetPlatformVersion(PJRT_Client* client, const PJRT_Api* api) {
  PJRT_Client_PlatformVersion_Args args;
  args.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = client;
  LogFatalIfPjrtError(api->PJRT_Client_PlatformVersion(&args), api);

  absl::string_view platform_version(args.platform_version,
                                     args.platform_version_size);
  return platform_version;
}

absl::string_view GetPlatformName(PJRT_Client* client, const PJRT_Api* api) {
  PJRT_Client_PlatformName_Args args;
  args.client = client;
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  pjrt::LogFatalIfPjrtError(api->PJRT_Client_PlatformName(&args), api);

  absl::string_view platform_name(args.platform_name, args.platform_name_size);
  return platform_name;
}

absl::StatusOr<PJRT_TopologyDescription*> GetTopologyDescription(
    PJRT_Client* client, const PJRT_Api* api) {
  PJRT_Client_TopologyDescription_Args args;
  args.struct_size = PJRT_Client_TopologyDescription_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = client;
  RETURN_STATUS_IF_PJRT_ERROR(api->PJRT_Client_TopologyDescription(&args), api);
  return args.topology;
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

PJRT_DeviceDescription* GetDeviceDescription(const PJRT_Api* api,
                                             PJRT_Device* device) {
  PJRT_Device_GetDescription_Args args;
  args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = device;
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_GetDescription(&args), api);
  return args.device_description;
}

absl::Span<PJRT_Memory* const> GetAddressableMemories(const PJRT_Api* api,
                                                      PJRT_Device* device) {
  PJRT_Device_AddressableMemories_Args args;
  args.struct_size = PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = device;
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_AddressableMemories(&args), api);
  return absl::MakeSpan(args.memories, args.num_memories);
}

int GetId(const PJRT_Api* api, PJRT_DeviceDescription* device_desc) {
  PJRT_DeviceDescription_Id_Args args = PJRT_DeviceDescription_Id_Args{
      PJRT_DeviceDescription_Id_Args_STRUCT_SIZE, nullptr, device_desc};
  pjrt::LogFatalIfPjrtError(api->PJRT_DeviceDescription_Id(&args), api);
  return args.id;
}

static void PjRtValueDeleterCallback(char* value) { delete[] value; }

static PJRT_KeyValueGetCFunc ToKVGetCFunc(
    xla::KeyValueStoreInterface* kv_store) {
  return [kv_store](PJRT_KeyValueGetCallback_Args* args) -> PJRT_Error* {
    absl::StatusOr<std::string> output =
        kv_store->Get(std::string_view(args->key, args->key_size),
                      absl::Milliseconds(args->timeout_in_ms));
    if (!output.ok()) {
      absl::string_view message = output.status().message();
      return (*args->callback_error)(
          StatusCodeToPjrtErrorCode(output.status().code()), message.data(),
          message.size());
    }
    args->value = new char[output->size()];
    std::copy(output->begin(), output->end(), args->value);
    args->value_size = output->size();
    args->value_deleter_callback = &PjRtValueDeleterCallback;
    return nullptr;
  };
}

static PJRT_KeyValuePutCFunc ToKVPutCFunc(
    xla::KeyValueStoreInterface* kv_store) {
  return [kv_store](PJRT_KeyValuePutCallback_Args* args) -> PJRT_Error* {
    absl::Status status =
        kv_store->Set(std::string_view(args->key, args->key_size),
                      std::string_view(args->value, args->value_size));
    if (!status.ok()) {
      absl::string_view message = status.message();
      return (*args->callback_error)(StatusCodeToPjrtErrorCode(status.code()),
                                     message.data(), message.size());
    }
    return nullptr;
  };
}

static PJRT_KeyValueGetCallback ToCKVGetCallback(
    PJRT_KeyValueGetCFunc* kv_get_c_func) {
  return [](PJRT_KeyValueGetCallback_Args* args) -> PJRT_Error* {
    PJRT_KeyValueGetCFunc* kv_get_c_func =
        reinterpret_cast<PJRT_KeyValueGetCFunc*>(args->user_arg);
    if (kv_get_c_func == nullptr) {
      absl::Status status = xla::InvalidArgument(
          "got nullptr for PJRT_KeyValueGet_Args.user_arg");
      return (*args->callback_error)(StatusCodeToPjrtErrorCode(status.code()),
                                     status.message().data(),
                                     status.message().size());
    }
    return (*kv_get_c_func)(args);
  };
}

static PJRT_KeyValuePutCallback ToCKVPutCallback(
    PJRT_KeyValuePutCFunc* kv_put_c_func) {
  return [](PJRT_KeyValuePutCallback_Args* args) -> PJRT_Error* {
    PJRT_KeyValuePutCFunc* kv_put_c_func =
        reinterpret_cast<PJRT_KeyValuePutCFunc*>(args->user_arg);
    if (kv_put_c_func == nullptr) {
      absl::Status status = xla::InvalidArgument(
          "got nullptr for PJRT_KeyValuePut_Args.user_arg");
      return (*args->callback_error)(StatusCodeToPjrtErrorCode(status.code()),
                                     status.message().data(),
                                     status.message().size());
    }
    return (*kv_put_c_func)(args);
  };
}

std::unique_ptr<PJRT_KeyValueCallbackData> ConvertToCKeyValueCallbacks(
    std::shared_ptr<xla::KeyValueStoreInterface> kv_store) {
  auto kv_callback_data = std::make_unique<PJRT_KeyValueCallbackData>();
  kv_callback_data->kv_get_c_func = ToKVGetCFunc(kv_store.get());
  kv_callback_data->kv_put_c_func = ToKVPutCFunc(kv_store.get());
  kv_callback_data->c_kv_get =
      ToCKVGetCallback(&kv_callback_data->kv_get_c_func);
  kv_callback_data->c_kv_put =
      ToCKVPutCallback(&kv_callback_data->kv_put_c_func);
  kv_callback_data->kv_store = std::move(kv_store);
  return kv_callback_data;
}

PJRT_SendCallbackInfo CppSendCallbackToCSendCallback(
    xla::SendCallback cpp_send_callback,
    PJRT_SendCallbackFunction* send_callback_function) {
  return PJRT_SendCallbackInfo{
      cpp_send_callback.channel_id,
      // this is the void* user_arg to capture `cpp_send_callback.callback`
      send_callback_function,
      // this is the function pointer, PJRT_SendCallback
      [](PJRT_Chunk* chunk, PJRT_CallbackError* callback_error,
         size_t total_size_in_bytes, bool done, void* user_arg) -> PJRT_Error* {
        // PJRT_SendCallback, `send_callback` is internal C interface callback
        // representation that cpatures the client C++ callback in void*
        // `user_arg` and reinterprets in the lower-level runtime for execution.
        // `user_arg` captures `send_callback_function` which is
        // SendCallbackFunction*.
        PJRT_SendCallbackFunction* send_callback =
            reinterpret_cast<PJRT_SendCallbackFunction*>(user_arg);
        return (*send_callback)(chunk, callback_error, total_size_in_bytes,
                                done);
      }};
}

PJRT_RecvCallbackInfo CppRecvCallbackToCRecvCallback(
    xla::RecvCallback cpp_recv_callback,
    PJRT_RecvCallbackFunction* recv_callback_function) {
  return PJRT_RecvCallbackInfo{
      cpp_recv_callback.channel_id,
      // this is the void* user_arg to capture `cpp_recv_callback.callback`
      recv_callback_function,
      // this is the function pointer, PJRT_RecvCallback
      [](PJRT_CopyToDeviceStream* stream, void* user_arg) {
        // PJRT_RecvCallback, `recv_callback` is internal C interface callback
        // representation that cpatures the client C++ callback in void*
        // `user_arg` and reinterprets in the lower-level runtime for execution.
        // `user_arg` captures `recv_callback_function` which is
        // RecvCallbackFunction*.
        auto* recv_callback =
            reinterpret_cast<std::function<void(PJRT_CopyToDeviceStream*)>*>(
                user_arg);
        (*recv_callback)(stream);
      }};
}

absl::StatusOr<BufferMemoryLayoutData> ConvertToBufferMemoryLayoutData(
    const xla::Layout& cpp_layout) {
  BufferMemoryLayoutData layout_data;
  layout_data.c_layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled;

  PJRT_Buffer_MemoryLayout_Tiled c_tiled;
  layout_data.minor_to_major.assign(cpp_layout.minor_to_major().begin(),
                                    cpp_layout.minor_to_major().end());
  c_tiled.minor_to_major = layout_data.minor_to_major.data();
  c_tiled.minor_to_major_size = layout_data.minor_to_major.size();
  c_tiled.num_tiles = cpp_layout.tiles().size();
  if (c_tiled.num_tiles >= 0) {
    layout_data.tile_dim_sizes.reserve(c_tiled.num_tiles);
    for (int i = 0; i < c_tiled.num_tiles; ++i) {
      absl::Span<const int64_t> tile_dim = cpp_layout.tiles()[i].dimensions();
      layout_data.tile_dims.insert(layout_data.tile_dims.end(),
                                   tile_dim.begin(), tile_dim.end());
      layout_data.tile_dim_sizes.push_back(tile_dim.size());
    }
    c_tiled.tile_dims = layout_data.tile_dims.data();
    c_tiled.tile_dim_sizes = layout_data.tile_dim_sizes.data();
  }
  layout_data.c_layout.tiled = c_tiled;
  return layout_data;
}

absl::StatusOr<BufferMemoryLayoutData> ConvertToBufferMemoryLayoutData(
    absl::Span<int64_t const> byte_strides) {
  BufferMemoryLayoutData layout_data;
  layout_data.c_layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Strides;
  layout_data.c_layout.strides.byte_strides = byte_strides.data();
  layout_data.c_layout.strides.num_byte_strides = byte_strides.size();
  return layout_data;
}

absl::StatusOr<xla::Layout> ConvertToLayout(
    const PJRT_Buffer_MemoryLayout_Tiled& c_tiled) {
  absl::Span<const int64_t> minor_to_major(c_tiled.minor_to_major,
                                           c_tiled.minor_to_major_size);
  absl::InlinedVector<xla::Tile, 1> tiles;
  tiles.reserve(c_tiled.num_tiles);
  const int64_t* current_tile = c_tiled.tile_dims;
  for (int i = 0; i < c_tiled.num_tiles; ++i) {
    tiles.push_back(xla::Tile(
        absl::Span<const int64_t>(current_tile, c_tiled.tile_dim_sizes[i])));
    current_tile += c_tiled.tile_dim_sizes[i];
  }
  xla::Layout layout = xla::Layout(minor_to_major);
  layout.mutable_tiles()->assign(tiles.begin(), tiles.end());
  return layout;
}

PJRT_Buffer_Type GetElementType(const PJRT_Api* api, PJRT_Buffer* buffer) {
  PJRT_Buffer_ElementType_Args args;
  args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  LogFatalIfPjrtError(api->PJRT_Buffer_ElementType(&args), api);
  return args.type;
}

absl::Span<const int64_t> GetDimensions(const PJRT_Api* api,
                                        PJRT_Buffer* buffer) {
  PJRT_Buffer_Dimensions_Args args;
  args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  LogFatalIfPjrtError(api->PJRT_Buffer_Dimensions(&args), api);
  return {args.dims, args.num_dims};
}

PJRT_Buffer_MemoryLayout GetMemoryLayout(const PJRT_Api* api,
                                         PJRT_Buffer* buffer) {
  PJRT_Buffer_GetMemoryLayout_Args args;
  args.struct_size = PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  LogFatalIfPjrtError(api->PJRT_Buffer_GetMemoryLayout(&args), api);
  return args.layout;
}

absl::StatusOr<xla::Shape> BuildXlaShapeFromC(
    PJRT_Buffer_Type element_type, const int64_t* dims, size_t num_dims,
    PJRT_Buffer_MemoryLayout* layout) {
  xla::Shape shape =
      xla::ShapeUtil::MakeShape(ConvertFromPjRtBufferType(element_type),
                                absl::Span<const int64_t>(dims, num_dims));
  xla::Layout cpp_layout;
  if (layout != nullptr) {
    switch (layout->type) {
      case PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled: {
        TF_ASSIGN_OR_RETURN(cpp_layout, ConvertToLayout(layout->tiled));
        break;
      }
      case PJRT_Buffer_MemoryLayout_Type::
          PJRT_Buffer_MemoryLayout_Type_Strides: {
        TF_RETURN_IF_ERROR(absl::InvalidArgumentError(
            "PJRT_Buffer_MemoryLayout_Type_Strides is not supported to be "
            "converted to a xla::Shape"));
        break;
      }
      default: {
        TF_RETURN_IF_ERROR(absl::InvalidArgumentError(absl::StrCat(
            "Unexpected PJRT_Buffer_MemoryLayout_Type type: ", layout->type)));
      }
    }
    *shape.mutable_layout() = cpp_layout;
  }
  return shape;
}

absl::string_view PlatformName(const PJRT_Api* api,
                               const PJRT_TopologyDescription* topo_desc) {
  PJRT_TopologyDescription_PlatformName_Args args;
  args.struct_size = PJRT_TopologyDescription_PlatformName_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = const_cast<PJRT_TopologyDescription*>(topo_desc);
  LogFatalIfPjrtError(api->PJRT_TopologyDescription_PlatformName(&args), api);
  return {args.platform_name, args.platform_name_size};
}

absl::Span<PJRT_DeviceDescription* const> DeviceDescriptions(
    const PJRT_Api* api, const PJRT_TopologyDescription* topo_desc) {
  PJRT_TopologyDescription_GetDeviceDescriptions_Args args;
  args.struct_size =
      PJRT_TopologyDescription_GetDeviceDescriptions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = const_cast<PJRT_TopologyDescription*>(topo_desc);
  LogFatalIfPjrtError(
      api->PJRT_TopologyDescription_GetDeviceDescriptions(&args), api);
  return {args.descriptions, args.num_descriptions};
}

absl::StatusOr<xla::CompiledMemoryStats> GetCompiledMemoryStats(
    const PJRT_Api* api, PJRT_Executable* executable) {
  // TODO(jieying): To be removed after 03/2024.
  if (api->pjrt_api_version.major_version == 0 &&
      api->pjrt_api_version.minor_version < 40) {
    return absl::UnimplementedError(
        "GetCompiledMemoryStats requires a plugin with PJRT C API version >= "
        "0.40");
  }
  PJRT_Executable_GetCompiledMemoryStats_Args args;
  args.struct_size = PJRT_Executable_GetCompiledMemoryStats_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = executable;
  RETURN_STATUS_IF_PJRT_ERROR(
      api->PJRT_Executable_GetCompiledMemoryStats(&args), api);
  xla::CompiledMemoryStats results;
  results.generated_code_size_in_bytes = args.generated_code_size_in_bytes;
  results.argument_size_in_bytes = args.argument_size_in_bytes;
  results.output_size_in_bytes = args.output_size_in_bytes;
  results.alias_size_in_bytes = args.alias_size_in_bytes;
  results.temp_size_in_bytes = args.temp_size_in_bytes;
  results.host_generated_code_size_in_bytes =
      args.host_generated_code_size_in_bytes;
  results.host_argument_size_in_bytes = args.host_argument_size_in_bytes;
  results.host_output_size_in_bytes = args.host_output_size_in_bytes;
  results.host_alias_size_in_bytes = args.host_alias_size_in_bytes;
  results.host_temp_size_in_bytes = args.host_temp_size_in_bytes;
  return results;
}

PJRT_Profiler_Extension CreatePjrtProfilerExtension(
    absl::string_view traceme_name) {
  tsl::profiler::TraceMeProducer producer(
      traceme_name, tsl::profiler::ContextType::kPjrtLibraryCall);
  int64_t traceme_context_id = producer.GetContextId();
  PJRT_Profiler_Extension profiler_extension{
      /*struct_size=*/PJRT_Profiler_Extension_STRUCT_SIZE,
      /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Profiler,
      /*next=*/nullptr,
      /*profiler_api=*/nullptr,
      /*traceme_context_id=*/traceme_context_id,
  };
  return profiler_extension;
}

}  // namespace pjrt
