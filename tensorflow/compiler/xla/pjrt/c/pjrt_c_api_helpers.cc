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

#include <memory>

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace pjrt {

const absl::string_view kHloFormat = "hlo";
const absl::string_view kMlirFormat = "mlir";

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

xla::Status PjrtErrorToStatus(const PJRT_Error* error, const PJRT_Api* api) {
  xla::Status status;
  if (error != nullptr) {
    status = xla::Status(PjrtErrorToStatusCode(error, api),
                         GetPjrtErrorMessage(error, api));
  }
  return status;
}

tensorflow::error::Code PjrtErrorToStatusCode(const PJRT_Error* error,
                                              const PJRT_Api* api) {
  PJRT_Error_GetCode_Args args;
  args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.error = error;
  api->PJRT_Error_GetCode(&args);
  PJRT_Error_Code code = args.code;
  switch (code) {
    case PJRT_Error_Code_CANCELLED:
      return tensorflow::error::CANCELLED;
    case PJRT_Error_Code_UNKNOWN:
      return tensorflow::error::UNKNOWN;
    case PJRT_Error_Code_INVALID_ARGUMENT:
      return tensorflow::error::INVALID_ARGUMENT;
    case PJRT_Error_Code_DEADLINE_EXCEEDED:
      return tensorflow::error::DEADLINE_EXCEEDED;
    case PJRT_Error_Code_NOT_FOUND:
      return tensorflow::error::NOT_FOUND;
    case PJRT_Error_Code_ALREADY_EXISTS:
      return tensorflow::error::ALREADY_EXISTS;
    case PJRT_Error_Code_PERMISSION_DENIED:
      return tensorflow::error::PERMISSION_DENIED;
    case PJRT_Error_Code_RESOURCE_EXHAUSTED:
      return tensorflow::error::RESOURCE_EXHAUSTED;
    case PJRT_Error_Code_FAILED_PRECONDITION:
      return tensorflow::error::FAILED_PRECONDITION;
    case PJRT_Error_Code_ABORTED:
      return tensorflow::error::ABORTED;
    case PJRT_Error_Code_OUT_OF_RANGE:
      return tensorflow::error::OUT_OF_RANGE;
    case PJRT_Error_Code_UNIMPLEMENTED:
      return tensorflow::error::UNIMPLEMENTED;
    case PJRT_Error_Code_INTERNAL:
      return tensorflow::error::INTERNAL;
    case PJRT_Error_Code_UNAVAILABLE:
      return tensorflow::error::UNAVAILABLE;
    case PJRT_Error_Code_DATA_LOSS:
      return tensorflow::error::DATA_LOSS;
    case PJRT_Error_Code_UNAUTHENTICATED:
      return tensorflow::error::UNAUTHENTICATED;
  }
}

PJRT_Error_Code StatusCodeToPjrtErrorCode(tensorflow::error::Code code) {
  switch (code) {
    case tensorflow::error::CANCELLED:
      return PJRT_Error_Code::PJRT_Error_Code_CANCELLED;
    case tensorflow::error::UNKNOWN:
      return PJRT_Error_Code::PJRT_Error_Code_UNKNOWN;
    case tensorflow::error::INVALID_ARGUMENT:
      return PJRT_Error_Code::PJRT_Error_Code_INVALID_ARGUMENT;
    case tensorflow::error::DEADLINE_EXCEEDED:
      return PJRT_Error_Code::PJRT_Error_Code_DEADLINE_EXCEEDED;
    case tensorflow::error::NOT_FOUND:
      return PJRT_Error_Code::PJRT_Error_Code_NOT_FOUND;
    case tensorflow::error::ALREADY_EXISTS:
      return PJRT_Error_Code::PJRT_Error_Code_ALREADY_EXISTS;
    case tensorflow::error::PERMISSION_DENIED:
      return PJRT_Error_Code::PJRT_Error_Code_PERMISSION_DENIED;
    case tensorflow::error::UNAUTHENTICATED:
      return PJRT_Error_Code::PJRT_Error_Code_UNAUTHENTICATED;
    case tensorflow::error::RESOURCE_EXHAUSTED:
      return PJRT_Error_Code::PJRT_Error_Code_RESOURCE_EXHAUSTED;
    case tensorflow::error::FAILED_PRECONDITION:
      return PJRT_Error_Code::PJRT_Error_Code_FAILED_PRECONDITION;
    case tensorflow::error::ABORTED:
      return PJRT_Error_Code::PJRT_Error_Code_ABORTED;
    case tensorflow::error::OUT_OF_RANGE:
      return PJRT_Error_Code::PJRT_Error_Code_OUT_OF_RANGE;
    case tensorflow::error::UNIMPLEMENTED:
      return PJRT_Error_Code::PJRT_Error_Code_UNIMPLEMENTED;
    case tensorflow::error::INTERNAL:
      return PJRT_Error_Code::PJRT_Error_Code_INTERNAL;
    case tensorflow::error::UNAVAILABLE:
      return PJRT_Error_Code::PJRT_Error_Code_UNAVAILABLE;
    case tensorflow::error::DATA_LOSS:
      return PJRT_Error_Code::PJRT_Error_Code_DATA_LOSS;
    case tensorflow::error::OK:
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
    LOG(FATAL) << "Unexpected error status " << _status.error_message();
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
    case PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kZeroCopy:
      return xla::PjRtClient::HostBufferSemantics::kZeroCopy;
  }
}

}  // namespace pjrt
