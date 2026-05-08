/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_status_utils.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/error_codes.pb.h"

namespace pjrt {

PJRT_ErrorDeleter MakeErrorDeleter(const PJRT_Api* api) {
  return [api](PJRT_Error* error) -> void {
    PJRT_Error_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.error = error;

    api->PJRT_Error_Destroy(&destroy_args);
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
    case PJRT_Error_Code_OK:
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
    case tsl::error::OK:
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

absl::Status PjrtErrorToStatus(const PJRT_Error* error, const PJRT_Api* api) {
  absl::Status status;
  if (error != nullptr) {
    status = absl::Status(PjrtErrorToStatusCode(error, api),
                          GetPjrtErrorMessage(error, api));
    if (api->struct_size >=
        PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Error_ForEachPayload)) {
      PJRT_Error_ForEachPayload_Args args{};
      args.struct_size = PJRT_Error_ForEachPayload_Args_STRUCT_SIZE;
      args.extension_start = nullptr;
      args.error = error;
      args.visitor = [](const char* key, size_t key_size, const char* value,
                        size_t value_size, void* user_arg) {
        absl::string_view key_str(key, key_size);
        absl::Cord value_cord(absl::string_view(value, value_size));
        static_cast<absl::Status*>(user_arg)->SetPayload(key_str,
                                                         std::move(value_cord));
      };
      args.user_arg = &status;
      pjrt::LogFatalIfPjrtError(api->PJRT_Error_ForEachPayload(&args), api);
    }
  }
  return status;
}

struct PJRT_Error_Impl : public PJRT_Error {
  absl::Status status;

  static void Destroy(PJRT_Error* error) {
    delete static_cast<PJRT_Error_Impl*>(error);
  }

  static void Message(const PJRT_Error* error, const char** message,
                      size_t* message_size) {
    const PJRT_Error_Impl* impl = static_cast<const PJRT_Error_Impl*>(error);
    *message = impl->status.message().data();
    *message_size = impl->status.message().size();
  }

  static PJRT_Error_Code GetCode(const PJRT_Error* error) {
    const PJRT_Error_Impl* impl = static_cast<const PJRT_Error_Impl*>(error);
    return StatusCodeToPjrtErrorCode(impl->status.code());
  }

  static void ForEachPayload(const PJRT_Error* error,
                             PJRT_Error_PayloadVisitor visitor,
                             void* user_arg) {
    const PJRT_Error_Impl* impl = static_cast<const PJRT_Error_Impl*>(error);
    impl->status.ForEachPayload(
        [&](absl::string_view key, const absl::Cord& value) {
          std::optional<absl::string_view> value_view = value.TryFlat();
          if (value_view.has_value()) {
            visitor(key.data(), key.size(), value_view->data(),
                    value_view->size(), user_arg);
          } else {
            std::string value_str(value);
            visitor(key.data(), key.size(), value_str.data(), value_str.size(),
                    user_arg);
          }
        });
  }
};

static constexpr PJRT_Error_FunctionTable kBuiltinErrorVTable = {
    /*struct_size=*/PJRT_Error_FunctionTable_STRUCT_SIZE,
    /*instance_size=*/PJRT_Error_STRUCT_SIZE,
    /*extension_start=*/nullptr,
    /*destroy=*/PJRT_Error_Impl::Destroy,
    /*message=*/PJRT_Error_Impl::Message,
    /*get_code=*/PJRT_Error_Impl::GetCode,
    /*for_each_payload=*/PJRT_Error_Impl::ForEachPayload,
};

absl::Status PjrtErrorToStatus(PJRT_Error* error) {
  if (error == nullptr) {
    return absl::OkStatus();
  }
  if (error->vtable == &kBuiltinErrorVTable) {
    PJRT_Error_Impl* impl = static_cast<PJRT_Error_Impl*>(error);
    absl::Status status = std::move(impl->status);
    delete impl;
    return status;
  }
  const char* message = nullptr;
  size_t message_size = 0;
  error->vtable->message(error, &message, &message_size);
  PJRT_Error_Code code = error->vtable->get_code(error);
  absl::Status status(PjrtErrorCodeToStatusCode(code),
                      absl::string_view(message, message_size));
  error->vtable->destroy(error);
  return status;
}

PJRT_Error* StatusToPjRtError(absl::Status s) {
  if (s.ok()) {
    return nullptr;
  }
  PJRT_Error_Impl* impl = new PJRT_Error_Impl();
  impl->vtable = &kBuiltinErrorVTable;
  impl->status = std::move(s);
  return impl;
}

void DestroyPjRtError(PJRT_Error* error) {
  if (error == nullptr) {
    return;
  }
  error->vtable->destroy(error);
}

}  // namespace pjrt
