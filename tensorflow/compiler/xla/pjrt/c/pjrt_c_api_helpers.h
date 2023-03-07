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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_HELPERS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_HELPERS_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace pjrt {

ABSL_CONST_INIT extern const absl::string_view kHloFormat;
ABSL_CONST_INIT extern const absl::string_view kMlirFormat;
ABSL_CONST_INIT extern const absl::string_view kHloWithConfigFormat;

using PJRT_ClientDeleter = std::function<void(PJRT_Client*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the client.
PJRT_ClientDeleter MakeClientDeleter(const PJRT_Api* api);

using PJRT_ErrorDeleter = std::function<void(PJRT_Error*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the error.
PJRT_ErrorDeleter MakeErrorDeleter(const PJRT_Api* api);

using PJRT_BufferDeleter = std::function<void(PJRT_Buffer*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the buffer.
PJRT_BufferDeleter MakeBufferDeleter(const PJRT_Api* api);

using PJRT_ExecutableDeleter = std::function<void(PJRT_Executable*)>;

// Creates a custom deleter for smart pointers.
// Pass in pointer `api` to the PJRT C API.
// The lifetime of the Api pointed to must be longer than the executable.
PJRT_ExecutableDeleter MakeExecutableDeleter(const PJRT_Api* api);

using PJRT_LoadedExecutableDeleter =
    std::function<void(PJRT_LoadedExecutable*)>;

// Creates a custom deleter for smart pointers.
// Pass in pointer `api` to the PJRT C API.
// The lifetime of the Api pointed to must be longer than the executable.
PJRT_LoadedExecutableDeleter MakeLoadedExecutableDeleter(const PJRT_Api* api);

using PJRT_EventDeleter = std::function<void(PJRT_Event*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the event.
PJRT_EventDeleter MakeEventDeleter(const PJRT_Api* api);

using PJRT_SerializedExecutableDeleter =
    std::function<void(PJRT_SerializedExecutable*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the serialized
// executable.
PJRT_SerializedExecutableDeleter MakeSerializedExecutableDeleter(
    const PJRT_Api* api);

using PJRT_DeviceTopologyDeleter = std::function<void(PJRT_DeviceTopology*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the client.
PJRT_DeviceTopologyDeleter MakeDeviceTopologyDeleter(const PJRT_Api* api);

// Fatal error logging if status is not success. This terminates the process
// and frees the PJRT_Error passed in.
void LogFatalIfPjrtError(PJRT_Error* error, const PJRT_Api* api);

absl::string_view GetPjrtErrorMessage(const PJRT_Error* error,
                                      const PJRT_Api* api);

xla::Status PjrtErrorToStatus(const PJRT_Error* error, const PJRT_Api* api);

tsl::error::Code PjrtErrorToStatusCode(const PJRT_Error* error,
                                       const PJRT_Api* api);

PJRT_Error_Code StatusCodeToPjrtErrorCode(tsl::error::Code code);

// Conversion helper from xla::PrimitiveType to PJRT_Buffer_Type.
PJRT_Buffer_Type ConvertToPjRtBufferType(xla::PrimitiveType type);

// Conversion helper from PJRT_Buffer_type to xla::PrimitiveType.
xla::PrimitiveType ConvertFromPjRtBufferType(PJRT_Buffer_Type type);

// Conversion helper from xla::PjRtClient::HostBufferSemantics to
// PJRT_HostBufferSemantics.
PJRT_HostBufferSemantics ConvertToPjRtHostBufferSemantics(
    xla::PjRtClient::HostBufferSemantics buffer_semantics);

// Conversion helper to xla::PjRtClient::HostBufferSemantics from
// PJRT_HostBufferSemantics.
xla::PjRtClient::HostBufferSemantics ConvertFromPjRtHostBufferSemantics(
    PJRT_HostBufferSemantics buffer_semantics);

// Create and return a `PjRtFuture`  which will be set when `c_event` is ready.
// This also deletes `c_event` when the `PjRtFuture` is set.
xla::PjRtFuture<xla::Status> ConvertCEventToCppFuture(PJRT_Event* c_event,
                                                      const PJRT_Api* c_api);

// The data of returned variable-length PJRT_NamedValue list is backed by
// `cpp_value_map`, so `cpp_value_map` must outlive the returned list. It will
// raise errors for unsupported PjRtValueType.
xla::StatusOr<std::vector<PJRT_NamedValue>> ConvertToPjRtNamedValueList(
    const absl::flat_hash_map<std::string, xla::PjRtValueType>& cpp_value_map);

absl::flat_hash_map<std::string, xla::PjRtValueType>
ConvertFromPjRtNamedValueList(PJRT_NamedValue* c_value_list, size_t list_size);

// Validates that all entries in value_map have a matching name and type in
// expected_name_and_type. expected_name_and_type may contain extra entries
// not in value_map without error.
xla::Status ValidateCreateOptions(
    const absl::flat_hash_map<std::string, xla::PjRtValueType>& value_map,
    const absl::flat_hash_map<std::string, PJRT_NamedValue_Type>&
        expected_name_and_types);

// Helper function for checking C API argument struct sizes. Returns a non-OK
// status if the expected and actual sizes aren't equal (i.e. no ABI
// compatibility guarantees).
xla::Status CheckMatchingStructSizes(absl::string_view struct_name,
                                     size_t expected_size, size_t actual_size);

absl::string_view GetPlatformVersion(PJRT_Client* client, const PJRT_Api* api);

}  // namespace pjrt

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_HELPERS_H_
