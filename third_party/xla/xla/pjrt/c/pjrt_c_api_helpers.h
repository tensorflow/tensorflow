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

#ifndef XLA_PJRT_C_PJRT_C_API_HELPERS_H_
#define XLA_PJRT_C_PJRT_C_API_HELPERS_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace pjrt {

ABSL_CONST_INIT extern const absl::string_view kHloFormat;
ABSL_CONST_INIT extern const absl::string_view kMlirFormat;
ABSL_CONST_INIT extern const absl::string_view kHloWithConfigFormat;

// Return error status if not success and frees the PJRT_Error returned by
// `expr`.
#define RETURN_STATUS_IF_PJRT_ERROR(expr, c_api)                         \
  do {                                                                   \
    PJRT_Error* error = (expr);                                          \
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> _error(         \
        error, pjrt::MakeErrorDeleter(c_api));                           \
    absl::Status _status = pjrt::PjrtErrorToStatus(_error.get(), c_api); \
    if (!_status.ok()) {                                                 \
      return _status;                                                    \
    }                                                                    \
  } while (false)

using PJRT_ClientDeleter = std::function<void(PJRT_Client*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the client.
PJRT_ClientDeleter MakeClientDeleter(const PJRT_Api* api);

using PJRT_AsyncHostToDeviceTransferManagerDeleter =
    std::function<void(PJRT_AsyncHostToDeviceTransferManager*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the transfer manager.
PJRT_AsyncHostToDeviceTransferManagerDeleter
MakeAsyncHostToDeviceTransferManagerDeleter(const PJRT_Api* api);

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

using PJRT_TopologyDescriptionDeleter =
    std::function<void(PJRT_TopologyDescription*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the client.
PJRT_TopologyDescriptionDeleter MakeTopologyDescriptionDeleter(
    const PJRT_Api* api);

using PJRT_Layouts_MemoryLayoutDeleter =
    std::function<void(PJRT_Layouts_MemoryLayout*)>;

// The lifetime of `api` must be longer than the layout object to be
// deleted. This function requires that `api` includes the PJRT_Layouts
// extension.
PJRT_Layouts_MemoryLayoutDeleter MakeMemoryLayoutDeleter(const PJRT_Api* api);

// Fatal error logging if status is not success. This terminates the process
// and frees the PJRT_Error passed in.
void LogFatalIfPjrtError(PJRT_Error* error, const PJRT_Api* api);

absl::string_view GetPjrtErrorMessage(const PJRT_Error* error,
                                      const PJRT_Api* api);

PJRT_Error_Code GetErrorCode(const PJRT_Error* error, const PJRT_Api* api);

absl::Status PjrtErrorToStatus(const PJRT_Error* error, const PJRT_Api* api);

absl::StatusCode PjrtErrorToStatusCode(const PJRT_Error* error,
                                       const PJRT_Api* api);

absl::StatusCode PjrtErrorCodeToStatusCode(PJRT_Error_Code code);
PJRT_Error_Code StatusCodeToPjrtErrorCode(absl::StatusCode code);

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
xla::PjRtFuture<> ConvertCEventToCppFuture(PJRT_Event* c_event,
                                           const PJRT_Api* c_api);

// The data of returned variable-length PJRT_NamedValue list is backed by
// `cpp_value_map`, so `cpp_value_map` must outlive the returned list. It will
// raise errors for unsupported PjRtValueType.
absl::StatusOr<std::vector<PJRT_NamedValue>> ConvertToPjRtNamedValueList(
    const absl::flat_hash_map<std::string, xla::PjRtValueType>& cpp_value_map);

absl::flat_hash_map<std::string, xla::PjRtValueType>
ConvertFromPjRtNamedValueList(const PJRT_NamedValue* c_value_list,
                              size_t list_size);

// Validates that all entries in value_map have a matching name and type in
// expected_name_and_type. expected_name_and_type may contain extra entries
// not in value_map without error.
absl::Status ValidateCreateOptions(
    const absl::flat_hash_map<std::string, xla::PjRtValueType>& value_map,
    const absl::flat_hash_map<std::string, PJRT_NamedValue_Type>&
        expected_name_and_types);

// Returns attributes for plugin that uses XLA compiler. The attributes have the
// lifetime of the process.
const std::vector<PJRT_NamedValue>& GetXlaPluginCAttributes();

// Helper function for checking the actual C API argument struct size is greater
// than or equal to the expected size. The actual struct size can be larger if
// it comes from a forwards-compatible caller built at a later version than this
// check. Returns a non-OK status if the expected is smaller.
absl::Status ActualStructSizeIsGreaterOrEqual(absl::string_view struct_name,
                                              size_t expected_size,
                                              size_t actual_size);

absl::string_view GetPlatformVersion(PJRT_Client* client, const PJRT_Api* api);
absl::string_view GetPlatformName(PJRT_Client* client, const PJRT_Api* api);

absl::StatusOr<PJRT_TopologyDescription*> GetTopologyDescription(
    PJRT_Client* client, const PJRT_Api* api);

// Releases `chunk`.
PJRT_Chunk ConvertFromCppChunk(xla::PjRtChunk chunk);

// Returned PjRtChunk takes ownership of data in PJRT_Chunk (i.e. chunk.deleter
// should not be called).
xla::PjRtChunk ConvertToCppChunk(const PJRT_Chunk& chunk);

PJRT_DeviceDescription* GetDeviceDescription(const PJRT_Api* api,
                                             PJRT_Device* device);

absl::Span<PJRT_Memory* const> GetAddressableMemories(const PJRT_Api* api,
                                                      PJRT_Device* device);

int GetId(const PJRT_Api* api, PJRT_DeviceDescription* device_desc);

using PJRT_KeyValueGetCFunc =
    std::function<PJRT_Error*(PJRT_KeyValueGetCallback_Args* args)>;

using PJRT_KeyValuePutCFunc =
    std::function<PJRT_Error*(PJRT_KeyValuePutCallback_Args* args)>;

// Groups data needed to support key value get/put callbacks.
struct PJRT_KeyValueCallbackData {
  PJRT_KeyValueCallbackData() = default;
  PJRT_KeyValueCallbackData(const PJRT_KeyValueCallbackData&) = delete;

  std::shared_ptr<xla::KeyValueStoreInterface> kv_store;

  // kv_get_c_func and kv_put_c_func are holding pointers to kv_store.
  pjrt::PJRT_KeyValueGetCFunc kv_get_c_func;
  pjrt::PJRT_KeyValuePutCFunc kv_put_c_func;
  // c_kv_get and c_kv_put are holding pointers to kv_get_c_func and
  // kv_put_c_func.
  PJRT_KeyValueGetCallback c_kv_get;
  PJRT_KeyValuePutCallback c_kv_put;
};

// The returned &kv_get_c_func and &kv_put_c_func must be set as
// PJRT_Client_Create_Args.kv_get_user_arg and
// PJRT_Client_Create_Args.kv_put_user_arg, respectively. The entire
// PJRT_KeyValueCallbackData must be kept alive as long as c_kv_get and c_kv_put
// may be called.
std::unique_ptr<PJRT_KeyValueCallbackData> ConvertToCKeyValueCallbacks(
    std::shared_ptr<xla::KeyValueStoreInterface> kv_store);

// std::function version of PJRT_SendCallback
using PJRT_SendCallbackFunction =
    std::function<PJRT_Error*(PJRT_Chunk*, PJRT_CallbackError*, size_t, bool)>;
// std::function version of PJRT_RecvCallback
using PJRT_RecvCallbackFunction = std::function<void(PJRT_CopyToDeviceStream*)>;

// Wraps original `xla::SendCallback` inside `PJRT_Callback` using
// 1) void* `user_arg` to capture `cpp_send_callback.callback` (std::function)
// 2) `PJRT_SendCallback` function pointer, which reinterprets and calls
// `user_arg` to call `cpp_send_callback.callback` function.
PJRT_SendCallbackInfo CppSendCallbackToCSendCallback(
    xla::SendCallback cpp_send_callback,
    PJRT_SendCallbackFunction* send_callback_function);

// Wraps original `xla::RecvCallback` inside `PJRT_Callback` using
// 1) void* `user_arg` to capture `cpp_send_callback.callback` (std::function)
// 2) `PJRT_RecvCallback` function pointer, which reinterprets and calls
// `user_arg` to call `cpp_recv_callback.callback` function.
PJRT_RecvCallbackInfo CppRecvCallbackToCRecvCallback(
    xla::RecvCallback cpp_recv_callback,
    PJRT_RecvCallbackFunction* recv_callback_function);

// Data needed to support PJRT_Buffer_MemoryLayout. `minor_to_major` holds the
// data in PJRT_Buffer_MemoryLayout_Tiled.minor_to_major. `tile_dims` and
// `tile_dim_sizes` holds the data in PJRT_Buffer_MemoryLayout_Tiled.tile_dims
// and PJRT_Buffer_MemoryLayout_Tiled.tile_dim_sizes.
struct BufferMemoryLayoutData {
  PJRT_Buffer_MemoryLayout c_layout;
  std::vector<int64_t> minor_to_major;
  std::vector<int64_t> tile_dims;
  std::vector<size_t> tile_dim_sizes;
};
absl::StatusOr<BufferMemoryLayoutData> ConvertToBufferMemoryLayoutData(
    const xla::Layout& cpp_layout);
absl::StatusOr<BufferMemoryLayoutData> ConvertToBufferMemoryLayoutData(
    absl::Span<int64_t const> byte_strides);

absl::StatusOr<xla::Layout> ConvertToLayout(
    const PJRT_Buffer_MemoryLayout_Tiled& c_tiled);

PJRT_Buffer_Type GetElementType(const PJRT_Api* api, PJRT_Buffer* buffer);
absl::Span<const int64_t> GetDimensions(const PJRT_Api* api,
                                        PJRT_Buffer* buffer);
std::unique_ptr<PJRT_Layouts_MemoryLayout, PJRT_Layouts_MemoryLayoutDeleter>
GetMemoryLayout(const PJRT_Api* api, PJRT_Buffer* buffer);

absl::StatusOr<xla::Shape> BuildXlaShapeFromC(PJRT_Buffer_Type element_type,
                                              const int64_t* dims,
                                              size_t num_dims,
                                              PJRT_Buffer_MemoryLayout* layout);

absl::string_view PlatformName(const PJRT_Api* api,
                               const PJRT_TopologyDescription* topo_desc);
absl::Span<PJRT_DeviceDescription* const> DeviceDescriptions(
    const PJRT_Api* api, const PJRT_TopologyDescription* topo_desc);

absl::StatusOr<xla::CompiledMemoryStats> GetCompiledMemoryStats(
    const PJRT_Api* api, PJRT_Executable* executable);

PJRT_ShapeSpec ConvertToPjRtShapeSpec(
    const xla::PjRtClient::ShapeSpec& shape_spec);

xla::PjRtClient::ShapeSpec ConvertFromPjrtShapeSpec(
    PJRT_ShapeSpec c_shape_spec);

// Creates a PJRT_Profiler_Extension and adds a producer trace with
// the given name. The created PJRT_Profiler_Extension will be used in argument
// structs to pass the producer traceme context id to add a corresponding
// consumer trace in the API implementation.
PJRT_Profiler_Extension CreatePjrtProfilerExtension(
    absl::string_view traceme_name);

// Traverses an extension chain to find an extension struct with type
// `type`. `in` can either be a PJRT_Api* or a pointer to an Args struct --
// anything with an `extension_start` field. The ExtType template parameter
// specifies the C extension type of the returned struct, if found (i.e. a
// specific extension struct that is layout-compatible with
// PJRT_Extension_Base).
template <typename ExtType, typename InputType>
ExtType* FindExtension(InputType* in, PJRT_Extension_Type type) {
  PJRT_Extension_Base* ext = in->extension_start;
  while (ext != nullptr) {
    if (ext->type == type) {
      return reinterpret_cast<ExtType*>(ext);
    }
    ext = ext->next;
  }
  // 'type' wasn't found in extension chain
  return nullptr;
}

// Gets a traceme context id attached to PJRT_Profiler_Extension.
// Returns -1 if there is no PJRT_Profiler_Extension in args.
template <typename InputType>
int64_t GetTracemeContextId(InputType* args) {
  PJRT_Profiler_Extension* profiler_extension =
      FindExtension<PJRT_Profiler_Extension>(
          args, PJRT_Extension_Type::PJRT_Extension_Type_Profiler);
  int64_t traceme_context_id = -1;
  if (profiler_extension != nullptr) {
    traceme_context_id = profiler_extension->traceme_context_id;
  }
  return traceme_context_id;
}

std::vector<xla::PjRtMemorySpaceDescription> GetMemorySpaceDescriptions(
    PJRT_DeviceDescription* device_description, const PJRT_Api* c_api);

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_HELPERS_H_
