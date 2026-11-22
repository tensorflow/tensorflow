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

#ifndef XLA_PJRT_C_PJRT_C_API_RAW_CLIENT_H_
#define XLA_PJRT_C_PJRT_C_API_RAW_CLIENT_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PJRT_OnceCallback PJRT_OnceCallback;

typedef struct PJRT_OnceCallback_FunctionTable {
  size_t struct_size;
  size_t instance_size;
  PJRT_Extension_Base* extension_start;
  // Invokes and frees `callback`.
  void (*invoke)(PJRT_OnceCallback* callback);
  // Frees `callback` without invoking it.
  void (*destroy)(PJRT_OnceCallback* callback);
} PJRT_OnceCallback_FunctionTable;

struct PJRT_OnceCallback {
  const PJRT_OnceCallback_FunctionTable* vtable;
};

PJRT_DEFINE_STRUCT_TRAITS(PJRT_OnceCallback_FunctionTable, destroy);
PJRT_DEFINE_STRUCT_TRAITS(PJRT_OnceCallback, vtable);

typedef struct PJRT_FulfillAliasRawBufferCallback
    PJRT_FulfillAliasRawBufferCallback;

typedef struct PJRT_FulfillAliasRawBufferCallback_FunctionTable {
  size_t struct_size;
  size_t instance_size;
  PJRT_Extension_Base* extension_start;
  // Binds the alias to `buffer` (ownership transferred) and frees `callback`.
  PJRT_NO_DISCARD PJRT_Error* (*fulfill_buffer)(
      PJRT_FulfillAliasRawBufferCallback* callback, PJRT_RawBuffer* buffer);
  // Sets non-null `error` (ownership transferred) on the alias and frees
  // `callback`.
  void (*fulfill_error)(PJRT_FulfillAliasRawBufferCallback* callback,
                        PJRT_Error* error);
  // Frees `callback` without fulfilling it.
  void (*destroy)(PJRT_FulfillAliasRawBufferCallback* callback);
} PJRT_FulfillAliasRawBufferCallback_FunctionTable;

struct PJRT_FulfillAliasRawBufferCallback {
  const PJRT_FulfillAliasRawBufferCallback_FunctionTable* vtable;
};

PJRT_DEFINE_STRUCT_TRAITS(PJRT_FulfillAliasRawBufferCallback_FunctionTable,
                          destroy);
PJRT_DEFINE_STRUCT_TRAITS(PJRT_FulfillAliasRawBufferCallback, vtable);

typedef struct PJRT_RawClient PJRT_RawClient;

typedef struct PJRT_RawClient_FunctionTable {
  size_t struct_size;
  size_t instance_size;
  PJRT_Extension_Base* extension_start;
  // Frees the PJRT_RawClient.
  void (*destroy)(PJRT_RawClient* client);
  // Quiesces any background work that may reference the owning client before
  // its members are destroyed.
  void (*quiesce)(PJRT_RawClient* client);
  // Returns true if memory allocations for async transfers in the memory space
  // with `memory_kind_id` should be deferred behind an allocation event.
  bool (*should_create_async_allocation_event)(const PJRT_RawClient* client,
                                               int memory_kind_id);
  // Experimental: Maps memory for fast transfers. May have backend-specific
  // alignment requirements (most backends will require at least a page).
  PJRT_NO_DISCARD PJRT_Error* (*dma_map)(PJRT_RawClient* client, void* data,
                                         size_t size);
  // Experimental: Unmaps memory for fast transfers. Caller is responsible to
  // ensure that all data transfers are complete before calling dma_unmap.
  PJRT_NO_DISCARD PJRT_Error* (*dma_unmap)(PJRT_RawClient* client, void* data);
  // Returns true if the host memory range [data, data + transfer_size) is
  // already DMA-mapped (e.g. pinned on GPU or in a premapped buffer on TPU).
  bool (*is_dma_mapped)(const PJRT_RawClient* client, const void* data,
                        int64_t transfer_size);
  // Returns the NUMA node associated with `local_device_id`.
  int (*get_numa_node)(const PJRT_RawClient* client, int32_t local_device_id);
  // Returns the required alignment for device memory addresses when slicing.
  PJRT_NO_DISCARD PJRT_Error* (*get_device_address_alignment)(
      const PJRT_RawClient* client, size_t* out_alignment);
  // Returns the required byte alignment for host memory when performing DMA.
  size_t (*get_dma_host_alignment)(const PJRT_RawClient* client);
  // Returns the required byte alignment for transfer sizes when performing DMA,
  // or false if unpadded DMA transfers are not supported.
  bool (*get_dma_size_alignment)(const PJRT_RawClient* client,
                                 size_t* out_alignment);
  // Backend-specific handlers for when an OOM is detected during execute.
  void (*call_oom_handlers)(const PJRT_RawClient* client);
  // Records memory statistics for profiling/monitoring.
  void (*record_memory_stats)(PJRT_RawClient* client);
  // Records the number of linearized bytes for H2D or D2H transfers.
  void (*record_linearize_bytes)(PJRT_RawClient* client, bool is_h2d,
                                 size_t bytes);
  // Returns the opaque platform-specific stream used for external ready events
  // on `local_device_id`.
  PJRT_NO_DISCARD PJRT_Error* (*get_stream_for_external_ready_events)(
      const PJRT_RawClient* client, int32_t local_device_id,
      intptr_t* out_stream);
  // Clears allocator memory statistics for `local_device_id`.
  PJRT_NO_DISCARD PJRT_Error* (*clear_memory_stats)(PJRT_RawClient* client,
                                                    int32_t local_device_id);
  // Allocates a caller-owned raw buffer in `memory_space` for execute.
  PJRT_NO_DISCARD PJRT_Error* (*allocate_raw_buffer_for_execute)(
      PJRT_RawClient* client, PJRT_Memory* memory_space,
      size_t on_device_bytes_count, bool retry_on_oom,
      PJRT_RawBuffer** out_buffer);
  // Allocates `num_buffers` caller-owned raw buffers in `memory_space` for
  // execute into caller-allocated `out_buffers[0..num_buffers)`.
  PJRT_NO_DISCARD PJRT_Error* (*allocate_raw_buffers_for_execute)(
      PJRT_RawClient* client, PJRT_Memory* memory_space,
      const size_t* on_device_bytes_counts, size_t num_buffers,
      bool retry_on_oom, PJRT_RawBuffer** out_buffers);
  // Creates a caller-owned linked device event promise and device event.
  PJRT_NO_DISCARD PJRT_Error* (*create_linked_event_promise)(
      PJRT_RawClient* client, int32_t local_device_id, int memory_kind_id,
      const char* debug_info, size_t debug_info_size,
      PJRT_DeviceEventPromise** out_promise, PJRT_DeviceEvent* out_event);
  // Creates a caller-owned device event signaling completion on `stream`.
  PJRT_NO_DISCARD PJRT_Error* (*create_device_event_for_stream)(
      PJRT_RawClient* client, int32_t local_device_id, intptr_t stream,
      PJRT_DeviceEvent* out_event);
  // Updates coordination process info across tasks.
  void (*update_global_process_info)(PJRT_RawClient* client,
                                     const PJRT_ProcessInfo* process_infos,
                                     size_t num_process_infos);
  // Poisons execution `launch_id` on `local_device_id`. Takes ownership of
  // `error`.
  PJRT_NO_DISCARD PJRT_Error* (*poison_execution)(PJRT_RawClient* client,
                                                  int32_t local_device_id,
                                                  int32_t launch_id,
                                                  PJRT_Error* error,
                                                  bool* out_poisoned);
  // Returns true if buffer donation can be recovered after `error`. Takes
  // ownership of `error`.
  bool (*can_recover_donation)(const PJRT_RawClient* client, PJRT_Error* error);
  // Enqueues a wait for `event` (ownership transferred) on `stream`.
  PJRT_NO_DISCARD PJRT_Error* (*wait_on_stream)(PJRT_RawClient* client,
                                                int32_t local_device_id,
                                                PJRT_DeviceEvent event,
                                                intptr_t stream);
  // Populates `out_stats` (`device` field unused) with allocator statistics for
  // `local_device_id`.
  PJRT_NO_DISCARD PJRT_Error* (*get_allocator_stats)(
      const PJRT_RawClient* client, int32_t local_device_id,
      PJRT_Device_MemoryStats_Args* out_stats);
  // Imports foreign memory as a caller-owned raw buffer. Takes ownership of
  // `on_delete_callback` (may be nullptr): on success, invokes it when done
  // with `device_ptr`; on error, destroys it without invoking.
  PJRT_NO_DISCARD PJRT_Error* (*import_foreign_memory)(
      PJRT_RawClient* client, PJRT_Memory* memory_space, void* device_ptr,
      size_t size, PJRT_OnceCallback* on_delete_callback, bool is_mutable,
      PJRT_RawBuffer** out_buffer);
  // Allocates a caller-owned raw buffer in `memory_space` after optional
  // `allocate_after` (ownership transferred; `{nullptr, nullptr}` if none).
  PJRT_NO_DISCARD PJRT_Error* (*allocate_raw_buffer)(
      PJRT_RawClient* client, PJRT_Memory* memory_space,
      size_t on_device_bytes_count, bool retry_on_oom,
      PJRT_DeviceEvent allocate_after, PJRT_RawBuffer** out_buffer);
  // Enqueues `execute_fn` (ownership transferred) to run on `local_device_id`.
  void (*launch_on_device)(const PJRT_RawClient* client,
                           int32_t local_device_id,
                           PJRT_OnceCallback* execute_fn);
  // Creates a caller-owned raw buffer alias and one-shot fulfillment callback.
  PJRT_NO_DISCARD PJRT_Error* (*create_raw_buffer_channel)(
      PJRT_RawClient* client, PJRT_Memory* memory_space,
      size_t on_device_bytes_count, PJRT_RawBuffer** out_buffer,
      PJRT_FulfillAliasRawBufferCallback** out_fulfill_callback);
} PJRT_RawClient_FunctionTable;

struct PJRT_RawClient {
  const PJRT_RawClient_FunctionTable* vtable;
};

PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawClient_FunctionTable,
                          create_raw_buffer_channel);
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawClient, vtable);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_RAW_CLIENT_H_
