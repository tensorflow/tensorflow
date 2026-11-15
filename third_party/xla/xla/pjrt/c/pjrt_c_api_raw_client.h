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

#ifdef __cplusplus
extern "C" {
#endif

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
} PJRT_RawClient_FunctionTable;

struct PJRT_RawClient {
  const PJRT_RawClient_FunctionTable* vtable;
};

PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawClient_FunctionTable, clear_memory_stats);
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawClient, vtable);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_RAW_CLIENT_H_
