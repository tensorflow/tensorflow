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

#ifndef XLA_FFI_API_COLLECTIVES_C_API_H_
#define XLA_FFI_API_COLLECTIVES_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/ffi/api/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// XLA FFI Collectives API
//===----------------------------------------------------------------------===//

// Exposes the XLA-owned host collective communicator and collective memory
// window to FFI handlers.
// `request_communicator` requests a clique in the Prepare stage;
// `get_communicator` returns the handle once cliques are acquired.
// `request_window` requests window registration for a batch of already-
// allocated buffers with the clique in Prepare; `get_window` returns the
// corresponding window handle once collective memory is acquired.
// The communicator and window are backend-defined and passed as opaque
// pointers (a handler on XLA:GPU reinterprets `XLA_FFI_Communicator*` as
// `ncclComm_t` and `XLA_FFI_Window*` as `ncclWindow_t`).
// This API is backend agnostic; backend wiring lives in the runtime that
// attaches the extension.

#define XLA_FFI_Extension_Collectives 129
#define XLA_FFI_Extension_Collectives_MajorVersion 0
#define XLA_FFI_Extension_Collectives_MinorVersion 2

// Mirrors `xla::CollectiveOpGroupMode`.
typedef enum XLA_FFI_CollectiveGroupMode {
  XLA_FFI_GROUP_CROSS_REPLICA = 0,
  XLA_FFI_GROUP_CROSS_PARTITION = 1,
  XLA_FFI_GROUP_CROSS_REPLICA_AND_PARTITION = 2,
  XLA_FFI_GROUP_FLATTENED_ID = 3,
} XLA_FFI_CollectiveGroupMode;

typedef struct XLA_FFI_ReplicaGroup {
  const int64_t* ids;
  size_t size;
} XLA_FFI_ReplicaGroup;

typedef struct XLA_FFI_Collectives_Extension XLA_FFI_Collectives_Extension;

// Opaque, backend-defined per-invocation collective state. Set by the runtime
// that attaches the extension and interpreted only by the callbacks below.
typedef struct XLA_FFI_CollectivesState XLA_FFI_CollectivesState;

// Opaque, non-owning communicator handle. The backend defines the concrete
// type; callers reinterpret it (e.g. as `ncclComm_t`).
typedef struct XLA_FFI_Communicator XLA_FFI_Communicator;

//===----------------------------------------------------------------------===//
// Communicator: request in Prepare, get in Initialize/Execute
//===----------------------------------------------------------------------===//

typedef struct XLA_FFI_Communicator_Request_Args {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_CollectiveGroupMode group_mode;
  const XLA_FFI_ReplicaGroup* groups;
  size_t num_groups;
  int64_t communication_id;
} XLA_FFI_Communicator_Request_Args;

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Communicator_Request_Args,
                             communication_id);

// Requests the collective clique so it is acquired before execution. Prepare
// stage only.
typedef XLA_FFI_Error* XLA_FFI_Communicator_Request(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Communicator_Request_Args* args);

typedef struct XLA_FFI_Communicator_Get_Args {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_CollectiveGroupMode group_mode;
  const XLA_FFI_ReplicaGroup* groups;
  size_t num_groups;
  int64_t communication_id;
  XLA_FFI_Communicator* communicator;  // out
} XLA_FFI_Communicator_Get_Args;

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Communicator_Get_Args, communicator);

// Returns the non-owning communicator handle for the clique. Valid once
// cliques are acquired (Initialize/Execute stages).
typedef XLA_FFI_Error* XLA_FFI_Communicator_Get(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Communicator_Get_Args* args);

//===----------------------------------------------------------------------===//
// Collective memory window
//===----------------------------------------------------------------------===//
//
// Handlers request window registration for a batch of already-allocated
// buffers with the clique via `request_window` in Prepare, then look up an
// opaque `XLA_FFI_Window` per buffer via `get_window` in Init/Execute.
// Handlers reinterpret the window and call the backend's collective device APIs
// directly to obtain local, peer, and multicast pointers.

// Opaque, non-owning collective memory window handle. The backend defines the
// concrete type; callers reinterpret it (e.g. as `ncclWindow_t`).
typedef struct XLA_FFI_Window XLA_FFI_Window;

typedef struct XLA_FFI_CollectiveMemoryRegion {
  const void* buffer;
  size_t byte_size;
  uint64_t flags;
} XLA_FFI_CollectiveMemoryRegion;

typedef struct XLA_FFI_Window_Request_Args {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_CollectiveGroupMode group_mode;
  const XLA_FFI_ReplicaGroup* groups;
  size_t num_groups;
  int64_t communication_id;

  const XLA_FFI_CollectiveMemoryRegion* regions;
  size_t num_regions;
} XLA_FFI_Window_Request_Args;

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Window_Request_Args, num_regions);

// Requests window registration for a batch of already-allocated buffers with
// the clique. Actual registration happens later once the clique is acquired;
// look up the resulting window handle with `get_window`. Prepare stage only.
typedef XLA_FFI_Error* XLA_FFI_Window_Request(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Window_Request_Args* args);

typedef struct XLA_FFI_Window_Get_Args {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_CollectiveGroupMode group_mode;
  const XLA_FFI_ReplicaGroup* groups;
  size_t num_groups;
  int64_t communication_id;

  const void* buffer;      // registered address (in)
  XLA_FFI_Window* window;  // out
  size_t window_offset;    // byte offset of `buffer` within `window` (out)
} XLA_FFI_Window_Get_Args;

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Window_Get_Args, window_offset);

// Returns the non-owning collective memory window handle for a buffer whose
// registration was previously requested via `request_window`, along with the
// byte offset of `buffer` within that window. Valid once collective memory is
// acquired (Initialize/Execute stages).
typedef XLA_FFI_Error* XLA_FFI_Window_Get(
    const XLA_FFI_Collectives_Extension* self, XLA_FFI_Window_Get_Args* args);

//===----------------------------------------------------------------------===//
// Extension struct
//===----------------------------------------------------------------------===//

struct XLA_FFI_Collectives_Extension {
  XLA_FFI_Extension extension_base;

  XLA_FFI_CollectivesState* state;

  XLA_FFI_Communicator_Request* request_communicator;
  XLA_FFI_Communicator_Get* get_communicator;

  XLA_FFI_Window_Request* request_window;
  XLA_FFI_Window_Get* get_window;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Collectives_Extension, get_window);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_FFI_API_COLLECTIVES_C_API_H_
