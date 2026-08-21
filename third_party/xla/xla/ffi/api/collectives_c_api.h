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

// Exposes the XLA-owned host collective communicator to FFI handlers.
// `request_communicator` requests a clique in the Prepare stage;
// `get_communicator` returns the handle once cliques are acquired. The
// communicator is backend-defined and passed as an opaque `void*` (a handler on
// XLA:GPU reinterprets it as `ncclComm_t`, XLA:CPU as its own comm type, etc.).
// This API is backend agnostic; backend wiring lives in the runtime that
// attaches the extension.

// Version macros for the collectives extension.
#define XLA_FFI_Extension_Collectives 129
#define XLA_FFI_Extension_Collectives_MajorVersion 0
#define XLA_FFI_Extension_Collectives_MinorVersion 1

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

// Returns the non-owning communicator handle for the clique. Valid once cliques
// are acquired (Initialize/Execute stages).
typedef XLA_FFI_Error* XLA_FFI_Communicator_Get(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Communicator_Get_Args* args);

// Main extension struct for the collectives API.
struct XLA_FFI_Collectives_Extension {
  XLA_FFI_Extension extension_base;

  XLA_FFI_CollectivesState* state;

  XLA_FFI_Communicator_Request* request_communicator;
  XLA_FFI_Communicator_Get* get_communicator;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Collectives_Extension, get_communicator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_FFI_API_COLLECTIVES_C_API_H_
