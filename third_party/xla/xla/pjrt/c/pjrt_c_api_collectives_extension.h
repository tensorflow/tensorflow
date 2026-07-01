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

#ifndef XLA_PJRT_C_PJRT_C_API_COLLECTIVES_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_COLLECTIVES_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_COLLECTIVES_EXTENSION_VERSION 1

typedef struct PJRT_Collectives PJRT_Collectives;
typedef struct PJRT_Collectives_Communicator PJRT_Collectives_Communicator;
typedef struct PJRT_Collectives_Communicators PJRT_Collectives_Communicators;
typedef struct PJRT_Collectives_ToString_Holder
    PJRT_Collectives_ToString_Holder;

struct PJRT_Collectives_Destroy_Args {
  size_t struct_size;
  PJRT_Collectives* collectives;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_Destroy_Args, collectives);

typedef PJRT_Error* PJRT_Collectives_Destroy(
    PJRT_Collectives_Destroy_Args* args);

struct PJRT_Collectives_CreateCommunicators_Args {
  size_t struct_size;
  PJRT_Collectives* collectives;

  // CliqueKey fields.
  const int64_t* device_ids;
  size_t num_device_ids;

  // CliqueIds fields.
  const char** clique_ids;  // May be null if clique_ids is not set.
  const size_t* clique_id_sizes;
  size_t num_clique_ids;

  // DeviceRank fields.
  int64_t* rank_ids;
  size_t num_device_ranks;

  PJRT_Collectives_Communicator** communicators;  // out array of pointers
  size_t num_communicators;                       // out

  // Holder for the communicator array. Individual Communicator will still be
  // valid after the array is destroyed.
  PJRT_Collectives_Communicators* communicators_holder;  // out
  void (*communicators_holder_deleter)(
      PJRT_Collectives_Communicators* ptr);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_CreateCommunicators_Args,
                          communicators_holder_deleter);

typedef PJRT_Error* PJRT_Collectives_CreateCommunicators(
    PJRT_Collectives_CreateCommunicators_Args* args);

struct PJRT_Collectives_Communicator_Destroy_Args {
  size_t struct_size;
  PJRT_Collectives_Communicator* communicator;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_Communicator_Destroy_Args,
                          communicator);

typedef PJRT_Error* PJRT_Collectives_Communicator_Destroy(
    PJRT_Collectives_Communicator_Destroy_Args* args);

// Communicator operations

enum PJRT_Collectives_ReductionKind : int32_t {
  PJRT_COLLECTIVES_REDUCTION_SUM = 0,
  PJRT_COLLECTIVES_REDUCTION_PRODUCT = 1,
  PJRT_COLLECTIVES_REDUCTION_MIN = 2,
  PJRT_COLLECTIVES_REDUCTION_MAX = 3,
};

enum PJRT_Collectives_CollectiveOpKind : int32_t {
  PJRT_COLLECTIVES_COLLECTIVE_OP_KIND_CROSS_MODULE = 0,
  PJRT_COLLECTIVES_COLLECTIVE_OP_KIND_CROSS_REPLICA = 1,
};

struct PJRT_Collectives_CpuExecutor {
  // RendezvousKey.
  int64_t run_id;
  int64_t* global_device_ids;
  size_t num_global_device_ids;
  int32_t num_local_participants;
  PJRT_Collectives_CollectiveOpKind collective_op_kind;
  int64_t op_id;

  int64_t timeout_in_ns;
};

struct PJRT_Collectives_Communicator_AllReduce_Args {
  size_t struct_size;
  PJRT_Collectives_Communicator* communicator;
  void* send_buffer_ptr;
  size_t send_buffer_size;
  void* recv_buffer_ptr;
  size_t recv_buffer_size;
  PJRT_Buffer_Type primitive_type;
  size_t count;
  PJRT_Collectives_ReductionKind reduction_kind;
  PJRT_Collectives_CpuExecutor* cpu_executor;
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_Communicator_AllReduce_Args, event);

typedef PJRT_Error* PJRT_Collectives_Communicator_AllReduce(
    PJRT_Collectives_Communicator_AllReduce_Args* args);

struct PJRT_Collectives_Communicator_ReduceScatter_Args {
  size_t struct_size;
  PJRT_Collectives_Communicator* communicator;
  void* send_buffer_ptr;
  size_t send_buffer_size;
  void* recv_buffer_ptr;
  size_t recv_buffer_size;
  PJRT_Buffer_Type primitive_type;
  size_t count;
  PJRT_Collectives_ReductionKind reduction_kind;
  PJRT_Collectives_CpuExecutor* cpu_executor;
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_Communicator_ReduceScatter_Args,
                          event);

typedef PJRT_Error* PJRT_Collectives_Communicator_ReduceScatter(
    PJRT_Collectives_Communicator_ReduceScatter_Args* args);

struct PJRT_Collectives_Communicator_AllGather_Args {
  size_t struct_size;
  PJRT_Collectives_Communicator* communicator;

  void* send_buffer_ptr;
  size_t send_buffer_size;

  void* recv_buffer_ptr;
  size_t recv_buffer_size;

  PJRT_Buffer_Type primitive_type;
  size_t count;
  PJRT_Collectives_CpuExecutor* cpu_executor;
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_Communicator_AllGather_Args, event);

typedef PJRT_Error* PJRT_Collectives_Communicator_AllGather(
    PJRT_Collectives_Communicator_AllGather_Args* args);

struct PJRT_Collectives_Communicator_CollectivePermute_Args {
  size_t struct_size;
  PJRT_Collectives_Communicator* communicator;

  void* send_buffer_ptr;
  size_t send_buffer_size;

  void* recv_buffer_ptr;
  size_t recv_buffer_size;

  PJRT_Buffer_Type primitive_type;
  size_t count;

  bool has_source_rank;
  int64_t source_rank;

  int64_t* target_ranks;
  size_t num_target_ranks;

  PJRT_Collectives_CpuExecutor* cpu_executor;
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_Communicator_CollectivePermute_Args,
                          event);

typedef PJRT_Error* PJRT_Collectives_Communicator_CollectivePermute(
    PJRT_Collectives_Communicator_CollectivePermute_Args* args);

struct PJRT_Collectives_Communicator_AllToAll_Args {
  size_t struct_size;
  PJRT_Collectives_Communicator* communicator;

  void** send_buffers_ptrs;
  const size_t* send_buffers_sizes;
  size_t num_send_buffers;

  void** recv_buffers_ptrs;
  const size_t* recv_buffers_sizes;
  size_t num_recv_buffers;

  PJRT_Buffer_Type primitive_type;
  size_t count;

  PJRT_Collectives_CpuExecutor* cpu_executor;

  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_Communicator_AllToAll_Args, event);

typedef PJRT_Error* PJRT_Collectives_Communicator_AllToAll(
    PJRT_Collectives_Communicator_AllToAll_Args* args);

struct PJRT_Collectives_Communicator_ToString_Args {
  size_t struct_size;
  PJRT_Collectives_Communicator* communicator;
  const char* str;  // out
  size_t str_size;  // out

  PJRT_Collectives_ToString_Holder* str_holder;                       // out
  void (*str_holder_deleter)(PJRT_Collectives_ToString_Holder* ptr);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_Communicator_ToString_Args,
                          str_holder_deleter);

typedef PJRT_Error* PJRT_Collectives_Communicator_ToString(
    PJRT_Collectives_Communicator_ToString_Args* args);

typedef struct PJRT_Collectives_Extension {
  PJRT_Extension_Base base;
  PJRT_Collectives_Destroy* collectives_destroy;
  PJRT_Collectives_CreateCommunicators* collectives_create_communicators;
  PJRT_Collectives_Communicator_Destroy* communicator_destroy;
  PJRT_Collectives_Communicator_AllReduce* communicator_all_reduce;
  PJRT_Collectives_Communicator_ReduceScatter* communicator_reduce_scatter;
  PJRT_Collectives_Communicator_AllGather* communicator_all_gather;
  PJRT_Collectives_Communicator_CollectivePermute*
      communicator_collective_permute;
  PJRT_Collectives_Communicator_AllToAll* communicator_all_to_all;
  PJRT_Collectives_Communicator_ToString* communicator_to_string;
} PJRT_Collectives_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Collectives_Extension, communicator_to_string);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_COLLECTIVES_EXTENSION_H_
