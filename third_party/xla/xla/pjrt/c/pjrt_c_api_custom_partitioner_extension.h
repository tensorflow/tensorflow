/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_CUSTOM_PARTITIONER_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_CUSTOM_PARTITIONER_EXTENSION_H_

#include <cstddef>
#include <cstdint>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_CUSTOM_PARTITIONER_EXTENSION_VERSION 0

struct JAX_CustomCallPartitioner_string {
  const char* data;
  size_t size;
};

struct JAX_CustomCallPartitioner_aval {
  JAX_CustomCallPartitioner_string shape;
  bool has_sharding;
  JAX_CustomCallPartitioner_string sharding;
};

// General callback information containing api versions, the result error
// message and the cleanup function to free any temporary memory that is backing
// the results. Arguments are always owned by the caller, and results are owned
// by the cleanup_fn. These should never be used directly. Args and results
// should be serialized via the PopulateArgs, ReadArgs, PopulateResults,
// ConsumeResults functions defined below.
struct JAX_CustomCallPartitioner_version_and_error {
  int64_t api_version;
  void* data;  // out
  // cleanup_fn cleans up any returned results. The caller must finish with all
  // uses by the point the cleanup is called.
  void (*cleanup_fn)(void* data);  // out
  bool has_error;
  PJRT_Error_Code code;                        // out
  JAX_CustomCallPartitioner_string error_msg;  // out
};

struct JAX_CustomCallPartitioner_Partition_Args {
  JAX_CustomCallPartitioner_version_and_error header;

  size_t num_args;
  JAX_CustomCallPartitioner_aval* op_args;
  JAX_CustomCallPartitioner_aval op_result;
  JAX_CustomCallPartitioner_string backend_config;

  // out
  JAX_CustomCallPartitioner_string mlir_module;
  JAX_CustomCallPartitioner_string* args_sharding;
  JAX_CustomCallPartitioner_string result_sharding;
};

struct JAX_CustomCallPartitioner_InferShardingFromOperands_Args {
  JAX_CustomCallPartitioner_version_and_error header;

  size_t num_args;
  JAX_CustomCallPartitioner_aval* op_args;
  JAX_CustomCallPartitioner_string result_shape;
  JAX_CustomCallPartitioner_string backend_config;

  bool has_result_sharding;
  JAX_CustomCallPartitioner_string result_sharding;
};

struct JAX_CustomCallPartitioner_PropagateUserSharding_Args {
  JAX_CustomCallPartitioner_version_and_error header;

  JAX_CustomCallPartitioner_string backend_config;

  JAX_CustomCallPartitioner_string result_shape;

  JAX_CustomCallPartitioner_string result_sharding;  // inout
};

struct JAX_CustomCallPartitioner_Callbacks {
  int64_t version;
  void* private_data;
  void (*dtor)(JAX_CustomCallPartitioner_Callbacks* data);
  void (*partition)(JAX_CustomCallPartitioner_Callbacks* data,
                    JAX_CustomCallPartitioner_Partition_Args* args);
  void (*infer_sharding)(
      JAX_CustomCallPartitioner_Callbacks* data,
      JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args);
  void (*propagate_user_sharding)(
      JAX_CustomCallPartitioner_Callbacks* data,
      JAX_CustomCallPartitioner_PropagateUserSharding_Args* args);
  bool can_side_effecting_have_replicated_sharding;
};

struct PJRT_Register_Custom_Partitioner_Args {
  size_t struct_size;
  const char* name;  // lifetime of the call.
  size_t name_size;
  JAX_CustomCallPartitioner_Callbacks* callbacks;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Register_Custom_Partitioner_Args, callbacks);

// Registers a custom partitioner.
typedef PJRT_Error* PJRT_Register_Custom_Partitioner(
    PJRT_Register_Custom_Partitioner_Args* args);

typedef struct PJRT_Custom_Partitioner_Extension {
  size_t struct_size;
  PJRT_Extension_Type type;
  PJRT_Extension_Base* next;
  PJRT_Register_Custom_Partitioner* register_custom_partitioner;
} PJRT_Custom_Partitioner_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Custom_Partitioner_Extension,
                          register_custom_partitioner);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_CUSTOM_PARTITIONER_EXTENSION_H_
