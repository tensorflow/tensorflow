/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_TPU_OPS_C_API_H_
#define TENSORFLOW_CORE_TPU_TPU_OPS_C_API_H_

#include <stddef.h>

#include <cstdint>

#include "absl/types/optional.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/proto_helper.h"
#include "tensorflow/core/tpu/libtftpu.h"

typedef struct TpuSerializedProto TpuSerializedProto;

namespace tensorflow {

class TpuMeshCommonState;
class TpuEmbeddingEngineState;
class ResourceMgr;

}  // namespace tensorflow

extern "C" {

typedef struct XLA_TpuProgram XLA_TpuProgram;

// Enum for choosing sharding/unsharding program from a `XLA_TpuProgram` obj.
enum TpuProgramShardingType { kInvalid = 0, kMain, kSharding, kUnsharding };

struct TpuProgramFingerprint {
  const char* bytes;
  size_t size;
};

struct TpuExecutableSerializedProto {
  const char* bytes;
  size_t size;
};

struct CompilerMetadataSerializedProto {
  const char* bytes;
  size_t size;
};

struct HostComputeMetadataSerializedProto {
  const char* bytes;
  size_t size;
};

typedef struct XLA_TpuMeshState XLA_TpuMeshState;

typedef struct XLA_TpuEmbeddingEngineState XLA_TpuEmbeddingEngineState;

typedef struct TpuEmbedding_TensorBatchFixedState
    TpuEmbedding_TensorBatchFixedState;

typedef struct TpuProfiler TpuProfiler;

typedef struct XLA_DeviceAssignment {
  const char* bytes;
  size_t size;
} XLA_DeviceAssignment;

// Property for creating compilation cache key.
struct CompilationCacheKeyProperty {
  const char* config_prefix;
  const char* shapes_prefix;
  const char* function_name;
  uint64_t mlir_module_fingerprint;
  const int32_t* device_ids;
  size_t device_ids_size;
  int32_t guaranteed_constants_size;
  uint64_t function_library_fingerprint;
  int32_t num_cores_per_replica;
  int32_t num_replicas;
  const XLA_TpuMeshState* mesh_state;
  uint64_t session_id;
  tensorflow::ResourceMgr* resource_mgr;
};

// Compilation cache key result returning both the key and a more verbose debug
// version.
struct CompilationCacheKeyResult {
  const char* key;
  const char* debug_string;
};

typedef struct XLA_TpuNodeContext XLA_TpuNodeContext;

typedef struct TfTpu_OrdinalSelector TfTpuOrdinalSelector;

struct TpuPartitionedCall_Params {
  bool input_shape_opt;
  bool group_tensors_for_packing;
  int32_t minimum_input_tensors_packing;
  int32_t minimum_output_tensors_packing;

  // Whether to attempt to automatically shard inputs by adding an
  // XlaSharding op after each input.
  bool enable_auto_xla_input_sharding;

  // The dimension of each input to shard if
  // enable_auto_xla_input_sharding is set to true. Negative numbers are
  // allowed and refers to dimensions starting from the end.
  int32_t auto_xla_input_sharding_dim;

  // If true, only create one variable on the TPU for each variable on the CPU.
  bool enable_variable_deduplication;
};

// Compiles Mlir or TF function computation by lowering into HLO IR and returns
// `count` number of TPU programs ready for execution.
// The API allocates the `XLA_TpuProgram*[]` array `tpu_programs` and creates
// `XLA_TpuProgram` object(s) using the `TpuProgram_New` API. The caller is
// responsible to deallocate both the `XLA_TpuProgram*[]` array and the
// `XLA_TpuProgram` object(s) using `TpuProgram_FreeArray` and `TpuProgram_Free`
// API respectively.
TFTPU_CAPI_EXPORT void TpuCompile_CompileAndBuild(
    TpuSerializedProto compilation_request, const XLA_TpuMeshState* mesh_state,
    XLA_TpuProgram** tpu_programs[], size_t* count, TF_Status* status);

// Compiles a HLO IR and returns `count` number of TPU programs ready for
// execution. The API allocates the `XLA_TpuProgram*[]` array `tpu_programs` and
// creates `XLA_TpuProgram` object(s) using the `TpuProgram_New` API. The caller
// is responsible to deallocate both the `XLA_TpuProgram*[]` array and the
// `XLA_TpuProgram` object(s) using `TpuProgram_FreeArray` and `TpuProgram_Free`
// API respectively.
TFTPU_CAPI_EXPORT void TpuCompile_XrtCompileAndBuild(
    TpuSerializedProto xrt_computation, const XLA_TpuMeshState* mesh_state,
    XLA_TpuProgram** tpu_programs[], size_t* count, TF_Status* status);

// Creates a TPU profiler that is ready to start profiling.
TFTPU_CAPI_EXPORT void TpuProfiler_Create(TpuProfiler** tpu_profiler,
                                          TF_Status* status);
// Destroys the given TPU profiler.
TFTPU_CAPI_EXPORT void TpuProfiler_Destroy(TpuProfiler* tpu_profiler);
// Starts profiling if not already started, returns an error otherwise.
TFTPU_CAPI_EXPORT void TpuProfiler_Start(TpuProfiler* tpu_profiler,
                                         TF_Status* status);
// Stops profiling if not already stopped, returns an error otherwise.
TFTPU_CAPI_EXPORT void TpuProfiler_Stop(TpuProfiler* tpu_profiler,
                                        TF_Status* status);
// Serializes profiled data into `buffer` and returns the size of `buffer`. The
// profile data held by the TPU driver will be cleared after retrieval.
//
// Step 1. Query the size of buffer required into `size_in_bytes`.
//
//   size_t size_in_bytes;
//   TpuProfiler_CollectData(profiler, status, nullptr, &size_in_bytes);
//
// Step 2. Retrieve the data into a `buffer` of size `size_in_bytes`.
//         Subsequently,The TPU driver clears its copy of the profile data.
//
//   uint8_t buffer = new uint8_t[size_in_bytes];
//   TpuProfiler_CollectData(profiler, status, buffer, size_in_bytes);
//
// Step 3. Unpack the data into an XSpace.
//
//   tensorflow::profiler::XSpace space;
//   space.ParseFromArray(buffer, size_in_bytes);
//
TFTPU_CAPI_EXPORT void TpuProfiler_CollectData(TpuProfiler* tpu_profiler,
                                               TF_Status* status,
                                               uint8_t* buffer,
                                               size_t* size_in_bytes);

// Creates a new TPU mesh state object.
TFTPU_CAPI_EXPORT XLA_TpuMeshState* TpuMeshState_Create();

// Deletes the given TPU `mesh_state` object. Once deleted the object is
// unusable.
TFTPU_CAPI_EXPORT void TpuMeshState_Free(XLA_TpuMeshState* mesh_state);

// Returns a pointer to an opaque mesh data structure used internally.
TFTPU_CAPI_EXPORT void* TpuMeshState_MeshCommonState(
    XLA_TpuMeshState* mesh_state);

// Creates a new TPU embedding engine state object.
TFTPU_CAPI_EXPORT XLA_TpuEmbeddingEngineState* TpuEmbeddingEngineState_Create();

// Delete the given TPU embedding engine state object. Once deleted the object
// is unusable.
TFTPU_CAPI_EXPORT void TpuEmbeddingEngineState_Free(
    XLA_TpuEmbeddingEngineState* engine_state);

// Returns a pointer to an opaque embedding engine state data structure used
// internally.
TFTPU_CAPI_EXPORT void* TpuEmbeddingEngineState_GetState(
    XLA_TpuEmbeddingEngineState* engine_state);

TFTPU_CAPI_EXPORT void TfTpuOrdinalSelector_Create(
    TfTpuOrdinalSelector** ordinal_selector, int num_cores_per_replica);

TFTPU_CAPI_EXPORT void TfTpuOrdinalSelector_Destroy(
    TfTpuOrdinalSelector* ordinal_selector);

TFTPU_CAPI_EXPORT void TfTpuOrdinalSelector_GetOrdinal(
    TfTpuOrdinalSelector* ordinal_selector, std::optional<uint64_t> key,
    int64_t* req_id, int64_t* ordinal);

TFTPU_CAPI_EXPORT void TfTpuOrdinalSelector_DequeueFromCoreSelector(
    TfTpuOrdinalSelector* ordinal_selector, int32_t device_ordinal,
    int64_t req_id);

TFTPU_CAPI_EXPORT void TfTpu_GetTpuPartitionedCallParams(
    TpuPartitionedCall_Params* params);

typedef struct TpuExecutable_LoadProgramAndEnqueueToStream_Params {
  int32_t struct_size;
  void* priv;

  const XLA_TpuProgram* program;
  SE_DeviceMemoryBase* arguments;
  size_t arguments_len;
  SE_DeviceMemoryBase* result;
  bool has_cross_program_prefetch_addr;
  SE_DeviceMemoryBase* cross_program_prefetch_addr;
  int32_t rng_seed;
  XLA_DeviceAssignment* device_assignment;
  SE_Stream* stream;

  TF_Status* status;  // out
} TpuExecutable_LoadProgramAndEnqueueToStream_Params;

#define TpuExecutable_LoadProgramAndEnqueueToStream_Params_SIZE \
  (sizeof(struct TpuExecutable_LoadProgramAndEnqueueToStream_Params))

TFTPU_CAPI_EXPORT void TpuExecutable_LoadProgramAndEnqueueToStream(
    TpuExecutable_LoadProgramAndEnqueueToStream_Params* params);

TFTPU_CAPI_EXPORT void HardwareLayout_HostShapeToDeviceShape(
    XLA_Shape* host_shape, XLA_Shape* device_shape);
TFTPU_CAPI_EXPORT int64_t HardwareLayout_ShapeSize(XLA_Shape* shape);
TFTPU_CAPI_EXPORT int64_t HardwareLayout_ShapeSizeCompact(XLA_Shape* shape);
TFTPU_CAPI_EXPORT int64_t HardwareLayout_ShapeSizeCompactRaw(XLA_Shape* shape);

typedef struct TpuExecute_RuntimeInputToPaddedData_Params {
  int32_t struct_size;
  void* priv;

  uint32_t* runtime_input_ptr;
  size_t runtime_input_size;
  int8_t* padded_data_ptr;
  size_t padded_data_size;
  XLA_Shape* runtime_shape;
  XLA_Shape* compile_time_shape;

  TF_Status* status;  // out
} TpuExecute_RuntimeInputToPaddedData_Params;

#define TpuExecute_RuntimeInputToPaddedData_Params_SIZE \
  (sizeof(struct TpuExecute_RuntimeInputToPaddedData_Params))

TFTPU_CAPI_EXPORT void TpuExecute_RuntimeInputToPaddedData(
    TpuExecute_RuntimeInputToPaddedData_Params* params);

typedef struct ConfigureDistributedTpuOp_DoWork_Params {
  int32_t struct_size;
  void* priv;

  size_t num_cores_per_host_size;
  const int32_t* num_cores_per_host;
  size_t server_address_size;
  const char* server_address;

  size_t* host_config_output_size;  // out
  char** host_config_output;        // out
  TF_Status* status;                // out
} ConfigureDistributedTpuOp_DoWork_Params;

#define ConfigureDistributedTpuOp_DoWork_Params_SIZE \
  (sizeof(struct ConfigureDistributedTpuOp_DoWork_Params))

TFTPU_CAPI_EXPORT void ConfigureDistributedTpuOp_DoWork(
    ConfigureDistributedTpuOp_DoWork_Params* params);

typedef struct WaitForDistributedTpuOp_DoWork_Params {
  int32_t struct_size;
  void* priv;

  size_t num_hosts;
  size_t num_cores_per_host;
  const int32_t** host_ordinal_to_global_core_id_map;
  tensorflow::TpuMeshCommonState* tpu_mesh_common_state;

  size_t* tpu_topology_output_size;  // out
  char** tpu_topology_output;        // out
  TF_Status* status;                 // out
} WaitForDistributedTpuOp_DoWork_Params;

#define WaitForDistributedTpuOp_DoWork_Params_SIZE \
  (sizeof(struct WaitForDistributedTpuOp_DoWork_Params))

TFTPU_CAPI_EXPORT void WaitForDistributedTpuOp_DoWork(
    WaitForDistributedTpuOp_DoWork_Params* params);

typedef struct InitializeHostForDistributedTpuOp_DoWork_Params {
  int32_t struct_size;
  void* priv;

  size_t tpu_host_config_size;
  const char* tpu_host_config;
  bool enable_whole_mesh_compilations;
  bool is_master_worker;

  size_t* core_id_output_size;  // out
  int32_t** core_id_output;     // out
  TF_Status* status;            // out
} InitializeHostForDistributedTpuOp_DoWork_Params;

#define InitializeHostForDistributedTpuOp_DoWork_Params_SIZE \
  (sizeof(struct InitializeHostForDistributedTpuOp_DoWork_Params))

TFTPU_CAPI_EXPORT void InitializeHostForDistributedTpuOp_DoWork(
    InitializeHostForDistributedTpuOp_DoWork_Params* params);

TFTPU_CAPI_EXPORT void SetGlobalTPUArrayOp_DoWork(
    const size_t tpu_topology_size, const char* tpu_topology,
    TF_Status* status);

TFTPU_CAPI_EXPORT void DisconnectDistributedTpuChipsOp_DoWork(
    int32_t* number_of_chips_output, TF_Status* status);

TFTPU_CAPI_EXPORT void TpuConfigurationApi_FreeCharArray(char* output);
TFTPU_CAPI_EXPORT void TpuConfigurationApi_FreeInt32Array(int32_t* output);

TFTPU_CAPI_EXPORT bool TpuConfigurationApi_HasTPUPodState();

TFTPU_CAPI_EXPORT void TpuConfigurationApi_TpusPerHost(int32_t* tpus,
                                                       TF_Status* status);
TFTPU_CAPI_EXPORT void TpuConfigurationApi_TpuMemoryLimit(int64_t* memory_limit,
                                                          TF_Status* status);
TFTPU_CAPI_EXPORT void TpuConfigurationApi_RemoteCompilationCacheSizeInBytes(
    int64_t* cache_size_in_bytes);

typedef struct TpuConfigurationApi_CompilationCacheServerAddrFromConfig_Params {
  int32_t struct_size;
  void* priv;

  size_t tpu_host_config_size;
  const char* tpu_host_config;

  size_t* server_address_output_size;  // out
  char** server_address_output;        // out
  TF_Status* status;                   // out
} TpuConfigurationApi_CompilationCacheServerAddressFromConfig_Params;

#define TpuConfigurationApi_CompilationCacheServerAddrFromConfig_Params_SIZE \
  (sizeof(                                                                   \
      struct TpuConfigurationApi_CompilationCacheServerAddrFromConfig_Params))

TFTPU_CAPI_EXPORT
void TpuConfigurationApi_CompilationCacheServerAddressFromConfig(
    TpuConfigurationApi_CompilationCacheServerAddrFromConfig_Params* params);

typedef struct TpuConfigurationApi_GetServerAddressAndPort_Params {
  int32_t struct_size;
  void* priv;

  size_t* server_address_output_size;  // out
  char** server_address_output;        // out
  int* port_output;                    // out
  TF_Status* status;                   // out
} TpuConfigurationApi_GetServerAddressAndPort_Params;

#define TpuConfigurationApi_GetServerAddressAndPort_Params_SIZE \
  (sizeof(struct TpuConfigurationApi_GetServerAddressAndPort_Params))

TFTPU_CAPI_EXPORT void TpuConfigurationApi_GetServerAddressAndPort(
    TpuConfigurationApi_GetServerAddressAndPort_Params* params);

// Creates a new TPU program.
TFTPU_CAPI_EXPORT XLA_TpuProgram* TpuProgram_New();

// Destroys the `tpu_program`.
TFTPU_CAPI_EXPORT void TpuProgram_Free(XLA_TpuProgram* tpu_program);

// Creates an array of `XLA_TpuProgram*`.
TFTPU_CAPI_EXPORT XLA_TpuProgram** TpuProgram_NewArray(size_t count);

// Destroys an array of `XLA_TpuProgram*`.
TFTPU_CAPI_EXPORT void TpuProgram_FreeArray(XLA_TpuProgram* tpu_program[]);

// Unloads and destroys the `tpu_program`. Once the TPU program is unloaded and
// destroyed, it is in an unusable state.
TFTPU_CAPI_EXPORT void TpuProgram_UnloadAndDestroy(XLA_TpuProgram* tpu_program,
                                                   TF_Status* status);

// Gets TPU program size in bytes from the `tpu_program`.
TFTPU_CAPI_EXPORT int64_t
TpuProgram_GetProgramSize(const XLA_TpuProgram* tpu_program);

// Logs the summary of current memory state snapshot of the `tpu_program`.
TFTPU_CAPI_EXPORT bool TpuProgram_LogProgramMemorySummary(
    const XLA_TpuProgram* tpu_program);

// Gets TPU program executable info from the `tpu_program`.
TFTPU_CAPI_EXPORT void TpuProgram_GetExecutableInfo(
    const XLA_TpuProgram* tpu_program, TpuSerializedProto* executable_info,
    TF_Status* status);

// Gets host transfer info proto.
TFTPU_CAPI_EXPORT void TpuProgram_GetHostTransferInfo(
    const XLA_TpuProgram* tpu_program, TpuSerializedProto* host_transfer_info,
    TF_Status* status);

// Gets HLO metadata proto.
TFTPU_CAPI_EXPORT void TpuProgram_GetHloMetadata(
    const XLA_TpuProgram* tpu_program, TpuSerializedProto* hlo_metadata,
    TF_Status* status);

// Gets may modify variables boolean value.
TFTPU_CAPI_EXPORT void TpuProgram_GetMayModifyVariables(
    const XLA_TpuProgram* tpu_program, bool* may_modify_variables);

// Checks if TPU program has sharding.
TFTPU_CAPI_EXPORT bool TpuProgram_HasSharding(
    const XLA_TpuProgram* tpu_program);

// Gets TPU program by sharding type. Return value is valid only when the
// `status.status()` returns `OK`.
TFTPU_CAPI_EXPORT XLA_TpuProgram* TpuProgram_GetTpuProgram(
    XLA_TpuProgram* tpu_program, TpuProgramShardingType type);

// Gets TPU executable proto from a `tpu_program`.
TFTPU_CAPI_EXPORT void TpuProgram_SerializeTpuExecutable(
    const XLA_TpuProgram* tpu_program, TpuExecutableSerializedProto* executable,
    TF_Status* status);

// Gets compilation metadata proto from a `tpu_program`.
TFTPU_CAPI_EXPORT void TpuProgram_SerializeCompilerMetadata(
    const XLA_TpuProgram* tpu_program,
    CompilerMetadataSerializedProto* compiler_metadata, TF_Status* status);

// Deserializes the `GetTpuProgramResponse` proto into an `XLA_TpuProgram`.
TFTPU_CAPI_EXPORT void TpuProgram_DeserializeFromGetTpuProgramResponseProto(
    TpuSerializedProto get_tpu_program_response, XLA_TpuProgram* tpu_program,
    TF_Status* status);

TFTPU_CAPI_EXPORT TpuProgramFingerprint
TpuProgram_GetFingerprint(const XLA_TpuProgram* tpu_program);

TFTPU_CAPI_EXPORT void TpuProgram_DestroyFingerprint(
    TpuProgramFingerprint fingerprint);

// Checks if whether a TPU compilation is enabled.
TFTPU_CAPI_EXPORT bool TpuCompile_IsTpuCompilationEnabled();

// XLA compilation cannot be cancelled. To avoid hanging the TF worker will exit
// when cancellation is requested for an XLA compile op. Some tests require this
// behavior to be disabled, and we test for this condition with the following
// flag function.
TFTPU_CAPI_EXPORT bool TpuCompile_ShouldTpuCompileOpIgnoreCancellation();

// Returns the number of available TPU core count.
TFTPU_CAPI_EXPORT int TpuTopology_AvailableCoreCount(
    const XLA_TpuMeshState* mesh_state, TpuCoreTypeEnum tpu_core_type);

// Recycle unused service port.
TFTPU_CAPI_EXPORT void TpuNetUtil_RecycleUnusedPort(int port);

// Creates a unique compilation cache `key` used for `put` and `get` operations.
// Returned buffers are heap-allocated and must be owned.
TFTPU_CAPI_EXPORT CompilationCacheKeyResult
TpuCompile_CreateCompilationCacheKey(CompilationCacheKeyProperty property);

// Destroys the CompilationCacheKeyResult returned by calling the
// `TpuCompile_CreateCompilationCacheKey` API.
TFTPU_CAPI_EXPORT void TpuCompile_DestroyCompilationCacheKey(
    CompilationCacheKeyResult result);

// Creates a guaranteed const fingerprint. Guarantee const is normally used in
// TPU inference to avoid re-copying unchanged variables onto the TPU device.
// It promises the value is identical for every execution in the same session
// even if the actual value changes in later executions.
TFTPU_CAPI_EXPORT uint64_t TpuCompile_CreateGuaranteedConstFingerprint(
    uint64_t fingerprint, const char* data, size_t size);

XLA_TpuNodeContext* TpuNodeContext_Create(int device_ordinal,
                                          TF_Status* status);
void TpuNodeContext_Free(XLA_TpuNodeContext* node_context);

void TpuNodeContext_StopChipHeartbeats(TF_Status* status);

void TpuNodeContext_CloseTpuHost(TF_Status* status);

void TpuNodeContext_Initialize(int device_ordinal, TF_Status* status);

bool TpuNodeContext_CompactionSupported(int device_ordinal);

// Globally initialize the TPU system for inference.
TFTPU_CAPI_EXPORT void TfTpu_InitializeTpuModelServer();

typedef struct TpuEmbeddingEngine_ExecutePartitioner_Params {
  int32_t struct_size;
  void* priv;
  TpuSerializedProto tpu_embedding_config;

  // out
  size_t* common_config_size;
  char** common_config;
  TF_Status* status;
} TpuEmbeddingEngine_ExecutePartitioner_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_ExecutePartitioner(
    TpuEmbeddingEngine_ExecutePartitioner_Params* params);

typedef struct TpuEmbeddingEngine_ConfigureMemory_Params {
  int32_t struct_size;
  void* priv;

  int num_inputs;
  size_t common_config_size;
  const char* common_config;

  // out
  size_t* memory_config_size;
  char** memory_config;
  TF_Status* status;
} TpuEmbeddingEngine_ConfigureMemory_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_ConfigureMemory(
    TpuEmbeddingEngine_ConfigureMemory_Params* params);

typedef struct TpuEmbeddingEngine_CollateMemory_Params {
  int32_t struct_size;
  void* priv;

  size_t memory_configs_size;
  const TpuSerializedProto* memory_configs;

  // out
  size_t* merged_memory_config_size;
  char** merged_memory_config;
  TF_Status* status;
} TpuEmbeddingEngine_CollateMemory_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_CollateMemory(
    TpuEmbeddingEngine_CollateMemory_Params* params);

typedef struct TpuEmbeddingEngine_ConfigureHost_Params {
  int32_t struct_size;
  void* priv;

  int num_inputs;
  size_t common_config_size;
  const char* common_config;
  size_t memory_config_size;
  const char* memory_config;
  TpuSerializedProto tpu_embedding_config;

  // out
  size_t* network_config_size;
  char** network_config;
  TF_Status* status;
} TpuEmbeddingEngine_ConfigureHost_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_ConfigureHost(
    TpuEmbeddingEngine_ConfigureHost_Params* params);

typedef struct TpuEmbeddingEngine_ConnectHosts_Params {
  int32_t struct_size;
  void* priv;

  size_t network_configs_size;
  const TpuSerializedProto* network_configs;

  // out
  TF_Status* status;
} TpuEmbeddingEngine_ConnectHosts_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_ConnectHosts(
    TpuEmbeddingEngine_ConnectHosts_Params* params);

typedef struct TpuEmbeddingEngine_Finalize_Params {
  int32_t struct_size;
  void* priv;
  const XLA_TpuMeshState* tpu_mesh_state;

  size_t common_config_size;
  const char* common_config;
  size_t memory_config_size;
  const char* memory_config;

  // out
  TF_Status* status;
} TpuEmbeddingEngine_Finalize_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_Finalize(
    TpuEmbeddingEngine_Finalize_Params* params);

typedef struct TpuEmbeddingEngine_IsInitialized_Params {
  int32_t struct_size;
  void* priv;

  size_t config_string_size;
  const char* config_string;

  // out
  bool* is_tpu_embedding_initialized;
  TF_Status* status;
} TpuEmbeddingEngine_IsInitialized_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_IsInitialized(
    TpuEmbeddingEngine_IsInitialized_Params* params);

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_WriteParameters(
    TpuEmbeddingEngineParameters* params, TF_Status* status);

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_ReadParameters(
    TpuEmbeddingEngineParameters* params, TF_Status* status);

typedef struct TpuEmbeddingEngine_EnqueueTensorBatch_Params {
  int32_t struct_size;
  void* priv;

  int32_t mode;
  int32_t local_device_ordinal;
  TpuEmbedding_TensorBatchFixedState* fixed_state;

  TF_Tensor** sample_indices_tensors;
  size_t sample_indices_tensors_size;
  TF_Tensor** embedding_indices_tensors;
  size_t embedding_indices_tensors_size;
  TF_Tensor** aggregation_weights_tensors;
  size_t aggregation_weights_tensors_size;
  TF_Status* status;
} TpuEmbeddingEngine_EnqueueTensorBatch_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_EnqueueTensorBatch(
    TpuEmbeddingEngine_EnqueueTensorBatch_Params* params);

typedef struct TpuEmbedding_TensorBatchFixedState_Create_Params {
  int32_t struct_size;
  void* priv;

  size_t combiners_size;
  char** combiners;

  // out
  TF_Status* status;
} TpuEmbedding_TensorBatchFixedState_Create_Params;

TFTPU_CAPI_EXPORT TpuEmbedding_TensorBatchFixedState*
TpuEmbeddingTensorBatchFixedState_Create(
    TpuEmbedding_TensorBatchFixedState_Create_Params* params);
TFTPU_CAPI_EXPORT void TpuEmbeddingTensorBatchFixedState_Destroy(
    TpuEmbedding_TensorBatchFixedState* fixed_state);

typedef struct TpuEmbeddingEngine_RecvActivationsComputation_Params {
  int32_t struct_size;
  void* priv;

  size_t config_string_size;
  XLA_Shape* deduplication_data_shape;
  const XLA_TpuMeshState* tpu_mesh_state;

  // out
  TpuSerializedProto* xla_computation;
  TF_Status* status;
} TpuEmbeddingEngine_RecvActivationsComputation_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_RecvActivationsComputation(
    TpuEmbeddingEngine_RecvActivationsComputation_Params* params);

typedef struct
    TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation_Params {
  int32_t struct_size;
  void* priv;

  const XLA_TpuMeshState* tpu_mesh_state;
  // out
  TpuSerializedProto* xla_computation;
  TF_Status* status;
} TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation_Params;

TFTPU_CAPI_EXPORT void
TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation(
    TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation_Params*
        params);

typedef struct TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation_Params {
  int32_t struct_size;
  void* priv;

  int32_t num_inputs;
  const XLA_TpuMeshState* tpu_mesh_state;
  XLA_Shape* learning_rate_tuple_shape;
  XLA_Shape* deduplication_data_shape;
  XLA_Shape* gradient_tuple_shape;
  // out
  TpuSerializedProto* xla_computation;
  TF_Status* status;
} TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation(
    TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation_Params* params);

struct TfTpu_OpsApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CompileAndBuild);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_XrtCompileAndBuild);

  TFTPU_ADD_FN_IN_STRUCT(TpuMeshState_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuMeshState_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuMeshState_MeshCommonState);

  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngineState_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngineState_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngineState_GetState);

  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_Destroy);
  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_Start);
  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_Stop);
  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_CollectData);

  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_LoadProgramAndEnqueueToStream);
  TFTPU_ADD_FN_IN_STRUCT(HardwareLayout_HostShapeToDeviceShape);
  TFTPU_ADD_FN_IN_STRUCT(HardwareLayout_ShapeSize);
  TFTPU_ADD_FN_IN_STRUCT(HardwareLayout_ShapeSizeCompact);
  TFTPU_ADD_FN_IN_STRUCT(HardwareLayout_ShapeSizeCompactRaw);

  TFTPU_ADD_FN_IN_STRUCT(TpuExecute_RuntimeInputToPaddedData);

  TFTPU_ADD_FN_IN_STRUCT(ConfigureDistributedTpuOp_DoWork);
  TFTPU_ADD_FN_IN_STRUCT(WaitForDistributedTpuOp_DoWork);
  TFTPU_ADD_FN_IN_STRUCT(InitializeHostForDistributedTpuOp_DoWork);
  TFTPU_ADD_FN_IN_STRUCT(SetGlobalTPUArrayOp_DoWork);
  TFTPU_ADD_FN_IN_STRUCT(DisconnectDistributedTpuChipsOp_DoWork);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_FreeCharArray);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_FreeInt32Array);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_HasTPUPodState);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_TpusPerHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_TpuMemoryLimit);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_RemoteCompilationCacheSizeInBytes);
  TFTPU_ADD_FN_IN_STRUCT(
      TpuConfigurationApi_CompilationCacheServerAddressFromConfig);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_GetServerAddressAndPort);

  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_NewArray);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_FreeArray);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_UnloadAndDestroy);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetProgramSize);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_LogProgramMemorySummary);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetExecutableInfo);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetHostTransferInfo);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetHloMetadata);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetMayModifyVariables);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_HasSharding);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetTpuProgram);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_SerializeTpuExecutable);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_SerializeCompilerMetadata);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_DeserializeFromGetTpuProgramResponseProto);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetFingerprint);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_DestroyFingerprint);

  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_IsTpuCompilationEnabled);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_ShouldTpuCompileOpIgnoreCancellation);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_AvailableCoreCount);
  TFTPU_ADD_FN_IN_STRUCT(TpuNetUtil_RecycleUnusedPort);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CreateCompilationCacheKey);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_DestroyCompilationCacheKey);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CreateGuaranteedConstFingerprint);

  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_StopChipHeartbeats);
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_CloseTpuHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_Initialize);
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_CompactionSupported);

  TFTPU_ADD_FN_IN_STRUCT(TfTpu_InitializeTpuModelServer);

  TFTPU_ADD_FN_IN_STRUCT(TfTpuOrdinalSelector_Create);
  TFTPU_ADD_FN_IN_STRUCT(TfTpuOrdinalSelector_Destroy);
  TFTPU_ADD_FN_IN_STRUCT(TfTpuOrdinalSelector_GetOrdinal);
  TFTPU_ADD_FN_IN_STRUCT(TfTpuOrdinalSelector_DequeueFromCoreSelector);
  TFTPU_ADD_FN_IN_STRUCT(TfTpu_GetTpuPartitionedCallParams);

  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_ExecutePartitioner);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_ConfigureMemory);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_CollateMemory);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_ConfigureHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_ConnectHosts);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_Finalize);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_IsInitialized);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_WriteParameters);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_ReadParameters);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingTensorBatchFixedState_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingTensorBatchFixedState_Destroy);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_EnqueueTensorBatch);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_RecvActivationsComputation);
  TFTPU_ADD_FN_IN_STRUCT(
      TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation);
  TFTPU_ADD_FN_IN_STRUCT(
      TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation);
};

}  // extern "C"

#endif  // TENSORFLOW_CORE_TPU_TPU_OPS_C_API_H_
