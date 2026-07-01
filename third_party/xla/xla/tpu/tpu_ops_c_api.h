/* Copyright 2020 The OpenXLA Authors.

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
#ifndef XLA_TPU_TPU_OPS_C_API_H_
#define XLA_TPU_TPU_OPS_C_API_H_

#include <stddef.h>

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "xla/tpu/c_api_decl.h"
#include "xla/tpu/libtftpu.h"

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

// An empty fingerprint is always represented as (bytes=nullptr, size=0).
// There is no other valid way to represent it.
struct TpuProgramFingerprint {
  const char* bytes;
  size_t size;
};

// An empty proto is always represented as (bytes=nullptr, size=0).
// There is no other valid way to represent it.
struct TpuExecutableSerializedProto {
  const char* bytes;
  size_t size;
};

// An empty proto is always represented as (bytes=nullptr, size=0).
// There is no other valid way to represent it.
struct CompilerMetadataSerializedProto {
  const char* bytes;
  size_t size;
};

// An empty proto is always represented as (bytes=nullptr, size=0).
// There is no other valid way to represent it.
struct HostComputeMetadataSerializedProto {
  const char* bytes;
  size_t size;
};

typedef struct XLA_TpuMeshState XLA_TpuMeshState;

typedef struct XLA_TpuEmbeddingEngineState XLA_TpuEmbeddingEngineState;

typedef struct TpuEmbedding_TensorBatchFixedState
    TpuEmbedding_TensorBatchFixedState;

// An empty device assignment is always represented as (bytes=nullptr, size=0).
// There is no other valid way to represent it.
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

TFTPU_CAPI_EXPORT void SetGlobalTPUArrayOp_DoWork(size_t tpu_topology_size,
                                                  const char* tpu_topology,
                                                  TF_Status* status);
TFTPU_CAPI_EXPORT void TpuConfigurationApi_FreeCharArray(char* output);

TFTPU_CAPI_EXPORT void TpuConfigurationApi_TpusPerHost(int32_t* tpus,
                                                       TF_Status* status);
TFTPU_CAPI_EXPORT void TpuConfigurationApi_TpuMemoryLimit(int64_t* memory_limit,
                                                          TF_Status* status);

// Returns the number of available TPU core count.
TFTPU_CAPI_EXPORT int TpuTopology_AvailableCoreCount(
    const XLA_TpuMeshState* mesh_state, TpuCoreTypeEnum tpu_core_type);

// Returns the number of cores per Chip.
TFTPU_CAPI_EXPORT int TpuTopology_AvailableCoresPerChip(
    TpuCoreTypeEnum tpu_core_type);

// Returns the number of cores per Chip or -1 if the TPU system is not
// available.
TFTPU_CAPI_EXPORT absl::StatusOr<int>
TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice(
    TpuCoreTypeEnum tpu_core_type);

// Returns a pointer to the TPU topology struct.
TFTPU_CAPI_EXPORT const SE_TpuTopology* TpuUtil_GetTopologyPtr();

// Returns XLA pad size from TPU topology.
TFTPU_CAPI_EXPORT size_t TpuUtil_GetXlaPadSizeFromTpuTopology();

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

typedef struct TF_Tensor TF_Tensor;

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

  TpuSerializedProto tpu_embedding_config;
  TpuSerializedProto embedding_partitions;
  TpuSerializedProto hbm_buffers_config;
  TpuSerializedProto tpu_topology;
  XLA_Shape* deduplication_data_shape;
  TpuSerializedProto* op_sharding;

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

  TpuSerializedProto tpu_embedding_config;
  TpuSerializedProto embedding_partitions;
  TpuSerializedProto hbm_buffers_config;
  TpuSerializedProto tpu_topology;
  TpuSerializedProto* op_sharding;
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
  TpuSerializedProto tpu_embedding_config;
  TpuSerializedProto embedding_partitions;
  TpuSerializedProto hbm_buffers_config;
  TpuSerializedProto tpu_topology;
  XLA_Shape* learning_rate_tuple_shape;
  XLA_Shape* deduplication_data_shape;
  XLA_Shape* gradient_tuple_shape;
  TpuSerializedProto* op_sharding;
  // out
  TpuSerializedProto* xla_computation;
  TF_Status* status;
} TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation(
    TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation_Params* params);

typedef struct TpuEmbeddingEngine_DedupDataSizeComputation_Params {
  int32_t struct_size;
  void* priv;

  TpuSerializedProto tpu_embedding_config;
  TpuSerializedProto embedding_partitions;
  TpuSerializedProto hbm_buffers_config;
  TpuSerializedProto tpu_topology;
  // out
  int32_t* num_elements;
  TF_Status* status;
} TpuEmbeddingEngine_DedupDataSizeComputation_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_DedupDataSizeComputation(
    TpuEmbeddingEngine_DedupDataSizeComputation_Params* params);

typedef struct TpuEmbeddingEngine_DedupDataTupleMaskComputation_Params {
  int32_t struct_size;
  void* priv;

  TpuSerializedProto tpu_embedding_config;
  TpuSerializedProto embedding_partitions;
  TpuSerializedProto hbm_buffers_config;
  TpuSerializedProto tpu_topology;
  // out
  TpuSerializedProto* xla_computation;
  TF_Status* status;
} TpuEmbeddingEngine_DedupDataTupleMaskComputation_Params;

TFTPU_CAPI_EXPORT void TpuEmbeddingEngine_DedupDataTupleMaskComputation(
    TpuEmbeddingEngine_DedupDataTupleMaskComputation_Params* params);

typedef struct SparseCore_GetMaxIdsAndUniques_Params {
  size_t struct_size;
  void* priv;
  const char* program_key;
  const char* table_name;
  int64_t num_samples_per_sparse_core;
  int64_t feature_width;
  // out
  TF_Status* status;
  int64_t max_ids_per_partition;
  int64_t max_unique_ids_per_partition;
} SparseCore_GetMaxIdsAndUniques_Params;

TFTPU_CAPI_EXPORT void SparseCore_GetMaxIdsAndUniques(
    SparseCore_GetMaxIdsAndUniques_Params* params);

struct TfTpu_OpsApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuMeshState_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuMeshState_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuMeshState_MeshCommonState);

  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngineState_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngineState_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngineState_GetState);

  TFTPU_ADD_FN_IN_STRUCT(SetGlobalTPUArrayOp_DoWork);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_FreeCharArray);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_TpusPerHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuConfigurationApi_TpuMemoryLimit);

  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_AvailableCoreCount);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_AvailableCoresPerChip);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice);
  TFTPU_ADD_FN_IN_STRUCT(TpuUtil_GetTopologyPtr);
  TFTPU_ADD_FN_IN_STRUCT(TpuUtil_GetXlaPadSizeFromTpuTopology);

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
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_DedupDataSizeComputation);
  TFTPU_ADD_FN_IN_STRUCT(TpuEmbeddingEngine_DedupDataTupleMaskComputation);

  TFTPU_ADD_FN_IN_STRUCT(SparseCore_GetMaxIdsAndUniques);
};

}  // extern "C"

#endif  // XLA_TPU_TPU_OPS_C_API_H_
