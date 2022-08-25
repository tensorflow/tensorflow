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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_DECL_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_DECL_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/tf_attrtype.h"
#include "tensorflow/core/tpu/libtftpu.h"

extern "C" {

struct TF_Status;
typedef struct TF_Status TF_Status;

// Maximum number of array elements to inline into structs for performance.
#define TPU_C_API_MAX_INLINED 6

enum TpuCoreTypeEnum {
  kTensorCore,
  kEmbeddingV1,
  kEmbeddingV2,
};

enum TpuVersionEnum {
  kUnknownTpuVersion,
  kTpuV2,
  kTpuV3,
  kTpuV4,
};

typedef struct TpuRuntimeVersion {
  // The three version numbers are: major, minor, patch
  int version[3];
  const char* metadata;
  size_t metadata_size;
} TpuRuntimeVersion;

typedef struct SE_Platform SE_Platform;
typedef struct SE_StreamExecutor SE_StreamExecutor;
typedef struct SE_Stream SE_Stream;
typedef struct SE_Event SE_Event;
typedef struct SE_Timer SE_Timer;

typedef struct TpuSerializedProto {
  const char* bytes;
  size_t size;
} TpuSerializedProto;

typedef struct SE_PlatformId {
  void* id;  // aka stream_executor::Platform::Id
} SE_PlatformId;
typedef struct SE_StreamExecutorConfig SE_StreamExecutorConfig;
typedef struct SE_DeviceOptions SE_DeviceOptions;
typedef TF_Status* (*SE_StatusCallback)(void*);

typedef struct SE_DeviceMemoryBase {
  void* opaque;
  uint64_t size;
  uint64_t payload;
} SE_DeviceMemoryBase;

typedef struct SE_ScopedDeviceMemory {
  SE_DeviceMemoryBase wrapped;
  int device_ordinal;
} SE_ScopedDeviceMemory;

typedef struct SE_AllocatorStats {
  int64_t num_allocs;
  int64_t bytes_in_use;
  int64_t peak_bytes_in_use;
  int64_t largest_alloc_size;

  bool has_bytes_limit;
  int64_t bytes_limit;

  int64_t bytes_reserved;
  int64_t peak_bytes_reserved;

  bool has_bytes_reservable_limit;
  int64_t bytes_reservable_limit;

  int64_t largest_free_block_bytes;
} SE_AllocatorStats;

// Note, due to the... odd way in which DeviceMemoryAllocator is used in TF, we
// cannot simply wrap an underlying pointer. Instead, we reverse the call
// direction and request memory via a callback.
typedef void (*SE_AllocateFn)(void* ctx, int device_ordinal, uint64_t size,
                              bool retry_on_failure, int64_t memory_space,
                              SE_ScopedDeviceMemory* result, TF_Status* status);

typedef void (*SE_DeallocateFn)(void* ctx, SE_DeviceMemoryBase* base,
                                int device_ordinal, TF_Status* status);

typedef struct SE_DeviceMemoryAllocator {
  SE_Platform* platform;
  void* ctx;
  SE_AllocateFn allocate;
  SE_DeallocateFn deallocate;
} SE_DeviceMemoryAllocator;

typedef struct SE_DeviceDescription {
  char* device_vendor;
  char* platform_version;
  char* driver_version;
  char* runtime_version;
  char* pci_bus_id;
  char* name;

  int64_t thread_dim_limit_x;
  int64_t thread_dim_limit_y;
  int64_t thread_dim_limit_z;
  int64_t block_dim_limit_x;
  int64_t block_dim_limit_y;
  int64_t block_dim_limit_z;

  int64_t threads_per_core_limit;
  int64_t threads_per_block_limit;
  int64_t threads_per_warp;

  int64_t registers_per_core_limit;
  int64_t registers_per_block_limit;

  int64_t device_address_bits;
  int64_t device_memory_size;
  int64_t memory_bandwidth;

  int64_t shared_memory_per_core;
  int64_t shared_memory_per_block;

  float clock_rate_ghz;

  int cuda_compute_capability_major;
  int cuda_compute_capability_minor;

  int numa_node;
  int core_count;
  bool ecc_enabled;
} SE_DeviceDescription;

typedef struct Tpu_Compiler Tpu_Compiler;
typedef struct SE_Executable SE_Executable;

typedef struct SE_ExecutableRunOptions {
  SE_DeviceMemoryAllocator allocator;
  int device_ordinal;
  SE_Stream* stream;
  SE_Stream* host_to_device_stream;
  TpuSerializedProto device_assignment;
  int rng_seed;
  int64_t run_id;
  int launch_id;
} SE_ExecutableRunOptions;

typedef struct SE_ExecutableSerializationHandle
    SE_ExecutableSerializationHandle;

typedef struct SE_MaybeOwningDeviceMemory {
  SE_DeviceMemoryBase memory;
  bool owned;

  // Set if owned
  int device_ordinal;
  SE_DeviceMemoryAllocator allocator;
} SE_MaybeOwningDeviceMemory;

struct IntList {
  union {
    int* heap;  // owned
    int inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
};

struct Int64List {
  union {
    int64_t* heap;  // owned
    int64_t inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
};

struct FloatList {
  union {
    float* heap;  // owned
    float inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
};

struct BoolList {
  union {
    bool* heap;  // owned
    bool inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
};

struct FloatListRef {
  float* ptr;  // not owned
  int64_t size;
};

typedef struct TpuEmbeddingEngineParameters {
  FloatListRef** parameters[8];
  size_t num_tables;
} TpuEmbeddingEngineParameters;

typedef struct XLA_Tile {
  Int64List dimensions;
} XLA_Tile;

struct TileList {
  union {
    XLA_Tile* heap;  // owned
    XLA_Tile inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
};

typedef struct XLA_Layout {
  Int64List minor_to_major;
  IntList dim_level_types;
  TileList tiles;
  int64_t element_size_in_bits;
  int64_t memory_space;
} XLA_Layout;

// Represents an XLA shape tree.
typedef struct XLA_Shape {
  int element_type;
  Int64List dimensions;
  BoolList dynamic_dimensions;
  XLA_Shape* tuple_shapes;  // owned
  int ntuple_shapes;
  bool has_layout;
  XLA_Layout layout;
} XLA_Shape;

// Represents a leaf node for a XLA shaped buffer.
typedef struct XLA_ShapedBuffer {
  XLA_Shape on_device_shape;
  int device_ordinal;

  SE_DeviceMemoryBase* bases;
  size_t count;
} XLA_ShapedBuffer;

// Represents a leaf XLA literal.
typedef struct XLA_Literal {
  char** buffers;
  size_t* sizes;
  size_t count;
  XLA_Shape shape;
} XLA_Literal;

typedef struct XLA_MaybeOwningDeviceMemoryShapeTree {
  XLA_Shape shape;
  SE_MaybeOwningDeviceMemory* buffers;
} XLA_MaybeOwningDeviceMemoryShapeTree;

typedef struct XLA_ShapeIndex {
  int64_t indices[8];
  int64_t count;
} XLA_ShapeIndex;

typedef struct SE_ExecutionInput {
  XLA_MaybeOwningDeviceMemoryShapeTree shape_tree;
  XLA_ShapeIndex* unowned_indices;
  int unowned_indices_size;
  XLA_Shape dynamic_shape;
} SE_ExecutionInput;

typedef struct SE_ExecutionOutput {
  XLA_ShapedBuffer result;
  SE_MaybeOwningDeviceMemory* to_be_released;
  int to_be_released_size;
  XLA_ShapeIndex* aliased_indices;
  int aliased_indices_size;
} SE_ExecutionOutput;

typedef struct XLA_ComputationLayout {
  int parameter_count;
  XLA_Shape* parameter_layouts;
  XLA_Shape result_layout;
} XLA_ComputationLayout;

typedef struct XLA_HloModuleConfig {
  uint64_t seed;
  int32_t launch_id;
  int64_t replica_count;
  int64_t num_partitions;
  bool use_spmd_partitioning;
  bool use_auto_spmd_partitioning;
  Int64List auto_spmd_partitioning_mesh_shape;
  Int64List auto_spmd_partitioning_mesh_ids;
  TpuSerializedProto debug_options;
  bool has_static_device_assignment;
  TpuSerializedProto static_device_assignment;
  bool has_entry_computation_layout;
  XLA_ComputationLayout entry_computation_layout;
} XLA_HloModuleConfig;

typedef struct SE_HloExecutionProfile SE_HloExecutionProfile;

struct SE_StreamExecutorList {
  SE_StreamExecutor** exec;
  int count;
};

typedef struct XLA_HloModuleGroup {
  TpuSerializedProto proto;
  XLA_HloModuleConfig* module_config;
} XLA_HloModuleGroup;

typedef struct XLA_HloModule {
  TpuSerializedProto proto;
  XLA_HloModuleConfig module_config;
} XLA_HloModule;

typedef struct XLA_TransferManager XLA_TransferManager;

typedef struct XLA_ComputationPlacer XLA_ComputationPlacer;

typedef void (*XLA_CallbackFn)(void*);
typedef void (*XLA_StatusCallbackFn)(void*, TF_Status*);

typedef struct SE_TpuTopology SE_TpuTopology;
typedef struct SE_TpuTopology_Core SE_TpuTopology_Core;
typedef struct SE_TpuTopology_Core SE_TpuTopology_Host;
}

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_DECL_H_
