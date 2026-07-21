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

#ifndef XLA_TPU_C_API_DECL_H_
#define XLA_TPU_C_API_DECL_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/tpu/libtftpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TSL_Status TF_Status;

// Maximum number of array elements to inline into structs for performance.
#define TPU_C_API_MAX_INLINED 6

typedef enum TpuCoreTypeEnum {
  kTensorCore,
  kEmbeddingV1,
  kEmbeddingV2,
} TpuCoreTypeEnum;

typedef enum TpuVersionEnum {
  kUnknownTpuVersion,
  kTpuV2,
  kTpuV3,
  kTpuV4,
  kTpuV5,
} TpuVersionEnum;

typedef struct TpuRuntimeVersion {
  // The three version numbers are: major, minor, patch
  int version[3];
  const char* metadata;
  size_t metadata_size;
} TpuRuntimeVersion;

// An empty proto is always represented as (bytes=nullptr, size=0).
// There is no other valid way to represent it.
typedef struct TpuSerializedProto {
  const char* bytes;
  size_t size;
} TpuSerializedProto;

typedef struct IntList {
  union {
    int* heap;  // owned
    int inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
} IntList;

typedef struct Int64List {
  union {
    int64_t* heap;  // owned
    int64_t inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
} Int64List;

typedef struct FloatList {
  union {
    float* heap;  // owned
    float inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
} FloatList;

typedef struct BoolList {
  union {
    bool* heap;  // owned
    bool inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
} BoolList;

typedef struct FloatListRef {
  float* ptr;  // not owned
  int64_t size;
} FloatListRef;

typedef struct TpuEmbeddingEngineParameters {
  FloatListRef** parameters[8];
  size_t num_tables;
} TpuEmbeddingEngineParameters;

typedef struct XLA_Tile {
  Int64List dimensions;
} XLA_Tile;

typedef struct TileList {
  union {
    XLA_Tile* heap;  // owned
    XLA_Tile inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
} TileList;

typedef struct XLA_Layout {
  Int64List minor_to_major;
  TileList tiles;
  int index_primitive_type;
  int pointer_primitive_type;
  int64_t element_size_in_bits;
  int64_t memory_space;
  int64_t dynamic_shape_metadata_prefix_bytes;
  int64_t tail_padding_alignment_in_elements;
} XLA_Layout;

// Represents an XLA shape tree.
typedef struct XLA_Shape {
  int element_type;
  Int64List dimensions;
  BoolList dynamic_dimensions;
  struct XLA_Shape* tuple_shapes;  // owned
  int ntuple_shapes;
  bool has_layout;
  XLA_Layout layout;
} XLA_Shape;

// Represents a leaf XLA literal.
typedef struct XLA_Literal {
  char** buffers;
  size_t* sizes;
  size_t count;
  XLA_Shape shape;
} XLA_Literal;

typedef struct XLA_ShapeIndex {
  int64_t indices[8];
  int64_t count;
} XLA_ShapeIndex;

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
  BoolList allow_spmd_sharding_propagation_to_parameters;
  BoolList allow_spmd_sharding_propagation_to_output;
} XLA_HloModuleConfig;

typedef struct SE_HloExecutionProfile SE_HloExecutionProfile;

typedef struct XLA_HloModuleGroup {
  TpuSerializedProto proto;
  XLA_HloModuleConfig* module_config;
} XLA_HloModuleGroup;

typedef struct XLA_HloModule {
  TpuSerializedProto proto;
  XLA_HloModuleConfig module_config;
} XLA_HloModule;

typedef struct XLA_TpuMeshState XLA_TpuMeshState;

typedef void (*XLA_CallbackFn)(void*);
typedef void (*XLA_StatusCallbackFn)(void*, TF_Status*);

typedef struct SE_TpuTopology SE_TpuTopology;
typedef struct SE_TpuTopology_Core SE_TpuTopology_Core;
typedef struct SE_TpuTopology_Core SE_TpuTopology_Host;

typedef struct SE_OutsideCompilationParams SE_OutsideCompilationParams;

#ifdef __cplusplus
}
#endif

#endif  // XLA_TPU_C_API_DECL_H_
