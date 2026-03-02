/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_TPU_TOPOLOGY_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_TPU_TOPOLOGY_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality related to TPU topology.

#define PJRT_API_TPU_TOPOLOGY_EXTENSION_VERSION 1

struct PJRT_TpuTopology_Subslice_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  const int32_t* chips_per_host_bounds;
  size_t chips_per_host_bounds_num_dims;
  const int32_t* host_bounds;
  size_t host_bounds_num_dims;

  // Owned by the caller. Should be destroyed by calling
  // PJRT_TpuTopology_Destroy.
  PJRT_TopologyDescription* subslice_topology;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_Subslice_Args, subslice_topology);

// Returns a subslice topology of the given topology.
typedef PJRT_Error* PJRT_TpuTopology_Subslice(
    PJRT_TpuTopology_Subslice_Args* args);

struct PJRT_TpuTopology_IsSubsliceTopology_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  bool is_subslice_topology;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_IsSubsliceTopology_Args,
                          is_subslice_topology);

// Returns true if the topology is a subslice topology.
typedef PJRT_Error* PJRT_TpuTopology_IsSubsliceTopology(
    PJRT_TpuTopology_IsSubsliceTopology_Args* args);

typedef struct PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* client_topology;
  const PJRT_TopologyDescription* subslice_topology;
  const int32_t* subslice_origin;
  size_t subslice_origin_dim_num;
  int32_t full_device_id;

  int32_t subslice_device_id;  // out
} PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args;
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args, subslice_device_id);

// Returns the subslice device id for the given full device id.
typedef PJRT_Error* PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId(
    PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args* args);

typedef struct PJRT_TpuTopology_ReplaceHostBounds_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  const int32_t* host_bounds;
  size_t host_bounds_dim_num;

  // Owned by the caller. Should be destroyed by calling
  // PJRT_TpuTopology_Destroy.
  PJRT_TopologyDescription* new_topology;  // out
} PJRT_TpuTopology_ReplaceHostBounds_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ReplaceHostBounds_Args,
                          new_topology);

// Returns a new PjRtTopologyDescription by replacing the host bounds of the
// input `topology` with the provided `host_bounds`.
typedef PJRT_Error* PJRT_TpuTopology_ReplaceHostBounds(
    PJRT_TpuTopology_ReplaceHostBounds_Args* args);

typedef struct PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  bool is_enhanced_barrier_enabled;  // out
} PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args,
                          is_enhanced_barrier_enabled);

// Returns true if the enhanced barrier is enabled in the given TPU topology.
typedef PJRT_Error* PJRT_TpuTopology_IsEnhancedBarrierEnabled(
    PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args* args);

typedef struct PJRT_TpuTopology_HasLimitedIciConnectivity_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  bool has_limited_ici_connectivity;  // out
} PJRT_TpuTopology_HasLimitedIciConnectivity_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_HasLimitedIciConnectivity_Args,
                          has_limited_ici_connectivity);

// Returns true if the given TPU topology has limited ICI connectivity.
typedef PJRT_Error* PJRT_TpuTopology_HasLimitedIciConnectivity(
    PJRT_TpuTopology_HasLimitedIciConnectivity_Args* args);

typedef struct PJRT_TpuTopology_IsReachableOverLimitedIci_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t source_chip_id;
  int32_t dest_chip_id;
  bool is_reachable_over_limited_ici;  // out
} PJRT_TpuTopology_IsReachableOverLimitedIci_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_IsReachableOverLimitedIci_Args,
                          is_reachable_over_limited_ici);

// Returns true if `source_chip_id` can directly reach `dest_chip_id` on a TPU
// topology with limited ICI routing.
typedef PJRT_Error* PJRT_TpuTopology_IsReachableOverLimitedIci(
    PJRT_TpuTopology_IsReachableOverLimitedIci_Args* args);

struct PJRT_TpuTopology_ProcessCount_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t process_count;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ProcessCount_Args, process_count);

// Returns the number of processes in this topology.
typedef PJRT_Error* PJRT_TpuTopology_ProcessCount(
    PJRT_TpuTopology_ProcessCount_Args* args);

struct PJRT_TpuTopology_ChipsPerProcess_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t chips_per_process;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ChipsPerProcess_Args,
                          chips_per_process);

// Returns the number of chips per process in this topology.
typedef PJRT_Error* PJRT_TpuTopology_ChipsPerProcess(
    PJRT_TpuTopology_ChipsPerProcess_Args* args);

struct PJRT_TpuTopology_CoreCountPerChip_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t core_count_of_default_type_per_chip;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_CoreCountPerChip_Args,
                          core_count_of_default_type_per_chip);

// Returns the number of cores of default type per chip in this topology.
typedef PJRT_Error* PJRT_TpuTopology_CoreCountPerChip(
    PJRT_TpuTopology_CoreCountPerChip_Args* args);

struct PJRT_TpuTopology_ChipCount_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t chip_count;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ChipCount_Args, chip_count);

// Returns the number of total chips in this topology.
typedef PJRT_Error* PJRT_TpuTopology_ChipCount(
    PJRT_TpuTopology_ChipCount_Args* args);

struct PJRT_TpuTopology_CoreCount_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t core_count_of_default_type;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_CoreCount_Args,
                          core_count_of_default_type);

// Returns the number of total cores of default type in this topology.
typedef PJRT_Error* PJRT_TpuTopology_CoreCount(
    PJRT_TpuTopology_CoreCount_Args* args);

struct PJRT_TpuTopology_LogiDeviceCount_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t logical_device_count_of_default_type;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_LogiDeviceCount_Args,
                          logical_device_count_of_default_type);

// Returns the number of total logical devices of default type in this topology.
typedef PJRT_Error* PJRT_TpuTopology_LogiDeviceCount(
    PJRT_TpuTopology_LogiDeviceCount_Args* args);

struct PJRT_TpuTopology_LogiDeviceCountPerProcess_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t logical_device_count_of_default_type_per_process;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_LogiDeviceCountPerProcess_Args,
                          logical_device_count_of_default_type_per_process);

// Returns the number of logical devices of default type per process in this
// topology.
typedef PJRT_Error* PJRT_TpuTopology_LogiDeviceCountPerProcess(
    PJRT_TpuTopology_LogiDeviceCountPerProcess_Args* args);

struct PJRT_TpuTopology_LogiDeviceCountPerChip_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t logical_device_count_of_default_type_per_chip;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_LogiDeviceCountPerChip_Args,
                          logical_device_count_of_default_type_per_chip);

// Returns the number of logical devices of default type per chip in this
// topology.
typedef PJRT_Error* PJRT_TpuTopology_LogiDeviceCountPerChip(
    PJRT_TpuTopology_LogiDeviceCountPerChip_Args* args);

struct PJRT_TpuTopology_CoreCountPerProcess_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t core_count_of_default_type_per_process;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_CoreCountPerProcess_Args,
                          core_count_of_default_type_per_process);

// Returns the number of cores per process in this topology.
typedef PJRT_Error* PJRT_TpuTopology_CoreCountPerProcess(
    PJRT_TpuTopology_CoreCountPerProcess_Args* args);

struct PJRT_TpuTopology_ProcessIds_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  // The maximum number of process IDs that can be returned. If the topology has
  // more than max_process_ids processes, an error is returned.
  int32_t max_process_ids;
  // Points to an array of size max_process_ids. The process IDs will be
  // filled in this array.
  int32_t* process_ids;
  size_t num_process_ids;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ProcessIds_Args, num_process_ids);

// Returns the process IDs in this topology.
typedef PJRT_Error* PJRT_TpuTopology_ProcessIds(
    PJRT_TpuTopology_ProcessIds_Args* args);

struct PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t process_id;
  // The maximum number of device IDs that can be returned. If the topology has
  // more than max_logical_device_ids devices on the process, an error is
  // returned.
  int32_t max_logical_device_ids;
  // Points to an array of size max_logical_device_ids. The device IDs will be
  // filled in this array.
  int32_t* logical_device_of_default_type_ids;
  size_t num_logical_device_ids;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args,
                          num_logical_device_ids);

// Returns the logical device of default type IDs on the given process.
typedef PJRT_Error* PJRT_TpuTopology_LogiDeviceIdsOnProcess(
    PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args* args);

struct PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t chip_id;
  int32_t process_id;        // out
  int32_t index_on_process;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args,
                          index_on_process);

// Returns the process ID and index on process for the given chip.
typedef PJRT_Error* PJRT_TpuTopology_ProcIdAndIdxOnProcForChip(
    PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args* args);

struct PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t device_id;
  int32_t process_id;        // out
  int32_t index_on_process;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args,
                          index_on_process);

// Returns the process ID and index on process for the given logical device of
// default type.
typedef PJRT_Error* PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice(
    PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args* args);

struct PJRT_TpuTopology_ProcessCoordFromId_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t process_id;
  // The maximum dimension of coordinates that can be returned.
  // If the process has more than max_coords dimensions, an error is returned.
  size_t coords_max_dims;
  // Points to an array of size max_dims. The coordinates of the process will
  // be stored in this array.
  int32_t* coords;
  size_t coords_num_dims;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ProcessCoordFromId_Args,
                          coords_num_dims);

// Returns the coordinates of the process with the given process ID.
typedef PJRT_Error* PJRT_TpuTopology_ProcessCoordFromId(
    PJRT_TpuTopology_ProcessCoordFromId_Args* args);

struct PJRT_TpuTopology_ChipIdFromCoord_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  const int32_t* coords;
  size_t coords_num_dims;
  int32_t chip_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ChipIdFromCoord_Args, chip_id);

// Returns the chip ID for the given coordinates.
typedef PJRT_Error* PJRT_TpuTopology_ChipIdFromCoord(
    PJRT_TpuTopology_ChipIdFromCoord_Args* args);

struct PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  const int32_t* chip_coords;
  size_t chip_coords_num_dims;
  int32_t logical_device_index_on_chip;
  int32_t logical_device_of_default_type_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args,
                          logical_device_of_default_type_id);

// Returns the logical device of default type ID for the chip with the given
// coordinates and logical device index on chip.
typedef PJRT_Error* PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx(
    PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args* args);

struct PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t device_id;
  // The maximum dimension of coordinates that can be returned.
  // If the device has more than max_coords dimensions, an error is returned.
  size_t chip_coords_max_dims;
  // Points to an array of size max_dims. The coordinates of the device will
  // be stored in this array.
  int32_t* chip_coords;
  size_t chip_coords_num_dims;   // out
  int32_t device_index_on_chip;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args,
                          device_index_on_chip);

// Returns the coordinates of the chip containing the given logical device of
// default type, and the index of the logical device on the chip.
typedef PJRT_Error* PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice(
    PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args* args);

struct PJRT_TpuTopology_ChipsPerProcessBounds_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  // The maximum dimension of coordinates that can be returned.
  // If the process has more than max_coords dimensions, an error is returned.
  size_t chip_per_process_bounds_max_dims;
  // Points to an array of size max_dims. The bounds of the chips per process
  // will be stored in this array.
  int32_t* chip_per_process_bounds;
  size_t chip_per_process_bounds_num_dims;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ChipsPerProcessBounds_Args,
                          chip_per_process_bounds_num_dims);

// Returns the bounds of the chips per process in this topology.
typedef PJRT_Error* PJRT_TpuTopology_ChipsPerProcessBounds(
    PJRT_TpuTopology_ChipsPerProcessBounds_Args* args);

struct PJRT_TpuTopology_ChipBounds_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  // The maximum dimension of coordinates that can be returned.
  // If the chip has more than max_coords dimensions, an error is returned.
  size_t chip_bounds_max_dims;
  // Points to an array of size max_dims. The bounds of the chip will be stored
  // in this array.
  int32_t* chip_bounds;
  size_t chip_bounds_num_dims;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ChipBounds_Args,
                          chip_bounds_num_dims);

// Returns the bounds of the chip in this topology.
typedef PJRT_Error* PJRT_TpuTopology_ChipBounds(
    PJRT_TpuTopology_ChipBounds_Args* args);

struct PJRT_TpuTopology_ProcessBounds_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  // The maximum dimension of coordinates that can be returned.
  // If the process has more than max_coords dimensions, an error is returned.
  size_t process_bounds_max_dims;
  // Points to an array of size max_dims. The bounds of the process will be
  // stored in this array.
  int32_t* process_bounds;
  size_t process_bounds_num_dims;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ProcessBounds_Args,
                          process_bounds_num_dims);

// Returns the bounds of the process in this topology.
typedef PJRT_Error* PJRT_TpuTopology_ProcessBounds(
    PJRT_TpuTopology_ProcessBounds_Args* args);

struct PJRT_TpuTopology_GetRoutingStrategy_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  char* routing_strategy;
  size_t routing_strategy_len;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_GetRoutingStrategy_Args,
                          routing_strategy_len);

// Returns the routing strategy as a string.
typedef PJRT_Error* PJRT_TpuTopology_GetRoutingStrategy(
    PJRT_TpuTopology_GetRoutingStrategy_Args* args);

typedef struct PJRT_TpuTopology_SliceConfig {
  size_t dim_size;
  int32_t dimensions[4];  // contains `dim_size` elements.
  bool wrap[4];           // contains `dim_size` elements.
  bool twist;
} PJRT_TpuTopology_SliceConfig;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_SliceConfig, twist);

struct PJRT_TpuTopology_GetSliceConfig_Args {
  size_t struct_size;
  const char* platform_type_name;
  size_t platform_type_name_len;
  const char* slice_name;
  size_t slice_name_len;
  PJRT_TpuTopology_SliceConfig* slice_config;  // in / out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_GetSliceConfig_Args, slice_config);

// Returns the slice config for the given platform and slice name.
typedef PJRT_Error* PJRT_TpuTopology_GetSliceConfig(
    PJRT_TpuTopology_GetSliceConfig_Args* args);

struct PJRT_TpuTopology_GetSliceConfigs_Args {
  size_t struct_size;
  const char* platform_type_name;
  size_t platform_type_name_len;
  PJRT_TpuTopology_SliceConfig* slice_configs;
  size_t max_slice_configs;
  size_t num_slice_configs;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_GetSliceConfigs_Args,
                          num_slice_configs);

// Returns all the slice configs for the given platform.
typedef PJRT_Error* PJRT_TpuTopology_GetSliceConfigs(
    PJRT_TpuTopology_GetSliceConfigs_Args* args);

struct PJRT_TpuTopology_GetDefaultPlatformConfig_Args {
  size_t struct_size;
  const char* platform_type_name;
  size_t platform_type_name_len;
  int64_t num_chips_per_tray;  // out
  int64_t num_trays;           // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_GetDefaultPlatformConfig_Args,
                          num_trays);

// Returns the default platform config for the given platform.
typedef PJRT_Error* PJRT_TpuTopology_GetDefaultPlatformConfig(
    PJRT_TpuTopology_GetDefaultPlatformConfig_Args* args);

typedef struct PJRT_TpuTopology_Extension {
  PJRT_Extension_Base base;
  PJRT_TpuTopology_Subslice* subslice;
  PJRT_TpuTopology_IsSubsliceTopology* is_subslice_topology;
  PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId*
      subslice_device_id_from_full_device_id;
  PJRT_TpuTopology_ReplaceHostBounds* replace_host_bounds;
  PJRT_TpuTopology_IsEnhancedBarrierEnabled* is_enhanced_barrier_enabled;
  PJRT_TpuTopology_HasLimitedIciConnectivity* has_limited_ici_connectivity;
  PJRT_TpuTopology_IsReachableOverLimitedIci* is_reachable_over_limited_ici;

  PJRT_TpuTopology_ProcessCount* process_count;
  PJRT_TpuTopology_ChipsPerProcess* chips_per_process;
  PJRT_TpuTopology_CoreCountPerChip* core_count_per_chip;
  PJRT_TpuTopology_ChipCount* chip_count;
  PJRT_TpuTopology_CoreCount* core_count;
  PJRT_TpuTopology_LogiDeviceCountPerProcess* logical_device_count_per_process;
  PJRT_TpuTopology_LogiDeviceCount* logical_device_count;
  PJRT_TpuTopology_LogiDeviceCountPerChip* logical_device_count_per_chip;
  PJRT_TpuTopology_CoreCountPerProcess* core_count_per_process;
  PJRT_TpuTopology_ProcessIds* process_ids;
  PJRT_TpuTopology_LogiDeviceIdsOnProcess* logical_device_ids_on_process;
  PJRT_TpuTopology_ProcIdAndIdxOnProcForChip* proc_id_and_idx_on_proc_for_chip;
  PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice*
      proc_id_and_idx_on_proc_for_logi_device;
  PJRT_TpuTopology_ProcessCoordFromId* process_coord_from_id;
  PJRT_TpuTopology_ChipIdFromCoord* chip_id_from_coord;
  PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx*
      logical_device_id_from_chip_coord_and_idx;
  PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice*
      chip_coord_and_idx_for_logi_device;
  PJRT_TpuTopology_ChipsPerProcessBounds* chips_per_process_bounds;
  PJRT_TpuTopology_ChipBounds* chip_bounds;
  PJRT_TpuTopology_ProcessBounds* process_bounds;
  PJRT_TpuTopology_GetRoutingStrategy* get_routing_strategy;
  PJRT_TpuTopology_GetSliceConfig* get_slice_config;
  PJRT_TpuTopology_GetSliceConfigs* get_slice_configs;
  PJRT_TpuTopology_GetDefaultPlatformConfig* get_default_platform_config;
} PJRT_TpuTopology_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_Extension,
                          get_default_platform_config);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_TPU_TOPOLOGY_EXTENSION_H_
