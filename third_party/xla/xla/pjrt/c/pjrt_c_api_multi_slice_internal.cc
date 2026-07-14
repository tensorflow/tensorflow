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

#include "xla/pjrt/c/pjrt_c_api_multi_slice_internal.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_multi_slice_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

struct PJRT_MultiSlice_SerializedConfig {
  std::string serialized_string;
};

struct PJRT_MultiSlice_NumDevicesPerSlice {
  std::vector<int32_t> slice_ids;
  std::vector<int32_t> num_devices;
};

namespace pjrt {

namespace {

void NumDevicesPerSliceDeleter(PJRT_MultiSlice_NumDevicesPerSlice* ptr) {
  delete ptr;
}

void SerializedConfigDeleter(PJRT_MultiSlice_SerializedConfig* ptr) {
  delete ptr;
}

PJRT_Error* ConfigDestroy(PJRT_MultiSlice_Config_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_MultiSlice_Config_Destroy_Args",
      PJRT_MultiSlice_Config_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->config;
  return nullptr;
}

PJRT_Error* ConfigNumSlices(PJRT_MultiSlice_Config_NumSlices_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_MultiSlice_Config_NumSlices_Args",
      PJRT_MultiSlice_Config_NumSlices_Args_STRUCT_SIZE, args->struct_size));
  args->num_slices = args->config->config->NumSlices();
  return nullptr;
}

PJRT_Error* ConfigSliceId(PJRT_MultiSlice_Config_SliceId_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_MultiSlice_Config_SliceId_Args",
      PJRT_MultiSlice_Config_SliceId_Args_STRUCT_SIZE, args->struct_size));
  args->slice_id = args->config->config->SliceId();
  return nullptr;
}

PJRT_Error* ConfigNumDevicesPerSlice(
    PJRT_MultiSlice_Config_NumDevicesPerSlice_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_MultiSlice_Config_NumDevicesPerSlice_Args",
      PJRT_MultiSlice_Config_NumDevicesPerSlice_Args_STRUCT_SIZE,
      args->struct_size));
  absl::flat_hash_map<int32_t, int32_t> devices_per_slice_map =
      args->config->config->NumDevicesPerSlice();
  args->num_devices_per_slice_map = devices_per_slice_map.size();

  auto devices_per_slice_map_ptr =
      std::make_unique<PJRT_MultiSlice_NumDevicesPerSlice>();
  devices_per_slice_map_ptr->slice_ids.resize(args->num_devices_per_slice_map);
  devices_per_slice_map_ptr->num_devices.resize(
      args->num_devices_per_slice_map);

  int i = 0;
  for (const auto& [slice_id, num_devices_val] : devices_per_slice_map) {
    devices_per_slice_map_ptr->slice_ids[i] = slice_id;
    devices_per_slice_map_ptr->num_devices[i] = num_devices_val;
    ++i;
  }

  args->slice_ids = devices_per_slice_map_ptr->slice_ids.data();
  args->num_devices = devices_per_slice_map_ptr->num_devices.data();
  args->devices_per_slice_map = devices_per_slice_map_ptr.release();
  args->devices_per_slice_map_deleter = NumDevicesPerSliceDeleter;
  return nullptr;
}

PJRT_Error* ConfigSerialize(PJRT_MultiSlice_Config_Serialize_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_MultiSlice_Config_Serialize_Args",
      PJRT_MultiSlice_Config_Serialize_Args_STRUCT_SIZE, args->struct_size));
  auto serialized_config = std::make_unique<PJRT_MultiSlice_SerializedConfig>();
  serialized_config->serialized_string = args->config->config->Serialize();

  args->serialized = serialized_config->serialized_string.c_str();
  args->size = serialized_config->serialized_string.size();
  args->serialized_config = serialized_config.release();
  args->serialized_config_deleter = SerializedConfigDeleter;
  return nullptr;
}

}  // namespace

PJRT_MultiSlice_Extension CreateMultiSliceExtension(PJRT_Extension_Base* next) {
  return PJRT_MultiSlice_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_MultiSlice_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_MultiSlice,
          /*next=*/next,
      },
      /*config_destroy=*/ConfigDestroy,
      /*config_num_slices=*/ConfigNumSlices,
      /*config_slice_id=*/ConfigSliceId,
      /*config_num_devices_per_slice=*/ConfigNumDevicesPerSlice,
      /*config_serialize=*/ConfigSerialize,
  };
}

}  // namespace pjrt
