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

#include "xla/pjrt/c_api_client/pjrt_c_api_multi_slice_config.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_multi_slice_extension.h"

namespace pjrt {

PjRtCApiMultiSliceConfig::~PjRtCApiMultiSliceConfig() {
  PJRT_MultiSlice_Config_Destroy_Args args;
  args.struct_size = PJRT_MultiSlice_Config_Destroy_Args_STRUCT_SIZE;
  args.config = config_;

  std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error(
      extension_->config_destroy(&args), pjrt::MakeErrorDeleter(c_api_));
  if (error != nullptr) {
    LOG(ERROR) << "Failed to delete PjRtCApiMultiSliceConfig: "
               << pjrt::PjrtErrorToStatus(error.get(), c_api_);
  }
}

int32_t PjRtCApiMultiSliceConfig::NumSlices() const {
  PJRT_MultiSlice_Config_NumSlices_Args args;
  args.struct_size = PJRT_MultiSlice_Config_NumSlices_Args_STRUCT_SIZE;
  args.config = config_;
  args.num_slices = 0;
  CHECK(extension_->config_num_slices(&args) == nullptr);
  return args.num_slices;
}

int32_t PjRtCApiMultiSliceConfig::SliceId() const {
  PJRT_MultiSlice_Config_SliceId_Args args;
  args.struct_size = PJRT_MultiSlice_Config_SliceId_Args_STRUCT_SIZE;
  args.config = config_;
  args.slice_id = 0;
  CHECK(extension_->config_slice_id(&args) == nullptr);
  return args.slice_id;
}

absl::flat_hash_map<int32_t, int32_t>
PjRtCApiMultiSliceConfig::NumDevicesPerSlice() const {
  PJRT_MultiSlice_Config_NumDevicesPerSlice_Args args;
  args.struct_size = PJRT_MultiSlice_Config_NumDevicesPerSlice_Args_STRUCT_SIZE;
  args.config = config_;
  CHECK(extension_->config_num_devices_per_slice(&args) == nullptr);

  absl::flat_hash_map<int32_t, int32_t> result;
  for (size_t i = 0; i < args.num_devices_per_slice_map; ++i) {
    result[args.slice_ids[i]] = args.num_devices[i];
  }

  args.devices_per_slice_map_deleter(args.devices_per_slice_map);

  return result;
}

std::string PjRtCApiMultiSliceConfig::Serialize() const {
  CHECK(extension_ != nullptr);

  PJRT_MultiSlice_Config_Serialize_Args args;
  args.struct_size = PJRT_MultiSlice_Config_Serialize_Args_STRUCT_SIZE;
  args.config = config_;
  CHECK(extension_->config_serialize(&args) == nullptr);

  std::string result(args.serialized, args.size);
  args.serialized_config_deleter(args.serialized_config);

  return result;
}

}  // namespace pjrt
