/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/utils.h"
#include "tensorflow/core/public/session_options.h"
#include "tsl/framework/device_id_utils.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace {
absl::StatusOr<xla::Shape> DeviceShapeRepresentation(
    const TensorShape& shape, DataType type, bool use_fast_memory,
    XlaLayoutPreference layout_preference) {
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(
      tensorflow::TensorShapeToXLAShape(type, shape, &xla_shape));
  ApiConverter::StackHelper<XLA_Shape> c_xla_shape(xla_shape);
  ApiConverter::StackHelper<XLA_Shape> c_device_shape;
  TF_Status* tf_status = TF_NewStatus();
  TfnpdApi()->TFNPD_XlaShapeToDeviceShapeRepresentation(
      &c_xla_shape.value, type, use_fast_memory,
      ConvertToCXlaLayoutPreference(layout_preference), &c_device_shape.value,
      tf_status);
  const Status status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  TF_RETURN_IF_ERROR(status);
  return c_device_shape.AsCpp<xla::Shape>();
}
}  // namespace

Status NextPluggableDeviceFactory::ListPhysicalDevices(
    std::vector<string>* devices) {
  TF_Status* c_status = TF_NewStatus();
  int32_t device_count = api_->TFNPD_GetDeviceCount(c_status);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status));
  TF_DeleteStatus(c_status);

  for (int i = 0; i < device_count; ++i) {
    const string device_name =
        absl::StrCat("/physical_device:", device_type_, ":", i);
    devices->push_back(device_name);
  }

  return absl::OkStatus();
}

Status NextPluggableDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const std::string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  TF_Status* c_status = TF_NewStatus();

  // Setup per-device states or resources that are internal to plugin.
  api_->TFNPD_InitPluginInternalDeviceStates(c_status);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status));

  const int32_t visible_device_count = api_->TFNPD_GetDeviceCount(c_status);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status));
  TF_DeleteStatus(c_status);

  if (visible_device_count <= 0) {
    return absl::OkStatus();
  }
  const absl::flat_hash_map<std::string, int64_t> device_count_map(
      session_options.config.device_count().begin(),
      session_options.config.device_count().end());
  const GPUOptions gpu_options = session_options.config.gpu_options();
  TF_ASSIGN_OR_RETURN(
      const size_t num_tf_devices,
      tsl::GetNumberTfDevicesAndConfigurePlatformDeviceId(
          device_count_map, device_type_, gpu_options.visible_device_list(),
          visible_device_count));

  if (!gpu_options.experimental().virtual_devices().empty()) {
    VLOG(2) << "NextPluggableDevice does not support virtual device setting.";
  }

  for (int i = 0; i < num_tf_devices; ++i) {
    NextPluggableDevice::Options options;
    options.device_name_prefix = name_prefix;
    options.device_name = device_type_;
    options.compilation_device_name = compilation_device_name_;
    options.device_ordinal = i;
    options.shape_determination_fns = {
        XlaShapeLayoutHelpers::ShapeDeterminationFns{
            UseNoPreferenceLayoutFn(), DeviceShapeRepresentation}};

    auto device =
        std::make_unique<NextPluggableDevice>(session_options, options);
    devices->push_back(std::move(device));
  }

  LOG(INFO) << "Created " << num_tf_devices
            << " TensorFlow NextPluggableDevices. "
            << "Physical device type: " << device_type_;
  return absl::OkStatus();
}

}  // namespace tensorflow
