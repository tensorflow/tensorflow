/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_GPU_TARGET_CONFIG_H_
#define XLA_SERVICE_GPU_GPU_TARGET_CONFIG_H_

#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

struct GpuTargetConfig {
  GpuTargetConfig() = default;
  explicit GpuTargetConfig(const stream_executor::GpuTargetConfigProto& proto);
  explicit GpuTargetConfig(stream_executor::StreamExecutor* s);

  stream_executor::GpuTargetConfigProto ToProto() const;

  stream_executor::DeviceDescription gpu_device_info;
  std::string platform_name;
  stream_executor::dnn::VersionInfo dnn_version_info;
  std::string device_description_str;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_TARGET_CONFIG_H_
