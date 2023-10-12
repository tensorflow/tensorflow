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

#include "xla/service/gpu/gpu_target_config.h"

#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

GpuTargetConfig::GpuTargetConfig(stream_executor::StreamExecutor* s)
    : gpu_device_info(s->GetDeviceDescription().ToGpuProto()),
      platform_name(s->platform()->Name()) {}

GpuTargetConfig::GpuTargetConfig(
    const stream_executor::GpuTargetConfigProto& proto)
    : gpu_device_info({proto.gpu_device_info()}),
      platform_name(proto.platform_name()),
      dnn_version_info(proto.dnn_version_info()),
      device_description_str(proto.device_description_str()) {}

stream_executor::GpuTargetConfigProto GpuTargetConfig::ToProto() const {
  stream_executor::GpuTargetConfigProto proto;
  *proto.mutable_gpu_device_info() = gpu_device_info.ToGpuProto();
  proto.set_platform_name(platform_name);
  *proto.mutable_dnn_version_info() = dnn_version_info.ToProto();
  proto.set_device_description_str(device_description_str);
  return proto;
}

}  // namespace gpu
}  // namespace xla
