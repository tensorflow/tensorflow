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
#ifndef TENSORFLOW_CORE_TFRT_COMMON_PJRT_CLIENT_FACTORY_OPTIONS_H_
#define TENSORFLOW_CORE_TFRT_COMMON_PJRT_CLIENT_FACTORY_OPTIONS_H_

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/gpu/gpu_helpers.h"

namespace xla {
// PjrtClientFactoryOptions store arguments to create PJRT client.
// Caller is responsible to set option value for corresponding PJRT client
// factory.
struct PjrtClientFactoryOptions {
  struct GpuClientCreateOptions {
    bool asynchronous = false;
    xla::GpuAllocatorConfig allocator_config = {};
    std::shared_ptr<xla::DistributedRuntimeClient> distributed_client = nullptr;
    int node_id = 0;
    std::optional<std::set<int>> allowed_devices = std::nullopt;
    std::optional<std::string> platform_name = std::nullopt;
  };

  struct CpuClientCreateOptions {
    bool asynchronous = false;
  };
  GpuClientCreateOptions gpu_options;
  CpuClientCreateOptions cpu_options;
};
}  // namespace xla

#endif  // TENSORFLOW_CORE_TFRT_COMMON_PJRT_CLIENT_FACTORY_OPTIONS_H_
